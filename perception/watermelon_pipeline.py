from __future__ import annotations

import time
from typing import Any

import numpy as np

from calibration.transform import Transform
from geometry.pose_estimator import estimate_pose_by_pca
from geometry.suction_grasp import choose_suction_grasp
from perception.pointcloud_builder import mask_depth_to_points
from perception.watermelon_segmenter import WatermelonSegmenter
from robot.injection_molding_robot import InjectionRobotCommandBuilder
from volume.volume_estimator import estimate_volume_by_ellipsoid


def to_point3d(point: np.ndarray) -> dict[str, float]:
    return {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}


class WatermelonVisionProcessor:
    def __init__(
        self,
        transform: Transform,
        camera_id: str = "gemini_435le_01",
        min_points: int = 300,
        min_depth_m: float = 0.2,
        max_depth_m: float = 3.0,
        pregrasp_offset_m: float = 0.08,
        robot_origin_base: tuple[float, float, float] = (0.0, 0.0, 0.0),
        robot_distance_norm_m: float = 1.5,
        same_height_band_m: float = 0.08,
        grasp_mode: str = "injection",
        tool_normal_base: tuple[float, float, float] = (0.0, 0.0, 1.0),
        robot_command_builder: InjectionRobotCommandBuilder | None = None,
        segmenter: WatermelonSegmenter | None = None,
    ) -> None:
        self.transform = transform
        self.camera_id = camera_id
        self.min_points = min_points
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.pregrasp_offset_m = pregrasp_offset_m
        self.robot_origin_base = np.asarray(robot_origin_base, dtype=np.float64).reshape(3)
        self.robot_distance_norm_m = robot_distance_norm_m
        self.same_height_band_m = same_height_band_m
        self.grasp_mode = "injection" if grasp_mode in ("injection", "visible") else "injection"
        self.tool_normal_base = _unit_vector(np.asarray(tool_normal_base, dtype=np.float64).reshape(3))
        self.robot_command_builder = robot_command_builder or InjectionRobotCommandBuilder()
        self.segmenter = segmenter or WatermelonSegmenter(
            depth_min_m=min_depth_m,
            depth_max_m=max_depth_m,
        )
        self.track_id = 0

    def process(
        self,
        color_bgr: np.ndarray,
        depth_mm: np.ndarray,
        intrinsic: dict[str, float],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        timestamp = time.time()
        detections = self.segmenter.segment(color_bgr, depth_mm)
        debug: dict[str, Any] = {"detections": detections}

        if not detections:
            return self._status("no_target", timestamp, "No watermelon mask was detected."), debug

        candidates = []
        low_point_counts: list[int] = []
        errors: list[str] = []
        for detection in detections:
            candidate = self._evaluate_detection(detection, depth_mm, intrinsic)
            if candidate.get("status") == "ok":
                candidates.append(candidate)
                continue
            if "point_count" in candidate:
                low_point_counts.append(int(candidate["point_count"]))
            if "message" in candidate:
                errors.append(str(candidate["message"]))

        debug["target_candidates"] = candidates
        if not candidates:
            best_count = max(low_point_counts) if low_point_counts else 0
            return (
                self._status(
                    "low_confidence",
                    timestamp,
                    f"No target passed grasp scoring. Best valid point count: {best_count}.",
                ),
                debug,
            )

        self._score_target_candidates(candidates)
        best = max(candidates, key=lambda item: float(item["target_score"]))
        detection = best["detection"]
        points_camera = best["points_camera"]
        pose = best["pose"]
        grasp = best["grasp"]
        volume = best["volume"]
        center_base = best["center_base"]
        contact_base = best["contact_base"]
        normal_base = best["normal_base"]
        pregrasp_base = best["pregrasp_base"]
        approach_base = -normal_base
        robot_command = best["robot_command"]

        self.track_id += 1
        result = {
            "status": "ok",
            "timestamp": timestamp,
            "camera_id": self.camera_id,
            "target": {
                "track_id": self.track_id,
                "frame_id": "robot_base",
                "class_name": str(detection["class_name"]),
                "detection_confidence": float(detection["score"]),
                "pose_confidence": float(best["pose_confidence"]),
                "grasp_confidence": float(grasp["score"]),
                "target_selection_score": float(best["target_score"]),
                "target_selection": best["target_score_breakdown"],
                "center_base_m": to_point3d(center_base),
                "axes_m": pose["axes_length_m"],
                "volume": volume,
                "grasp": {
                    "contact_point_base_m": to_point3d(contact_base),
                    "surface_normal_base": to_point3d(normal_base),
                    "approach_vector_base": to_point3d(approach_base),
                    "pregrasp_point_base_m": to_point3d(pregrasp_base),
                    "pregrasp_offset_m": float(grasp["pregrasp_offset_m"]),
                    "score": float(grasp["score"]),
                    "contact_pixel": grasp.get("contact_pixel"),
                    "score_breakdown": grasp.get("score_breakdown", {}),
                    "method": grasp.get("method", "unknown"),
                    "is_inferred": bool(grasp.get("is_inferred", False)),
                },
                "grasp_mode": self.grasp_mode,
                "robot_command": robot_command,
            },
        }
        debug.update(
            {
                "detection": detection,
                "mask": detection["mask"],
                "points_camera": points_camera,
                "pose": pose,
                "grasp": grasp,
                "volume": volume,
                "result": result,
            }
        )
        return result, debug

    def _evaluate_detection(
        self,
        detection: dict[str, Any],
        depth_mm: np.ndarray,
        intrinsic: dict[str, float],
    ) -> dict[str, Any]:
        mask = detection["mask"]
        points_camera = mask_depth_to_points(
            depth_mm,
            mask,
            intrinsic,
            self.min_depth_m,
            self.max_depth_m,
        )

        if points_camera.shape[0] < self.min_points:
            return {
                "status": "low_confidence",
                "point_count": int(points_camera.shape[0]),
                "message": "Too few valid 3D points.",
            }

        try:
            pose = estimate_pose_by_pca(points_camera)
            grasp = choose_suction_grasp(
                points_camera,
                pregrasp_offset_m=self.pregrasp_offset_m,
                depth_mm=depth_mm,
                mask=mask,
                intrinsic=intrinsic,
                desired_normal_camera=self._tool_normal_camera(),
            )
            volume = estimate_volume_by_ellipsoid(pose["axes_length_m"])
        except ValueError as exc:
            return {
                "status": "low_confidence",
                "point_count": int(points_camera.shape[0]),
                "message": str(exc),
            }

        center_base = self.transform.point_camera_to_base(pose["center_camera"])
        contact_base = self.transform.point_camera_to_base(grasp["contact_point_camera"])
        normal_base = self.transform.vector_camera_to_base(grasp["surface_normal_camera"])
        pregrasp_base = self.transform.point_camera_to_base(grasp["pregrasp_point_camera"])
        pose_confidence = self._pose_confidence(points_camera.shape[0])
        robot_command = self.robot_command_builder.build_from_grasp(
            contact_point_base_m=contact_base,
            pregrasp_point_base_m=pregrasp_base,
        )
        return {
            "status": "ok",
            "detection": detection,
            "points_camera": points_camera,
            "pose": pose,
            "grasp": grasp,
            "volume": volume,
            "center_base": center_base,
            "contact_base": contact_base,
            "normal_base": normal_base,
            "pregrasp_base": pregrasp_base,
            "pose_confidence": pose_confidence,
            "point_count": int(points_camera.shape[0]),
            "robot_command": robot_command,
            "target_score": 0.0,
            "target_score_breakdown": {},
        }

    def _tool_normal_camera(self) -> np.ndarray:
        normal_camera = self.transform.R.T @ self.tool_normal_base
        return _unit_vector(normal_camera)

    def _score_target_candidates(self, candidates: list[dict[str, Any]]) -> None:
        if not candidates:
            return

        heights = [float(item["center_base"][2]) for item in candidates]
        same_height_mode = len(candidates) > 1 and max(heights) - min(heights) <= self.same_height_band_m
        for item in candidates:
            target_score, target_score_breakdown = self._target_selection_score(
                detection=item["detection"],
                grasp=item["grasp"],
                point_count=int(item["point_count"]),
                center_base=item["center_base"],
                pose_confidence=float(item["pose_confidence"]),
                same_height_mode=same_height_mode,
            )
            item["target_score"] = target_score
            item["target_score_breakdown"] = target_score_breakdown

    def _target_selection_score(
        self,
        detection: dict[str, Any],
        grasp: dict[str, Any],
        point_count: int,
        center_base: np.ndarray,
        pose_confidence: float,
        same_height_mode: bool,
    ) -> tuple[float, dict[str, float]]:
        detection_score = float(detection.get("score", 0.0))
        grasp_score = float(grasp.get("score", 0.0))
        point_count_score = float(np.clip(point_count / 6000.0, 0.0, 1.0))

        horizontal_distance = float(np.linalg.norm(center_base[:2] - self.robot_origin_base[:2]))
        robot_distance_score = float(
            np.clip(1.0 - horizontal_distance / max(self.robot_distance_norm_m, 1e-6), 0.0, 1.0)
        )

        if same_height_mode:
            weights = {
                "grasp": 0.42,
                "detection": 0.16,
                "pose": 0.10,
                "point_count": 0.10,
                "robot_distance": 0.22,
            }
        else:
            weights = {
                "grasp": 0.50,
                "detection": 0.18,
                "pose": 0.12,
                "point_count": 0.10,
                "robot_distance": 0.10,
            }

        score = (
            weights["grasp"] * grasp_score
            + weights["detection"] * detection_score
            + weights["pose"] * pose_confidence
            + weights["point_count"] * point_count_score
            + weights["robot_distance"] * robot_distance_score
        )
        return float(score), {
            "grasp": float(grasp_score),
            "detection": float(detection_score),
            "pose": float(pose_confidence),
            "point_count": float(point_count_score),
            "robot_distance": float(robot_distance_score),
            "horizontal_distance_to_robot_m": horizontal_distance,
            "same_height_mode": float(same_height_mode),
        }

    @staticmethod
    def _pose_confidence(point_count: int) -> float:
        return float(np.clip(point_count / 5000.0, 0.35, 0.9))

    def _status(self, status: str, timestamp: float, message: str) -> dict[str, Any]:
        return {
            "status": status,
            "timestamp": timestamp,
            "camera_id": self.camera_id,
            "target": None,
            "message": message,
        }


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vector / norm
