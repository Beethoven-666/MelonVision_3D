from __future__ import annotations

import time
from typing import Any

import numpy as np

from calibration.transform import Transform
from geometry.pose_estimator import estimate_pose_by_pca
from geometry.suction_grasp import choose_suction_grasp
from perception.pointcloud_builder import mask_depth_to_points
from perception.watermelon_segmenter import WatermelonSegmenter
from robot.dual_arm_planner import DualArmInjectionPlanner
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
        robot_config: dict[str, Any] | None = None,
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
        self.grasp_mode = "injection"
        self.tool_normal_base = _unit_vector(np.asarray(tool_normal_base, dtype=np.float64).reshape(3))
        self.robot_command_builder = robot_command_builder or InjectionRobotCommandBuilder()
        self.robot_config = robot_config or self.robot_command_builder.config
        self.dual_arm_planner = DualArmInjectionPlanner(
            transform=self.transform,
            command_builder=self.robot_command_builder,
            config=self.robot_config,
            pregrasp_offset_m=self.pregrasp_offset_m,
        )
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
        for target_id, detection in enumerate(detections, start=1):
            candidate = self._evaluate_detection(target_id, detection, depth_mm, intrinsic)
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
        dual_arm_plan = self.dual_arm_planner.plan(candidates)
        primary_target_id = int(dual_arm_plan["selected_targets"][0]["target_id"])
        best = next(candidate for candidate in candidates if int(candidate["target_id"]) == primary_target_id)
        detection = best["detection"]
        points_camera = best["points_camera"]
        pose = best["pose"]
        volume = best["volume"]
        center_base = best["center_base"]
        primary_grasp = dual_arm_plan["primary_grasp"]
        contact_base = _point_from_dict(primary_grasp["contact_point_base_m"])
        normal_base = _point_from_dict(primary_grasp["surface_normal_base"])
        pregrasp_base = _point_from_dict(primary_grasp["pregrasp_point_base_m"])
        approach_base = -normal_base
        robot_command = dual_arm_plan["robot_command"]

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
                "grasp_confidence": float(primary_grasp["score"]),
                "target_selection_score": float(best["target_score"]),
                "target_selection": best["target_score_breakdown"],
                "center_base_m": to_point3d(center_base),
                "camera_depth_m": float(best["camera_depth_m"]),
                "camera_distance_m": float(best["camera_distance_m"]),
                "axes_m": pose["axes_length_m"],
                "volume": volume,
                "predicted_weight_kg": float(best["predicted_weight_kg"]),
                "grasp": {
                    "contact_point_base_m": to_point3d(contact_base),
                    "surface_normal_base": to_point3d(normal_base),
                    "approach_vector_base": to_point3d(approach_base),
                    "pregrasp_point_base_m": to_point3d(pregrasp_base),
                    "pregrasp_offset_m": self.pregrasp_offset_m,
                    "score": float(primary_grasp["score"]),
                    "contact_pixel": None,
                    "score_breakdown": primary_grasp.get("score_breakdown", {}),
                    "method": primary_grasp.get("method", "unknown"),
                    "is_inferred": bool(primary_grasp.get("is_inferred", False)),
                    "orientation_enabled": bool(primary_grasp.get("orientation_enabled", False)),
                    "orientation_axis": primary_grasp.get("orientation_axis"),
                    "orientation_angle_deg": primary_grasp.get("orientation_angle_deg"),
                    "orientation_was_clipped": bool(primary_grasp.get("orientation_was_clipped", False)),
                },
                "grasp_mode": self.grasp_mode,
                "robot_command": robot_command,
                "dual_arm_plan": dual_arm_plan,
            },
        }
        debug.update(
            {
                "detection": detection,
                "mask": detection["mask"],
                "points_camera": points_camera,
                "pose": pose,
                "grasp": best["visible_grasp"],
                "dual_arm_plan": dual_arm_plan,
                "volume": volume,
                "result": result,
            }
        )
        return result, debug

    def _evaluate_detection(
        self,
        target_id: int,
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
        center_camera = np.asarray(pose["center_camera"], dtype=np.float64).reshape(3)
        pose_confidence = self._pose_confidence(points_camera.shape[0])
        return {
            "status": "ok",
            "target_id": target_id,
            "detection": detection,
            "points_camera": points_camera,
            "pose": pose,
            "visible_grasp": grasp,
            "volume": volume,
            "center_base": center_base,
            "camera_depth_m": float(center_camera[2]),
            "camera_distance_m": float(np.linalg.norm(center_camera)),
            "pose_confidence": pose_confidence,
            "point_count": int(points_camera.shape[0]),
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
        depths = [float(item["camera_depth_m"]) for item in candidates]
        closest_depth = min(depths)
        depth_range = max(depths) - closest_depth
        same_height_mode = len(candidates) > 1 and max(heights) - min(heights) <= self.same_height_band_m
        for item in candidates:
            camera_depth_m = float(item["camera_depth_m"])
            if depth_range <= 1e-6:
                depth_priority_score = 1.0
            else:
                depth_priority_score = float(np.clip(1.0 - (camera_depth_m - closest_depth) / depth_range, 0.0, 1.0))
            target_score, target_score_breakdown = self._target_selection_score(
                detection=item["detection"],
                grasp=item["visible_grasp"],
                point_count=int(item["point_count"]),
                center_base=item["center_base"],
                camera_depth_m=camera_depth_m,
                camera_distance_m=float(item["camera_distance_m"]),
                depth_priority_score=depth_priority_score,
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
        camera_depth_m: float,
        camera_distance_m: float,
        depth_priority_score: float,
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
                "depth": 0.70,
                "grasp": 0.12,
                "detection": 0.06,
                "pose": 0.04,
                "point_count": 0.04,
                "robot_distance": 0.04,
            }
        else:
            weights = {
                "depth": 0.75,
                "grasp": 0.10,
                "detection": 0.05,
                "pose": 0.04,
                "point_count": 0.03,
                "robot_distance": 0.03,
            }

        score = (
            weights["depth"] * depth_priority_score
            + weights["grasp"] * grasp_score
            + weights["detection"] * detection_score
            + weights["pose"] * pose_confidence
            + weights["point_count"] * point_count_score
            + weights["robot_distance"] * robot_distance_score
        )
        return float(score), {
            "depth_priority": float(depth_priority_score),
            "camera_depth_m": float(camera_depth_m),
            "camera_distance_m": float(camera_distance_m),
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


def _point_from_dict(point: dict[str, float]) -> np.ndarray:
    return np.array([float(point["x"]), float(point["y"]), float(point["z"])], dtype=np.float64)
