from __future__ import annotations

import time
from typing import Any

import numpy as np

from calibration.transform import Transform
from geometry.pose_estimator import estimate_pose_by_pca
from geometry.suction_grasp import choose_suction_grasp
from perception.pointcloud_builder import mask_depth_to_points
from perception.watermelon_segmenter import WatermelonSegmenter
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
        segmenter: WatermelonSegmenter | None = None,
    ) -> None:
        self.transform = transform
        self.camera_id = camera_id
        self.min_points = min_points
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.pregrasp_offset_m = pregrasp_offset_m
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

        detection = max(detections, key=lambda item: float(item.get("area", 0.0)))
        mask = detection["mask"]
        points_camera = mask_depth_to_points(
            depth_mm,
            mask,
            intrinsic,
            self.min_depth_m,
            self.max_depth_m,
        )
        debug.update({"detection": detection, "mask": mask, "points_camera": points_camera})

        if points_camera.shape[0] < self.min_points:
            return (
                self._status(
                    "low_confidence",
                    timestamp,
                    f"Too few valid 3D points: {points_camera.shape[0]}.",
                ),
                debug,
            )

        try:
            pose = estimate_pose_by_pca(points_camera)
            grasp = choose_suction_grasp(
                points_camera,
                pregrasp_offset_m=self.pregrasp_offset_m,
            )
            volume = estimate_volume_by_ellipsoid(pose["axes_length_m"])
        except ValueError as exc:
            return self._status("low_confidence", timestamp, str(exc)), debug

        center_base = self.transform.point_camera_to_base(pose["center_camera"])
        contact_base = self.transform.point_camera_to_base(grasp["contact_point_camera"])
        normal_base = self.transform.vector_camera_to_base(grasp["surface_normal_camera"])
        pregrasp_base = self.transform.point_camera_to_base(grasp["pregrasp_point_camera"])
        approach_base = -normal_base

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
                "pose_confidence": 0.75,
                "grasp_confidence": float(grasp["score"]),
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
                },
            },
        }
        debug.update({"pose": pose, "grasp": grasp, "volume": volume, "result": result})
        return result, debug

    def _status(self, status: str, timestamp: float, message: str) -> dict[str, Any]:
        return {
            "status": status,
            "timestamp": timestamp,
            "camera_id": self.camera_id,
            "target": None,
            "message": message,
        }
