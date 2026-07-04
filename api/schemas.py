from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Point3D(BaseModel):
    x: float
    y: float
    z: float


class Vector3D(BaseModel):
    x: float
    y: float
    z: float


class VolumeResult(BaseModel):
    volume_m3: float
    volume_liter: float
    method: str
    confidence: float


class GraspResult(BaseModel):
    contact_point_base_m: Point3D
    surface_normal_base: Vector3D
    approach_vector_base: Vector3D
    pregrasp_point_base_m: Point3D
    pregrasp_offset_m: float
    score: float
    contact_pixel: Optional[list[int]] = None
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    method: str = "unknown"
    is_inferred: bool = False


class WatermelonTarget(BaseModel):
    track_id: int
    frame_id: str
    class_name: str
    detection_confidence: float
    pose_confidence: float
    grasp_confidence: float
    target_selection_score: float
    target_selection: dict[str, float] = Field(default_factory=dict)
    center_base_m: Point3D
    axes_m: dict[str, float]
    volume: VolumeResult
    predicted_weight_kg: float
    grasp: GraspResult
    grasp_mode: str = "injection"
    robot_command: dict[str, object] = Field(default_factory=dict)
    dual_arm_plan: dict[str, object] = Field(default_factory=dict)


class BestTargetResponse(BaseModel):
    status: str
    timestamp: float
    camera_id: str
    target: Optional[WatermelonTarget] = None
    message: Optional[str] = None
