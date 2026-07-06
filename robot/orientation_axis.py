from __future__ import annotations

import math
from typing import Any

import numpy as np


ORIENTATION_AXIS_IDS = ("r1", "r2")
ORIENTATION_FIELD_ORDER = (
    "position_deg",
    "velocity_deg_s",
    "acceleration_deg_s2",
    "jerk_deg_s3",
)


DEFAULT_ORIENTATION_AXES_CONFIG: dict[str, Any] = {
    "enabled": False,
    "axis_order_when_enabled": ["x", "y1", "y2", "z1", "z2", "r1", "r2"],
    "unit": "deg",
    "r1": {
        "enabled": True,
        "source_arm": "arm1",
        "rotation_axis_base": [0.0, 0.0, 1.0],
        "zero_direction_base": [1.0, 0.0, 0.0],
        "angle_offset_deg": 0.0,
        "min_deg": -90.0,
        "max_deg": 90.0,
        "idle_deg": 0.0,
    },
    "r2": {
        "enabled": True,
        "source_arm": "arm2",
        "rotation_axis_base": [0.0, 0.0, 1.0],
        "zero_direction_base": [1.0, 0.0, 0.0],
        "angle_offset_deg": 0.0,
        "min_deg": -90.0,
        "max_deg": 90.0,
        "idle_deg": 0.0,
    },
}


def orientation_axes_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("orientation_axes", {}).get("enabled", False))


def active_axis_order(config: dict[str, Any], fallback_axis_order: tuple[str, ...]) -> tuple[str, ...]:
    orientation_cfg = config.get("orientation_axes", {})
    if not orientation_axes_enabled(config):
        return fallback_axis_order
    return tuple(orientation_cfg.get("axis_order_when_enabled", list(fallback_axis_order) + list(ORIENTATION_AXIS_IDS)))


def is_orientation_axis(axis: str) -> bool:
    return axis in ORIENTATION_AXIS_IDS


def compute_orientation_axes(
    arm_assignments: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not orientation_axes_enabled(config):
        return {}

    orientation_cfg = config.get("orientation_axes", {})
    results: dict[str, dict[str, Any]] = {}
    for axis_id in ORIENTATION_AXIS_IDS:
        axis_cfg = orientation_cfg.get(axis_id, {})
        source_arm = str(axis_cfg.get("source_arm", "arm1" if axis_id == "r1" else "arm2"))
        assignment = arm_assignments.get(source_arm, {})
        results[axis_id] = _compute_single_axis(axis_id, axis_cfg, source_arm, assignment)
    return results


def _compute_single_axis(
    axis_id: str,
    axis_cfg: dict[str, Any],
    source_arm: str,
    assignment: dict[str, Any],
) -> dict[str, Any]:
    idle_deg = float(axis_cfg.get("idle_deg", 0.0))
    base_result = {
        "axis": axis_id,
        "enabled": False,
        "source_arm": source_arm,
        "angle_deg": idle_deg,
        "raw_angle_deg": None,
        "angle_offset_deg": float(axis_cfg.get("angle_offset_deg", 0.0)),
        "min_deg": float(axis_cfg.get("min_deg", -90.0)),
        "max_deg": float(axis_cfg.get("max_deg", 90.0)),
        "idle_deg": idle_deg,
        "was_clipped": False,
        "rotation_axis_base": _point_dict(_unit_vector(axis_cfg.get("rotation_axis_base", [0.0, 0.0, 1.0]))),
        "zero_direction_base": _point_dict(_unit_vector(axis_cfg.get("zero_direction_base", [1.0, 0.0, 0.0]))),
        "desired_direction_base": None,
        "reason": None,
    }

    if not bool(axis_cfg.get("enabled", True)):
        base_result["reason"] = "axis_config_disabled"
        return base_result
    if not bool(assignment.get("enabled", False)):
        base_result["reason"] = "source_arm_idle"
        return base_result

    desired_direction = _assignment_approach_direction(assignment)
    if desired_direction is None:
        base_result["reason"] = "missing_approach_direction"
        return base_result

    rotation_axis = _unit_vector(axis_cfg.get("rotation_axis_base", [0.0, 0.0, 1.0]))
    zero_direction = _unit_vector(axis_cfg.get("zero_direction_base", [1.0, 0.0, 0.0]))
    angle_offset = float(axis_cfg.get("angle_offset_deg", 0.0))
    min_deg = float(axis_cfg.get("min_deg", -90.0))
    max_deg = float(axis_cfg.get("max_deg", 90.0))

    try:
        raw_angle = _signed_projected_angle_deg(
            desired_direction=desired_direction,
            rotation_axis=rotation_axis,
            zero_direction=zero_direction,
        )
    except ValueError as exc:
        base_result["reason"] = str(exc)
        return base_result

    commanded_angle = raw_angle + angle_offset
    clipped_angle = float(np.clip(commanded_angle, min_deg, max_deg))
    base_result.update(
        {
            "enabled": True,
            "angle_deg": clipped_angle,
            "raw_angle_deg": float(raw_angle),
            "was_clipped": bool(abs(clipped_angle - commanded_angle) > 1e-9),
            "desired_direction_base": _point_dict(_unit_vector(desired_direction)),
            "reason": "ok",
        }
    )
    return base_result


def _assignment_approach_direction(assignment: dict[str, Any]) -> np.ndarray | None:
    approach = _vector_from_value(assignment.get("approach_vector_base"))
    if approach is not None:
        return approach
    normal = _vector_from_value(assignment.get("surface_normal_base"))
    if normal is not None:
        return -normal
    return None


def _signed_projected_angle_deg(
    desired_direction: np.ndarray,
    rotation_axis: np.ndarray,
    zero_direction: np.ndarray,
) -> float:
    axis = _unit_vector(rotation_axis)
    desired = _project_to_plane(_unit_vector(desired_direction), axis)
    zero = _project_to_plane(_unit_vector(zero_direction), axis)
    if desired is None:
        raise ValueError("desired_direction_parallel_to_rotation_axis")
    if zero is None:
        raise ValueError("zero_direction_parallel_to_rotation_axis")

    cross = np.cross(zero, desired)
    sin_value = float(np.dot(axis, cross))
    cos_value = float(np.dot(zero, desired))
    return math.degrees(math.atan2(sin_value, cos_value))


def _project_to_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    projected = vector - normal * float(np.dot(vector, normal))
    norm = float(np.linalg.norm(projected))
    if norm <= 1e-9:
        return None
    return projected / norm


def _vector_from_value(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict):
        vector = np.array(
            [float(value.get("x", 0.0)), float(value.get("y", 0.0)), float(value.get("z", 0.0))],
            dtype=np.float64,
        )
    else:
        vector = np.asarray(value, dtype=np.float64).reshape(3)
    if np.linalg.norm(vector) <= 1e-9:
        return None
    return vector


def _unit_vector(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vector / norm


def _point_dict(vector: np.ndarray) -> dict[str, float]:
    return {
        "x": float(vector[0]),
        "y": float(vector[1]),
        "z": float(vector[2]),
    }
