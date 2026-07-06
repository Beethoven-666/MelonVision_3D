from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot.orientation_axis import (
    DEFAULT_ORIENTATION_AXES_CONFIG,
    ORIENTATION_FIELD_ORDER,
    active_axis_order,
    compute_orientation_axes,
    is_orientation_axis,
    orientation_axes_enabled,
)


AXIS_ORDER = ("x", "y1", "y2", "z1", "z2")
FIELD_ORDER = ("position_mm", "velocity_mm_s", "acceleration_mm_s2", "jerk_mm_s3")
ARM_AXIS_MAP = {
    "arm1": {"y": "y1", "z": "z1"},
    "arm2": {"y": "y2", "z": "z2"},
}


DEFAULT_CONFIG: dict[str, Any] = {
    "robot_type": "five_axis_injection_molding",
    "axis_order": list(AXIS_ORDER),
    "field_order": list(FIELD_ORDER),
    "coordinate_unit": "mm",
    "axis_mapping": {
        "x": {"source": "x", "offset_mm": 0.0, "min_mm": -9999.0, "max_mm": 9999.0},
        "y1": {"source": "y", "offset_mm": 0.0, "min_mm": -9999.0, "max_mm": 9999.0},
        "y2": {"source": "y", "offset_mm": 0.0, "min_mm": -9999.0, "max_mm": 9999.0},
        "z1": {"source": "z", "offset_mm": 0.0, "min_mm": -9999.0, "max_mm": 9999.0},
        "z2": {"source": "z", "offset_mm": 0.0, "min_mm": -9999.0, "max_mm": 9999.0},
    },
    "idle_axis_positions_mm": {
        "x": 0.0,
        "y1": 0.0,
        "y2": 0.0,
        "z1": 0.0,
        "z2": 0.0,
    },
    "motion_profile": {
        "x": {"velocity_mm_s": 300.0, "acceleration_mm_s2": 1000.0, "jerk_mm_s3": 5000.0},
        "y1": {"velocity_mm_s": 300.0, "acceleration_mm_s2": 1000.0, "jerk_mm_s3": 5000.0},
        "y2": {"velocity_mm_s": 300.0, "acceleration_mm_s2": 1000.0, "jerk_mm_s3": 5000.0},
        "z1": {"velocity_mm_s": 200.0, "acceleration_mm_s2": 800.0, "jerk_mm_s3": 4000.0},
        "z2": {"velocity_mm_s": 200.0, "acceleration_mm_s2": 800.0, "jerk_mm_s3": 4000.0},
        "r1": {"velocity_deg_s": 90.0, "acceleration_deg_s2": 180.0, "jerk_deg_s3": 500.0},
        "r2": {"velocity_deg_s": 90.0, "acceleration_deg_s2": 180.0, "jerk_deg_s3": 500.0},
    },
    "planning": {
        "density_kg_per_liter": 0.95,
        "dual_arm_weight_threshold_kg": 4.0,
        "x_shared_tolerance_mm": 80.0,
        "single_target_two_arm_min_score": 0.30,
    },
    "side_grasp": {
        "arm1_normal_base": [0.0, -1.0, 0.0],
        "arm2_normal_base": [0.0, 1.0, 0.0],
    },
    "orientation_axes": DEFAULT_ORIENTATION_AXES_CONFIG,
    "plc_registers": {
        "scale": 10.0,
        "round_digits": 0,
        "write_order": "axis_major",
        "start_address": 0,
    },
}


class InjectionRobotCommandBuilder:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = _deep_merge(DEFAULT_CONFIG, config or {})
        self.linear_axis_order = tuple(
            axis for axis in self.config.get("axis_order", AXIS_ORDER) if not is_orientation_axis(axis)
        )
        self.axis_order = active_axis_order(self.config, self.linear_axis_order)
        self.field_order = tuple(self.config.get("field_order", FIELD_ORDER))

    def build_from_grasp(
        self,
        contact_point_base_m: np.ndarray,
        pregrasp_point_base_m: np.ndarray | None = None,
    ) -> dict[str, Any]:
        point_mm = _point_m_to_mm(contact_point_base_m)
        approach_vector_base = _approach_from_pregrasp(contact_point_base_m, pregrasp_point_base_m)
        surface_normal_base = -approach_vector_base if approach_vector_base is not None else None
        arm_assignments = {
            "arm1": _assignment_from_point("arm1", point_mm, None, "legacy_single_grasp"),
            "arm2": _idle_assignment("arm2"),
        }
        arm_assignments["arm1"]["approach_vector_base"] = (
            _point_dict(approach_vector_base) if approach_vector_base is not None else None
        )
        arm_assignments["arm1"]["surface_normal_base"] = (
            _point_dict(surface_normal_base) if surface_normal_base is not None else None
        )
        return self.build_from_arm_assignments(
            arm_assignments=arm_assignments,
            plan_type="single_arm_single_target",
            plan_summary="Legacy single-grasp command; arm2 is idle.",
        )

    def build_from_arm_assignments(
        self,
        arm_assignments: dict[str, dict[str, Any]],
        plan_type: str,
        plan_summary: str,
    ) -> dict[str, Any]:
        arm_assignments = copy.deepcopy(arm_assignments)
        orientation_axes = compute_orientation_axes(arm_assignments, self.config)
        self._apply_orientation_to_arm_assignments(arm_assignments, orientation_axes)
        axis_values = self._axis_values_from_arm_assignments(arm_assignments, orientation_axes)
        register_values = self._register_values(axis_values)
        return {
            "robot_type": self.config.get("robot_type", "five_axis_injection_molding"),
            "coordinate_unit": "mixed_mm_deg" if orientation_axes_enabled(self.config) else "mm",
            "plan_type": plan_type,
            "plan_summary": plan_summary,
            "axis_order": list(self.axis_order),
            "field_order": list(self.field_order),
            "field_order_by_axis": {
                axis: list(self._field_order_for_axis(axis))
                for axis in self.axis_order
            },
            "arm_assignments": arm_assignments,
            "axis_values": axis_values,
            "axis_positions": self._axis_positions(axis_values),
            "register_values": register_values,
            "register_count": len(register_values),
            "plc_register_start": int(self.config.get("plc_registers", {}).get("start_address", 0)),
            "orientation_enabled": orientation_axes_enabled(self.config),
            "orientation_axes": orientation_axes,
            "is_command_valid": True,
        }

    def _axis_values_from_arm_assignments(
        self,
        arm_assignments: dict[str, dict[str, Any]],
        orientation_axes: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, float]]:
        axis_positions = dict(self.config.get("idle_axis_positions_mm", {}))
        active_points = []

        for arm_id, assignment in arm_assignments.items():
            if not assignment.get("enabled", False):
                continue
            point_mm = np.asarray(assignment["contact_point_mm_array"], dtype=np.float64).reshape(3)
            active_points.append(point_mm)
            axes = ARM_AXIS_MAP[arm_id]
            axis_positions[axes["y"]] = float(point_mm[1])
            axis_positions[axes["z"]] = float(point_mm[2])

        if active_points:
            axis_positions["x"] = float(np.mean([point[0] for point in active_points]))

        axis_values: dict[str, dict[str, float]] = {}
        profile = self.config["motion_profile"]
        mapping = self.config["axis_mapping"]
        for axis in self.axis_order:
            if is_orientation_axis(axis):
                axis_values[axis] = self._orientation_axis_values_from_result(axis, orientation_axes)
                continue

            axis_cfg = mapping[axis]
            raw_position = float(axis_positions.get(axis, self.config["idle_axis_positions_mm"].get(axis, 0.0)))
            raw_position += float(axis_cfg.get("offset_mm", 0.0))
            min_mm = float(axis_cfg.get("min_mm", -9999.0))
            max_mm = float(axis_cfg.get("max_mm", 9999.0))
            clipped_position = float(np.clip(raw_position, min_mm, max_mm))
            axis_profile = profile[axis]
            axis_values[axis] = {
                "position_mm": clipped_position,
                "velocity_mm_s": float(axis_profile["velocity_mm_s"]),
                "acceleration_mm_s2": float(axis_profile["acceleration_mm_s2"]),
                "jerk_mm_s3": float(axis_profile["jerk_mm_s3"]),
                "was_clipped": bool(abs(clipped_position - raw_position) > 1e-6),
            }
        return axis_values

    def _register_values(self, axis_values: dict[str, dict[str, float]]) -> list[int]:
        register_cfg = self.config.get("plc_registers", {})
        round_digits = int(register_cfg.get("round_digits", 0))
        values: list[int] = []
        for axis in self.axis_order:
            scale = self._register_scale_for_axis(axis)
            for field in self._field_order_for_axis(axis):
                scaled = axis_values[axis][field] * scale
                values.append(int(round(scaled, round_digits)))
        return values

    def _orientation_axis_values_from_result(
        self,
        axis: str,
        orientation_axes: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        orientation = orientation_axes.get(axis, {})
        if not orientation:
            orientation = {
                "enabled": False,
                "angle_deg": float(
                    self.config.get("orientation_axes", {}).get(axis, {}).get("idle_deg", 0.0)
                ),
                "was_clipped": False,
                "source_arm": "arm1" if axis == "r1" else "arm2",
            }
        axis_profile = self.config["motion_profile"][axis]
        return {
            "position_deg": float(orientation.get("angle_deg", 0.0)),
            "velocity_deg_s": float(axis_profile["velocity_deg_s"]),
            "acceleration_deg_s2": float(axis_profile["acceleration_deg_s2"]),
            "jerk_deg_s3": float(axis_profile["jerk_deg_s3"]),
            "was_clipped": bool(orientation.get("was_clipped", False)),
            "enabled": bool(orientation.get("enabled", False)),
            "source_arm": str(orientation.get("source_arm", "")),
        }

    def _field_order_for_axis(self, axis: str) -> tuple[str, ...]:
        if is_orientation_axis(axis):
            return ORIENTATION_FIELD_ORDER
        return self.field_order

    def _register_scale_for_axis(self, axis: str) -> float:
        register_cfg = self.config.get("plc_registers", {})
        fallback = float(register_cfg.get("scale", 1.0))
        if is_orientation_axis(axis):
            return float(register_cfg.get("angular_scale", 100.0))
        return float(register_cfg.get("linear_scale", fallback))

    def _axis_positions(self, axis_values: dict[str, dict[str, Any]]) -> dict[str, float]:
        positions: dict[str, float] = {}
        for axis, values in axis_values.items():
            if is_orientation_axis(axis):
                positions[axis] = float(values.get("position_deg", 0.0))
            else:
                positions[axis] = float(values.get("position_mm", 0.0))
        return positions

    def _apply_orientation_to_arm_assignments(
        self,
        arm_assignments: dict[str, dict[str, Any]],
        orientation_axes: dict[str, dict[str, Any]],
    ) -> None:
        for assignment in arm_assignments.values():
            assignment.setdefault("orientation_enabled", False)
            assignment.setdefault("orientation_axis", None)
            assignment.setdefault("orientation_angle_deg", None)
            assignment.setdefault("orientation_was_clipped", False)

        for axis, orientation in orientation_axes.items():
            source_arm = str(orientation.get("source_arm", ""))
            assignment = arm_assignments.get(source_arm)
            if assignment is None:
                continue
            assignment["orientation_axis"] = axis
            assignment["orientation_enabled"] = bool(orientation.get("enabled", False))
            assignment["orientation_angle_deg"] = (
                float(orientation["angle_deg"]) if bool(orientation.get("enabled", False)) else None
            )
            assignment["orientation_was_clipped"] = bool(orientation.get("was_clipped", False))


def build_injection_robot_command(
    contact_point_base_m: np.ndarray,
    pregrasp_point_base_m: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return InjectionRobotCommandBuilder(config).build_from_grasp(
        contact_point_base_m=contact_point_base_m,
        pregrasp_point_base_m=pregrasp_point_base_m,
    )


def load_injection_robot_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return DEFAULT_CONFIG
    config_path = Path(path)
    if not config_path.exists():
        return DEFAULT_CONFIG
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return _deep_merge(DEFAULT_CONFIG, data)


def make_arm_assignment(
    arm_id: str,
    contact_point_base_m: np.ndarray,
    target_id: int | None,
    method: str,
    score: float,
    is_inferred: bool,
    weight_kg: float | None = None,
    surface_normal_base: np.ndarray | None = None,
    approach_vector_base: np.ndarray | None = None,
) -> dict[str, Any]:
    point_mm = _point_m_to_mm(contact_point_base_m)
    assignment = _assignment_from_point(arm_id, point_mm, target_id, method)
    assignment.update(
        {
            "score": float(score),
            "is_inferred": bool(is_inferred),
            "predicted_weight_kg": float(weight_kg) if weight_kg is not None else None,
            "surface_normal_base": _point_dict(surface_normal_base) if surface_normal_base is not None else None,
            "approach_vector_base": _point_dict(approach_vector_base) if approach_vector_base is not None else None,
        }
    )
    return assignment


def make_idle_assignment(arm_id: str) -> dict[str, Any]:
    return _idle_assignment(arm_id)


def _assignment_from_point(
    arm_id: str,
    point_mm: np.ndarray,
    target_id: int | None,
    method: str,
) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "enabled": True,
        "target_id": target_id,
        "method": method,
        "contact_point_base_m": _point_dict(point_mm / 1000.0),
        "contact_point_mm": _point_dict(point_mm),
        "contact_point_mm_array": point_mm.tolist(),
        "surface_normal_base": None,
        "approach_vector_base": None,
        "orientation_enabled": False,
        "orientation_axis": None,
        "orientation_angle_deg": None,
        "orientation_was_clipped": False,
    }


def _idle_assignment(arm_id: str) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "enabled": False,
        "target_id": None,
        "method": "idle",
        "contact_point_base_m": None,
        "contact_point_mm": None,
        "contact_point_mm_array": [0.0, 0.0, 0.0],
        "surface_normal_base": None,
        "approach_vector_base": None,
        "orientation_enabled": False,
        "orientation_axis": None,
        "orientation_angle_deg": None,
        "orientation_was_clipped": False,
        "score": 0.0,
        "is_inferred": False,
        "predicted_weight_kg": None,
    }


def _point_m_to_mm(point_m: np.ndarray | None) -> np.ndarray:
    if point_m is None:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(point_m, dtype=np.float64).reshape(3) * 1000.0


def _approach_from_pregrasp(
    contact_point_base_m: np.ndarray,
    pregrasp_point_base_m: np.ndarray | None,
) -> np.ndarray | None:
    if pregrasp_point_base_m is None:
        return None
    contact = np.asarray(contact_point_base_m, dtype=np.float64).reshape(3)
    pregrasp = np.asarray(pregrasp_point_base_m, dtype=np.float64).reshape(3)
    vector = contact - pregrasp
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return None
    return vector / norm


def _point_dict(point_mm: np.ndarray) -> dict[str, float]:
    return {
        "x": float(point_mm[0]),
        "y": float(point_mm[1]),
        "z": float(point_mm[2]),
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, dict):
            result[key] = _deep_merge(value, {})
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
