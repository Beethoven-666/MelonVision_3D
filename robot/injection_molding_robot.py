from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


AXIS_ORDER = ("x", "y1", "y2", "z1", "z2")
FIELD_ORDER = ("position_mm", "velocity_mm_s", "acceleration_mm_s2", "jerk_mm_s3")


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
    "motion_profile": {
        "x": {"velocity_mm_s": 300.0, "acceleration_mm_s2": 1000.0, "jerk_mm_s3": 5000.0},
        "y1": {"velocity_mm_s": 300.0, "acceleration_mm_s2": 1000.0, "jerk_mm_s3": 5000.0},
        "y2": {"velocity_mm_s": 300.0, "acceleration_mm_s2": 1000.0, "jerk_mm_s3": 5000.0},
        "z1": {"velocity_mm_s": 200.0, "acceleration_mm_s2": 800.0, "jerk_mm_s3": 4000.0},
        "z2": {"velocity_mm_s": 200.0, "acceleration_mm_s2": 800.0, "jerk_mm_s3": 4000.0},
    },
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
        self.axis_order = tuple(self.config.get("axis_order", AXIS_ORDER))
        self.field_order = tuple(self.config.get("field_order", FIELD_ORDER))

    def build_from_grasp(
        self,
        contact_point_base_m: np.ndarray,
        pregrasp_point_base_m: np.ndarray | None = None,
    ) -> dict[str, Any]:
        contact_mm = _point_m_to_mm(contact_point_base_m)
        pregrasp_mm = _point_m_to_mm(pregrasp_point_base_m) if pregrasp_point_base_m is not None else None
        axis_values = self._axis_values(contact_mm)
        register_values = self._register_values(axis_values)
        return {
            "robot_type": self.config.get("robot_type", "five_axis_injection_molding"),
            "coordinate_unit": "mm",
            "axis_order": list(self.axis_order),
            "field_order": list(self.field_order),
            "axis_values": axis_values,
            "register_values": register_values,
            "register_count": len(register_values),
            "plc_register_start": int(self.config.get("plc_registers", {}).get("start_address", 0)),
            "contact_point_mm": _point_dict(contact_mm),
            "pregrasp_point_mm": _point_dict(pregrasp_mm) if pregrasp_mm is not None else None,
            "is_command_valid": True,
        }

    def _axis_values(self, point_mm: np.ndarray) -> dict[str, dict[str, float]]:
        axis_values: dict[str, dict[str, float]] = {}
        mapping = self.config["axis_mapping"]
        profile = self.config["motion_profile"]
        for axis in self.axis_order:
            axis_cfg = mapping[axis]
            position = _source_value(point_mm, axis_cfg.get("source", axis))
            position += float(axis_cfg.get("offset_mm", 0.0))
            min_mm = float(axis_cfg.get("min_mm", -9999.0))
            max_mm = float(axis_cfg.get("max_mm", 9999.0))
            clipped_position = float(np.clip(position, min_mm, max_mm))
            axis_profile = profile[axis]
            axis_values[axis] = {
                "position_mm": clipped_position,
                "velocity_mm_s": float(axis_profile["velocity_mm_s"]),
                "acceleration_mm_s2": float(axis_profile["acceleration_mm_s2"]),
                "jerk_mm_s3": float(axis_profile["jerk_mm_s3"]),
                "was_clipped": bool(abs(clipped_position - position) > 1e-6),
            }
        return axis_values

    def _register_values(self, axis_values: dict[str, dict[str, float]]) -> list[int]:
        register_cfg = self.config.get("plc_registers", {})
        scale = float(register_cfg.get("scale", 1.0))
        round_digits = int(register_cfg.get("round_digits", 0))
        values: list[int] = []
        for axis in self.axis_order:
            for field in self.field_order:
                scaled = axis_values[axis][field] * scale
                values.append(int(round(scaled, round_digits)))
        return values


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


def _point_m_to_mm(point_m: np.ndarray | None) -> np.ndarray:
    if point_m is None:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(point_m, dtype=np.float64).reshape(3) * 1000.0


def _point_dict(point_mm: np.ndarray) -> dict[str, float]:
    return {
        "x": float(point_mm[0]),
        "y": float(point_mm[1]),
        "z": float(point_mm[2]),
    }


def _source_value(point_mm: np.ndarray, source: str) -> float:
    source = source.lower()
    if source == "x":
        return float(point_mm[0])
    if source == "y":
        return float(point_mm[1])
    if source == "z":
        return float(point_mm[2])
    raise ValueError(f"Unsupported axis source: {source}")


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
