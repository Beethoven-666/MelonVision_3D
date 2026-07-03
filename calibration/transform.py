from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


class Transform:
    def __init__(self, R: np.ndarray, t: np.ndarray) -> None:
        self.R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(t, dtype=np.float64).reshape(3)

    def point_camera_to_base(self, p_camera: np.ndarray) -> np.ndarray:
        p_camera = np.asarray(p_camera, dtype=np.float64).reshape(3)
        return self.R @ p_camera + self.t

    def points_camera_to_base(self, points_camera: np.ndarray) -> np.ndarray:
        points_camera = np.asarray(points_camera, dtype=np.float64)
        return (self.R @ points_camera.T).T + self.t

    def vector_camera_to_base(self, v_camera: np.ndarray) -> np.ndarray:
        v_camera = np.asarray(v_camera, dtype=np.float64).reshape(3)
        v_base = self.R @ v_camera
        norm = np.linalg.norm(v_base)
        return v_base / norm if norm > 1e-9 else v_base


def identity_transform() -> Transform:
    return Transform(np.eye(3), np.zeros(3))


def load_transform_from_yaml(path: str | Path | None) -> Transform:
    if path is None:
        return identity_transform()

    config_path = Path(path)
    if not config_path.exists():
        return identity_transform()

    data: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    rotation = data.get("rotation_matrix", np.eye(3).tolist())
    translation = data.get("translation_m", [0.0, 0.0, 0.0])
    if isinstance(translation, dict):
        translation = [translation.get("x", 0.0), translation.get("y", 0.0), translation.get("z", 0.0)]
    return Transform(np.asarray(rotation, dtype=np.float64), np.asarray(translation, dtype=np.float64))
