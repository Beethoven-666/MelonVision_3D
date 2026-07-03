from __future__ import annotations

import numpy as np


def mask_depth_to_points(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    intrinsic: dict[str, float],
    min_depth_m: float = 0.2,
    max_depth_m: float = 3.0,
    max_points: int = 20000,
) -> np.ndarray:
    if depth_mm.shape[:2] != mask.shape[:2]:
        raise ValueError("depth_mm and mask must have the same height and width.")

    fx = float(intrinsic["fx"])
    fy = float(intrinsic["fy"])
    cx = float(intrinsic["cx"])
    cy = float(intrinsic["cy"])

    depth_m = depth_mm.astype(np.float32) / 1000.0
    valid = (mask > 0) & (depth_m > min_depth_m) & (depth_m < max_depth_m)
    v, u = np.where(valid)
    if v.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    if max_points > 0 and v.size > max_points:
        step = int(np.ceil(v.size / max_points))
        v = v[::step]
        u = u[::step]

    z = depth_m[v, u]
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)
