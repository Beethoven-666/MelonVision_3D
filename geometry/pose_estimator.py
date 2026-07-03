from __future__ import annotations

import numpy as np


def estimate_pose_by_pca(points_camera: np.ndarray) -> dict[str, object]:
    if points_camera.shape[0] < 100:
        raise ValueError("Too few valid points to estimate pose.")

    q_low = np.percentile(points_camera, 2, axis=0)
    q_high = np.percentile(points_camera, 98, axis=0)
    valid = np.all((points_camera >= q_low) & (points_camera <= q_high), axis=1)
    pts = points_camera[valid]
    if pts.shape[0] < 100:
        raise ValueError("Too few points remain after outlier filtering.")

    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1

    projected = centered @ eigvecs
    p_low = np.percentile(projected, 2, axis=0)
    p_high = np.percentile(projected, 98, axis=0)
    axes_length = p_high - p_low
    center_local = (p_low + p_high) / 2.0
    center_camera = mean + eigvecs @ center_local

    return {
        "center_camera": center_camera,
        "R_camera_object": eigvecs,
        "axes_length_m": {
            "major": float(axes_length[0]),
            "middle": float(axes_length[1]),
            "minor": float(axes_length[2]),
        },
    }
