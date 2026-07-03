from __future__ import annotations

import numpy as np


def estimate_local_plane_normal(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, np.argmin(eigvals)]
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    return center, normal


def choose_suction_grasp(
    points_camera: np.ndarray,
    camera_origin: np.ndarray | None = None,
    pregrasp_offset_m: float = 0.08,
) -> dict[str, object]:
    if points_camera.shape[0] < 200:
        raise ValueError("Too few points to calculate a suction grasp.")

    if camera_origin is None:
        camera_origin = np.zeros(3)

    z = points_camera[:, 2]
    z_threshold = np.percentile(z, 10)
    visible_surface = points_camera[z <= z_threshold]
    if visible_surface.shape[0] < 50:
        visible_surface = points_camera

    contact, normal = estimate_local_plane_normal(visible_surface)
    to_camera = camera_origin - contact
    if np.dot(normal, to_camera) < 0:
        normal = -normal

    pregrasp = contact + normal * pregrasp_offset_m
    return {
        "contact_point_camera": contact,
        "surface_normal_camera": normal,
        "pregrasp_point_camera": pregrasp,
        "pregrasp_offset_m": float(pregrasp_offset_m),
        "score": 0.75,
    }
