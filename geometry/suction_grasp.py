from __future__ import annotations

from typing import Any

import cv2
import numpy as np


DEFAULT_GRASP_WEIGHTS = {
    "front": 0.25,
    "edge": 0.20,
    "flatness": 0.20,
    "normal": 0.20,
    "depth_stability": 0.15,
}

DEFAULT_SIDE_GRASP_PRIOR = 0.70


def estimate_local_plane_normal(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    normal = eigvecs[:, 0]
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    return center, normal, eigvals


def choose_suction_grasp(
    points_camera: np.ndarray,
    camera_origin: np.ndarray | None = None,
    pregrasp_offset_m: float = 0.08,
    depth_mm: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    intrinsic: dict[str, float] | None = None,
    desired_normal_camera: np.ndarray | None = None,
    suction_radius_px: int = 18,
    local_radius_px: int = 9,
    max_candidates: int = 180,
) -> dict[str, object]:
    if points_camera.shape[0] < 200:
        raise ValueError("Too few points to calculate a suction grasp.")

    if camera_origin is None:
        camera_origin = np.zeros(3)

    if depth_mm is not None and mask is not None and intrinsic is not None:
        try:
            return choose_suction_grasp_from_mask(
                depth_mm=depth_mm,
                mask=mask,
                intrinsic=intrinsic,
                camera_origin=camera_origin,
                desired_normal_camera=desired_normal_camera,
                pregrasp_offset_m=pregrasp_offset_m,
                suction_radius_px=suction_radius_px,
                local_radius_px=local_radius_px,
                max_candidates=max_candidates,
            )
        except ValueError:
            pass

    z = points_camera[:, 2]
    z_threshold = np.percentile(z, 10)
    visible_surface = points_camera[z <= z_threshold]
    if visible_surface.shape[0] < 50:
        visible_surface = points_camera

    contact, normal, eigvals = estimate_local_plane_normal(visible_surface)
    to_camera = camera_origin - contact
    if np.dot(normal, to_camera) < 0:
        normal = -normal

    pregrasp = contact + normal * pregrasp_offset_m
    curvature = _surface_curvature(eigvals)
    return {
        "contact_point_camera": contact,
        "surface_normal_camera": normal,
        "pregrasp_point_camera": pregrasp,
        "pregrasp_offset_m": float(pregrasp_offset_m),
        "score": 0.55,
        "contact_pixel": None,
        "candidate_count": int(visible_surface.shape[0]),
        "score_breakdown": {
            "front": 0.55,
            "edge": 0.0,
            "flatness": _flatness_score(curvature),
            "normal": _normal_alignment_score(normal, _unit_vector(to_camera)),
            "depth_stability": 0.5,
            "curvature": float(curvature),
        },
        "method": "visible_surface_fallback",
    }


def choose_ellipsoid_side_grasp(
    pose: dict[str, object],
    side_direction_camera: np.ndarray,
    pregrasp_offset_m: float = 0.08,
    min_radius_m: float = 0.06,
    max_radius_m: float = 0.20,
) -> dict[str, object]:
    center = np.asarray(pose["center_camera"], dtype=np.float64).reshape(3)
    rotation = np.asarray(pose["R_camera_object"], dtype=np.float64).reshape(3, 3)
    axes_length = pose["axes_length_m"]
    radii = np.array(
        [
            float(axes_length["major"]) / 2.0,
            float(axes_length["middle"]) / 2.0,
            float(axes_length["minor"]) / 2.0,
        ],
        dtype=np.float64,
    )
    radii = np.clip(radii, min_radius_m, max_radius_m)

    desired_normal = _unit_vector(side_direction_camera)
    if np.linalg.norm(desired_normal) <= 1e-9:
        raise ValueError("Invalid side direction for ellipsoid side grasp.")

    normal_local = rotation.T @ desired_normal
    denom = np.sqrt(np.sum((radii * normal_local) ** 2))
    if denom <= 1e-9:
        raise ValueError("Unable to project side direction onto ellipsoid.")

    point_local = (radii**2 * normal_local) / denom
    contact = center + rotation @ point_local
    normal = _unit_vector(rotation @ (point_local / np.maximum(radii**2, 1e-9)))
    if np.dot(normal, desired_normal) < 0:
        normal = -normal

    pregrasp = contact + normal * pregrasp_offset_m
    axis_balance = _ellipsoid_axis_balance_score(radii)
    side_alignment = _normal_alignment_score(normal, desired_normal)
    size_prior = _ellipsoid_size_prior_score(radii, min_radius_m, max_radius_m)
    inferred_surface_penalty = 0.78
    model_quality = (
        0.45 * side_alignment
        + 0.25 * axis_balance
        + 0.20 * size_prior
        + 0.10 * inferred_surface_penalty
    )
    score = DEFAULT_SIDE_GRASP_PRIOR * model_quality

    return {
        "contact_point_camera": contact,
        "surface_normal_camera": normal,
        "pregrasp_point_camera": pregrasp,
        "pregrasp_offset_m": float(pregrasp_offset_m),
        "score": float(np.clip(score, 0.0, 1.0)),
        "contact_pixel": None,
        "candidate_count": 1,
        "score_breakdown": {
            "side_alignment": float(side_alignment),
            "axis_balance": float(axis_balance),
            "size_prior": float(size_prior),
            "inferred_surface_penalty": float(inferred_surface_penalty),
            "model_quality": float(model_quality),
            "r_major_m": float(radii[0]),
            "r_middle_m": float(radii[1]),
            "r_minor_m": float(radii[2]),
            "model_prior": DEFAULT_SIDE_GRASP_PRIOR,
        },
        "method": "ellipsoid_model_side_grasp",
        "is_inferred": True,
    }


def select_grasp_by_mode(
    visible_grasp: dict[str, object] | None,
    side_grasp: dict[str, object] | None,
    grasp_mode: str,
    side_min_score: float = 0.35,
    auto_side_bonus: float = 0.12,
) -> dict[str, object]:
    if grasp_mode == "visible":
        if visible_grasp is None:
            raise ValueError("Visible grasp is not available.")
        return visible_grasp

    if grasp_mode == "side":
        if side_grasp is None:
            raise ValueError("Side grasp is not available.")
        if float(side_grasp.get("score", 0.0)) < side_min_score:
            raise ValueError("Side grasp score is below the minimum threshold.")
        return side_grasp

    if grasp_mode != "auto":
        raise ValueError(f"Unsupported grasp mode: {grasp_mode}")

    if visible_grasp is None and side_grasp is None:
        raise ValueError("No grasp candidate is available.")
    if visible_grasp is None:
        return side_grasp
    if side_grasp is None:
        return visible_grasp

    visible_score = float(visible_grasp.get("score", 0.0))
    side_score = float(side_grasp.get("score", 0.0))
    if side_score >= side_min_score and side_score + auto_side_bonus >= visible_score:
        return side_grasp
    return visible_grasp


def choose_suction_grasp_from_mask(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    intrinsic: dict[str, float],
    camera_origin: np.ndarray | None = None,
    desired_normal_camera: np.ndarray | None = None,
    pregrasp_offset_m: float = 0.08,
    suction_radius_px: int = 18,
    local_radius_px: int = 9,
    max_candidates: int = 180,
    min_local_points: int = 35,
    weights: dict[str, float] | None = None,
) -> dict[str, object]:
    if depth_mm.shape[:2] != mask.shape[:2]:
        raise ValueError("depth_mm and mask must have the same height and width.")

    if camera_origin is None:
        camera_origin = np.zeros(3)
    if weights is None:
        weights = DEFAULT_GRASP_WEIGHTS

    valid_depth = depth_mm > 0
    valid_mask = ((mask > 0) & valid_depth).astype(np.uint8)
    if int(valid_mask.sum()) < min_local_points:
        raise ValueError("Too few valid mask pixels to score suction candidates.")

    edge_distance = cv2.distanceTransform(valid_mask, cv2.DIST_L2, 5)
    candidate_mask = (valid_mask > 0) & (edge_distance >= max(2, suction_radius_px // 2))
    if int(candidate_mask.sum()) < min_local_points:
        candidate_mask = valid_mask > 0

    candidate_pixels = _sample_candidate_pixels(
        candidate_mask=candidate_mask,
        depth_mm=depth_mm,
        edge_distance=edge_distance,
        max_candidates=max_candidates,
    )
    if not candidate_pixels:
        raise ValueError("No valid suction candidates were found.")

    valid_depth_values_m = depth_mm[valid_mask > 0].astype(np.float32) / 1000.0
    z_min = float(np.percentile(valid_depth_values_m, 2))
    z_max = float(np.percentile(valid_depth_values_m, 98))
    z_range = max(z_max - z_min, 1e-6)

    best: dict[str, Any] | None = None
    evaluated_count = 0
    for u, v in candidate_pixels:
        local_points, local_depth_m = _local_points_from_depth(
            depth_mm=depth_mm,
            mask=valid_mask,
            intrinsic=intrinsic,
            center_u=u,
            center_v=v,
            radius_px=local_radius_px,
        )
        if local_points.shape[0] < min_local_points:
            continue

        contact, normal, eigvals = estimate_local_plane_normal(local_points)
        to_camera = camera_origin - contact
        if np.dot(normal, to_camera) < 0:
            normal = -normal

        desired_normal = _unit_vector(desired_normal_camera) if desired_normal_camera is not None else _unit_vector(to_camera)
        curvature = _surface_curvature(eigvals)
        front = _front_score(float(contact[2]), z_min, z_range)
        edge = _edge_score(float(edge_distance[v, u]), suction_radius_px)
        flatness = _flatness_score(curvature)
        normal_score = _normal_alignment_score(normal, desired_normal)
        depth_stability = _depth_stability_score(local_depth_m)
        total = (
            weights["front"] * front
            + weights["edge"] * edge
            + weights["flatness"] * flatness
            + weights["normal"] * normal_score
            + weights["depth_stability"] * depth_stability
        )

        evaluated_count += 1
        candidate = {
            "total": float(total),
            "contact": contact,
            "normal": normal,
            "pixel": (int(u), int(v)),
            "breakdown": {
                "front": float(front),
                "edge": float(edge),
                "flatness": float(flatness),
                "normal": float(normal_score),
                "depth_stability": float(depth_stability),
                "curvature": float(curvature),
                "edge_distance_px": float(edge_distance[v, u]),
                "local_depth_std_m": float(np.std(local_depth_m)) if local_depth_m.size else 0.0,
                "local_points": int(local_points.shape[0]),
            },
        }
        if best is None or candidate["total"] > best["total"]:
            best = candidate

    if best is None:
        raise ValueError("No suction candidate had enough local 3D points.")

    pregrasp = best["contact"] + best["normal"] * pregrasp_offset_m
    return {
        "contact_point_camera": best["contact"],
        "surface_normal_camera": best["normal"],
        "pregrasp_point_camera": pregrasp,
        "pregrasp_offset_m": float(pregrasp_offset_m),
        "score": float(best["total"]),
        "contact_pixel": best["pixel"],
        "candidate_count": int(evaluated_count),
        "score_breakdown": best["breakdown"],
        "method": "candidate_score_visible_point_cloud",
    }


def _sample_candidate_pixels(
    candidate_mask: np.ndarray,
    depth_mm: np.ndarray,
    edge_distance: np.ndarray,
    max_candidates: int,
) -> list[tuple[int, int]]:
    ys, xs = np.where(candidate_mask)
    if ys.size == 0:
        return []

    depth_m = depth_mm[ys, xs].astype(np.float32) / 1000.0
    edge = edge_distance[ys, xs].astype(np.float32)
    front_rank = 1.0 - _normalize_01(depth_m)
    edge_rank = _normalize_01(edge)
    pre_score = 0.65 * front_rank + 0.35 * edge_rank
    order = np.argsort(pre_score)[::-1]
    if max_candidates > 0:
        order = order[:max_candidates]
    return [(int(xs[i]), int(ys[i])) for i in order]


def _local_points_from_depth(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    intrinsic: dict[str, float],
    center_u: int,
    center_v: int,
    radius_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_mm.shape[:2]
    x_min = max(center_u - radius_px, 0)
    x_max = min(center_u + radius_px + 1, width)
    y_min = max(center_v - radius_px, 0)
    y_max = min(center_v + radius_px + 1, height)

    local_depth = depth_mm[y_min:y_max, x_min:x_max]
    local_mask = mask[y_min:y_max, x_min:x_max] > 0
    valid = local_mask & (local_depth > 0)
    vv, uu = np.where(valid)
    if vv.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    uu = uu + x_min
    vv = vv + y_min
    z = depth_mm[vv, uu].astype(np.float32) / 1000.0
    x = (uu.astype(np.float32) - float(intrinsic["cx"])) * z / float(intrinsic["fx"])
    y = (vv.astype(np.float32) - float(intrinsic["cy"])) * z / float(intrinsic["fy"])
    return np.stack([x, y, z], axis=1), z


def _front_score(z_m: float, z_min: float, z_range: float) -> float:
    return float(np.clip(1.0 - (z_m - z_min) / z_range, 0.0, 1.0))


def _edge_score(edge_distance_px: float, suction_radius_px: int) -> float:
    required = max(float(suction_radius_px), 1.0)
    return float(np.clip(edge_distance_px / required, 0.0, 1.0))


def _flatness_score(curvature: float) -> float:
    return float(np.clip(1.0 - curvature / 0.06, 0.0, 1.0))


def _normal_alignment_score(normal: np.ndarray, desired_normal: np.ndarray) -> float:
    alignment = float(np.dot(_unit_vector(normal), _unit_vector(desired_normal)))
    return float(np.clip((alignment + 1.0) * 0.5, 0.0, 1.0))


def _depth_stability_score(depth_m: np.ndarray) -> float:
    if depth_m.size == 0:
        return 0.0
    return float(np.clip(1.0 - float(np.std(depth_m)) / 0.015, 0.0, 1.0))


def _surface_curvature(eigvals: np.ndarray) -> float:
    eigvals = np.maximum(np.asarray(eigvals, dtype=np.float64), 0.0)
    total = float(np.sum(eigvals))
    if total <= 1e-12:
        return 0.0
    return float(eigvals[0] / total)


def _ellipsoid_axis_balance_score(radii: np.ndarray) -> float:
    radii = np.asarray(radii, dtype=np.float64)
    min_radius = float(np.min(radii))
    max_radius = float(np.max(radii))
    if max_radius <= 1e-9:
        return 0.0
    ratio = min_radius / max_radius
    return float(np.clip(0.45 + 0.55 * ratio, 0.0, 1.0))


def _ellipsoid_size_prior_score(radii: np.ndarray, min_radius_m: float, max_radius_m: float) -> float:
    radii = np.asarray(radii, dtype=np.float64)
    center = (min_radius_m + max_radius_m) / 2.0
    half_range = max((max_radius_m - min_radius_m) / 2.0, 1e-6)
    normalized_error = np.abs(radii - center) / half_range
    return float(np.clip(1.0 - 0.25 * np.mean(normalized_error), 0.55, 1.0))


def _unit_vector(vector: np.ndarray | None) -> np.ndarray:
    if vector is None:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(vector)
    if norm <= 1e-9:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return vector / norm


def _normalize_01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low <= 1e-9:
        return np.ones_like(values, dtype=np.float32) * 0.5
    return (values - low) / (high - low)
