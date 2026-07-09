from __future__ import annotations

from typing import Any

import numpy as np

from calibration.transform import Transform
from geometry.suction_grasp import choose_ellipsoid_side_grasp
from robot.injection_molding_robot import (
    InjectionRobotCommandBuilder,
    make_arm_assignment,
    make_idle_assignment,
)


class DualArmInjectionPlanner:
    def __init__(
        self,
        transform: Transform,
        command_builder: InjectionRobotCommandBuilder,
        config: dict[str, Any],
        pregrasp_offset_m: float = 0.08,
    ) -> None:
        self.transform = transform
        self.command_builder = command_builder
        self.config = config
        self.pregrasp_offset_m = pregrasp_offset_m
        self.planning = config.get("planning", {})
        self.side_grasp = config.get("side_grasp", {})

    def plan(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if not candidates:
            raise ValueError("No target candidates are available for dual-arm planning.")

        enriched = sorted(
            (self._with_weight_and_side_grasps(candidate) for candidate in candidates),
            key=self._distance_priority_key,
        )
        nearest = enriched[0]
        if nearest["predicted_weight_kg"] >= self._dual_arm_weight_threshold_kg():
            return self._plan_two_arms_one_target(nearest)

        light_candidates = [
            candidate
            for candidate in enriched
            if candidate["predicted_weight_kg"] < self._dual_arm_weight_threshold_kg()
        ]
        if len(light_candidates) >= 2:
            pair_plan = self._plan_two_arms_two_targets(
                light_candidates,
                anchor_target_id=int(nearest["target_id"]),
            )
            if pair_plan is not None:
                return pair_plan

        reason = "nearest_light_target_no_feasible_pair" if len(light_candidates) >= 2 else "nearest_available_target"
        return self._plan_single_arm(nearest, reason=reason)

    def _with_weight_and_side_grasps(self, candidate: dict[str, Any]) -> dict[str, Any]:
        volume_liter = float(candidate["volume"]["volume_liter"])
        candidate["predicted_weight_kg"] = volume_liter * self._density_kg_per_liter()
        candidate["arm_side_grasps"] = {
            "arm1": self._side_grasp_for_arm(candidate, "arm1"),
            "arm2": self._side_grasp_for_arm(candidate, "arm2"),
        }
        return candidate

    def _side_grasp_for_arm(self, candidate: dict[str, Any], arm_id: str) -> dict[str, Any] | None:
        normal_base = np.asarray(
            self.side_grasp.get(f"{arm_id}_normal_base", [0.0, -1.0 if arm_id == "arm1" else 1.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)
        normal_camera = self.transform.R.T @ _unit_vector(normal_base)
        try:
            return choose_ellipsoid_side_grasp(
                pose=candidate["pose"],
                side_direction_camera=normal_camera,
                pregrasp_offset_m=self.pregrasp_offset_m,
            )
        except ValueError:
            return None

    def _plan_two_arms_one_target(self, candidate: dict[str, Any]) -> dict[str, Any]:
        arm1 = candidate["arm_side_grasps"].get("arm1")
        arm2 = candidate["arm_side_grasps"].get("arm2")
        min_score = self._single_target_two_arm_min_score()
        if arm1 is None or arm2 is None:
            return self._plan_single_arm(candidate, reason="side_grasp_unavailable_for_two_arm_heavy_target")
        if float(arm1.get("score", 0.0)) < min_score or float(arm2.get("score", 0.0)) < min_score:
            return self._plan_single_arm(candidate, reason="side_grasp_score_too_low_for_two_arm_heavy_target")

        arm_assignments = {
            "arm1": self._assignment_from_grasp("arm1", candidate, arm1),
            "arm2": self._assignment_from_grasp("arm2", candidate, arm2),
        }
        command = self.command_builder.build_from_arm_assignments(
            arm_assignments=arm_assignments,
            plan_type="dual_arm_single_heavy_watermelon",
            plan_summary="Predicted heavy target; use both arms on one watermelon.",
        )
        return self._plan_result(
            command=command,
            plan_type="dual_arm_single_heavy_watermelon",
            selected_targets=[candidate],
            arm_grasps={"arm1": arm1, "arm2": arm2},
        )

    def _plan_two_arms_two_targets(
        self,
        candidates: list[dict[str, Any]],
        anchor_target_id: int | None = None,
    ) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        for i, first in enumerate(candidates):
            for second in candidates[i + 1 :]:
                if anchor_target_id is not None and not _pair_has_target(first, second, anchor_target_id):
                    continue

                pair_options = [
                    ("arm1", first, "arm2", second),
                    ("arm1", second, "arm2", first),
                ]
                for arm_a, candidate_a, arm_b, candidate_b in pair_options:
                    grasp_a = candidate_a["arm_side_grasps"].get(arm_a)
                    grasp_b = candidate_b["arm_side_grasps"].get(arm_b)
                    if grasp_a is None or grasp_b is None:
                        continue

                    point_a = self.transform.point_camera_to_base(grasp_a["contact_point_camera"]) * 1000.0
                    point_b = self.transform.point_camera_to_base(grasp_b["contact_point_camera"]) * 1000.0
                    x_delta = abs(float(point_a[0] - point_b[0]))
                    if x_delta > self._x_shared_tolerance_mm():
                        continue

                    grasp_pair_score = (
                        float(grasp_a.get("score", 0.0))
                        + float(grasp_b.get("score", 0.0))
                        - x_delta / max(self._x_shared_tolerance_mm(), 1.0) * 0.2
                    )
                    pair_key = _pair_distance_priority_key(
                        candidate_a,
                        candidate_b,
                        x_delta_mm=x_delta,
                        grasp_pair_score=grasp_pair_score,
                    )
                    if best is None or pair_key < best["pair_key"]:
                        best = {
                            "pair_key": pair_key,
                            "grasp_pair_score": grasp_pair_score,
                            "arm_a": arm_a,
                            "candidate_a": candidate_a,
                            "grasp_a": grasp_a,
                            "arm_b": arm_b,
                            "candidate_b": candidate_b,
                            "grasp_b": grasp_b,
                            "x_delta_mm": x_delta,
                        }

        if best is None:
            return None

        selected_targets = sorted(
            [best["candidate_a"], best["candidate_b"]],
            key=self._distance_priority_key,
        )
        primary_target_id = int(selected_targets[0]["target_id"])
        primary_arm_id = best["arm_a"] if int(best["candidate_a"]["target_id"]) == primary_target_id else best["arm_b"]
        arm_assignments = {
            best["arm_a"]: self._assignment_from_grasp(best["arm_a"], best["candidate_a"], best["grasp_a"]),
            best["arm_b"]: self._assignment_from_grasp(best["arm_b"], best["candidate_b"], best["grasp_b"]),
        }
        command = self.command_builder.build_from_arm_assignments(
            arm_assignments=arm_assignments,
            plan_type="dual_arm_two_light_watermelons",
            plan_summary="Predicted light targets; each arm grabs one watermelon.",
        )
        return self._plan_result(
            command=command,
            plan_type="dual_arm_two_light_watermelons",
            selected_targets=selected_targets,
            arm_grasps={best["arm_a"]: best["grasp_a"], best["arm_b"]: best["grasp_b"]},
            primary_arm_id=primary_arm_id,
            extra={
                "x_delta_mm": float(best["x_delta_mm"]),
                "distance_priority": "camera_depth_m_nearest_first",
            },
        )

    def _plan_single_arm(self, candidate: dict[str, Any], reason: str = "single_available_target") -> dict[str, Any]:
        grasp = candidate["visible_grasp"]
        arm_assignments = {
            "arm1": self._assignment_from_grasp("arm1", candidate, grasp),
            "arm2": make_idle_assignment("arm2"),
        }
        command = self.command_builder.build_from_arm_assignments(
            arm_assignments=arm_assignments,
            plan_type="single_arm_single_watermelon",
            plan_summary=f"Fallback single-arm plan: {reason}.",
        )
        return self._plan_result(
            command=command,
            plan_type="single_arm_single_watermelon",
            selected_targets=[candidate],
            arm_grasps={"arm1": grasp, "arm2": None},
            extra={"fallback_reason": reason},
        )

    def _assignment_from_grasp(
        self,
        arm_id: str,
        candidate: dict[str, Any],
        grasp: dict[str, Any],
    ) -> dict[str, Any]:
        contact_base = self.transform.point_camera_to_base(grasp["contact_point_camera"])
        normal_base = self.transform.vector_camera_to_base(grasp["surface_normal_camera"])
        approach_base = -normal_base
        return make_arm_assignment(
            arm_id=arm_id,
            contact_point_base_m=contact_base,
            target_id=int(candidate["target_id"]),
            method=str(grasp.get("method", "unknown")),
            score=float(grasp.get("score", 0.0)),
            is_inferred=bool(grasp.get("is_inferred", False)),
            weight_kg=float(candidate["predicted_weight_kg"]),
            surface_normal_base=normal_base,
            approach_vector_base=approach_base,
        )

    def _plan_result(
        self,
        command: dict[str, Any],
        plan_type: str,
        selected_targets: list[dict[str, Any]],
        arm_grasps: dict[str, dict[str, Any] | None],
        primary_arm_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        primary_target_id = int(selected_targets[0]["target_id"])
        primary_arm_id = primary_arm_id or _arm_id_for_target(command, primary_target_id)
        target_summaries = [
            {
                "target_id": int(candidate["target_id"]),
                "predicted_weight_kg": float(candidate["predicted_weight_kg"]),
                "volume_liter": float(candidate["volume"]["volume_liter"]),
                "center_base_m": _point_dict(candidate["center_base"]),
                "camera_depth_m": _candidate_depth_m(candidate),
                "camera_distance_m": _candidate_distance_m(candidate),
            }
            for candidate in selected_targets
        ]
        orientation_by_arm = self._orientation_by_arm(command)
        result = {
            "plan_type": plan_type,
            "robot_command": command,
            "primary_target_id": primary_target_id,
            "primary_arm_id": primary_arm_id,
            "selected_targets": target_summaries,
            "arm_grasps": {
                arm_id: self._grasp_summary(arm_id, grasp, orientation_by_arm.get(arm_id))
                for arm_id, grasp in arm_grasps.items()
            },
            "primary_grasp": self._primary_grasp(arm_grasps, orientation_by_arm, primary_arm_id),
            "planning_config": {
                "density_kg_per_liter": self._density_kg_per_liter(),
                "dual_arm_weight_threshold_kg": self._dual_arm_weight_threshold_kg(),
                "x_shared_tolerance_mm": self._x_shared_tolerance_mm(),
                "distance_priority": "camera_depth_m_nearest_first",
            },
        }
        if extra:
            result.update(extra)
        return result

    def _grasp_summary(
        self,
        arm_id: str,
        grasp: dict[str, Any] | None,
        orientation: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if grasp is None:
            return None
        contact_base = self.transform.point_camera_to_base(grasp["contact_point_camera"])
        normal_base = self.transform.vector_camera_to_base(grasp["surface_normal_camera"])
        approach_base = -normal_base
        pregrasp_base = self.transform.point_camera_to_base(grasp["pregrasp_point_camera"])
        orientation_enabled = bool(orientation.get("enabled", False)) if orientation else False
        return {
            "arm_id": arm_id,
            "method": grasp.get("method", "unknown"),
            "score": float(grasp.get("score", 0.0)),
            "is_inferred": bool(grasp.get("is_inferred", False)),
            "score_breakdown": grasp.get("score_breakdown", {}),
            "contact_point_base_m": _point_dict(contact_base),
            "surface_normal_base": _point_dict(normal_base),
            "approach_vector_base": _point_dict(approach_base),
            "pregrasp_point_base_m": _point_dict(pregrasp_base),
            "orientation_enabled": orientation_enabled,
            "orientation_axis": orientation.get("axis") if orientation else None,
            "orientation_angle_deg": (
                float(orientation["angle_deg"]) if orientation_enabled and orientation else None
            ),
            "orientation_was_clipped": bool(orientation.get("was_clipped", False)) if orientation else False,
        }

    def _primary_grasp(
        self,
        arm_grasps: dict[str, dict[str, Any] | None],
        orientation_by_arm: dict[str, dict[str, Any]],
        preferred_arm_id: str | None = None,
    ) -> dict[str, Any]:
        arm_order = [preferred_arm_id] if preferred_arm_id else []
        arm_order.extend(arm_id for arm_id in ("arm1", "arm2") if arm_id not in arm_order)
        for arm_id in arm_order:
            grasp = arm_grasps.get(arm_id)
            summary = self._grasp_summary(arm_id, grasp, orientation_by_arm.get(arm_id))
            if summary is not None:
                return summary
        raise ValueError("No primary grasp is available.")

    @staticmethod
    def _orientation_by_arm(command: dict[str, Any]) -> dict[str, dict[str, Any]]:
        by_arm: dict[str, dict[str, Any]] = {}
        for orientation in (command.get("orientation_axes") or {}).values():
            if not isinstance(orientation, dict):
                continue
            source_arm = orientation.get("source_arm")
            if source_arm:
                by_arm[str(source_arm)] = orientation
        return by_arm

    def _density_kg_per_liter(self) -> float:
        return float(self.planning.get("density_kg_per_liter", 0.95))

    def _dual_arm_weight_threshold_kg(self) -> float:
        return float(self.planning.get("dual_arm_weight_threshold_kg", 4.0))

    def _x_shared_tolerance_mm(self) -> float:
        return float(self.planning.get("x_shared_tolerance_mm", 80.0))

    def _single_target_two_arm_min_score(self) -> float:
        return float(self.planning.get("single_target_two_arm_min_score", 0.30))

    @staticmethod
    def _distance_priority_key(candidate: dict[str, Any]) -> tuple[float, int]:
        return (_candidate_depth_m(candidate), int(candidate["target_id"]))


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-9:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return vector / norm


def _candidate_depth_m(candidate: dict[str, Any]) -> float:
    if "camera_depth_m" in candidate:
        return float(candidate["camera_depth_m"])
    pose = candidate.get("pose") or {}
    if "center_camera" in pose:
        center_camera = np.asarray(pose["center_camera"], dtype=np.float64).reshape(3)
        return float(center_camera[2])
    return float("inf")


def _candidate_distance_m(candidate: dict[str, Any]) -> float:
    if "camera_distance_m" in candidate:
        return float(candidate["camera_distance_m"])
    pose = candidate.get("pose") or {}
    if "center_camera" in pose:
        center_camera = np.asarray(pose["center_camera"], dtype=np.float64).reshape(3)
        return float(np.linalg.norm(center_camera))
    return float("inf")


def _pair_has_target(first: dict[str, Any], second: dict[str, Any], target_id: int) -> bool:
    return int(first["target_id"]) == target_id or int(second["target_id"]) == target_id


def _pair_distance_priority_key(
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
    x_delta_mm: float,
    grasp_pair_score: float,
) -> tuple[float, float, float, float]:
    depth_a = _candidate_depth_m(candidate_a)
    depth_b = _candidate_depth_m(candidate_b)
    return (
        max(depth_a, depth_b),
        depth_a + depth_b,
        float(x_delta_mm),
        -float(grasp_pair_score),
    )


def _arm_id_for_target(command: dict[str, Any], target_id: int) -> str | None:
    assignments = command.get("arm_assignments") or {}
    for arm_id in ("arm1", "arm2"):
        assignment = assignments.get(arm_id) or {}
        assignment_target_id = assignment.get("target_id")
        if assignment_target_id is not None and assignment.get("enabled", False) and int(assignment_target_id) == target_id:
            return arm_id
    for arm_id, assignment in assignments.items():
        assignment_target_id = assignment.get("target_id")
        if assignment_target_id is not None and assignment.get("enabled", False) and int(assignment_target_id) == target_id:
            return str(arm_id)
    return None


def _point_dict(point_m: np.ndarray) -> dict[str, float]:
    point = np.asarray(point_m, dtype=np.float64).reshape(3)
    return {
        "x": float(point[0]),
        "y": float(point[1]),
        "z": float(point[2]),
    }
