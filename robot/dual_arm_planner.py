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

        enriched = [self._with_weight_and_side_grasps(candidate) for candidate in candidates]
        heavy_candidates = [
            candidate
            for candidate in enriched
            if candidate["predicted_weight_kg"] >= self._dual_arm_weight_threshold_kg()
        ]
        if heavy_candidates:
            best_heavy = max(heavy_candidates, key=self._heavy_score)
            return self._plan_two_arms_one_target(best_heavy)

        if len(enriched) >= 2:
            pair_plan = self._plan_two_arms_two_targets(enriched)
            if pair_plan is not None:
                return pair_plan

        best_single = max(enriched, key=lambda candidate: float(candidate["visible_grasp"].get("score", 0.0)))
        return self._plan_single_arm(best_single)

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

    def _plan_two_arms_two_targets(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        for i, first in enumerate(candidates):
            for second in candidates[i + 1 :]:
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

                    score = (
                        float(grasp_a.get("score", 0.0))
                        + float(grasp_b.get("score", 0.0))
                        - x_delta / max(self._x_shared_tolerance_mm(), 1.0) * 0.2
                    )
                    if best is None or score > best["score"]:
                        best = {
                            "score": score,
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
            selected_targets=[best["candidate_a"], best["candidate_b"]],
            arm_grasps={best["arm_a"]: best["grasp_a"], best["arm_b"]: best["grasp_b"]},
            extra={"x_delta_mm": float(best["x_delta_mm"])},
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
        return make_arm_assignment(
            arm_id=arm_id,
            contact_point_base_m=contact_base,
            target_id=int(candidate["target_id"]),
            method=str(grasp.get("method", "unknown")),
            score=float(grasp.get("score", 0.0)),
            is_inferred=bool(grasp.get("is_inferred", False)),
            weight_kg=float(candidate["predicted_weight_kg"]),
        )

    def _plan_result(
        self,
        command: dict[str, Any],
        plan_type: str,
        selected_targets: list[dict[str, Any]],
        arm_grasps: dict[str, dict[str, Any] | None],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target_summaries = [
            {
                "target_id": int(candidate["target_id"]),
                "predicted_weight_kg": float(candidate["predicted_weight_kg"]),
                "volume_liter": float(candidate["volume"]["volume_liter"]),
                "center_base_m": _point_dict(candidate["center_base"]),
            }
            for candidate in selected_targets
        ]
        result = {
            "plan_type": plan_type,
            "robot_command": command,
            "selected_targets": target_summaries,
            "arm_grasps": {
                arm_id: self._grasp_summary(grasp)
                for arm_id, grasp in arm_grasps.items()
            },
            "primary_grasp": self._primary_grasp(arm_grasps),
            "planning_config": {
                "density_kg_per_liter": self._density_kg_per_liter(),
                "dual_arm_weight_threshold_kg": self._dual_arm_weight_threshold_kg(),
                "x_shared_tolerance_mm": self._x_shared_tolerance_mm(),
            },
        }
        if extra:
            result.update(extra)
        return result

    def _grasp_summary(self, grasp: dict[str, Any] | None) -> dict[str, Any] | None:
        if grasp is None:
            return None
        contact_base = self.transform.point_camera_to_base(grasp["contact_point_camera"])
        normal_base = self.transform.vector_camera_to_base(grasp["surface_normal_camera"])
        pregrasp_base = self.transform.point_camera_to_base(grasp["pregrasp_point_camera"])
        return {
            "method": grasp.get("method", "unknown"),
            "score": float(grasp.get("score", 0.0)),
            "is_inferred": bool(grasp.get("is_inferred", False)),
            "score_breakdown": grasp.get("score_breakdown", {}),
            "contact_point_base_m": _point_dict(contact_base),
            "surface_normal_base": _point_dict(normal_base),
            "pregrasp_point_base_m": _point_dict(pregrasp_base),
        }

    def _primary_grasp(self, arm_grasps: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
        for arm_id in ("arm1", "arm2"):
            grasp = arm_grasps.get(arm_id)
            summary = self._grasp_summary(grasp)
            if summary is not None:
                summary["arm_id"] = arm_id
                return summary
        raise ValueError("No primary grasp is available.")

    def _density_kg_per_liter(self) -> float:
        return float(self.planning.get("density_kg_per_liter", 0.95))

    def _dual_arm_weight_threshold_kg(self) -> float:
        return float(self.planning.get("dual_arm_weight_threshold_kg", 4.0))

    def _x_shared_tolerance_mm(self) -> float:
        return float(self.planning.get("x_shared_tolerance_mm", 80.0))

    def _single_target_two_arm_min_score(self) -> float:
        return float(self.planning.get("single_target_two_arm_min_score", 0.30))

    @staticmethod
    def _heavy_score(candidate: dict[str, Any]) -> float:
        return float(candidate["predicted_weight_kg"]) + float(candidate["visible_grasp"].get("score", 0.0))


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-9:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return vector / norm


def _point_dict(point_m: np.ndarray) -> dict[str, float]:
    point = np.asarray(point_m, dtype=np.float64).reshape(3)
    return {
        "x": float(point[0]),
        "y": float(point[1]),
        "z": float(point[2]),
    }
