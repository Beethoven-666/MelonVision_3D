from __future__ import annotations

import argparse
import json
from typing import Any

import cv2
import numpy as np

from calibration.transform import load_transform_from_yaml
from camera.orbbec_camera import list_orbbec_devices
from camera.rgbd_stream_worker import RGBDStreamWorker
from perception.segmenter_factory import add_segmenter_args, build_segmenter_from_args
from perception.watermelon_pipeline import WatermelonVisionProcessor
from robot.injection_molding_robot import InjectionRobotCommandBuilder, load_injection_robot_config
from scripts.test_camera import depth_to_colormap


TARGET_COLORS = (
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 255),
    (255, 255, 0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug view for Orbbec watermelon perception.")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--hw-d2c", action="store_true")
    parser.add_argument("--startup-timeout-ms", type=int, default=10000)
    parser.add_argument("--frame-timeout-ms", type=int, default=2000)
    parser.add_argument("--camera-restart-timeouts", type=int, default=10)
    parser.add_argument("--camera-restart-wait-sec", type=float, default=2.0)
    parser.add_argument("--no-full-frame-require", action="store_true")
    parser.add_argument("--transform", default="configs/T_base_camera.yaml")
    parser.add_argument("--camera-id", default="gemini_435le_01")
    parser.add_argument("--min-points", type=int, default=300)
    parser.add_argument("--robot-origin-base", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--robot-distance-norm", type=float, default=1.5)
    parser.add_argument("--same-height-band", type=float, default=0.08)
    parser.add_argument("--grasp-mode", choices=("injection",), default="injection")
    parser.add_argument("--tool-normal-base", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    parser.add_argument("--robot-config", default="configs/injection_robot.yaml")
    add_segmenter_args(parser)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def project_point(point_camera: np.ndarray, intrinsic: dict[str, float]) -> tuple[int, int] | None:
    z = float(point_camera[2])
    if z <= 1e-6:
        return None
    u = int(round(intrinsic["fx"] * float(point_camera[0]) / z + intrinsic["cx"]))
    v = int(round(intrinsic["fy"] * float(point_camera[1]) / z + intrinsic["cy"]))
    return u, v


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray | None,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    overlay = image.copy()
    if mask is None:
        return overlay
    color_layer = np.zeros_like(overlay)
    color_layer[:] = color
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(overlay, 0.55, color_layer, 0.45, 0)[mask_bool]
    return overlay


def _selected_target_views(result: dict, debug: dict) -> list[dict[str, Any]]:
    target = result.get("target") or {}
    plan = target.get("dual_arm_plan") or debug.get("dual_arm_plan") or {}
    selected_targets = plan.get("selected_targets") or []
    if len(selected_targets) < 2:
        return []

    candidates = debug.get("target_candidates") or []
    candidates_by_id = {
        target_id: candidate
        for candidate in candidates
        if (target_id := _safe_int(candidate.get("target_id"))) is not None
    }

    views = []
    for index, selected in enumerate(selected_targets, start=1):
        target_id = _safe_int(selected.get("target_id")) if isinstance(selected, dict) else None
        if target_id is None:
            continue
        candidate = candidates_by_id.get(target_id)
        if candidate is None:
            continue
        arm_id = _arm_id_for_target(plan, target_id)
        color = _target_color(index, arm_id)
        views.append(
            {
                "index": index,
                "target_id": target_id,
                "candidate": candidate,
                "arm_id": arm_id,
                "grasp": _grasp_for_arm(candidate, arm_id),
                "color": color,
            }
        )
    return views if len(views) >= 2 else []


def _safe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _arm_id_for_target(plan: dict, target_id: int) -> str | None:
    assignments = (plan.get("robot_command") or {}).get("arm_assignments") or {}
    for arm_id in ("arm1", "arm2"):
        assignment = assignments.get(arm_id) or {}
        if assignment.get("enabled", False) and _safe_int(assignment.get("target_id")) == target_id:
            return arm_id
    for arm_id, assignment in assignments.items():
        if assignment.get("enabled", False) and _safe_int(assignment.get("target_id")) == target_id:
            return str(arm_id)
    return None


def _target_color(index: int, arm_id: str | None) -> tuple[int, int, int]:
    if arm_id == "arm1":
        return (0, 255, 0)
    if arm_id == "arm2":
        return (0, 165, 255)
    return TARGET_COLORS[(index - 1) % len(TARGET_COLORS)]


def _grasp_for_arm(candidate: dict[str, Any], arm_id: str | None) -> dict[str, Any] | None:
    if arm_id is not None:
        arm_grasp = (candidate.get("arm_side_grasps") or {}).get(arm_id)
        if arm_grasp is not None:
            return arm_grasp
    return candidate.get("visible_grasp")


def _draw_selected_target_overlays(
    image: np.ndarray,
    selected_views: list[dict[str, Any]],
) -> np.ndarray:
    overlay = image.copy()
    for view in selected_views:
        detection = view["candidate"].get("detection") or {}
        overlay = draw_mask_overlay(overlay, detection.get("mask"), view["color"])
    return overlay


def _draw_selected_targets(
    overlay: np.ndarray,
    selected_views: list[dict[str, Any]],
    intrinsic: dict[str, float],
) -> None:
    for view in selected_views:
        candidate = view["candidate"]
        detection = candidate.get("detection") or {}
        label = _target_label(view)
        color = view["color"]
        _draw_labeled_bbox(overlay, detection, label, color)
        _draw_candidate_pose_and_grasp(overlay, candidate, view.get("grasp"), intrinsic, color, view["index"])


def _target_label(view: dict[str, Any]) -> str:
    arm_id = view.get("arm_id")
    target_id = view.get("target_id")
    if arm_id:
        return f"{arm_id} T{target_id}"
    return f"T{target_id}"


def _draw_labeled_bbox(
    overlay: np.ndarray,
    detection: dict[str, Any],
    label: str,
    color: tuple[int, int, int],
) -> None:
    bbox = detection.get("bbox_xywh")
    if not bbox or len(bbox) < 4:
        return
    x, y, w, h = [int(round(float(value))) for value in bbox[:4]]
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_x = max(0, x)
    label_y = y - 8 if y >= text_h + 12 else y + text_h + 12
    cv2.rectangle(
        overlay,
        (label_x, max(0, label_y - text_h - baseline - 4)),
        (label_x + text_w + 8, label_y + baseline + 4),
        color,
        -1,
    )
    cv2.putText(overlay, label, (label_x + 4, label_y), font, font_scale, (0, 0, 0), thickness)


def _draw_candidate_pose_and_grasp(
    overlay: np.ndarray,
    candidate: dict[str, Any],
    grasp: dict[str, Any] | None,
    intrinsic: dict[str, float],
    color: tuple[int, int, int],
    index: int,
) -> None:
    pose = candidate.get("pose")
    if pose:
        center_px = project_point(pose["center_camera"], intrinsic)
        if center_px:
            cv2.circle(overlay, center_px, 6, (255, 255, 255), -1)
            cv2.putText(
                overlay,
                f"T{index}",
                (center_px[0] + 8, center_px[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    if not grasp:
        return
    contact_px = project_point(grasp["contact_point_camera"], intrinsic)
    pregrasp_px = project_point(grasp["pregrasp_point_camera"], intrinsic)
    if contact_px:
        cv2.drawMarker(overlay, contact_px, color, cv2.MARKER_CROSS, 16, 2)
        cv2.putText(
            overlay,
            "contact",
            (contact_px[0] + 8, contact_px[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    if contact_px and pregrasp_px:
        cv2.arrowedLine(overlay, contact_px, pregrasp_px, color, 2, tipLength=0.25)


def _selected_mask_panel(
    shape_like: np.ndarray,
    selected_views: list[dict[str, Any]],
) -> np.ndarray:
    mask_vis = np.zeros_like(shape_like)
    for view in selected_views:
        detection = view["candidate"].get("detection") or {}
        mask = detection.get("mask")
        if mask is None:
            continue
        mask_vis[mask > 0] = view["color"]
    return mask_vis


def _draw_result_text(
    overlay: np.ndarray,
    result: dict,
    selected_views: list[dict[str, Any]],
) -> None:
    status = result.get("status", "unknown")
    cv2.putText(overlay, f"status: {status}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    target = result.get("target")
    if not target:
        if result.get("message"):
            cv2.putText(overlay, str(result["message"])[:80], (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        return

    command = target.get("robot_command", {})
    command_count = command.get("register_count", 0)
    plan_type = target.get("dual_arm_plan", {}).get("plan_type", "unknown")
    if selected_views:
        for row, view in enumerate(selected_views[:2]):
            cv2.putText(
                overlay,
                _selected_target_metric_text(view),
                (16, 58 + row * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
            )
        cv2.putText(overlay, f"plan: {plan_type}  plc values: {command_count}", (16, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(overlay, _orientation_text(command), (16, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        return

    volume = target["volume"]["volume_liter"]
    weight = target.get("predicted_weight_kg", 0.0)
    grasp_score = target["grasp_confidence"]
    method = target.get("grasp", {}).get("method", "unknown")
    cv2.putText(overlay, f"volume: {volume:.2f} L  weight: {weight:.2f}kg  grasp: {grasp_score:.2f}", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(overlay, f"method: {method}", (16, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(overlay, f"plan: {plan_type}  plc values: {command_count}", (16, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(overlay, _orientation_text(command), (16, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)


def _selected_target_metric_text(view: dict[str, Any]) -> str:
    candidate = view["candidate"]
    volume = candidate.get("volume") or {}
    volume_liter = float(volume.get("volume_liter", 0.0))
    weight_kg = float(candidate.get("predicted_weight_kg", 0.0))
    grasp = view.get("grasp") or {}
    grasp_score = float(grasp.get("score", 0.0))
    return f"{_target_label(view)}: volume {volume_liter:.2f} L  weight {weight_kg:.2f}kg  grasp {grasp_score:.2f}"


def draw_debug_overlay(
    color_bgr: np.ndarray,
    result: dict,
    debug: dict,
    intrinsic: dict[str, float],
    show_result_text: bool = True,
) -> np.ndarray:
    selected_views = _selected_target_views(result, debug)
    if selected_views:
        overlay = _draw_selected_target_overlays(color_bgr, selected_views)
        _draw_selected_targets(overlay, selected_views, intrinsic)
    else:
        overlay = draw_mask_overlay(color_bgr, debug.get("mask"))
        detection = debug.get("detection")
        if detection:
            x, y, w, h = detection["bbox_xywh"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

        pose = debug.get("pose")
        grasp = debug.get("grasp")
        if pose:
            center_px = project_point(pose["center_camera"], intrinsic)
            if center_px:
                cv2.circle(overlay, center_px, 6, (255, 255, 255), -1)
                cv2.putText(
                    overlay,
                    "center",
                    (center_px[0] + 8, center_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        if grasp:
            contact_px = project_point(grasp["contact_point_camera"], intrinsic)
            pregrasp_px = project_point(grasp["pregrasp_point_camera"], intrinsic)
            if contact_px:
                cv2.drawMarker(overlay, contact_px, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
                cv2.putText(
                    overlay,
                    "contact",
                    (contact_px[0] + 8, contact_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
            if contact_px and pregrasp_px:
                cv2.arrowedLine(overlay, contact_px, pregrasp_px, (255, 0, 0), 2, tipLength=0.25)

    if show_result_text:
        _draw_result_text(overlay, result, selected_views)
    return overlay


def draw_debug(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    result: dict,
    debug: dict,
    intrinsic: dict[str, float],
) -> np.ndarray:
    selected_views = _selected_target_views(result, debug)
    overlay = draw_debug_overlay(color_bgr, result, debug, intrinsic)

    depth_vis = depth_to_colormap(depth_mm)
    if selected_views:
        mask_vis = _selected_mask_panel(overlay, selected_views)
    else:
        mask = debug.get("mask")
        if mask is None:
            mask_vis = np.zeros_like(overlay)
        else:
            mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return np.hstack([overlay, depth_vis, mask_vis])


def _orientation_text(command: dict) -> str:
    if not command.get("orientation_enabled", False):
        return "orientation: off"
    axes = command.get("orientation_axes") or {}
    parts = []
    for axis in ("r1", "r2"):
        info = axes.get(axis) or {}
        angle = float(info.get("angle_deg", 0.0))
        state = "on" if info.get("enabled", False) else "idle"
        clipped = " clipped" if info.get("was_clipped", False) else ""
        parts.append(f"{axis}={angle:.1f}deg({state}{clipped})")
    return "orientation: " + " ".join(parts)


def main() -> None:
    args = parse_args()
    devices = list_orbbec_devices()
    if not devices:
        print("[ERROR] No Orbbec camera was detected by pyorbbecsdk.")
        print("[HINT] Run: python .\\scripts\\test_camera.py --list-devices")
        return

    transform = load_transform_from_yaml(args.transform)
    robot_config = load_injection_robot_config(args.robot_config)
    robot_command_builder = InjectionRobotCommandBuilder(robot_config)
    segmenter = build_segmenter_from_args(args)
    processor = WatermelonVisionProcessor(
        transform=transform,
        camera_id=args.camera_id,
        min_points=args.min_points,
        robot_origin_base=tuple(args.robot_origin_base),
        robot_distance_norm_m=args.robot_distance_norm,
        same_height_band_m=args.same_height_band,
        grasp_mode=args.grasp_mode,
        tool_normal_base=tuple(args.tool_normal_base),
        robot_command_builder=robot_command_builder,
        robot_config=robot_config,
        segmenter=segmenter,
    )
    stream = RGBDStreamWorker(
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_hw_d2c=args.hw_d2c,
        full_frame_require=not args.no_full_frame_require,
        startup_timeout_ms=args.startup_timeout_ms,
        frame_timeout_ms=args.frame_timeout_ms,
        restart_after_timeouts=args.camera_restart_timeouts,
        restart_wait_sec=args.camera_restart_wait_sec,
    )
    frame_count = 0
    last_sequence = 0
    last_status_print = 0.0

    try:
        print("[INFO] Starting RGB-D stream...")
        stream.start()
        print("Press q or Esc to exit.")
        while True:
            frame = stream.get_latest()
            intrinsic = stream.get_color_intrinsic()
            if frame is None or intrinsic is None:
                last_status_print = _print_stream_status_periodically(stream, last_status_print)
                if cv2.waitKey(20) & 0xFF in (ord("q"), 27):
                    break
                continue
            if frame.sequence == last_sequence:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            last_sequence = frame.sequence

            frame_count += 1
            result, debug = processor.process(frame.color_bgr, frame.depth_mm, intrinsic)
            if args.print_json and result.get("status") == "ok":
                print(json.dumps(result, ensure_ascii=False))

            cv2.imshow(
                "Watermelon Vision Debug",
                draw_debug(frame.color_bgr, frame.depth_mm, result, debug, intrinsic),
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()


def _print_stream_status_periodically(stream: RGBDStreamWorker, last_print_time: float) -> float:
    now = cv2.getTickCount() / cv2.getTickFrequency()
    if now - last_print_time < 2.0:
        return last_print_time
    status = stream.get_status()
    print(f"[WARN] RGB-D stream status={status.get('state')} message={status.get('message')}")
    return now


if __name__ == "__main__":
    main()
