from __future__ import annotations

import argparse
import json
import time
from threading import Thread
from typing import Any

import cv2
import uvicorn

from api.server import update_latest_frame_jpeg, update_latest_result
from calibration.transform import load_transform_from_yaml
from camera.rgbd_stream_worker import RGBDStreamWorker
from main_debug_view import draw_debug_overlay
from perception.segmenter_factory import add_segmenter_args, build_segmenter_from_args
from perception.watermelon_pipeline import WatermelonVisionProcessor
from robot.injection_molding_robot import InjectionRobotCommandBuilder, load_injection_robot_config
from robot.modbus_tcp_client import write_holding_registers

#监控看板：http://127.0.0.1:8000/dashboard
#启动命令：python .\main_api.py --write-modbus --modbus-host 192.168.1.88 --modbus-start-address 0
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Watermelon Vision FastAPI service.")
    parser.add_argument("--host", default="0.0.0.0")
    #设置 FastAPI 端口，默认 8000
    parser.add_argument("--port", type=int, default=8000)
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
    #是否真的把结果写给 PLC。默认不写，避免调试时误动作
    parser.add_argument("--write-modbus", action="store_true", help="Write 20 robot command values to PLC via Modbus TCP.")
    #Modbus TCP 的 IP、端口、设备 ID、起始寄存器地址
    parser.add_argument("--modbus-host", default=None)
    parser.add_argument("--modbus-port", type=int, default=None)
    parser.add_argument("--modbus-unit-id", type=int, default=None)
    parser.add_argument("--modbus-start-address", type=int, default=None)
    parser.add_argument("--loop-sleep", type=float, default=0.02)
    parser.add_argument("--print-interval", type=float, default=1.0, help="Seconds between console result prints.")
    parser.add_argument("--print-json", action="store_true", help="Print the full latest result as JSON.")
    parser.add_argument("--dashboard-jpeg-quality", type=int, default=82)
    parser.add_argument("--dashboard-fps", type=float, default=20.0)
    parser.add_argument("--no-dashboard-frame", action="store_true", help="Disable frame publishing for /dashboard.")
    parser.add_argument(
        "--dashboard-frame-mode",
        choices=("debug", "rgb"),
        default="debug",
        help="Use debug for the annotated YOLO/debug view, or rgb for the raw color frame.",
    )
    parser.add_argument(
        "--no-print-result",
        dest="print_result",
        action="store_false",
        default=True,
        help="Disable console printing of perception and robot command results.",
    )
    return parser.parse_args()

#视觉循环函数：读相机、识别西瓜、生成机械手命令、更新 API 最新结果
def perception_loop(args: argparse.Namespace) -> None:
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
    print_state = {
        "last_print_time": 0.0,
        "last_signature": None,
    }
    dashboard_state = {
        "last_update_time": 0.0,
        "last_warn_time": 0.0,
    }

    try:
        stream.start()
        if args.print_result:
            print(
                "[INFO] Console perception output is enabled. "
                "Use --print-json for full JSON or --no-print-result to disable it.",
                flush=True,
            )
        last_sequence = 0
        while True:
            frame = stream.get_latest()
            intrinsic = stream.get_color_intrinsic()
            if frame is None or intrinsic is None:
                status = stream.get_status()
                result = _camera_status_result(args, status)
                update_latest_result(result)
                _print_result_if_needed(args, result, print_state)
                time.sleep(args.loop_sleep)
                continue

            if frame.sequence == last_sequence:
                time.sleep(args.loop_sleep)
                continue
            last_sequence = frame.sequence

            result, debug = processor.process(frame.color_bgr, frame.depth_mm, intrinsic)
            _attach_dashboard_targets(result, debug)
            if not args.no_dashboard_frame:
                _update_dashboard_frame_if_needed(
                    args,
                    frame.color_bgr,
                    frame.depth_mm,
                    result,
                    debug,
                    intrinsic,
                    dashboard_state,
                )
            if args.write_modbus and result.get("target"):
                _write_robot_command_to_plc(args, robot_config, result)
            update_latest_result(result)
            _print_result_if_needed(args, result, print_state)
            time.sleep(args.loop_sleep)
    except Exception as exc:
        result = {
            "status": "camera_error",
            "timestamp": time.time(),
            "camera_id": args.camera_id,
            "target": None,
            "message": str(exc),
        }
        update_latest_result(result)
        _print_result_if_needed(args, result, print_state)
        raise
    finally:
        stream.stop()


def main() -> None:
    args = parse_args()
    worker = Thread(target=perception_loop, args=(args,), daemon=True)
    worker.start()
    uvicorn.run("api.server:app", host=args.host, port=args.port, reload=False)


def _write_robot_command_to_plc(args: argparse.Namespace, robot_config: dict, result: dict) -> None:
    target = result.get("target") or {}
    command = target.get("robot_command") or {}
    values = command.get("register_values")
    if not values:
        return

    modbus_cfg = robot_config.get("modbus_tcp", {})
    register_cfg = robot_config.get("plc_registers", {})
    #确定 PLC IP 地址。命令行优先，其次 YAML，最后默认 192.168.1.10
    host = args.modbus_host or modbus_cfg.get("host", "192.168.1.88")
    #确定 Modbus TCP 端口，默认 502
    port = int(args.modbus_port or modbus_cfg.get("port", 502))
    unit_id = int(args.modbus_unit_id or modbus_cfg.get("unit_id", 1))
    start_address = int(
        args.modbus_start_address
        if args.modbus_start_address is not None
        else register_cfg.get("start_address", 0)
    )
    write_holding_registers(
        host=host,
        port=port,
        unit_id=unit_id,
        start_address=start_address,
        values=values,
    )


def _attach_dashboard_targets(result: dict[str, Any], debug: dict[str, Any]) -> None:
    target = result.get("target")
    if not isinstance(target, dict):
        return

    candidates = debug.get("target_candidates") or []
    if not candidates:
        return

    plan = target.get("dual_arm_plan") or {}
    selected_ids = {
        target_id
        for item in plan.get("selected_targets", [])
        if isinstance(item, dict) and (target_id := _safe_int(item.get("target_id"))) is not None
    }
    primary_target_id = _safe_int(plan.get("primary_target_id"))
    arm_by_target = _arm_by_target((target.get("robot_command") or {}).get("arm_assignments") or {})

    dashboard_targets = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        target_id = _safe_int(candidate.get("target_id"))
        detection = candidate.get("detection") or {}
        volume = candidate.get("volume") or {}
        arm_id = arm_by_target.get(target_id)
        grasp = _dashboard_grasp_for_candidate(candidate, arm_id)
        dashboard_targets.append(
            {
                "target_id": target_id,
                "selected": target_id in selected_ids,
                "primary": primary_target_id is not None and target_id == primary_target_id,
                "arm_id": arm_id,
                "class_name": detection.get("class_name"),
                "confidence": _float_value(detection.get("score")),
                "grasp_confidence": _float_value((grasp or {}).get("score")),
                "target_selection_score": _float_value(candidate.get("target_score")),
                "weight_kg": _float_value(candidate.get("predicted_weight_kg")),
                "volume_liter": _float_value(volume.get("volume_liter")),
                "center_base_m": _point_dict(candidate.get("center_base")),
                "camera_depth_m": _float_or_none(candidate.get("camera_depth_m")),
                "camera_distance_m": _float_or_none(candidate.get("camera_distance_m")),
            }
        )

    dashboard_targets.sort(
        key=lambda item: (
            not bool(item.get("selected")),
            item.get("camera_depth_m") if item.get("camera_depth_m") is not None else float("inf"),
            item.get("target_id") if item.get("target_id") is not None else float("inf"),
        )
    )
    for index, item in enumerate(dashboard_targets):
        item["label"] = _watermelon_label(index)
    target["dashboard_targets"] = dashboard_targets


def _update_dashboard_frame_if_needed(
    args: argparse.Namespace,
    color_bgr,
    depth_mm,
    result: dict[str, Any],
    debug: dict[str, Any],
    intrinsic: dict[str, float],
    dashboard_state: dict[str, float],
) -> None:
    now = time.monotonic()
    interval = 1.0 / max(float(args.dashboard_fps), 0.1)
    if now - float(dashboard_state.get("last_update_time", 0.0)) < interval:
        return

    dashboard_state["last_update_time"] = now
    if args.dashboard_frame_mode == "rgb":
        _update_dashboard_frame(color_bgr, args.dashboard_jpeg_quality)
        return

    try:
        dashboard_image = draw_debug_overlay(color_bgr, result, debug, intrinsic, show_result_text=False)
    except Exception as exc:
        last_warn = float(dashboard_state.get("last_warn_time", 0.0))
        if now - last_warn >= 2.0:
            print(f"[WARN] Dashboard debug frame render failed: {exc}", flush=True)
            dashboard_state["last_warn_time"] = now
        dashboard_image = color_bgr
    _update_dashboard_frame(dashboard_image, args.dashboard_jpeg_quality)


def _update_dashboard_frame(color_bgr, jpeg_quality: int) -> None:
    quality = int(max(35, min(jpeg_quality, 95)))
    ok, encoded = cv2.imencode(".jpg", color_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        update_latest_frame_jpeg(encoded.tobytes(), time.time())


def _arm_by_target(assignments: dict[str, Any]) -> dict[int, str]:
    result: dict[int, str] = {}
    for arm_id, assignment in assignments.items():
        if not isinstance(assignment, dict) or not assignment.get("enabled", False):
            continue
        target_id = _safe_int(assignment.get("target_id"))
        if target_id is not None:
            result[target_id] = str(arm_id)
    return result


def _dashboard_grasp_for_candidate(candidate: dict[str, Any], arm_id: str | None) -> dict[str, Any] | None:
    if arm_id:
        arm_grasp = (candidate.get("arm_side_grasps") or {}).get(arm_id)
        if isinstance(arm_grasp, dict):
            return arm_grasp
    visible_grasp = candidate.get("visible_grasp")
    return visible_grasp if isinstance(visible_grasp, dict) else None


def _point_dict(point: Any) -> dict[str, float] | None:
    if point is None:
        return None
    try:
        return {
            "x": float(point[0]),
            "y": float(point[1]),
            "z": float(point[2]),
        }
    except (TypeError, ValueError, IndexError):
        return None


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _watermelon_label(index: int) -> str:
    if 0 <= index < 26:
        return f"西瓜{chr(ord('A') + index)}"
    return f"西瓜{index + 1}"


def _camera_status_result(args: argparse.Namespace, status: dict[str, Any]) -> dict[str, Any]:
    state = status.get("state", "camera_error")
    return {
        "status": "camera_error" if state in {"camera_error", "timeout", "restarting"} else "camera_starting",
        "timestamp": time.time(),
        "camera_id": args.camera_id,
        "target": None,
        "message": str(status.get("message", "No valid RGB-D frame is available.")),
        "camera_stream": status,
    }


def _print_result_if_needed(
    args: argparse.Namespace,
    result: dict[str, Any],
    print_state: dict[str, Any],
) -> None:
    if not args.print_result:
        return

    now = time.monotonic()
    interval = max(float(args.print_interval), 0.0)
    signature = _result_signature(result)
    should_print = (
        print_state.get("last_signature") != signature
        or now - float(print_state.get("last_print_time", 0.0)) >= interval
    )
    if not should_print:
        return

    print_state["last_signature"] = signature
    print_state["last_print_time"] = now
    if args.print_json:
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    else:
        print(_format_result_summary(result), flush=True)


def _result_signature(result: dict[str, Any]) -> tuple[Any, ...]:
    target = result.get("target") or {}
    command = target.get("robot_command") or {}
    plan = target.get("dual_arm_plan") or {}
    return (
        result.get("status"),
        plan.get("plan_type"),
        command.get("register_count"),
        result.get("message"),
    )


def _format_result_summary(result: dict[str, Any]) -> str:
    timestamp = float(result.get("timestamp", time.time()))
    time_text = time.strftime("%H:%M:%S", time.localtime(timestamp))
    status = result.get("status", "unknown")
    target = result.get("target")
    if not target:
        message = result.get("message", "")
        return f"[VISION {time_text}] status={status} target=None message={message}"

    volume_liter = float((target.get("volume") or {}).get("volume_liter", 0.0))
    weight_kg = float(target.get("predicted_weight_kg", 0.0))
    grasp_score = float(target.get("grasp_confidence", 0.0))
    target_score = float(target.get("target_selection_score", 0.0))
    plan = target.get("dual_arm_plan") or {}
    plan_type = plan.get("plan_type", "unknown")
    command = target.get("robot_command") or {}
    register_values = command.get("register_values") or []

    lines = [
        (
            f"[VISION {time_text}] status=ok track={target.get('track_id')} "
            f"plan={plan_type} weight={weight_kg:.3f}kg volume={volume_liter:.3f}L "
            f"grasp_score={grasp_score:.3f} target_score={target_score:.3f}"
        ),
        (
            "  center_base_m="
            f"{_format_point(target.get('center_base_m'))} "
            "contact_base_m="
            f"{_format_point((target.get('grasp') or {}).get('contact_point_base_m'))} "
            "normal_base="
            f"{_format_point((target.get('grasp') or {}).get('surface_normal_base'))}"
        ),
        (
            f"  plc_registers count={command.get('register_count', len(register_values))} "
            f"start={command.get('plc_register_start', 0)} values={register_values}"
        ),
    ]

    orientation_text = _format_orientation_axes(command)
    if orientation_text:
        lines.append(f"  {orientation_text}")

    axis_text = _format_axis_positions(command.get("axis_values") or {}, command.get("axis_order") or [])
    if axis_text:
        lines.append(f"  axis_positions {axis_text}")

    arm_text = _format_arm_assignments(command.get("arm_assignments") or {})
    if arm_text:
        lines.append(f"  arm_assignments {arm_text}")

    selected_targets = plan.get("selected_targets") or []
    if selected_targets:
        lines.append(f"  selected_targets={selected_targets}")

    return "\n".join(lines)


def _format_point(point: Any) -> str:
    if not isinstance(point, dict):
        return "(n/a)"
    return (
        f"({float(point.get('x', 0.0)):.4f}, "
        f"{float(point.get('y', 0.0)):.4f}, "
        f"{float(point.get('z', 0.0)):.4f})"
    )


def _format_axis_positions(axis_values: dict[str, Any], axis_order: list[str]) -> str:
    parts = []
    ordered_axes = axis_order or list(axis_values.keys())
    for axis in ordered_axes:
        values = axis_values.get(axis)
        if not isinstance(values, dict):
            continue
        if "position_deg" in values:
            parts.append(f"{axis}={float(values.get('position_deg', 0.0)):.2f}deg")
        else:
            parts.append(f"{axis}={float(values.get('position_mm', 0.0)):.1f}mm")
    return " ".join(parts)


def _format_orientation_axes(command: dict[str, Any]) -> str:
    if not command.get("orientation_enabled", False):
        return "orientation enabled=False"
    axes = command.get("orientation_axes") or {}
    parts = []
    for axis in ("r1", "r2"):
        info = axes.get(axis) or {}
        angle = float(info.get("angle_deg", 0.0))
        state = "enabled" if info.get("enabled", False) else "idle"
        clipped = " clipped=True" if info.get("was_clipped", False) else ""
        parts.append(f"{axis}={angle:.2f}deg({state}{clipped})")
    return "orientation enabled=True " + " ".join(parts)


def _format_arm_assignments(assignments: dict[str, Any]) -> str:
    parts = []
    for arm_id in ("arm1", "arm2"):
        assignment = assignments.get(arm_id)
        if not isinstance(assignment, dict):
            continue
        enabled = bool(assignment.get("enabled", False))
        target_id = assignment.get("target_id")
        method = assignment.get("method", "unknown")
        contact = assignment.get("contact_point_mm")
        orientation_angle = assignment.get("orientation_angle_deg")
        orientation_text = (
            f"{float(orientation_angle):.2f}deg"
            if orientation_angle is not None
            else "None"
        )
        parts.append(
            f"{arm_id}(enabled={enabled}, target={target_id}, method={method}, "
            f"contact_mm={_format_point(contact)}, orientation={orientation_text})"
        )
    return "; ".join(parts)


if __name__ == "__main__":
    main()
