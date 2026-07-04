from __future__ import annotations

import argparse
import json
import time
from threading import Thread
from typing import Any

import uvicorn

from api.server import update_latest_result
from calibration.transform import load_transform_from_yaml
from camera.orbbec_camera import OrbbecCamera
from perception.watermelon_pipeline import WatermelonVisionProcessor
from robot.injection_molding_robot import InjectionRobotCommandBuilder, load_injection_robot_config
from robot.modbus_tcp_client import write_holding_registers


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
    )
    camera = OrbbecCamera(
        args.width,
        args.height,
        args.fps,
        use_hw_d2c=args.hw_d2c,
        full_frame_require=not args.no_full_frame_require,
        startup_timeout_ms=args.startup_timeout_ms,
    )
    print_state = {
        "last_print_time": 0.0,
        "last_signature": None,
    }

    try:
        camera.start()
        intrinsic = camera.get_color_intrinsic()
        if args.print_result:
            print(
                "[INFO] Console perception output is enabled. "
                "Use --print-json for full JSON or --no-print-result to disable it.",
                flush=True,
            )
        while True:
            color_bgr, depth_mm, _timestamp = camera.get_rgbd(args.frame_timeout_ms)
            if color_bgr is None or depth_mm is None:
                result = {
                    "status": "camera_error",
                    "timestamp": time.time(),
                    "camera_id": args.camera_id,
                    "target": None,
                    "message": "No valid RGB-D frame was received.",
                }
                update_latest_result(result)
                _print_result_if_needed(args, result, print_state)
                time.sleep(args.loop_sleep)
                continue

            result, _debug = processor.process(color_bgr, depth_mm, intrinsic)
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
        camera.stop()


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
    host = args.modbus_host or modbus_cfg.get("host", "192.168.1.10")
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

    axis_text = _format_axis_positions(command.get("axis_values") or {})
    if axis_text:
        lines.append(f"  axis_positions_mm {axis_text}")

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


def _format_axis_positions(axis_values: dict[str, Any]) -> str:
    parts = []
    for axis in ("x", "y1", "y2", "z1", "z2"):
        values = axis_values.get(axis)
        if not isinstance(values, dict):
            continue
        parts.append(f"{axis}={float(values.get('position_mm', 0.0)):.1f}")
    return " ".join(parts)


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
        parts.append(
            f"{arm_id}(enabled={enabled}, target={target_id}, method={method}, "
            f"contact_mm={_format_point(contact)})"
        )
    return "; ".join(parts)


if __name__ == "__main__":
    main()
