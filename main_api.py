from __future__ import annotations

import argparse
import time
from threading import Thread

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
    parser.add_argument("--grasp-mode", choices=("injection", "visible"), default="injection")
    parser.add_argument("--tool-normal-base", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    parser.add_argument("--robot-config", default="configs/injection_robot.yaml")
    parser.add_argument("--write-modbus", action="store_true", help="Write 20 robot command values to PLC via Modbus TCP.")
    parser.add_argument("--modbus-host", default=None)
    parser.add_argument("--modbus-port", type=int, default=None)
    parser.add_argument("--modbus-unit-id", type=int, default=None)
    parser.add_argument("--modbus-start-address", type=int, default=None)
    parser.add_argument("--loop-sleep", type=float, default=0.02)
    return parser.parse_args()


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
    )
    camera = OrbbecCamera(
        args.width,
        args.height,
        args.fps,
        use_hw_d2c=args.hw_d2c,
        full_frame_require=not args.no_full_frame_require,
        startup_timeout_ms=args.startup_timeout_ms,
    )

    try:
        camera.start()
        intrinsic = camera.get_color_intrinsic()
        while True:
            color_bgr, depth_mm, _timestamp = camera.get_rgbd(args.frame_timeout_ms)
            if color_bgr is None or depth_mm is None:
                update_latest_result(
                    {
                        "status": "camera_error",
                        "timestamp": time.time(),
                        "camera_id": args.camera_id,
                        "target": None,
                        "message": "No valid RGB-D frame was received.",
                    }
                )
                time.sleep(args.loop_sleep)
                continue

            result, _debug = processor.process(color_bgr, depth_mm, intrinsic)
            if args.write_modbus and result.get("target"):
                _write_robot_command_to_plc(args, robot_config, result)
            update_latest_result(result)
            time.sleep(args.loop_sleep)
    except Exception as exc:
        update_latest_result(
            {
                "status": "camera_error",
                "timestamp": time.time(),
                "camera_id": args.camera_id,
                "target": None,
                "message": str(exc),
            }
        )
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
    host = args.modbus_host or modbus_cfg.get("host", "192.168.1.10")
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


if __name__ == "__main__":
    main()
