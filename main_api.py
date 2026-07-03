from __future__ import annotations

import argparse
import time
from threading import Thread

import uvicorn

from api.server import update_latest_result
from calibration.transform import load_transform_from_yaml
from camera.orbbec_camera import OrbbecCamera
from perception.watermelon_pipeline import WatermelonVisionProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Watermelon Vision FastAPI service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--hw-d2c", action="store_true")
    parser.add_argument("--transform", default="configs/T_base_camera.yaml")
    parser.add_argument("--camera-id", default="gemini_435le_01")
    parser.add_argument("--min-points", type=int, default=300)
    parser.add_argument("--loop-sleep", type=float, default=0.02)
    return parser.parse_args()


def perception_loop(args: argparse.Namespace) -> None:
    transform = load_transform_from_yaml(args.transform)
    processor = WatermelonVisionProcessor(
        transform=transform,
        camera_id=args.camera_id,
        min_points=args.min_points,
    )
    camera = OrbbecCamera(args.width, args.height, args.fps, use_hw_d2c=args.hw_d2c)

    try:
        camera.start()
        intrinsic = camera.get_color_intrinsic()
        while True:
            color_bgr, depth_mm, _timestamp = camera.get_rgbd()
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


if __name__ == "__main__":
    main()
