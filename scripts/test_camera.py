from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camera.orbbec_camera import OrbbecCamera, list_orbbec_devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview Orbbec RGB-D frames.")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--hw-d2c", action="store_true", help="Use hardware depth-to-color alignment.")
    parser.add_argument("--startup-timeout-ms", type=int, default=10000)
    parser.add_argument("--frame-timeout-ms", type=int, default=2000)
    parser.add_argument("--no-full-frame-require", action="store_true")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--list-devices", action="store_true", help="Only list Orbbec devices and exit.")
    return parser.parse_args()


def depth_to_colormap(depth_mm: np.ndarray) -> np.ndarray:
    valid = depth_mm[(depth_mm > 0) & (depth_mm < 10000)]
    if valid.size == 0:
        return np.zeros((*depth_mm.shape[:2], 3), dtype=np.uint8)
    clipped = np.clip(depth_mm, np.percentile(valid, 2), np.percentile(valid, 98))
    depth_u8 = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)


def main() -> None:
    args = parse_args()
    devices = list_orbbec_devices()
    if args.list_devices:
        print_devices(devices)
        return
    if not devices:
        print("[ERROR] No Orbbec camera was detected by pyorbbecsdk.")
        print("[HINT] Check camera power/cable/network, Orbbec Viewer, device IP, and driver/SDK setup.")
        return

    print_devices(devices)
    camera = OrbbecCamera(
        args.width,
        args.height,
        args.fps,
        use_hw_d2c=args.hw_d2c,
        full_frame_require=not args.no_full_frame_require,
        startup_timeout_ms=args.startup_timeout_ms,
    )
    try:
        print("[INFO] Starting RGB-D stream...")
        camera.start()
    except Exception as exc:
        print(f"[ERROR] Failed to start Orbbec camera: {exc}")
        return
    frame_count = 0

    try:
        print("Intrinsics:", camera.get_intrinsics())
        print("Press q or Esc to exit.")
        while True:
            color_bgr, depth_mm, _timestamp = camera.get_rgbd(args.frame_timeout_ms)
            if color_bgr is None or depth_mm is None:
                print("[WARN] No valid RGB-D frame received.")
                continue

            frame_count += 1
            print(f"frame={frame_count} color={color_bgr.shape} depth={depth_mm.shape}")

            if not args.no_window:
                depth_vis = depth_to_colormap(depth_mm)
                preview = np.hstack([color_bgr, depth_vis])
                cv2.imshow("Orbbec RGB-D Test", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    finally:
        camera.stop()
        if not args.no_window:
            cv2.destroyAllWindows()


def print_devices(devices: list[dict[str, str]]) -> None:
    print(f"[INFO] Found {len(devices)} Orbbec device(s).")
    for device in devices:
        name = device.get("name") or "unknown"
        serial = device.get("serial_number") or "unknown"
        connection = device.get("connection_type") or "unknown"
        print(
            f"[INFO] Device {device['index']}: name={name}, serial={serial}, connection={connection}"
        )


if __name__ == "__main__":
    main()
