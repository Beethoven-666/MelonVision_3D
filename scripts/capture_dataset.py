from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camera.orbbec_camera import OrbbecCamera
from scripts.test_camera import depth_to_colormap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture RGB-D dataset frames from an Orbbec camera.")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--hw-d2c", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera = OrbbecCamera(args.width, args.height, args.fps, use_hw_d2c=args.hw_d2c)
    camera.start()
    index = 1

    try:
        print("Press s to save a frame, q or Esc to exit.")
        while True:
            color_bgr, depth_mm, timestamp = camera.get_rgbd()
            if color_bgr is None or depth_mm is None:
                continue

            depth_vis = depth_to_colormap(depth_mm)
            cv2.imshow("Capture RGB-D", np.hstack([color_bgr, depth_vis]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != ord("s"):
                continue

            stem = f"{index:06d}"
            cv2.imwrite(str(output_dir / f"color_{stem}.png"), color_bgr)
            np.save(output_dir / f"depth_{stem}.npy", depth_mm)
            cv2.imwrite(str(output_dir / f"depth_vis_{stem}.png"), depth_vis)
            meta = {
                "index": index,
                "saved_at": time.time(),
                "timestamp": timestamp,
                "intrinsics": camera.get_intrinsics(),
            }
            (output_dir / f"meta_{stem}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved frame {stem} to {output_dir}")
            index += 1
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
