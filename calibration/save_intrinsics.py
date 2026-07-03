from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camera.orbbec_camera import OrbbecCamera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save Orbbec camera intrinsics to YAML.")
    parser.add_argument("--output", default="configs/camera_intrinsics.yaml")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--hw-d2c", action="store_true", help="Use hardware depth-to-color alignment.")
    return parser.parse_args()


def _intrinsic_dict(intrinsic) -> dict[str, float]:
    return {
        "width": int(intrinsic.width),
        "height": int(intrinsic.height),
        "fx": float(intrinsic.fx),
        "fy": float(intrinsic.fy),
        "cx": float(intrinsic.cx),
        "cy": float(intrinsic.cy),
    }


def _distortion_dict(distortion) -> dict[str, float]:
    return {
        "k1": float(distortion.k1),
        "k2": float(distortion.k2),
        "k3": float(distortion.k3),
        "p1": float(distortion.p1),
        "p2": float(distortion.p2),
    }


def _extrinsic_dict(extrinsic) -> dict[str, object]:
    return {
        "rotation_matrix": np.asarray(extrinsic.rot, dtype=float).reshape(3, 3).tolist(),
        "translation_mm": np.asarray(extrinsic.transform, dtype=float).reshape(3).tolist(),
    }


def main() -> None:
    args = parse_args()
    camera = OrbbecCamera(args.width, args.height, args.fps, use_hw_d2c=args.hw_d2c)
    camera.start()
    try:
        param = camera.camera_param
        data = {
            "rgb": {
                "intrinsic": _intrinsic_dict(param.rgb_intrinsic),
                "distortion": _distortion_dict(param.rgb_distortion),
            },
            "depth": {
                "intrinsic": _intrinsic_dict(param.depth_intrinsic),
                "distortion": _distortion_dict(param.depth_distortion),
            },
            "depth_to_color": _extrinsic_dict(param.transform),
        }
    finally:
        camera.stop()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Saved camera intrinsics to {output_path}")


if __name__ == "__main__":
    main()
