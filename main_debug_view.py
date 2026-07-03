from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from calibration.transform import load_transform_from_yaml
from camera.orbbec_camera import OrbbecCamera, list_orbbec_devices
from perception.watermelon_pipeline import WatermelonVisionProcessor
from robot.injection_molding_robot import InjectionRobotCommandBuilder, load_injection_robot_config
from scripts.test_camera import depth_to_colormap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug view for Orbbec watermelon perception.")
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


def draw_mask_overlay(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    overlay = image.copy()
    if mask is None:
        return overlay
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(overlay, 0.55, green, 0.45, 0)[mask_bool]
    return overlay


def draw_debug(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    result: dict,
    debug: dict,
    intrinsic: dict[str, float],
) -> np.ndarray:
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
            cv2.putText(overlay, "center", (center_px[0] + 8, center_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if grasp:
        contact_px = project_point(grasp["contact_point_camera"], intrinsic)
        pregrasp_px = project_point(grasp["pregrasp_point_camera"], intrinsic)
        if contact_px:
            cv2.drawMarker(overlay, contact_px, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(overlay, "contact", (contact_px[0] + 8, contact_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if contact_px and pregrasp_px:
            cv2.arrowedLine(overlay, contact_px, pregrasp_px, (255, 0, 0), 2, tipLength=0.25)

    status = result.get("status", "unknown")
    cv2.putText(overlay, f"status: {status}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    target = result.get("target")
    if target:
        volume = target["volume"]["volume_liter"]
        grasp_score = target["grasp_confidence"]
        target_score = target.get("target_selection_score", 0.0)
        method = target.get("grasp", {}).get("method", "unknown")
        command_count = target.get("robot_command", {}).get("register_count", 0)
        cv2.putText(overlay, f"volume: {volume:.2f} L  grasp: {grasp_score:.2f}  target: {target_score:.2f}", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(overlay, f"method: {method}", (16, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(overlay, f"plc values: {command_count}", (16, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    elif result.get("message"):
        cv2.putText(overlay, str(result["message"])[:80], (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    depth_vis = depth_to_colormap(depth_mm)
    mask = debug.get("mask")
    if mask is None:
        mask_vis = np.zeros_like(overlay)
    else:
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return np.hstack([overlay, depth_vis, mask_vis])


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
        print("[INFO] Starting RGB-D stream...")
        camera.start()
    except Exception as exc:
        print(f"[ERROR] Failed to start Orbbec camera: {exc}")
        return
    frame_count = 0

    try:
        intrinsic = camera.get_color_intrinsic()
        print("Press q or Esc to exit.")
        while True:
            color_bgr, depth_mm, _timestamp = camera.get_rgbd(args.frame_timeout_ms)
            if color_bgr is None or depth_mm is None:
                continue

            frame_count += 1
            result, debug = processor.process(color_bgr, depth_mm, intrinsic)
            if args.print_json and result.get("status") == "ok":
                print(json.dumps(result, ensure_ascii=False))

            cv2.imshow("Watermelon Vision Debug", draw_debug(color_bgr, depth_mm, result, debug, intrinsic))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
