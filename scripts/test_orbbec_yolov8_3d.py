from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camera.orbbec_camera import OrbbecCamera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orbbec Gemini + YOLOv8 target 3D coordinate output.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model path.")
    parser.add_argument("--width", type=int, default=0, help="Preferred color stream width; 0 uses SDK default.")
    parser.add_argument("--height", type=int, default=0, help="Preferred color stream height; 0 uses SDK default.")
    parser.add_argument("--fps", type=int, default=0, help="Preferred stream FPS; 0 uses SDK default.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--device", default=None, help="YOLO device, for example cpu, 0, cuda:0.")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="Optional YOLO class IDs.")
    parser.add_argument("--interval", type=float, default=0.0, help="Seconds between detections.")
    parser.add_argument("--depth-window", type=int, default=5, help="Odd pixel window for median depth.")
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--no-window", action="store_true", help="Run without OpenCV preview.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means unlimited.")
    parser.add_argument("--hw-d2c", action="store_true", help="Use hardware depth-to-color alignment.")
    return parser.parse_args()


def normalize_depth_window(value: int) -> int:
    if value < 1:
        return 1
    return value if value % 2 == 1 else value + 1


def get_median_depth_m(depth_mm: np.ndarray, center_x: int, center_y: int, window_size: int) -> float:
    height, width = depth_mm.shape[:2]
    half = window_size // 2
    x_min = max(center_x - half, 0)
    x_max = min(center_x + half + 1, width)
    y_min = max(center_y - half, 0)
    y_max = min(center_y + half + 1, height)
    region = depth_mm[y_min:y_max, x_min:x_max]
    valid = region[region > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid) / 1000.0)


def deproject_pixel(intrinsic: dict[str, float], pixel: tuple[int, int], depth_m: float) -> np.ndarray:
    u, v = pixel
    x = (float(u) - intrinsic["cx"]) * depth_m / intrinsic["fx"]
    y = (float(v) - intrinsic["cy"]) * depth_m / intrinsic["fy"]
    return np.array([x, y, depth_m], dtype=np.float64)


def deproject_target(
    intrinsic: dict[str, float],
    depth_mm: np.ndarray,
    center_x: int,
    center_y: int,
    window_size: int,
) -> tuple[np.ndarray | None, float]:
    depth_m = get_median_depth_m(depth_mm, center_x, center_y, window_size)
    if depth_m <= 0:
        return None, depth_m
    xyz_mm = np.round(deproject_pixel(intrinsic, (center_x, center_y), depth_m) * 1000.0, 1)
    return xyz_mm, depth_m


def open_csv_writer(csv_path: str | None):
    if not csv_path:
        return None, None

    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = output_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "timestamp",
            "class_id",
            "class_name",
            "confidence",
            "center_x",
            "center_y",
            "depth_m",
            "x_mm",
            "y_mm",
            "z_mm",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
    )
    return csv_file, writer


def draw_target(
    image: np.ndarray,
    class_name: str,
    confidence: float,
    center_x: int,
    center_y: int,
    xyz_mm: np.ndarray,
    depth_m: float,
) -> None:
    x_mm, y_mm, z_mm = xyz_mm.tolist()
    label = f"{class_name} {confidence:.2f} X:{x_mm:.1f} Y:{y_mm:.1f} Z:{z_mm:.1f}mm"
    depth_label = f"D:{depth_m:.3f}m"
    cv2.circle(image, (center_x, center_y), 4, (255, 255, 255), -1)
    cv2.putText(image, label, (center_x + 8, center_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(image, depth_label, (center_x + 8, center_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)


def main() -> None:
    args = parse_args()
    depth_window = normalize_depth_window(args.depth_window)
    model = YOLO(args.model)

    camera = OrbbecCamera(args.width, args.height, args.fps, use_hw_d2c=args.hw_d2c)
    camera.start()
    intrinsic = camera.get_color_intrinsic()
    csv_file, csv_writer = open_csv_writer(args.csv)
    last_detection_time = 0.0
    frame_count = 0

    print("Press q or Esc to exit.")
    print("Coordinate unit: millimeter. Origin: Orbbec color camera optical center.")

    try:
        while True:
            color_bgr, depth_mm, _timestamp = camera.get_rgbd()
            if color_bgr is None or depth_mm is None:
                continue

            frame_count += 1
            annotated_image = color_bgr.copy()
            current_time = time.time()
            should_detect = args.interval <= 0 or current_time - last_detection_time >= args.interval

            if should_detect:
                last_detection_time = current_time
                results = model.predict(
                    color_bgr,
                    conf=args.conf,
                    classes=args.classes,
                    device=args.device,
                    verbose=False,
                )

                if results:
                    result = results[0]
                    annotated_image = result.plot()
                    boxes = result.boxes
                    boxes_xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else []
                    boxes_conf = boxes.conf.cpu().numpy() if boxes is not None else []
                    boxes_cls = boxes.cls.cpu().numpy().astype(int) if boxes is not None else []

                    for xyxy, confidence, class_id in zip(boxes_xyxy, boxes_conf, boxes_cls):
                        x1, y1, x2, y2 = xyxy.astype(int).tolist()
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        xyz_mm, depth_m = deproject_target(
                            intrinsic,
                            depth_mm,
                            center_x,
                            center_y,
                            depth_window,
                        )
                        class_name = result.names.get(class_id, str(class_id))
                        if xyz_mm is None:
                            print(f"{class_name} conf={confidence:.2f} center=({center_x},{center_y}) depth=invalid")
                            continue

                        draw_target(annotated_image, class_name, confidence, center_x, center_y, xyz_mm, depth_m)
                        x_mm, y_mm, z_mm = xyz_mm.tolist()
                        print(
                            f"{class_name} conf={confidence:.2f} center=({center_x},{center_y}) "
                            f"depth={depth_m:.3f}m xyz_mm=({x_mm:.1f},{y_mm:.1f},{z_mm:.1f})"
                        )

                        if csv_writer:
                            csv_writer.writerow(
                                [
                                    time.strftime("%Y-%m-%d %H:%M:%S"),
                                    class_id,
                                    class_name,
                                    f"{confidence:.4f}",
                                    center_x,
                                    center_y,
                                    f"{depth_m:.6f}",
                                    f"{x_mm:.1f}",
                                    f"{y_mm:.1f}",
                                    f"{z_mm:.1f}",
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                ]
                            )
                            csv_file.flush()

            if not args.no_window:
                cv2.imshow("Orbbec YOLOv8 3D", annotated_image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    finally:
        camera.stop()
        if csv_file:
            csv_file.close()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
