import argparse
import csv
import time
from pathlib import Path

try:
    import cv2
    import numpy as np
    import pyrealsense2 as rs
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency. Install requirements first: python -m pip install -r requirements.txt"
    ) from exc

try:
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency ultralytics. Install it first: python -m pip install ultralytics"
    ) from exc


def parse_args():
    """解析命令行参数，允许用户在运行时配置模型、相机、检测和输出选项。"""
    parser = argparse.ArgumentParser(
        description="RealSense D455f + YOLOv8 target 3D coordinate output"
    )
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model path")
    parser.add_argument("--width", type=int, default=640, help="camera stream width")
    parser.add_argument("--height", type=int, default=480, help="camera stream height")
    parser.add_argument("--fps", type=int, default=30, help="camera stream FPS")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--device", default=None, help="YOLO device, for example cpu, 0, cuda:0")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="optional YOLO class ids to detect",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="seconds between detections; 0 means every frame",
    )
    parser.add_argument(
        "--depth-window",
        type=int,
        default=5,
        help="odd pixel window used to find valid median depth around target center",
    )
    parser.add_argument("--csv", default=None, help="optional CSV path for coordinate output")
    parser.add_argument("--no-window", action="store_true", help="run without OpenCV preview window")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames; 0 means unlimited")
    return parser.parse_args()


def normalize_depth_window(value):
    """将深度采样窗口修正为可用的正奇数，保证中心点两侧采样范围对称。"""
    if value < 1:
        return 1
    return value if value % 2 == 1 else value + 1


def create_pipeline(width, height, fps):
    """创建并启动 RealSense 深度和彩色流，同时返回对齐器和深度比例尺。"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)
    return pipeline, align, depth_scale


def get_aligned_frame_data(pipeline, align):
    """读取一组相机帧，将深度帧对齐到彩色帧，并返回内参、彩色图和深度图。"""
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None

    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    return depth_intrinsics, color_image, depth_image, depth_frame


def get_median_depth(depth_image, depth_scale, center_x, center_y, window_size):
    """在目标中心点附近取有效深度的中位数，减少单个像素深度空洞或噪声的影响。"""
    height, width = depth_image.shape[:2]
    half_window = window_size // 2
    x_min = max(center_x - half_window, 0)
    x_max = min(center_x + half_window + 1, width)
    y_min = max(center_y - half_window, 0)
    y_max = min(center_y + half_window + 1, height)

    depth_region = depth_image[y_min:y_max, x_min:x_max]
    valid_depth_values = depth_region[depth_region > 0]
    if valid_depth_values.size == 0:
        return 0.0
    return float(np.median(valid_depth_values) * depth_scale)


def deproject_target(depth_intrinsics, depth_image, depth_scale, center_x, center_y, window_size):
    """把目标中心像素和对应深度反投影为相机坐标系下的三维坐标，单位为毫米。"""
    depth_m = get_median_depth(depth_image, depth_scale, center_x, center_y, window_size)
    if depth_m <= 0:
        return None, depth_m

    xyz_m = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [center_x, center_y], depth_m)
    xyz_mm = np.round(np.array(xyz_m, dtype=float) * 1000.0, 1)
    return xyz_mm, depth_m


def open_csv_writer(csv_path):
    """按需创建 CSV 输出文件，并写入目标检测和三维坐标结果的表头。"""
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


def draw_target(image, class_name, confidence, center_x, center_y, xyz_mm, depth_m):
    """在预览图上标出目标中心点、类别、置信度、深度和三维坐标。"""
    x_mm, y_mm, z_mm = xyz_mm.tolist()
    label = f"{class_name} {confidence:.2f} X:{x_mm:.1f} Y:{y_mm:.1f} Z:{z_mm:.1f}mm"
    depth_label = f"D:{depth_m:.3f}m"
    cv2.circle(image, (center_x, center_y), 4, (255, 255, 255), -1)
    cv2.putText(image, label, (center_x + 8, center_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(image, depth_label, (center_x + 8, center_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)


def main():
    """程序主入口：加载 YOLO 模型、启动相机循环、检测目标并输出三维坐标。"""
    args = parse_args()
    depth_window = normalize_depth_window(args.depth_window)
    model = YOLO(args.model)

    pipeline, align, depth_scale = create_pipeline(args.width, args.height, args.fps)
    csv_file, csv_writer = open_csv_writer(args.csv)
    last_detection_time = 0.0
    frame_count = 0

    print("Press q or Esc to exit.")
    print("Coordinate unit: millimeter. Origin: RealSense color camera optical center.")

    try:
        while True:
            frame_data = get_aligned_frame_data(pipeline, align)
            if frame_data is None:
                continue

            frame_count += 1
            depth_intrinsics, color_image, depth_image, _depth_frame = frame_data
            annotated_image = color_image.copy()
            current_time = time.time()

            should_detect = args.interval <= 0 or current_time - last_detection_time >= args.interval
            if should_detect:
                last_detection_time = current_time
                results = model.predict(
                    color_image,
                    conf=args.conf,
                    classes=args.classes,
                    device=args.device,
                    verbose=False,
                )

                if results:
                    result = results[0]
                    annotated_image = result.plot()
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
                    boxes_conf = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
                    boxes_cls = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else []

                    for xyxy, confidence, class_id in zip(boxes_xyxy, boxes_conf, boxes_cls):
                        x1, y1, x2, y2 = xyxy.astype(int).tolist()
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        xyz_mm, depth_m = deproject_target(
                            depth_intrinsics,
                            depth_image,
                            depth_scale,
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
                cv2.imshow("RealSense YOLOv8 3D", annotated_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break

    finally:
        pipeline.stop()
        if csv_file:
            csv_file.close()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()