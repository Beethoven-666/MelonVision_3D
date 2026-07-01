# RealSense D455f + YOLOv8 输出目标三维坐标说明

本文档配套脚本：[text/realsense_yolov8_3d.py](../text/realsense_yolov8_3d.py)。

## 功能

脚本使用 Intel RealSense D455f 获取深度流和彩色流，用 YOLOv8 在彩色图像中检测目标，然后读取目标框中心点附近的深度值，并通过 RealSense SDK 的 `rs.rs2_deproject_pixel_to_point()` 将二维像素点转换为相机坐标系下的三维坐标。

输出坐标单位为毫米，坐标原点为 RealSense 彩色相机光心：

- `X`：相机视角下的左右方向
- `Y`：相机视角下的上下方向
- `Z`：目标到相机的前方深度方向

## 安装依赖

在项目根目录运行：

```powershell
python -m pip install -r requirements.txt
```

如果 `pip install ...` 又出现 launcher 路径错误，继续使用 `python -m pip ...`，不要直接调用 `pip.exe`。

## 基本运行

使用 YOLOv8n 官方权重：

```powershell
python .\text\realsense_yolov8_3d.py --model yolov8n.pt
```

第一次运行时，`ultralytics` 会自动下载 `yolov8n.pt`。如果你已经有自己训练的权重，例如 `best.pt`：

```powershell
python .\text\realsense_yolov8_3d.py --model .\weights\best.pt
```

退出窗口：按 `q` 或 `Esc`。

## 常用参数

```powershell
python .\text\realsense_yolov8_3d.py --model yolov8n.pt --conf 0.4 --interval 0.5 --csv .\outputs\coords.csv
```

参数说明：

- `--model`：YOLOv8 权重路径，默认 `yolov8n.pt`
- `--conf`：检测置信度阈值，默认 `0.25`
- `--width` / `--height` / `--fps`：RealSense 视频流参数，默认 `640x480@30fps`
- `--interval`：检测间隔秒数，默认 `0`，表示每帧检测；机械臂抓取场景可设置为 `0.5`、`1` 或更大
- `--depth-window`：目标中心点周围取深度的窗口大小，默认 `5`，用于避开单个像素深度空洞
- `--classes`：只检测指定类别 ID，例如只检测 COCO 中的 person：`--classes 0`
- `--device`：YOLO 推理设备，例如 `cpu`、`0`、`cuda:0`
- `--csv`：保存检测结果和三维坐标到 CSV
- `--no-window`：不显示 OpenCV 窗口，只在命令行输出结果
- `--max-frames`：处理指定帧数后自动退出，默认 `0` 表示一直运行

## 输出示例

命令行会输出类似内容：

```text
bottle conf=0.82 center=(321,244) depth=0.635m xyz_mm=(12.4,8.7,635.0)
```

含义：YOLO 检测到 `bottle`，检测框中心像素为 `(321,244)`，该点深度为 `0.635m`，转换后的相机坐标为 `(X=12.4mm, Y=8.7mm, Z=635.0mm)`。

如果使用 `--csv`，CSV 会包含时间、类别、置信度、中心像素、深度、三维坐标和检测框坐标。

无窗口测试一帧并自动退出：

```powershell
python .\text\realsense_yolov8_3d.py --model yolov8n.pt --no-window --max-frames 1
```

## 实现流程

1. 启动 RealSense depth/color 流。
2. 使用 `rs.align(rs.stream.color)` 将深度帧对齐到彩色帧。
3. 将彩色帧传给 YOLOv8 做目标检测。
4. 对每个检测框取中心点 `(center_x, center_y)`。
5. 在中心点附近取有效深度的中位数，减少深度空洞影响。
6. 使用 `rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth_m)` 得到相机坐标系三维坐标。
7. 在画面中标注中心点和坐标，并可选写入 CSV。

## 注意事项

- 该坐标是相机坐标，不是机械臂基坐标。要给机械臂使用，还需要做手眼标定或相机到机械臂基座的外参变换。
- 深度值为 `0` 通常表示该像素没有有效深度，脚本会在中心点附近找有效深度；如果仍找不到，会输出 `depth=invalid`。
- D455f 与 D435i 的 SDK 调用方式基本一致，关键是保证深度和彩色流分辨率可被你的设备支持。
- 如果窗口卡住或相机被占用，关闭其它 RealSense Viewer、相机程序或重新插拔相机后再运行。