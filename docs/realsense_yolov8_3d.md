# Orbbec Gemini 435Le + 西瓜三维感知说明

本文档替代旧的 RealSense 说明。项目现在按“纯 Python 视觉感知服务”组织，底层相机从 `pyrealsense2` 切换为 `pyorbbecsdk2`，主线目标是：

```text
Gemini 435Le 采集 RGB-D
-> 西瓜分割
-> mask + depth 生成局部点云
-> PCA 估计三维中心、主轴和尺寸
-> 估计吸盘接触点、表面法向和预抓取点
-> 椭球体积预测
-> FastAPI 向机械臂提供最新结果
```

## 目录

```text
D:\MelonVision_3D
├── main_debug_view.py              # 调试界面：RGB、Depth、mask、中心点、抓取点、体积
├── main_api.py                     # FastAPI 服务
├── camera
│   ├── orbbec_camera.py            # pyorbbecsdk 相机封装
│   └── frame_utils.py              # Orbbec frame 转 OpenCV / numpy
├── calibration
│   ├── save_intrinsics.py          # 保存相机内参
│   └── transform.py                # camera -> robot_base 坐标变换
├── perception
│   ├── watermelon_segmenter.py     # HSV MVP 西瓜分割
│   ├── pointcloud_builder.py       # mask + depth -> 点云
│   └── watermelon_pipeline.py      # 单帧感知流程
├── geometry
│   ├── pose_estimator.py           # PCA 姿态和尺寸
│   └── suction_grasp.py            # 吸盘接触点和法向
├── volume
│   └── volume_estimator.py         # 椭球体积估计
├── api
│   ├── schemas.py
│   └── server.py
├── scripts
│   ├── test_camera.py              # RGB-D 相机测试
│   ├── test_orbbec_yolov8_3d.py    # 类似旧 realsense_yolov8_3d.py 的 YOLO 测试
│   └── capture_dataset.py          # 按键保存 RGB-D 数据
└── configs
    ├── camera.yaml
    └── T_base_camera.yaml
```

## 安装依赖

```powershell
python -m pip install -r requirements.txt
```

`pyorbbecsdk2` 的导入名仍然是 `pyorbbecsdk`。

## 1. 测试 Orbbec 相机

```powershell
python .\scripts\test_camera.py
```

只列出 SDK 能发现的设备：

```powershell
python .\scripts\test_camera.py --list-devices
```

无窗口处理一帧：

```powershell
python .\scripts\test_camera.py --no-window --max-frames 1
```

旧入口仍可用：

```powershell
python .\text\open_camera.py
```

## 2. 保存相机内参

```powershell
python .\calibration\save_intrinsics.py --output .\configs\camera_intrinsics.yaml
```

保存内容包括 RGB / Depth 内参、畸变参数和 depth 到 color 的外参。

## 3. 运行调试界面

```powershell
python .\main_debug_view.py
```

窗口会显示三列画面：RGB 叠加 mask、深度伪彩色图、mask。检测到西瓜后会显示中心点、吸盘接触点、法向箭头、体积和抓取置信度。

当前 MVP 使用 HSV 阈值做西瓜分割，现场需要根据光照调整 `perception/watermelon_segmenter.py` 中的阈值。

## 4. 运行 API 服务

```powershell
python .\main_api.py --host 0.0.0.0 --port 8000
```

接口：

```http
GET /api/v1/system/status
GET /api/v1/watermelon/best_target
```

机械臂主要读取：

```http
GET /api/v1/watermelon/best_target
```

返回目标坐标默认位于 `robot_base` 坐标系，单位为米。当前 `configs/T_base_camera.yaml` 是单位矩阵占位，正式抓取前必须完成相机到机械臂基坐标系的标定。

## 5. 类似旧文件的 YOLO 测试程序

如果你想像旧 `text/realsense_yolov8_3d.py` 一样用 YOLO 检测框中心输出三维坐标，运行：

```powershell
python .\scripts\test_orbbec_yolov8_3d.py --model yolov8n.pt
```

常用参数：

```powershell
python .\scripts\test_orbbec_yolov8_3d.py --model yolov8n.pt --conf 0.4 --interval 0.5 --csv .\outputs\coords.csv
```

无窗口测试一帧：

```powershell
python .\scripts\test_orbbec_yolov8_3d.py --model yolov8n.pt --no-window --max-frames 1
```

为了保留旧习惯，下面这个命令也会调用新的 Orbbec YOLO 测试程序：

```powershell
python .\text\realsense_yolov8_3d.py --model yolov8n.pt
```

输出格式保持接近旧版：

```text
bottle conf=0.82 center=(321,244) depth=0.635m xyz_mm=(12.4,8.7,635.0)
```

## 6. 采集数据

```powershell
python .\scripts\capture_dataset.py --output-dir .\data\raw
```

按 `s` 保存一组数据：

```text
color_000001.png
depth_000001.npy
depth_vis_000001.png
meta_000001.json
```

深度会以 `.npy` 保存毫米值，不只是 8-bit 可视化图。

## 注意事项

- 当前分割是 HSV MVP，适合先跑通链路；后续可以把 `WatermelonSegmenter.segment()` 内部替换为 YOLO-Seg。
- `configs/T_base_camera.yaml` 现在是占位的单位变换，不能直接用于真实机械臂抓取。
- 单视角体积估计只看得到西瓜可见表面，体积结果需要后续用真实数据做回归校正。
- 修改相机流配置时，如果设备不支持指定分辨率或帧率，代码会尽量回退到 SDK 默认 profile。

## 排查：只显示 `load extensions` 后退出

如果命令行只显示类似下面一行，然后回到提示符：

```text
load extensions from ...\site-packages\pyorbbecsdk\extensions
```

先运行：

```powershell
python .\scripts\test_camera.py --list-devices
```

如果输出 `Found 0 Orbbec device(s)`，说明 `pyorbbecsdk` 当前没有发现相机。优先检查：

- 相机供电、USB/网线连接、网口 IP 配置。
- Orbbec Viewer 是否能看到相机。
- 是否有其它程序正在占用设备。
- 当前 PyCharm/PowerShell 使用的 Python 环境是否就是安装 `pyorbbecsdk2` 的 `Orbbec` 环境。
