# Orbbec Gemini 435Le + 西瓜三维感知说明

本文档替代旧的 RealSense 说明。项目现在按“纯 Python 视觉感知服务”组织，底层相机从 `pyrealsense2` 切换为 `pyorbbecsdk2`，主线目标是：

```text
Gemini 435Le 采集 RGB-D
-> 西瓜分割
-> mask + depth 生成局部点云
-> PCA 估计三维中心、主轴和尺寸
-> 估计吸盘接触点、表面法向和预抓取点
-> 生成五轴注塑机械手 x/y1/y2/z1/z2 的 20 个 PLC 命令值
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
├── robot
│   ├── injection_molding_robot.py  # 五轴注塑机械手命令生成
│   └── modbus_tcp_client.py        # 可选 Modbus TCP 写 PLC
├── api
│   ├── schemas.py
│   └── server.py
├── scripts
│   ├── test_camera.py              # RGB-D 相机测试
│   ├── test_orbbec_yolov8_3d.py    # 类似旧 realsense_yolov8_3d.py 的 YOLO 测试
│   └── capture_dataset.py          # 按键保存 RGB-D 数据
└── configs
    ├── camera.yaml
    ├── injection_robot.yaml
    └── T_base_camera.yaml
```

## 安装依赖

```powershell
python -m pip install -r requirements.txt
```

`pyorbbecsdk2` 的导入名仍然是 `pyorbbecsdk`。

如果启用 `--write-modbus`，需要 `pymodbus`，它已经写入 `requirements.txt`。

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

吸盘点现在使用可见点云候选点评分，不再只是取最近一片点云的均值。评分项包括：

- `front`：候选点越靠近相机，越像暴露表面的前端，分越高。
- `edge`：候选点离 mask 边缘越远，越不容易漏气，分越高。
- `flatness`：局部 PCA 曲率越小，吸盘越容易密封，分越高。
- `normal`：局部法向越朝向相机或指定接近方向，分越高。
- `depth_stability`：局部深度越稳定，越不像噪声点，分越高。

多西瓜情况下，程序会对每个西瓜分别计算吸盘点和目标选择分。目标选择分主要看抓取点质量，同时加入检测置信度、点云数量、姿态置信度和到机械臂的水平距离。多个候选西瓜高度接近时，会提高“离机械臂近”的权重。

当前机械臂规格按“五轴注塑机械手”处理，程序不再默认做侧面椭球吸取。执行策略是：在相机真实可见、且符合固定吸盘方向的表面上选择吸点，然后生成 PLC 所需的 20 个值：

```text
x, y1, y2, z1, z2
每轴：position_mm, velocity_mm_s, acceleration_mm_s2, jerk_mm_s3
总数：5 x 4 = 20
```

默认配置见 `configs/injection_robot.yaml`。默认映射为 `x <- X`、`y1/y2 <- Y`、`z1/z2 <- Z`，实际设备如果 y1/y2 或 z1/z2 有偏置、限位或不同速度，需要在该 YAML 中修改。

可以通过下面参数调机械臂相关偏好：

```powershell
python .\main_debug_view.py --robot-origin-base 0 0 0 --tool-normal-base 0 0 1 --robot-config .\configs\injection_robot.yaml
python .\main_api.py --robot-origin-base 0 0 0 --tool-normal-base 0 0 1 --robot-config .\configs\injection_robot.yaml
```

`--tool-normal-base` 表示吸盘希望接触的表面法向，默认 `0 0 1`。如果你的 `robot_base` 中 Z 轴向下或机械手吸盘方向不同，需要按实际坐标系调整。

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

当 `status == "ok"` 时，`target.robot_command.register_values` 是准备写给 PLC 的 20 个整数值。默认缩放比例为 `10`，即 1 mm 会写成 10；可在 `configs/injection_robot.yaml` 的 `plc_registers.scale` 修改。

如果要让程序直接通过 Modbus TCP 写 PLC，需要显式打开：

```powershell
python .\main_api.py --write-modbus --modbus-host 192.168.1.10 --modbus-port 502 --modbus-start-address 0
```

默认不写 PLC，避免调试视觉时误动作。

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

## 排查：`Wait for frame timeout`

如果看到：

```text
Wait for frame timeout, you can try to increase the wait time! current timeout=1000
Camera started, but no valid frame set was received.
```

说明 SDK 已经发现设备，但启动 RGB-D 流后没有在等待时间内收到有效帧。先按下面顺序试：

```powershell
python .\scripts\test_camera.py --startup-timeout-ms 20000 --frame-timeout-ms 5000
```

如果仍然超时，放宽“必须 RGB 和 Depth 成对输出”的要求：

```powershell
python .\scripts\test_camera.py --startup-timeout-ms 20000 --frame-timeout-ms 5000 --no-full-frame-require
```

如果你的设备支持硬件 D2C，再试：

```powershell
python .\scripts\test_camera.py --hw-d2c --startup-timeout-ms 20000 --frame-timeout-ms 5000
```

调试界面也可以带同样参数：

```powershell
python .\main_debug_view.py --startup-timeout-ms 20000 --frame-timeout-ms 5000 --no-full-frame-require
```

日志里的 `Current firmware version ... < minimum required version ... CCP is not supported` 是固件能力 warning，不一定会阻止取流；但如果 Orbbec Viewer 也无法稳定出图，建议升级相机固件或切换到 Viewer 推荐的流配置。
