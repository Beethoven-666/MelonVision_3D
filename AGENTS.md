# 项目说明

## 项目概览

这是一个基于 Orbbec Gemini 435Le、`pyorbbecsdk2` 和 OpenCV/YOLO 的纯 Python 西瓜三维感知项目。程序从 Orbbec 相机获取对齐后的 RGB-D 数据，先用 HSV MVP 分割西瓜，再通过 mask 和 depth 生成局部点云，计算三维中心、PCA 尺寸、吸盘抓取点、表面法向、预抓取点和椭球体积，并为五轴注塑机械手生成 `x,y1,y2,z1,z2` 每轴 4 个值的 PLC 命令。

旧的 RealSense 入口仍保留为兼容包装，但底层已经切换为 Orbbec。

## 关键入口

- `main_debug_view.py`：调试界面，显示 RGB、Depth、mask、中心点、吸盘接触点、法向箭头和体积。
- `main_api.py`：正式服务入口，后台循环更新最新感知结果，并通过 FastAPI 提供 HTTP 接口。
- `scripts/test_camera.py`：Orbbec RGB-D 相机连通性和画面测试。
- `scripts/test_orbbec_yolov8_3d.py`：类似旧 `text/realsense_yolov8_3d.py` 的 YOLO 检测框中心三维坐标测试程序。
- `scripts/capture_dataset.py`：按键保存 RGB、Depth、Depth 可视化图和元数据。
- `calibration/save_intrinsics.py`：读取并保存 Orbbec RGB / Depth 内参、畸变和 depth-to-color 外参。

## 核心模块

- `camera/orbbec_camera.py`：`OrbbecCamera` 封装，负责启动 RGB-D 流、对齐和输出 `color_bgr` / `depth_mm`。
- `camera/frame_utils.py`：Orbbec color/depth frame 转 NumPy / OpenCV 格式。
- `perception/watermelon_segmenter.py`：HSV 西瓜分割 MVP，后续可替换为 YOLO-Seg。
- `perception/pointcloud_builder.py`：根据 mask、深度和内参反投影生成相机坐标系点云。
- `perception/watermelon_pipeline.py`：单帧感知流程编排。
- `geometry/pose_estimator.py`：PCA 估计三维中心、姿态和三轴尺寸。
- `geometry/suction_grasp.py`：估计吸盘接触点、表面法向和预抓取点。
- `volume/volume_estimator.py`：椭球体积估计。
- `robot/injection_molding_robot.py`：五轴注塑机械手命令生成，输出 `x,y1,y2,z1,z2` 的 20 个值。
- `robot/modbus_tcp_client.py`：可选 Modbus TCP 写 holding registers。
- `calibration/transform.py`：相机坐标系到 `robot_base` 坐标系的刚体变换。
- `api/server.py`：FastAPI 路由和最新结果缓存。

## 安装依赖

```powershell
python -m pip install -r requirements.txt
```

当前依赖包括：

- `pyorbbecsdk2`，导入名为 `pyorbbecsdk`
- `opencv-python`
- `numpy`
- `ultralytics`
- `pyyaml`
- `fastapi`
- `uvicorn`
- `pydantic`
- `pymodbus`

## 常用命令

测试相机：

```powershell
python .\scripts\test_camera.py
```

保存内参：

```powershell
python .\calibration\save_intrinsics.py --output .\configs\camera_intrinsics.yaml
```

运行调试界面：

```powershell
python .\main_debug_view.py
```

运行 API 服务：

```powershell
python .\main_api.py --host 0.0.0.0 --port 8000
```

YOLO 三维坐标测试：

```powershell
python .\scripts\test_orbbec_yolov8_3d.py --model yolov8n.pt
```

旧入口兼容：

```powershell
python .\text\realsense_yolov8_3d.py --model yolov8n.pt
python .\text\open_camera.py
```

## 坐标约定

- `depth_mm`：Orbbec 深度图，单位为毫米。
- 点云与抓取计算内部使用相机坐标系，单位为米。
- 对外给机械臂的结果使用 `robot_base` 坐标系，单位为米。
- `configs/T_base_camera.yaml` 当前是单位矩阵占位。真实抓取前必须完成相机到机械臂基坐标系的外参标定。

## API

状态接口：

```http
GET /api/v1/system/status
```

目标接口：

```http
GET /api/v1/watermelon/best_target
```

目标接口返回 `status`、`timestamp`、`camera_id` 和可选的 `target`。当 `status == "ok"` 时，`target` 内包含：

- `center_base_m`
- `target_selection_score`
- `target_selection`
- `axes_m`
- `volume`
- `grasp.contact_point_base_m`
- `grasp.surface_normal_base`
- `grasp.approach_vector_base`
- `grasp.pregrasp_point_base_m`
- `grasp.score_breakdown`
- `grasp.method`
- `grasp.is_inferred`
- `robot_command`

## 开发注意事项

- 当前 HSV 分割只是 MVP，现场需要根据光照调阈值。后续训练 YOLO-Seg 时，保持 `WatermelonSegmenter.segment()` 输出格式不变即可。
- 当前机械臂按五轴注塑机械手处理，默认不做侧面椭球吸取。吸盘点选择使用真实可见点云候选点评分，并通过 `--tool-normal-base` 约束固定吸盘方向。
- `configs/injection_robot.yaml` 控制五轴映射、每轴速度/加速度/加加速度、寄存器缩放和 Modbus 地址。PLC 侧真实地址、比例、限位必须现场确认后再启用 `--write-modbus`。
- 单视角体积估计受可见面限制，工程使用前需要真实数据校正。
- 修改相机流配置时要确认 Gemini 435Le 实际支持；代码会尽量回退到 SDK 默认 profile。
- 普通 CI 或无相机环境无法完整跑通相机链路；至少可以运行 Python 语法编译和合成数据链路测试。
- 不要把未标定的相机坐标直接交给机械臂。
