# Orbbec Watermelon Vision 快速入口

当前项目主入口已经切换到 Orbbec 相机、调试界面和 API 服务；旧 `text/` 目录入口不再保留。

最常用命令：

```powershell
python .\scripts\test_camera.py
python .\main_debug_view.py
python .\main_api.py --host 0.0.0.0 --port 8000
python .\main_api.py --host 0.0.0.0 --port 8000 --print-json
python .\scripts\test_orbbec_yolov8_3d.py --model yolov8n.pt
```

`main_api.py` 默认会在控制台打印感知摘要和 PLC 寄存器值；需要完整 API 返回结构时加 `--print-json`。

长时间运行时，相机取流由后台线程独立完成，只保留最新 RGB-D 帧。连续超时后会自动重启相机 pipeline：

```powershell
python .\main_api.py --camera-restart-timeouts 10 --camera-restart-wait-sec 2.0
python .\main_debug_view.py --camera-restart-timeouts 10 --camera-restart-wait-sec 2.0
```

这两个参数不改变图像分辨率、深度值或点云计算，只提升取流稳定性。

启动 `main_api.py` 后，在浏览器打开：

```http
http://127.0.0.1:8000/dashboard
```

该界面显示深度相机 RGB 画面、西瓜识别状态、机械臂轴状态、PLC 寄存器值和抓取计数。抓取计数通过界面按钮手动记录，避免相机每帧重复计数。

网页 RGB 画面由独立线程发布，不受西瓜识别耗时影响。需要提高或降低网页画面帧率时使用：

```powershell
python .\main_api.py --dashboard-fps 20 --dashboard-jpeg-quality 80
```

`--dashboard-fps` 只影响网页 RGB 刷新，不改变识别精度。

## PLC 输出格式

默认配置保持当前五轴格式：

```text
x, y1, y2, z1, z2
每轴 4 个值：position, velocity, acceleration, jerk
总计 20 个 PLC 值
```

以后增加吸盘旋转舵机时，先完成 PLC 地址映射和旋转零位标定，再把 `configs/injection_robot.yaml` 中的 `orientation_axes.enabled` 改成 `true`。打开后输出：

```text
x, y1, y2, z1, z2, r1, r2
每轴 4 个值
总计 28 个 PLC 值
```

`r1` 对应 arm1 吸盘旋转角，`r2` 对应 arm2。角度默认使用 `plc_registers.angular_scale: 100.0`，即 `1.23 deg -> 123`。当前 PLC 只支持 20 个值时必须保持关闭。
