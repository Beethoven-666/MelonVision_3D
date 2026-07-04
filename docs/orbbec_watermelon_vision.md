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
