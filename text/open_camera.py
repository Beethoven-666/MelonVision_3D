try:
	import pyrealsense2 as rs
except ModuleNotFoundError as exc:
	raise SystemExit(
		"Missing dependency pyrealsense2. Install it first, for example: pip install pyrealsense2"
	) from exc


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

try:
	frames = pipeline.wait_for_frames()
	depth_frame = frames.get_depth_frame()
	color_frame = frames.get_color_frame()

	if not depth_frame:
		raise RuntimeError("Failed to get depth frame")
	if not color_frame:
		raise RuntimeError("Failed to get color frame")

	# 获取相机内参
	depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
	color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

	# 命令行打印相机内参显示
	print("Depth Intrinsics:", depth_intrinsics)
	print("Color Intrinsics:", color_intrinsics)
finally:
	pipeline.stop()