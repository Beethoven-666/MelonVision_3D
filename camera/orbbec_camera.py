from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2

try:
    from pyorbbecsdk import (
        AlignFilter,
        Config,
        Context,
        OBAlignMode,
        OBFormat,
        OBFrameAggregateOutputMode,
        OBLogLevel,
        OBSensorType,
        OBStreamType,
        Pipeline,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency pyorbbecsdk. Install requirements first: python -m pip install -r requirements.txt"
    ) from exc

from camera.frame_utils import depth_frame_to_mm, frame_to_bgr_image


@dataclass
class RGBDFrame:
    color_bgr: Any
    depth_mm: Any
    timestamp: dict[str, float]


class OrbbecCamera:
    def __init__(
        self,
        width: int = 0,
        height: int = 0,
        fps: int = 0,
        align_to_color: bool = True,
        use_hw_d2c: bool = False,
        enable_sync: bool = False,
        full_frame_require: bool = True,
        startup_timeout_ms: int = 10000,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.align_to_color = align_to_color
        self.use_hw_d2c = use_hw_d2c
        self.enable_sync = enable_sync
        self.full_frame_require = full_frame_require
        self.startup_timeout_ms = startup_timeout_ms
        self.pipeline: Optional[Pipeline] = None
        self.config: Optional[Config] = None
        self.align_filter: Optional[AlignFilter] = None
        self.camera_param = None
        self.running = False

    def start(self) -> None:
        ctx = Context()
        try:
            ctx.set_logger_level(OBLogLevel.WARNING)
        except Exception:
            pass

        device_list = ctx.query_devices()
        if device_list.get_count() == 0:
            raise RuntimeError(
                "No Orbbec camera was found. Check power, cable/network, driver, and SDK setup."
            )

        self.pipeline = Pipeline()

        if self.enable_sync:
            try:
                self.pipeline.enable_frame_sync()
            except Exception:
                pass

        if self.use_hw_d2c:
            self.config = self._build_hw_d2c_config()
            if self.config is None:
                raise RuntimeError("Hardware D2C is not supported by the active Orbbec device.")
        else:
            self.config = Config()
            color_profile = self._select_color_profile()
            depth_profile = self._select_depth_profile()
            self.config.enable_stream(color_profile)
            self.config.enable_stream(depth_profile)
            if self.full_frame_require:
                try:
                    self.config.set_frame_aggregate_output_mode(
                        OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
                    )
                except Exception:
                    pass

        self.pipeline.start(self.config)
        self.running = True
        if self.align_to_color and not self.use_hw_d2c:
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        frames = self._wait_for_first_frames()
        if frames is None:
            self.stop()
            raise RuntimeError(
                f"Camera started, but no valid frame set was received within {self.startup_timeout_ms} ms."
            )

        self.camera_param = self.pipeline.get_camera_param()

    def stop(self) -> None:
        if not self.running:
            return
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        finally:
            self.running = False

    def get_rgbd(self, timeout_ms: int = 1000) -> tuple[Any, Any, dict[str, float]] | tuple[None, None, None]:
        if self.pipeline is None:
            raise RuntimeError("Camera pipeline is not initialized. Call start() first.")

        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None, None, None

        if self.align_filter is not None:
            frames = self.align_filter.process(frames)
            if frames is None:
                return None, None, None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None, None, None

        color_bgr = frame_to_bgr_image(color_frame)
        if color_bgr is None:
            return None, None, None

        depth_mm = depth_frame_to_mm(depth_frame)
        color_height, color_width = color_bgr.shape[:2]
        if depth_mm.shape[:2] != (color_height, color_width):
            depth_mm = cv2.resize(depth_mm, (color_width, color_height), interpolation=cv2.INTER_NEAREST)

        timestamp = {
            "color_timestamp": float(color_frame.get_timestamp()),
            "depth_timestamp": float(depth_frame.get_timestamp()),
        }
        return color_bgr, depth_mm, timestamp

    def _wait_for_first_frames(self):
        if self.pipeline is None:
            raise RuntimeError("Camera pipeline is not initialized.")

        timeout_ms = max(int(self.startup_timeout_ms), 1000)
        deadline = cv2.getTickCount() / cv2.getTickFrequency() + timeout_ms / 1000.0
        while cv2.getTickCount() / cv2.getTickFrequency() < deadline:
            frames = self.pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            if not self.full_frame_require:
                return frames
            try:
                if frames.get_color_frame() is not None and frames.get_depth_frame() is not None:
                    return frames
            except Exception:
                return frames
        return None

    def get_intrinsics(self) -> dict[str, dict[str, float]]:
        if self.camera_param is None:
            raise RuntimeError("Camera parameters are not initialized. Call start() first.")

        return {
            "rgb": self._intrinsic_to_dict(self.camera_param.rgb_intrinsic),
            "depth": self._intrinsic_to_dict(self.camera_param.depth_intrinsic),
        }

    def get_color_intrinsic(self) -> dict[str, float]:
        return self.get_intrinsics()["rgb"]

    def _select_color_profile(self):
        if self.pipeline is None:
            raise RuntimeError("Camera pipeline is not initialized.")
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        formats = [OBFormat.RGB, OBFormat.MJPG, OBFormat.YUYV]
        if hasattr(OBFormat, "BGR"):
            formats.insert(1, OBFormat.BGR)
        for fmt in formats:
            try:
                return profile_list.get_video_stream_profile(self.width, self.height, fmt, self.fps)
            except Exception:
                continue
        return profile_list.get_default_video_stream_profile()

    def _select_depth_profile(self):
        if self.pipeline is None:
            raise RuntimeError("Camera pipeline is not initialized.")
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if self.width <= 0 and self.height <= 0 and self.fps <= 0:
            return profile_list.get_default_video_stream_profile()

        try:
            for index in range(len(profile_list)):
                profile = profile_list[index]
                width_ok = self.width <= 0 or profile.get_width() == self.width
                height_ok = self.height <= 0 or profile.get_height() == self.height
                fps_ok = self.fps <= 0 or profile.get_fps() == self.fps
                if width_ok and height_ok and fps_ok:
                    return profile
        except Exception:
            pass
        return profile_list.get_default_video_stream_profile()

    def _build_hw_d2c_config(self) -> Optional[Config]:
        if self.pipeline is None:
            raise RuntimeError("Camera pipeline is not initialized.")
        config = Config()
        try:
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            for index in range(len(color_profiles)):
                color_profile = color_profiles[index]
                if color_profile.get_format() != OBFormat.RGB:
                    continue
                depth_profiles = self.pipeline.get_d2c_depth_profile_list(
                    color_profile, OBAlignMode.HW_MODE
                )
                if len(depth_profiles) == 0:
                    continue
                config.enable_stream(depth_profiles[0])
                config.enable_stream(color_profile)
                config.set_align_mode(OBAlignMode.HW_MODE)
                return config
        except Exception:
            return None
        return None

    @staticmethod
    def _intrinsic_to_dict(intrinsic) -> dict[str, float]:
        return {
            "width": int(intrinsic.width),
            "height": int(intrinsic.height),
            "fx": float(intrinsic.fx),
            "fy": float(intrinsic.fy),
            "cx": float(intrinsic.cx),
            "cy": float(intrinsic.cy),
        }


def list_orbbec_devices() -> list[dict[str, str]]:
    ctx = Context()
    device_list = ctx.query_devices()
    devices: list[dict[str, str]] = []
    for index in range(device_list.get_count()):
        device = device_list.get_device_by_index(index)
        info = device.get_device_info()
        item = {
            "index": str(index),
            "name": _safe_info_value(info, "get_name"),
            "serial_number": _safe_info_value(info, "get_serial_number"),
            "connection_type": _safe_info_value(info, "get_connection_type"),
        }
        devices.append(item)
    return devices


def _safe_info_value(info, method_name: str) -> str:
    method = getattr(info, method_name, None)
    if method is None:
        return ""
    try:
        return str(method())
    except Exception:
        return ""
