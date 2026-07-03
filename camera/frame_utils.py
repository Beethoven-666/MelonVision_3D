from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

try:
    from pyorbbecsdk import OBFormat
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency pyorbbecsdk. Install requirements first: python -m pip install -r requirements.txt"
    ) from exc


def _frame_bytes(frame) -> np.ndarray:
    return np.frombuffer(frame.get_data(), dtype=np.uint8)


def _i420_to_bgr(data: np.ndarray, width: int, height: int) -> np.ndarray:
    yuv = data.reshape((height * 3 // 2, width))
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)


def _nv12_to_bgr(data: np.ndarray, width: int, height: int) -> np.ndarray:
    yuv = data.reshape((height * 3 // 2, width))
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)


def _nv21_to_bgr(data: np.ndarray, width: int, height: int) -> np.ndarray:
    yuv = data.reshape((height * 3 // 2, width))
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)


def frame_to_bgr_image(frame) -> Optional[np.ndarray]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = _frame_bytes(frame)

    if color_format == OBFormat.RGB:
        rgb = data.reshape((height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if hasattr(OBFormat, "BGR") and color_format == OBFormat.BGR:
        return data.reshape((height, width, 3)).copy()

    if color_format == OBFormat.YUYV:
        yuyv = data.reshape((height, width, 2))
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)

    if hasattr(OBFormat, "UYVY") and color_format == OBFormat.UYVY:
        uyvy = data.reshape((height, width, 2))
        return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)

    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    if color_format == OBFormat.I420:
        return _i420_to_bgr(data, width, height)

    if color_format == OBFormat.NV12:
        return _nv12_to_bgr(data, width, height)

    if color_format == OBFormat.NV21:
        return _nv21_to_bgr(data, width, height)

    return None


def depth_frame_to_mm(depth_frame) -> np.ndarray:
    depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
        depth_frame.get_height(),
        depth_frame.get_width(),
    )
    return depth_raw.astype(np.float32) * float(depth_frame.get_depth_scale())
