from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from camera.orbbec_camera import OrbbecCamera


@dataclass(frozen=True)
class LatestRGBDFrame:
    color_bgr: Any
    depth_mm: Any
    timestamp: dict[str, float]
    sequence: int
    received_at: float


class RGBDStreamWorker:
    def __init__(
        self,
        width: int = 0,
        height: int = 0,
        fps: int = 0,
        use_hw_d2c: bool = False,
        full_frame_require: bool = True,
        startup_timeout_ms: int = 10000,
        frame_timeout_ms: int = 2000,
        restart_after_timeouts: int = 10,
        restart_wait_sec: float = 2.0,
    ) -> None:
        self.camera = OrbbecCamera(
            width,
            height,
            fps,
            use_hw_d2c=use_hw_d2c,
            full_frame_require=full_frame_require,
            startup_timeout_ms=startup_timeout_ms,
        )
        self.frame_timeout_ms = frame_timeout_ms
        self.restart_after_timeouts = restart_after_timeouts
        self.restart_wait_sec = restart_wait_sec
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_frame: LatestRGBDFrame | None = None
        self._intrinsic: dict[str, float] | None = None
        self._sequence = 0
        self._consecutive_timeouts = 0
        self._restart_count = 0
        self._status: dict[str, Any] = {
            "state": "stopped",
            "message": "RGB-D stream has not started.",
            "consecutive_timeouts": 0,
            "restart_count": 0,
            "last_frame_time": None,
        }

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="rgbd-stream-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.camera.stop()
        self._set_status("stopped", "RGB-D stream stopped.")

    def wait_until_ready(self, timeout_sec: float = 10.0) -> bool:
        return self._ready_event.wait(timeout_sec)

    def get_latest(self) -> LatestRGBDFrame | None:
        with self._lock:
            return self._latest_frame

    def get_color_intrinsic(self) -> dict[str, float] | None:
        with self._lock:
            return dict(self._intrinsic) if self._intrinsic is not None else None

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if not self._ensure_camera_running():
                self._sleep_restart_interval()
                continue

            try:
                color_bgr, depth_mm, timestamp = self.camera.get_rgbd(self.frame_timeout_ms)
            except Exception as exc:
                self._set_status("camera_error", f"RGB-D get frame failed: {exc}")
                self._restart_camera("exception_during_get_rgbd")
                continue

            if color_bgr is None or depth_mm is None:
                self._handle_timeout()
                continue

            self._sequence += 1
            frame = LatestRGBDFrame(
                color_bgr=color_bgr,
                depth_mm=depth_mm,
                timestamp=timestamp or {},
                sequence=self._sequence,
                received_at=time.time(),
            )
            with self._lock:
                self._latest_frame = frame
                self._consecutive_timeouts = 0
                self._status = {
                    "state": "running",
                    "message": "RGB-D stream is running.",
                    "consecutive_timeouts": 0,
                    "restart_count": self._restart_count,
                    "last_frame_time": frame.received_at,
                    "sequence": frame.sequence,
                }

    def _ensure_camera_running(self) -> bool:
        if self.camera.running:
            return True
        try:
            self._set_status("starting", "Starting RGB-D stream.")
            self.camera.start()
            intrinsic = self.camera.get_color_intrinsic()
            with self._lock:
                self._intrinsic = intrinsic
            self._ready_event.set()
            self._set_status("running", "RGB-D stream started.")
            return True
        except Exception as exc:
            self._ready_event.clear()
            self.camera.stop()
            self._set_status("camera_error", f"Failed to start RGB-D stream: {exc}")
            return False

    def _handle_timeout(self) -> None:
        self._consecutive_timeouts += 1
        self._set_status(
            "timeout",
            f"No valid RGB-D frame was received. consecutive_timeouts={self._consecutive_timeouts}",
        )
        if self.restart_after_timeouts > 0 and self._consecutive_timeouts >= self.restart_after_timeouts:
            self._restart_camera("consecutive_frame_timeouts")

    def _restart_camera(self, reason: str) -> None:
        self._restart_count += 1
        self._ready_event.clear()
        self._set_status(
            "restarting",
            f"Restarting RGB-D stream after {self._consecutive_timeouts} timeouts. reason={reason}",
        )
        self.camera.stop()
        self._sleep_restart_interval()
        self._consecutive_timeouts = 0

    def _sleep_restart_interval(self) -> None:
        deadline = time.time() + max(float(self.restart_wait_sec), 0.0)
        while not self._stop_event.is_set() and time.time() < deadline:
            time.sleep(0.05)

    def _set_status(self, state: str, message: str) -> None:
        with self._lock:
            self._status = {
                "state": state,
                "message": message,
                "consecutive_timeouts": self._consecutive_timeouts,
                "restart_count": self._restart_count,
                "last_frame_time": (
                    self._latest_frame.received_at if self._latest_frame is not None else None
                ),
                "sequence": self._latest_frame.sequence if self._latest_frame is not None else 0,
            }
