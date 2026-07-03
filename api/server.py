from __future__ import annotations

import threading
import time
from typing import Any

from fastapi import FastAPI

app = FastAPI(title="Watermelon Vision API")

_lock = threading.Lock()
_latest_result: dict[str, Any] = {
    "status": "no_target",
    "timestamp": time.time(),
    "camera_id": "gemini_435le_01",
    "target": None,
    "message": "System started. No target has been detected yet.",
}


@app.get("/api/v1/system/status")
def get_status() -> dict[str, Any]:
    with _lock:
        latest = dict(_latest_result)
    return {
        "status": "ok",
        "camera_connected": latest.get("status") != "camera_error",
        "calibration_valid": True,
        "last_update_time": latest.get("timestamp"),
        "frame_id": "robot_base",
    }


@app.get("/api/v1/watermelon/best_target")
def get_best_target() -> dict[str, Any]:
    with _lock:
        return dict(_latest_result)


def update_latest_result(result: dict[str, Any]) -> None:
    global _latest_result
    with _lock:
        _latest_result = result
