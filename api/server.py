from __future__ import annotations

import threading
import time
from typing import Any

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse

app = FastAPI(title="Watermelon Vision API")

_lock = threading.Lock()
_latest_result: dict[str, Any] = {
    "status": "no_target",
    "timestamp": time.time(),
    "camera_id": "gemini_435le_01",
    "target": None,
    "message": "System started. No target has been detected yet.",
}
_latest_frame_jpeg: bytes | None = None
_latest_frame_timestamp: float | None = None
_grabbed_count = 0
_grabbed_events: list[dict[str, Any]] = []


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


@app.get("/api/v1/dashboard/state")
def get_dashboard_state() -> dict[str, Any]:
    with _lock:
        latest = dict(_latest_result)
        frame_timestamp = _latest_frame_timestamp
        grabbed_count = _grabbed_count
        grabbed_events = list(_grabbed_events[-10:])

    target = latest.get("target") or {}
    command = target.get("robot_command") or {}
    plan = target.get("dual_arm_plan") or {}
    return {
        "timestamp": time.time(),
        "camera": {
            "camera_id": latest.get("camera_id"),
            "status": latest.get("status"),
            "message": latest.get("message"),
            "last_result_time": latest.get("timestamp"),
            "last_frame_time": frame_timestamp,
            "has_rgb_frame": frame_timestamp is not None,
        },
        "watermelon": _watermelon_summary(target, plan),
        "robot": {
            "plan_type": plan.get("plan_type"),
            "register_count": command.get("register_count", 0),
            "plc_register_start": command.get("plc_register_start", 0),
            "register_values": command.get("register_values", []),
            "axis_order": command.get("axis_order", []),
            "axis_values": command.get("axis_values", {}),
            "axis_positions": command.get("axis_positions", {}),
            "arm_assignments": command.get("arm_assignments", {}),
            "orientation_enabled": command.get("orientation_enabled", False),
            "orientation_axes": command.get("orientation_axes", {}),
            "is_command_valid": command.get("is_command_valid", False),
        },
        "counters": {
            "grabbed_count": grabbed_count,
            "planned_grab_count": _planned_grab_count(plan, command),
            "events": grabbed_events,
        },
        "raw_status": latest.get("status"),
    }


@app.get("/api/v1/dashboard/rgb.jpg")
def get_dashboard_rgb() -> Response:
    with _lock:
        image = _latest_frame_jpeg
    if image is None:
        return Response(status_code=204)
    return Response(content=image, media_type="image/jpeg")


@app.post("/api/v1/dashboard/grabbed/increment")
def increment_grabbed_count() -> dict[str, Any]:
    global _grabbed_count
    with _lock:
        target = (_latest_result.get("target") or {})
        command = target.get("robot_command") or {}
        plan = target.get("dual_arm_plan") or {}
        amount = max(1, _planned_grab_count(plan, command))
        _grabbed_count += amount
        event = {
            "timestamp": time.time(),
            "amount": amount,
            "total": _grabbed_count,
            "plan_type": plan.get("plan_type"),
            "register_count": command.get("register_count", 0),
        }
        _grabbed_events.append(event)
        if len(_grabbed_events) > 100:
            del _grabbed_events[:-100]
        return {"grabbed_count": _grabbed_count, "event": event}


@app.post("/api/v1/dashboard/grabbed/reset")
def reset_grabbed_count() -> dict[str, Any]:
    global _grabbed_count
    with _lock:
        _grabbed_count = 0
        _grabbed_events.clear()
        return {"grabbed_count": _grabbed_count}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML


def update_latest_result(result: dict[str, Any]) -> None:
    global _latest_result
    with _lock:
        _latest_result = result


def update_latest_frame_jpeg(frame_jpeg: bytes, timestamp: float | None = None) -> None:
    global _latest_frame_jpeg, _latest_frame_timestamp
    with _lock:
        _latest_frame_jpeg = frame_jpeg
        _latest_frame_timestamp = timestamp if timestamp is not None else time.time()


def _watermelon_summary(target: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    targets = _watermelon_target_summaries(target, plan)
    if not target:
        return {
            "detected": False,
            "detected_count": 0,
            "track_id": None,
            "class_name": None,
            "confidence": 0.0,
            "weight_kg": 0.0,
            "volume_liter": 0.0,
            "center_base_m": None,
            "grasp": None,
            "targets": [],
            "selected_targets": [],
        }
    volume = target.get("volume") or {}
    return {
        "detected": True,
        "detected_count": len(targets) if targets else 1,
        "track_id": target.get("track_id"),
        "class_name": target.get("class_name"),
        "confidence": target.get("detection_confidence", 0.0),
        "grasp_confidence": target.get("grasp_confidence", 0.0),
        "target_selection_score": target.get("target_selection_score", 0.0),
        "weight_kg": target.get("predicted_weight_kg", 0.0),
        "volume_liter": volume.get("volume_liter", 0.0),
        "center_base_m": target.get("center_base_m"),
        "grasp": target.get("grasp"),
        "targets": targets,
        "selected_targets": plan.get("selected_targets", []),
    }


def _watermelon_target_summaries(target: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    if not target:
        return []

    raw_targets = target.get("dashboard_targets") or []
    if raw_targets:
        normalized = [_normalize_dashboard_target(item, plan) for item in raw_targets if isinstance(item, dict)]
        return _visible_task_targets(normalized)

    selected_targets = plan.get("selected_targets") or []
    if selected_targets:
        return [_normalize_plan_target(item, target, plan) for item in selected_targets if isinstance(item, dict)]

    return [_primary_target_summary(target, plan)]


def _visible_task_targets(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not targets:
        return []
    selected = [item for item in targets if item.get("selected")]
    waiting = [item for item in targets if not item.get("selected")]
    if len(selected) >= 2:
        return selected[:2]
    if len(selected) == 1:
        return selected + waiting[:1]
    return targets[:2]


def _normalize_dashboard_target(item: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    target_id = _safe_int(item.get("target_id"))
    primary_target_id = _safe_int(plan.get("primary_target_id"))
    arm_id = item.get("arm_id") or _arm_id_for_target(plan, target_id)
    return {
        "target_id": target_id,
        "label": item.get("label") or _target_label(target_id, arm_id),
        "selected": bool(item.get("selected", False)),
        "primary": bool(item.get("primary", primary_target_id is not None and target_id == primary_target_id)),
        "arm_id": arm_id,
        "class_name": item.get("class_name"),
        "confidence": item.get("confidence", 0.0),
        "grasp_confidence": item.get("grasp_confidence", 0.0),
        "target_selection_score": item.get("target_selection_score", 0.0),
        "weight_kg": item.get("weight_kg", 0.0),
        "volume_liter": item.get("volume_liter", 0.0),
        "center_base_m": item.get("center_base_m"),
        "camera_depth_m": item.get("camera_depth_m"),
        "camera_distance_m": item.get("camera_distance_m"),
    }


def _normalize_plan_target(item: dict[str, Any], target: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    target_id = _safe_int(item.get("target_id"))
    primary_target_id = _safe_int(plan.get("primary_target_id"))
    arm_id = _arm_id_for_target(plan, target_id)
    return {
        "target_id": target_id,
        "label": _target_label(target_id, arm_id),
        "selected": True,
        "primary": primary_target_id is not None and target_id == primary_target_id,
        "arm_id": arm_id,
        "class_name": target.get("class_name"),
        "confidence": target.get("detection_confidence", 0.0),
        "grasp_confidence": ((plan.get("arm_grasps") or {}).get(arm_id) or {}).get("score", 0.0) if arm_id else 0.0,
        "target_selection_score": item.get("target_selection_score", 0.0),
        "weight_kg": item.get("predicted_weight_kg", 0.0),
        "volume_liter": item.get("volume_liter", 0.0),
        "center_base_m": item.get("center_base_m"),
        "camera_depth_m": item.get("camera_depth_m"),
        "camera_distance_m": item.get("camera_distance_m"),
    }


def _primary_target_summary(target: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    target_id = _safe_int(plan.get("primary_target_id"))
    arm_id = _arm_id_for_target(plan, target_id)
    volume = target.get("volume") or {}
    return {
        "target_id": target_id,
        "label": _target_label(target_id, arm_id),
        "selected": True,
        "primary": True,
        "arm_id": arm_id,
        "class_name": target.get("class_name"),
        "confidence": target.get("detection_confidence", 0.0),
        "grasp_confidence": target.get("grasp_confidence", 0.0),
        "target_selection_score": target.get("target_selection_score", 0.0),
        "weight_kg": target.get("predicted_weight_kg", 0.0),
        "volume_liter": volume.get("volume_liter", 0.0),
        "center_base_m": target.get("center_base_m"),
        "camera_depth_m": target.get("camera_depth_m"),
        "camera_distance_m": target.get("camera_distance_m"),
    }


def _arm_id_for_target(plan: dict[str, Any], target_id: int | None) -> str | None:
    if target_id is None:
        return None
    assignments = ((plan.get("robot_command") or {}).get("arm_assignments") or {})
    for arm_id, assignment in assignments.items():
        if not isinstance(assignment, dict) or not assignment.get("enabled", False):
            continue
        if _safe_int(assignment.get("target_id")) == target_id:
            return str(arm_id)
    return None


def _target_label(target_id: int | None, arm_id: str | None) -> str:
    if target_id is None:
        return arm_id or "目标"
    if arm_id:
        return f"{arm_id} T{target_id}"
    return f"T{target_id}"


def _safe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _planned_grab_count(plan: dict[str, Any], command: dict[str, Any]) -> int:
    selected = plan.get("selected_targets") or []
    if selected:
        return len(selected)
    assignments = command.get("arm_assignments") or {}
    enabled = sum(1 for item in assignments.values() if isinstance(item, dict) and item.get("enabled"))
    return enabled


DASHBOARD_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Watermelon Vision Dashboard</title>
  <style>
    :root {
      --bg: #0f1412;
      --panel: #151d19;
      --panel-2: #1b2520;
      --line: #2c3832;
      --text: #eef7f0;
      --muted: #91a399;
      --good: #35d07f;
      --warn: #f2bd4a;
      --bad: #ff6b6b;
      --accent: #53d5ff;
      --shadow: 0 18px 50px rgba(0, 0, 0, 0.32);
    }
    * { box-sizing: border-box; }
    html {
      height: 100%;
      overflow: hidden;
    }
    body {
      margin: 0;
      height: 100%;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
    }
    header {
      height: 52px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(15, 20, 18, 0.96);
      position: relative;
      top: 0;
      z-index: 5;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .top-status {
      display: flex;
      align-items: center;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--muted);
      box-shadow: 0 0 0 4px rgba(255,255,255,0.04);
    }
    .dot.ok { background: var(--good); }
    .dot.warn { background: var(--warn); }
    .dot.bad { background: var(--bad); }
    main {
      height: calc(100vh - 52px);
      padding: 12px;
      display: grid;
      grid-template-columns: minmax(420px, 1.45fr) minmax(360px, 1fr);
      gap: 12px;
      overflow: hidden;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
      min-height: 0;
    }
    .section-title {
      height: 36px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 12px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }
    .camera-panel {
      display: flex;
      flex-direction: column;
      min-height: 0;
    }
    .video {
      flex: 1 1 auto;
      min-height: 0;
      background: #070a09;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    .video img {
      width: 100%;
      height: 100%;
      max-height: none;
      object-fit: contain;
      display: block;
    }
    .placeholder {
      color: var(--muted);
      font-size: 14px;
      text-align: center;
      padding: 24px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      padding: 10px;
      background: var(--panel-2);
      border-top: 1px solid var(--line);
      flex: 0 0 auto;
    }
    .metric {
      min-height: 60px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,0.025);
    }
    .metric label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .metric strong {
      display: block;
      font-size: 20px;
      font-weight: 650;
      line-height: 1.05;
      white-space: pre-line;
    }
    .metric strong.compact {
      font-size: 14px;
      line-height: 1.25;
    }
    .side {
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      gap: 12px;
      align-content: stretch;
      min-height: 0;
      overflow: hidden;
    }
    .content { padding: 10px 12px; }
    .axis-panel {
      min-height: 0;
    }
    .axis-panel .content {
      overflow: hidden;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      text-align: right;
      padding: 7px 8px;
      border-bottom: 1px solid var(--line);
      white-space: nowrap;
    }
    th:first-child, td:first-child { text-align: left; }
    th {
      color: var(--muted);
      font-weight: 600;
      background: rgba(255,255,255,0.025);
    }
    tr:last-child td { border-bottom: 0; }
    .kv {
      display: grid;
      grid-template-columns: 132px minmax(0, 1fr);
      gap: 8px 12px;
      font-size: 13px;
      color: var(--text);
    }
    .kv span:nth-child(odd) { color: var(--muted); }
    .watermelon-list {
      display: grid;
      gap: 8px;
      margin-top: 0;
    }
    .watermelon-list.two-targets {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .target-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,0.025);
      padding: 8px;
      min-width: 0;
    }
    .target-card.selected {
      border-color: rgba(83, 213, 255, 0.58);
      background: rgba(83, 213, 255, 0.065);
    }
    .target-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
      margin-bottom: 6px;
      font-size: 12px;
    }
    .target-title strong {
      font-size: 13px;
      font-weight: 650;
      line-height: 1.2;
    }
    .target-grid {
      display: grid;
      grid-template-columns: max-content minmax(0, 1fr);
      gap: 4px 12px;
      font-size: 12px;
    }
    .target-grid span {
      white-space: nowrap;
    }
    .target-grid span:nth-child(odd) { color: var(--muted); }
    .target-grid span:nth-child(even) {
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .mono {
      font-family: Consolas, "Cascadia Mono", monospace;
      font-size: 12px;
      color: #d8e8df;
      word-break: break-all;
      line-height: 1.5;
    }
    .buttons {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .grab-panel .content {
      height: calc(100% - 36px);
      display: flex;
      align-items: center;
      padding: 18px 20px;
    }
    .grab-panel .event-spacer,
    .grab-panel #events {
      display: none;
    }
    button {
      border: 1px solid var(--line);
      background: #223129;
      color: var(--text);
      height: 32px;
      padding: 0 12px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
    }
    button.primary {
      background: #1f7a4a;
      border-color: #31945f;
    }
    button:hover { filter: brightness(1.1); }
    .grab-panel .buttons {
      gap: 14px;
    }
    .grab-panel button {
      height: 42px;
      padding: 0 20px;
      font-size: 16px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 0 9px;
      border-radius: 999px;
      background: rgba(255,255,255,0.07);
      color: var(--muted);
      font-size: 12px;
    }
    .badge.ok { color: var(--good); }
    .badge.warn { color: var(--warn); }
    .badge.bad { color: var(--bad); }
    @media (max-width: 980px) {
      html, body { overflow: auto; }
      main { height: auto; grid-template-columns: 1fr; padding: 12px; overflow: visible; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      header { padding: 0 14px; }
      .top-status { display: none; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Watermelon Vision Dashboard</h1>
    <div class="top-status"><span id="status-dot" class="dot"></span><span id="status-text">连接中</span><span id="clock"></span></div>
  </header>
  <main>
    <section class="camera-panel">
      <div class="section-title"><span>调试实时画面</span><span id="frame-time" class="badge">无画面</span></div>
      <div class="video">
        <img id="rgb" alt="Debug camera frame" />
        <div id="no-frame" class="placeholder">等待相机调试图像</div>
      </div>
      <div class="grid">
        <div class="metric"><label>目标状态</label><strong id="target-state">--</strong></div>
        <div class="metric"><label>预测重量</label><strong id="weight">--</strong></div>
        <div class="metric"><label>体积</label><strong id="volume">--</strong></div>
        <div class="metric"><label>累计抓取</label><strong id="grabbed">0</strong></div>
      </div>
    </section>

    <div class="side">
      <section class="watermelon-panel">
        <div class="section-title"><span>西瓜状态</span><span id="plan-badge" class="badge">--</span></div>
        <div class="content">
          <div class="watermelon-list" id="watermelon-list"></div>
        </div>
      </section>

      <section class="axis-panel">
        <div class="section-title"><span>机械臂轴状态</span><span id="register-badge" class="badge">0 values</span></div>
        <div class="content">
          <table>
            <thead><tr><th>轴</th><th>位置</th><th>速度</th><th>加速度</th><th>加加速度</th></tr></thead>
            <tbody id="axis-body"></tbody>
          </table>
        </div>
      </section>

      <section class="grab-panel">
        <div class="section-title"><span>抓取计数</span><span id="planned-count" class="badge">本次计划 0</span></div>
        <div class="content">
          <div class="buttons">
            <button class="primary" onclick="incrementGrabbed()">记录本次抓取</button>
            <button onclick="resetGrabbed()">重置计数</button>
          </div>
          <div class="event-spacer" style="height:12px"></div>
          <div class="mono" id="events"></div>
        </div>
      </section>
    </div>
  </main>
  <script>
    const fmt = (v, digits = 3) => Number.isFinite(Number(v)) ? Number(v).toFixed(digits) : "--";
    const point = p => p ? `(${fmt(p.x)}, ${fmt(p.y)}, ${fmt(p.z)})` : "--";
    const setText = (id, value) => { document.getElementById(id).textContent = value; };
    const setMetric = (id, value, compact = false) => {
      const el = document.getElementById(id);
      el.textContent = value;
      el.classList.toggle("compact", compact);
    };
    const escapeHtml = value => String(value ?? "--")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
    let hasRgbFrame = false;

    async function fetchState() {
      try {
        const res = await fetch(`/api/v1/dashboard/state?ts=${Date.now()}`);
        const data = await res.json();
        render(data);
      } catch (err) {
        setTopStatus("bad", "Dashboard 数据连接失败");
      }
    }

    function refreshImage() {
      const img = document.getElementById("rgb");
      const noFrame = document.getElementById("no-frame");
      if (!hasRgbFrame) {
        img.style.display = "none";
        noFrame.style.display = "block";
        return;
      }
      img.style.display = "block";
      noFrame.style.display = "none";
      img.src = `/api/v1/dashboard/rgb.jpg?ts=${Date.now()}`;
    }

    function render(data) {
      const cam = data.camera || {};
      const wm = data.watermelon || {};
      const robot = data.robot || {};
      const counters = data.counters || {};
      const targets = normalizedTargets(wm);
      const taskTargets = visibleTaskTargets(targets);
      const detectedCount = wm.detected ? (taskTargets.length || 1) : 0;
      const status = cam.status || "unknown";
      const ok = status === "ok";
      const cameraBad = status === "camera_error";
      setTopStatus(ok ? "ok" : cameraBad ? "bad" : "warn", `状态 ${status}`);
      setText("clock", new Date().toLocaleTimeString());
      setMetric("target-state", detectedCount > 0 ? `${detectedCount} 个目标` : "无目标", false);
      setMetric("weight", wm.detected ? targetMetricSummary(taskTargets, "weight_kg", "kg") : "--", taskTargets.length > 1);
      setMetric("volume", wm.detected ? targetMetricSummary(taskTargets, "volume_liter", "L") : "--", taskTargets.length > 1);
      setMetric("grabbed", counters.grabbed_count ?? 0, false);
      setText("planned-count", `本次计划 ${counters.planned_grab_count || 0}`);
      setText("frame-time", cam.last_frame_time ? new Date(cam.last_frame_time * 1000).toLocaleTimeString() : "无画面");
      setText("plan-badge", robot.plan_type || "--");
      setText("register-badge", `${robot.register_count || 0} values`);
      hasRgbFrame = Boolean(cam.has_rgb_frame);
      refreshImage();
      renderWatermelon(wm);
      renderAxes(robot);
      renderEvents(counters.events || []);
    }

    function setTopStatus(kind, text) {
      const dot = document.getElementById("status-dot");
      dot.className = `dot ${kind}`;
      setText("status-text", text);
    }

    function renderWatermelon(wm) {
      renderWatermelonTargets(wm, visibleTaskTargets(normalizedTargets(wm)));
    }

    function normalizedTargets(wm) {
      if (Array.isArray(wm.targets) && wm.targets.length) {
        return wm.targets;
      }
      if (Array.isArray(wm.selected_targets) && wm.selected_targets.length) {
        return wm.selected_targets.map(item => ({
          target_id: item.target_id,
          label: item.target_id == null ? "目标" : `T${item.target_id}`,
          selected: true,
          primary: false,
          arm_id: null,
          class_name: wm.class_name,
          confidence: wm.confidence,
          grasp_confidence: wm.grasp_confidence,
          target_selection_score: item.target_selection_score ?? wm.target_selection_score,
          weight_kg: item.predicted_weight_kg,
          volume_liter: item.volume_liter,
          center_base_m: item.center_base_m,
          camera_depth_m: item.camera_depth_m,
          camera_distance_m: item.camera_distance_m,
        }));
      }
      if (wm.detected) {
        return [{
          target_id: null,
          label: "主目标",
          selected: true,
          primary: true,
          arm_id: null,
          class_name: wm.class_name,
          confidence: wm.confidence,
          grasp_confidence: wm.grasp_confidence,
          target_selection_score: wm.target_selection_score,
          weight_kg: wm.weight_kg,
          volume_liter: wm.volume_liter,
          center_base_m: wm.center_base_m,
          camera_depth_m: null,
          camera_distance_m: null,
        }];
      }
      return [];
    }

    function watermelonName(target, index, targets) {
      if (targets.length > 1) {
        return `西瓜${String.fromCharCode(65 + index)}`;
      }
      return target.label || "西瓜A";
    }

    function visibleTaskTargets(targets) {
      const selected = targets.filter(target => target.selected);
      const waiting = targets.filter(target => !target.selected);
      if (selected.length >= 2) {
        return selected.slice(0, 2);
      }
      if (selected.length === 1) {
        return selected.concat(waiting.slice(0, 1));
      }
      return targets.slice(0, 2);
    }

    function targetMetricSummary(targets, key, unit) {
      if (!targets.length) {
        return "--";
      }
      if (targets.length === 1) {
        return `${fmt(targets[0][key], 2)} ${unit}`;
      }
      return targets.slice(0, 2)
        .map((target, index) => `${watermelonName(target, index, targets)} ${fmt(target[key], 2)} ${unit}`)
        .join("\n");
    }

    function renderWatermelonTargets(wm, targets) {
      const list = document.getElementById("watermelon-list");
      if (!wm.detected || !targets.length) {
        list.classList.remove("two-targets");
        list.innerHTML = `<div class="target-card"><div class="target-title"><strong>暂无目标</strong><span class="badge warn">waiting</span></div></div>`;
        return;
      }
      list.classList.toggle("two-targets", targets.length > 1);
      list.innerHTML = targets.map((target, index) => {
        const name = watermelonName(target, index, targets);
        const label = name;
        const state = target.selected ? "正在抓取" : "等待抓取";
        const badgeClass = target.selected ? "ok" : "warn";
        const arm = target.arm_id || "--";
        const rows = [
          ["机械臂", arm],
          ["预测重量", `${fmt(target.weight_kg, 2)} kg`],
          ["体积", `${fmt(target.volume_liter, 2)} L`],
          ["抓取置信", fmt(target.grasp_confidence, 3)],
          ["目标评分", fmt(target.target_selection_score, 3)],
        ];
        return `
          <div class="target-card ${target.selected ? "selected" : ""}">
            <div class="target-title">
              <strong>${escapeHtml(label)}</strong>
              <span class="badge ${badgeClass}">${escapeHtml(state)}</span>
            </div>
            <div class="target-grid">
              ${rows.map(([k, v]) => `<span>${escapeHtml(k)}</span><span>${escapeHtml(v)}</span>`).join("")}
            </div>
          </div>
        `;
      }).join("");
    }

    function renderAxes(robot) {
      const body = document.getElementById("axis-body");
      const axes = robot.axis_values || {};
      const order = robot.axis_order && robot.axis_order.length ? robot.axis_order : Object.keys(axes);
      body.innerHTML = order.map(axis => {
        const v = axes[axis] || {};
        const angular = Object.prototype.hasOwnProperty.call(v, "position_deg");
        const pos = angular ? `${fmt(v.position_deg, 2)} deg` : `${fmt(v.position_mm, 1)} mm`;
        const vel = angular ? `${fmt(v.velocity_deg_s, 1)} deg/s` : `${fmt(v.velocity_mm_s, 1)} mm/s`;
        const acc = angular ? `${fmt(v.acceleration_deg_s2, 1)} deg/s2` : `${fmt(v.acceleration_mm_s2, 1)} mm/s2`;
        const jerk = angular ? `${fmt(v.jerk_deg_s3, 1)} deg/s3` : `${fmt(v.jerk_mm_s3, 1)} mm/s3`;
        return `<tr><td>${axis}</td><td>${pos}</td><td>${vel}</td><td>${acc}</td><td>${jerk}</td></tr>`;
      }).join("") || `<tr><td colspan="5">暂无机械臂命令</td></tr>`;
    }

    function renderEvents(events) {
      document.getElementById("events").textContent = events.length
        ? events.slice().reverse().map(e => `${new Date(e.timestamp * 1000).toLocaleTimeString()} +${e.amount} total=${e.total} ${e.plan_type || ""}`).join("\n")
        : "暂无记录";
    }

    async function incrementGrabbed() {
      await fetch("/api/v1/dashboard/grabbed/increment", { method: "POST" });
      fetchState();
    }

    async function resetGrabbed() {
      await fetch("/api/v1/dashboard/grabbed/reset", { method: "POST" });
      fetchState();
    }

    fetchState();
    setInterval(fetchState, 700);
    setInterval(refreshImage, 50);
  </script>
</body>
</html>"""
