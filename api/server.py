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
    if not target:
        return {
            "detected": False,
            "track_id": None,
            "class_name": None,
            "confidence": 0.0,
            "weight_kg": 0.0,
            "volume_liter": 0.0,
            "center_base_m": None,
            "grasp": None,
            "selected_targets": [],
        }
    volume = target.get("volume") or {}
    return {
        "detected": True,
        "track_id": target.get("track_id"),
        "class_name": target.get("class_name"),
        "confidence": target.get("detection_confidence", 0.0),
        "grasp_confidence": target.get("grasp_confidence", 0.0),
        "target_selection_score": target.get("target_selection_score", 0.0),
        "weight_kg": target.get("predicted_weight_kg", 0.0),
        "volume_liter": volume.get("volume_liter", 0.0),
        "center_base_m": target.get("center_base_m"),
        "grasp": target.get("grasp"),
        "selected_targets": plan.get("selected_targets", []),
    }


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
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
    }
    header {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(15, 20, 18, 0.96);
      position: sticky;
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
      padding: 20px;
      display: grid;
      grid-template-columns: minmax(420px, 1.45fr) minmax(360px, 1fr);
      gap: 18px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .section-title {
      height: 44px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 14px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }
    .video {
      min-height: 420px;
      background: #070a09;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    .video img {
      width: 100%;
      height: 100%;
      max-height: calc(100vh - 160px);
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
      gap: 12px;
      padding: 14px;
      background: var(--panel-2);
      border-top: 1px solid var(--line);
    }
    .metric {
      min-height: 74px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,0.025);
    }
    .metric label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }
    .metric strong {
      display: block;
      font-size: 22px;
      font-weight: 650;
      line-height: 1.05;
    }
    .side {
      display: grid;
      gap: 18px;
      align-content: start;
    }
    .content { padding: 14px; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      text-align: right;
      padding: 9px 8px;
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
    button {
      border: 1px solid var(--line);
      background: #223129;
      color: var(--text);
      height: 34px;
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
      main { grid-template-columns: 1fr; padding: 12px; }
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
    <section>
      <div class="section-title"><span>RGB 实时画面</span><span id="frame-time" class="badge">无画面</span></div>
      <div class="video">
        <img id="rgb" alt="RGB camera frame" />
        <div id="no-frame" class="placeholder">等待相机 RGB 图像</div>
      </div>
      <div class="grid">
        <div class="metric"><label>目标状态</label><strong id="target-state">--</strong></div>
        <div class="metric"><label>预测重量</label><strong id="weight">--</strong></div>
        <div class="metric"><label>体积</label><strong id="volume">--</strong></div>
        <div class="metric"><label>累计抓取</label><strong id="grabbed">0</strong></div>
      </div>
    </section>

    <div class="side">
      <section>
        <div class="section-title"><span>西瓜状态</span><span id="plan-badge" class="badge">--</span></div>
        <div class="content kv" id="watermelon-kv"></div>
      </section>

      <section>
        <div class="section-title"><span>机械臂轴状态</span><span id="register-badge" class="badge">0 values</span></div>
        <div class="content">
          <table>
            <thead><tr><th>轴</th><th>位置</th><th>速度</th><th>加速度</th><th>加加速度</th></tr></thead>
            <tbody id="axis-body"></tbody>
          </table>
        </div>
      </section>

      <section>
        <div class="section-title"><span>双臂与 PLC</span><span id="orientation-badge" class="badge">orientation off</span></div>
        <div class="content">
          <div class="kv" id="arm-kv"></div>
          <div style="height:12px"></div>
          <div class="mono" id="registers"></div>
        </div>
      </section>

      <section>
        <div class="section-title"><span>抓取计数</span><span id="planned-count" class="badge">本次计划 0</span></div>
        <div class="content">
          <div class="buttons">
            <button class="primary" onclick="incrementGrabbed()">记录本次抓取</button>
            <button onclick="resetGrabbed()">重置计数</button>
          </div>
          <div style="height:12px"></div>
          <div class="mono" id="events"></div>
        </div>
      </section>
    </div>
  </main>
  <script>
    const fmt = (v, digits = 3) => Number.isFinite(Number(v)) ? Number(v).toFixed(digits) : "--";
    const point = p => p ? `(${fmt(p.x)}, ${fmt(p.y)}, ${fmt(p.z)})` : "--";
    const setText = (id, value) => { document.getElementById(id).textContent = value; };
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
      const status = cam.status || "unknown";
      const ok = status === "ok";
      const cameraBad = status === "camera_error";
      setTopStatus(ok ? "ok" : cameraBad ? "bad" : "warn", `状态 ${status}`);
      setText("clock", new Date().toLocaleTimeString());
      setText("target-state", wm.detected ? "已识别" : "无目标");
      setText("weight", wm.detected ? `${fmt(wm.weight_kg, 2)} kg` : "--");
      setText("volume", wm.detected ? `${fmt(wm.volume_liter, 2)} L` : "--");
      setText("grabbed", counters.grabbed_count ?? 0);
      setText("planned-count", `本次计划 ${counters.planned_grab_count || 0}`);
      setText("frame-time", cam.last_frame_time ? new Date(cam.last_frame_time * 1000).toLocaleTimeString() : "无画面");
      setText("plan-badge", robot.plan_type || "--");
      setText("register-badge", `${robot.register_count || 0} values`);
      setText("orientation-badge", robot.orientation_enabled ? "orientation on" : "orientation off");
      hasRgbFrame = Boolean(cam.has_rgb_frame);
      refreshImage();
      renderWatermelon(wm, cam);
      renderAxes(robot);
      renderArms(robot);
      renderEvents(counters.events || []);
    }

    function setTopStatus(kind, text) {
      const dot = document.getElementById("status-dot");
      dot.className = `dot ${kind}`;
      setText("status-text", text);
    }

    function renderWatermelon(wm, cam) {
      const rows = [
        ["相机", cam.camera_id || "--"],
        ["消息", cam.message || "--"],
        ["Track ID", wm.track_id ?? "--"],
        ["类别", wm.class_name || "--"],
        ["检测置信度", fmt(wm.confidence, 3)],
        ["抓取置信度", fmt(wm.grasp_confidence, 3)],
        ["目标评分", fmt(wm.target_selection_score, 3)],
        ["中心 robot_base", point(wm.center_base_m)],
        ["吸点 robot_base", point((wm.grasp || {}).contact_point_base_m)],
        ["法向", point((wm.grasp || {}).surface_normal_base)],
      ];
      document.getElementById("watermelon-kv").innerHTML = rows.map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join("");
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

    function renderArms(robot) {
      const assignments = robot.arm_assignments || {};
      const rows = [];
      for (const arm of ["arm1", "arm2"]) {
        const a = assignments[arm] || {};
        rows.push([`${arm} 启用`, a.enabled ? "true" : "false"]);
        rows.push([`${arm} 目标`, a.target_id ?? "--"]);
        rows.push([`${arm} 吸点 mm`, point(a.contact_point_mm)]);
        rows.push([`${arm} 旋转角`, a.orientation_angle_deg == null ? "--" : `${fmt(a.orientation_angle_deg, 2)} deg`]);
      }
      document.getElementById("arm-kv").innerHTML = rows.map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join("");
      const values = robot.register_values || [];
      document.getElementById("registers").textContent = values.length ? `PLC registers: [${values.join(", ")}]` : "PLC registers: --";
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
