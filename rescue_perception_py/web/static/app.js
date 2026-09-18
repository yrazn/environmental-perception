"use strict";

// 页面字段映射：融合权重和可信度在后端使用不同键名，不能共用一个 key。
const SENSORS = [
  { weight: "rgb", credibility: "rgb", label: "可见光", color: "#64748b" },
  { weight: "thermal", credibility: "thermal", label: "热成像", color: "#ef4444" },
  { weight: "lidar", credibility: "lidar_3d", label: "3D LiDAR", color: "#3b82f6" },
  { weight: "radar", credibility: "radar_4d", label: "4D 毫米波", color: "#22c55e" },
  { weight: "lidar2d", credibility: "lidar_2d", label: "2D LiDAR", color: "#8b5cf6" },
  { weight: "gas", credibility: "gas", label: "气体", color: "#f59e0b" },
];

const GAS_NAMES = { 0: "SAFE", 1: "CAUTION", 2: "DANGER", 3: "DEADLY" };
const CLASS_TAGS = [
  { min: 1, max: 2, cls: "fire", text: "火/烟" },
  { min: 3, max: 7, cls: "person", text: "人员" },
  { min: 8, max: 12, cls: "vehicle", text: "车辆" },
  { min: 19, max: 19, cls: "person", text: "人员候选" },
  { min: 20, max: 20, cls: "fire", text: "火源候选" },
];
const SOURCE_NAMES = [
  [1, "RGB"], [2, "热"], [4, "LiDAR"], [8, "雷达"], [16, "气体"],
];

let latest = null;
let paused = false;
let previousTime = -1;
// 只在事件从“未出现”变为“出现”时写入日志，避免 SSE 每帧重复刷屏。
let activeEvents = new Set();
let connected = false;
const history = {
  time: [], smoke: [], rgb: [], thermal: [], lidar: [], radar: [], gas: [],
};
const MAX_POINTS = 600;

const els = {
  runMode: document.getElementById("runMode"),
  mode: document.getElementById("mode"),
  status: document.getElementById("status"),
  envRisk: document.getElementById("envRisk"),
  safetyAction: document.getElementById("safetyAction"),
  speedLimit: document.getElementById("speedLimit"),
  clock: document.getElementById("clock"),
  sensorWeights: document.getElementById("sensorWeights"),
  envMetrics: document.getElementById("envMetrics"),
  thermalMetrics: document.getElementById("thermalMetrics"),
  targetList: document.getElementById("targetList"),
  inspectionList: document.getElementById("inspectionList"),
  eventList: document.getElementById("eventList"),
  map: document.getElementById("map"),
  chart: document.getElementById("chart"),
  pauseBtn: document.getElementById("pauseBtn"),
  stepBtn: document.getElementById("stepBtn"),
  resetBtn: document.getElementById("resetBtn"),
  speedSelect: document.getElementById("speedSelect"),
  connectionDot: document.getElementById("connectionDot"),
  connectionText: document.getElementById("connectionText"),
  modeChip: document.getElementById("modeChip"),
  statusChip: document.getElementById("statusChip"),
  riskChip: document.getElementById("riskChip"),
  actionChip: document.getElementById("actionChip"),
  speedChip: document.getElementById("speedChip"),
  mapSummary: document.getElementById("mapSummary"),
  targetCount: document.getElementById("targetCount"),
  inspectionCount: document.getElementById("inspectionCount"),
  eventCount: document.getElementById("eventCount"),
};

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function fitCanvas(canvas, aspect) {
  // 按设备像素比扩大实际画布，避免高分屏 Canvas 模糊。
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(280, rect.width);
  const h = Math.max(200, rect.height || w / aspect);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function setText(id, value) {
  els[id].textContent = value;
}

function setConnection(isConnected, text) {
  connected = isConnected;
  els.connectionDot.className = `connection-dot ${isConnected ? "online" : "offline"}`;
  els.connectionText.textContent = text;
}

function setTone(element, tone) {
  element.dataset.tone = tone;
}

function riskTone(level) {
  return ["ok", "warning", "high", "critical"][level] || "neutral";
}

function renderSensors() {
  // 权重决定融合贡献，可信度决定当前传感器是否健康，两者并列显示。
  els.sensorWeights.innerHTML = "";
  for (const sensor of SENSORS) {
    const weight = Number(latest.weights[sensor.weight] || 0);
    const cred = Number(latest.credibilities[sensor.credibility] || 0);
    const credTone = cred < 0.1 ? "failed" : cred < 0.3 ? "degraded" : "healthy";
    const row = document.createElement("div");
    row.className = `sensor-row ${credTone}`;
    row.innerHTML =
      `<span class="label">${sensor.label}</span>` +
      `<div class="bar-track"><div class="bar-fill" style="width:${clamp(weight * 100, 0, 100)}%;--bar-color:${sensor.color}"></div></div>` +
      `<span class="num">${weight.toFixed(2)}</span>` +
      `<small class="cred"><span class="health-dot"></span>可信度 ${cred.toFixed(2)} · ${credTone === "healthy" ? "正常" : credTone === "degraded" ? "退化" : "不可用"}</small>`;
    els.sensorWeights.appendChild(row);
  }
}

function renderEnv() {
  const rows = [
    ["烟雾指数", latest.smoke_score.toFixed(2)],
    ["可见度", latest.smoke_visibility_name],
    ["气体风险", GAS_NAMES[latest.gas_risk_level] || "-"],
    ["温度", `${latest.ambient_temperature.toFixed(1)} °C`],
    ["湿度", `${latest.ambient_humidity.toFixed(1)} %`],
    ["综合风险得分", Number(latest.risk_score || 0).toFixed(2)],
  ];
  els.envMetrics.innerHTML = rows.map(([label, value]) =>
    `<div class="metric-row"><span class="label">${label}</span><span>${value}</span></div>`
  ).join("");
}

function renderThermal() {
  const rows = [
    ["热风险", String(latest.thermal_risk_level)],
    ["最高温度", `${latest.thermal_max_temperature.toFixed(1)} °C`],
    ["热点数量", String(latest.thermal_hotspot_count)],
    ["风险置信度", latest.thermal_confidence.toFixed(2)],
  ];
  els.thermalMetrics.innerHTML = rows.map(([label, value]) =>
    `<div class="metric-row"><span class="label">${label}</span><span>${value}</span></div>`
  ).join("");
}

function tagFor(classId) {
  for (const rule of CLASS_TAGS) {
    if (classId >= rule.min && classId <= rule.max) {
      return `<span class="tag tag-${rule.cls}">${rule.text}</span>`;
    }
  }
  return `<span class="tag tag-other">其他</span>`;
}

function sourceText(mask) {
  return SOURCE_NAMES.filter(([bit]) => (mask & bit) === bit)
    .map(([, name]) => name).join("|") || "-";
}

function renderTargets() {
  els.targetCount.textContent = String(latest.targets.length);
  if (!latest.targets.length) {
    els.targetList.innerHTML = `<div class="metric-row"><span class="label">暂无目标</span></div>`;
    return;
  }
  const head =
    "<table><thead><tr><th>ID</th><th>类别</th><th>置信度</th><th>位置</th><th>来源</th><th>确认</th><th>风险</th></tr></thead><tbody>";
  const body = latest.targets.map((t) => {
    const pos = `${t.position[0].toFixed(1)},${t.position[1].toFixed(1)}`;
    return `<tr>` +
      `<td>${t.track_id}</td>` +
      `<td>${tagFor(t.class_id)}</td>` +
      `<td>${t.confidence.toFixed(2)}</td>` +
      `<td>${pos}</td>` +
      `<td>${sourceText(t.source_mask)}</td>` +
      `<td>${t.confirmed ? "是" : "否"}</td>` +
      `<td>${t.risk_level}</td>` +
      `</tr>`;
  }).join("");
  els.targetList.innerHTML = head + body + "</tbody></table>";
}

function renderInspection() {
  // rescue 模式通常为空；inspection 模式显示数量和最多三条异常详情。
  const cracks = latest.cracks || [];
  const water = latest.water_regions || [];
  const facilities = latest.facility_anomalies || [];
  const total = cracks.length + water.length + facilities.length;
  els.inspectionCount.textContent = String(total);
  const summary = `<div class="inspection-summary">` +
    `<div><strong>${cracks.length}</strong><span>裂缝</span></div>` +
    `<div><strong>${water.length}</strong><span>积水</span></div>` +
    `<div><strong>${facilities.length}</strong><span>设施</span></div>` +
    `</div>`;
  const details = [];
  for (const item of cracks.slice(0, 3)) {
    details.push(`<div class="inspection-item crack"><b>裂缝 #${item.crack_id}</b><span>${item.length_m.toFixed(2)} m · ${item.width_max_mm.toFixed(1)} mm · 可信 ${item.confidence.toFixed(2)}</span></div>`);
  }
  for (const item of water.slice(0, 3)) {
    details.push(`<div class="inspection-item water"><b>${item.kind === "seepage" ? "渗水" : "积水"} #${item.region_id}</b><span>${item.depth_level} · 可信 ${item.confidence.toFixed(2)}</span></div>`);
  }
  for (const item of facilities.slice(0, 3)) {
    details.push(`<div class="inspection-item facility"><b>${item.facility_class} #${item.anomaly_id}</b><span>${item.anomaly_type} · ${item.thermal_temperature.toFixed(1)} °C</span></div>`);
  }
  els.inspectionList.innerHTML = summary + (details.join("") || `<div class="empty-state">当前模式无巡检异常</div>`);
}

function renderEvents() {
  if (latest.time < previousTime) {
    els.eventList.innerHTML = "";
    activeEvents.clear();
  }
  previousTime = latest.time;
  const currentEvents = new Set(latest.events);
  for (const event of currentEvents) {
    if (activeEvents.has(event)) continue;
    const li = document.createElement("li");
    li.textContent = `[${latest.time.toFixed(1)}s] ${event}`;
    li.className = "new";
    els.eventList.prepend(li);
  }
  activeEvents = currentEvents;
  while (els.eventList.children.length > 40) {
    els.eventList.removeChild(els.eventList.lastChild);
  }
  els.eventCount.textContent = String(els.eventList.children.length);
}

function drawMap() {
  // 世界坐标约定：x 为隧道前进方向，y 为横向；屏幕向上对应 x 增大。
  const { ctx, w, h } = fitCanvas(els.map, 900 / 560);
  ctx.clearRect(0, 0, w, h);
  const scale = Math.min(w, h) / 32;
  const cx = w / 2;
  const cy = h * 0.72;

  const tunnelHalfWidth = 3.5 * scale;
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = "#eef2f6";
  ctx.fillRect(cx - tunnelHalfWidth, 0, tunnelHalfWidth * 2, h);
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 6]);
  ctx.beginPath();
  ctx.moveTo(cx, 0);
  ctx.lineTo(cx, h);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = "#dbe2e8";
  ctx.lineWidth = 1;
  for (let m = -40; m <= 40; m += 5) {
    const x = cx + m * scale;
    const y = cy - m * scale;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  ctx.fillStyle = "#9ca3af";
  ctx.font = "10px sans-serif";
  ctx.fillText("5m", cx + 5 * scale + 4, cy - 5 * scale + 4);

  const smoke = clamp(latest.smoke_score, 0, 1);
  // 灰色渐变仅表达总体烟雾强度，并非真实烟雾空间分割结果。
  if (smoke > 0.02) {
    const radius = (6 + smoke * 24) * scale;
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
    grad.addColorStop(0, `rgba(148,163,184,${0.08 + smoke * 0.30})`);
    grad.addColorStop(1, "rgba(148,163,184,0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  for (const zone of latest.hot_zones) {
    // 红色半透明圆表示火源高温禁入区，半径来自火源融合模块。
    const px = cx + zone.center[1] * scale;
    const py = cy - zone.center[0] * scale;
    const r = zone.radius_m * scale;
    ctx.fillStyle = "rgba(220,38,38,0.14)";
    ctx.strokeStyle = "rgba(220,38,38,0.75)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  ctx.fillStyle = "#0e7490";
  // 青蓝色三角形表示巡检车，当前页面把车辆固定绘制在 map 原点。
  ctx.beginPath();
  ctx.moveTo(cx, cy - 10);
  ctx.lineTo(cx - 8, cy + 8);
  ctx.lineTo(cx + 8, cy + 8);
  ctx.closePath();
  ctx.fill();

  for (const target of latest.targets) {
    const px = cx + target.position[1] * scale;
    const py = cy - target.position[0] * scale;
    const color = targetColor(target.class_id);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(px, py, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();

    const vx = target.velocity[0];
    const vy = target.velocity[1];
    const speed = Math.hypot(vx, vy);
    if (speed > 0.1) {
      const ax = vy * scale * 0.8;
      const ay = -vx * scale * 0.8;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px + ax, py + ay);
      ctx.stroke();
    }

    ctx.fillStyle = "#1f2933";
    ctx.font = "10px sans-serif";
    ctx.fillText(`#${target.track_id}`, px + 10, py - 6);
    ctx.fillStyle = "#6b7280";
    ctx.fillText(target.class_name_override || target.class_name, px + 10, py + 6);
  }

  const legend = [
    ["#dc2626", "火源"], ["#f59e0b", "人员"], ["#2563eb", "车辆"], ["#64748b", "其他"],
  ];
  ctx.font = "11px sans-serif";
  let legendX = 14;
  for (const [color, label] of legend) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(legendX + 5, 17, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#475569";
    ctx.fillText(label, legendX + 14, 21);
    legendX += 58;
  }
}

function targetColor(classId) {
  for (const rule of CLASS_TAGS) {
    if (classId >= rule.min && classId <= rule.max) {
      return rule.cls === "fire" ? "#dc2626" : rule.cls === "person" ? "#d97706" : "#2563eb";
    }
  }
  return "#6b7280";
}

function pushHistory() {
  // 暂停状态仍可能收到相同 SSE 快照，相同时间戳不重复写入趋势数组。
  const lastIndex = history.time.length - 1;
  if (lastIndex >= 0 && history.time[lastIndex] === latest.time) return;
  history.time.push(latest.time);
  history.smoke.push(latest.smoke_score);
  history.rgb.push(latest.weights.rgb);
  history.thermal.push(latest.weights.thermal);
  history.lidar.push(latest.weights.lidar);
  history.radar.push(latest.weights.radar);
  history.gas.push(latest.gas_risk_level / 3);
  for (const key of Object.keys(history)) {
    if (history[key].length > MAX_POINTS) {
      history[key].shift();
    }
  }
}

function drawChart() {
  const { ctx, w, h } = fitCanvas(els.chart, 900 / 240);
  ctx.clearRect(0, 0, w, h);
  if (history.time.length < 2) {
    ctx.fillStyle = "#9ca3af";
    ctx.font = "12px sans-serif";
    ctx.fillText("等待数据", 12, 22);
    return;
  }

  const pad = { left: 34, right: 10, top: 26, bottom: 20 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const t0 = history.time[0];
  const t1 = history.time[history.time.length - 1];
  const span = Math.max(t1 - t0, 1e-6);

  ctx.strokeStyle = "#e5e7eb";
  ctx.fillStyle = "#9ca3af";
  ctx.font = "10px sans-serif";
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + plotH * i / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
    ctx.fillText((1 - i / 4).toFixed(1), 4, y + 3);
  }

  const series = [
    { key: "smoke", color: "#0e7490", label: "烟雾" },
    { key: "rgb", color: "#9ca3af", label: "RGB" },
    { key: "thermal", color: "#dc2626", label: "热" },
    { key: "lidar", color: "#2563eb", label: "LiDAR" },
    { key: "radar", color: "#16a34a", label: "雷达" },
    { key: "gas", color: "#b45309", label: "气体" },
  ];

  for (const item of series) {
    const values = history[item.key];
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
      const x = pad.left + (history.time[i] - t0) / span * plotW;
      const y = pad.top + (1 - clamp(values[i], 0, 1)) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  ctx.font = "10px sans-serif";
  let lx = pad.left + 6;
  for (const item of series) {
    ctx.fillStyle = item.color;
    ctx.fillRect(lx, 8, 8, 8);
    ctx.fillStyle = "#1f2933";
    ctx.fillText(item.label, lx + 11, 16);
    lx += 42;
  }
}

function render() {
  setText("runMode", (latest.run_mode || "unknown").toUpperCase());
  setText("mode", latest.mode_name);
  setText("status", latest.status_name);
  setText("envRisk", latest.env_risk_name);
  setText("safetyAction", latest.safety_action);
  setText("speedLimit", `${Math.round(latest.speed_limit * 100)}%`);
  setText("clock", `${latest.time.toFixed(1)}s`);
  const tone = riskTone(Number(latest.env_risk || 0));
  setTone(els.riskChip, tone);
  setTone(els.actionChip, Number(latest.safety_level || 0) >= 2 ? "critical" : tone);
  setTone(els.speedChip, Number(latest.speed_limit || 0) <= 0.2 ? "critical" : Number(latest.speed_limit || 0) < 1 ? "warning" : "ok");
  setTone(els.statusChip, latest.status_name === "OK" ? "ok" : latest.status_name === "LOST" ? "critical" : "warning");
  setTone(els.modeChip, Number(latest.mode || 0) >= 2 ? "critical" : Number(latest.mode || 0) === 1 ? "warning" : "ok");
  document.body.dataset.risk = tone;
  els.mapSummary.textContent = `${latest.targets.length} 个目标 · 烟雾 ${latest.smoke_score.toFixed(2)} · 风险 ${latest.env_risk_name}`;
  renderSensors();
  renderEnv();
  renderThermal();
  renderTargets();
  renderInspection();
  renderEvents();
  drawMap();
  pushHistory();
  drawChart();
}

async function sendControl(action, value) {
  // 控制结果以服务端状态为准，避免按钮文字与真实暂停状态不一致。
  try {
    const response = await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, value }),
    });
    const result = await response.json();
    if (!result.ok) throw new Error(result.error || "control rejected");
    paused = Boolean(result.paused);
    els.pauseBtn.textContent = paused ? "继续" : "暂停";
    return result;
  } catch (err) {
    console.error("control failed", err);
  }
}

function connect() {
  // SSE 负责服务端单向实时推送；连接中断后一秒自动重连。
  const source = new EventSource("/events");
  source.onopen = () => setConnection(true, "实时数据已连接");
  source.onmessage = (event) => {
    const parsed = JSON.parse(event.data);
    if (!parsed || !parsed.weights || !Array.isArray(parsed.targets)) return;
    latest = parsed;
    paused = Boolean(latest.paused);
    els.pauseBtn.textContent = paused ? "继续" : "暂停";
    render();
  };
  source.onerror = () => {
    setConnection(false, "连接中断，正在重试");
    source.close();
    setTimeout(connect, 1000);
  };
}

els.pauseBtn.addEventListener("click", async () => {
  await sendControl(paused ? "resume" : "pause");
});

els.stepBtn.addEventListener("click", async () => {
  paused = true;
  await sendControl("step");
  els.pauseBtn.textContent = "继续";
});

els.resetBtn.addEventListener("click", async () => {
  history.time.length = 0;
  for (const key of Object.keys(history)) history[key].length = 0;
  activeEvents.clear();
  els.eventList.innerHTML = "";
  previousTime = -1;
  await sendControl("reset");
});

els.speedSelect.addEventListener("change", () => {
  sendControl("speed", parseFloat(els.speedSelect.value));
});

window.addEventListener("resize", () => {
  if (latest) {
    drawMap();
    drawChart();
  }
});

setConnection(false, "正在连接");
connect();
