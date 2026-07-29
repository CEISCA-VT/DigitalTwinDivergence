"use strict";

const $ = (id) => document.getElementById(id);
const state = { data: null, index: 0, playing: false, speed: 1, lastFrame: 0, carry: 0 };
const colors = { ink: "#172126", grid: "#dfe6e7", teal: "#087f79", coral: "#dc604c", gold: "#a96f12", muted: "#819096" };

function setText(id, value) { $(id).textContent = value; }
function finite(value, fallback = 0) { return Number.isFinite(Number(value)) ? Number(value) : fallback; }
function fmt(value, digits = 2) { return finite(value).toFixed(digits); }
function timeLabel(seconds) {
  const value = Math.max(0, finite(seconds));
  const minutes = Math.floor(value / 60);
  return `${String(minutes).padStart(2, "0")}:${(value % 60).toFixed(1).padStart(4, "0")}`;
}
function human(value) { return String(value || "--").replaceAll("_", " "); }

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

async function initialize() {
  try {
    const result = await fetchJson("/api/logs");
    const select = $("runSelect");
    select.innerHTML = "";
    for (const log of result.logs) {
      const option = document.createElement("option");
      option.value = log.id;
      option.textContent = log.label;
      select.appendChild(option);
    }
    if (!result.logs.length) throw new Error("No accepted benign logs were found.");
    select.disabled = false;
    select.addEventListener("change", () => loadReplay(select.value));
    await loadReplay(result.logs[0].id);
  } catch (error) {
    showError(error);
  }
}

async function loadReplay(runId) {
  stopPlayback();
  setSystem("Loading replay", "Running the accepted log through the digital twin", "loading");
  $("runSelect").disabled = true;
  try {
    state.data = await fetchJson(`/api/replay?id=${encodeURIComponent(runId)}`);
    state.index = 0;
    $("timeline").max = Math.max(0, state.data.points.length - 1);
    $("timeline").value = 0;
    populateRun();
    render();
    setSystem("Replay ready", `${state.data.summary.updates} synchronized updates`, "ready");
    $("errorView").hidden = true;
  } catch (error) {
    showError(error);
  } finally {
    $("runSelect").disabled = false;
  }
}

function setSystem(title, detail, status) {
  setText("systemState", title);
  setText("systemDetail", detail);
  $("stateDot").className = `state-dot ${status === "ready" ? "" : status}`.trim();
}

function showError(error) {
  setSystem("Replay error", "See the message below", "error");
  setText("errorText", error.message || String(error));
  $("errorView").hidden = false;
}

function populateRun() {
  const { metadata: meta, summary } = state.data;
  setText("metaSurface", human(meta.surface));
  setText("metaSpeed", human(meta.speed));
  setText("metaRoute", `${human(meta.route)} · ${meta.repeats} loops`);
  setText("metaNetwork", human(meta.network));
  setText("metaTrial", meta.trial);
  setText("runRmse", `${fmt(summary.agreement_rmse_m, 3)} m`);
  setText("runP95", `${fmt(summary.security_agreement_rmse_m, 3)} m`);
  setText("sumUpdates", summary.updates);
  setText("sumDuration", `${fmt(summary.duration_s, 1)} s`);
  setText("sumLoss", summary.packet_loss);
  setText("sumStale", summary.stale_packets);
  setText("sumLatency", `${fmt(summary.latency_median_ms, 0)} ms`);
  setText("sumSat", `${summary.satellite_min}–${summary.satellite_max}`);
}

function current() {
  return state.data?.points[Math.min(state.index, state.data.points.length - 1)] || null;
}

function render() {
  if (!state.data || !state.data.points.length) return;
  const point = current();
  $("timeline").value = state.index;
  const total = state.data.points[state.data.points.length - 1].t;
  setText("timelineTime", `${timeLabel(point.t)} / ${timeLabel(total)}`);
  setText("mapTime", timeLabel(point.t));
  setText("mapPosition", `E ${fmt(point.ekf_x)} m · N ${fmt(point.ekf_y)} m`);
  setText("poseX", fmt(point.ekf_x));
  setText("poseY", fmt(point.ekf_y));
  setText("poseTheta", fmt(point.path_heading * 180 / Math.PI, 1));
  setText("poseVelocity", fmt(point.velocity));

  const difference = Math.hypot(point.ekf_x - point.gps_x, point.ekf_y - point.gps_y);
  setText("currentResidual", `${fmt(difference, 3)} m`);
  setText("residualNow", `${fmt(difference, 3)} m`);
  $("residualMeter").style.width = `${Math.min(100, difference / 2 * 100)}%`;

  setText("currentNis", fmt(point.nis));
  setText("nisThreshold", fmt(point.threshold));
  setText("confidence", `${fmt(point.confidence * 100, 0)}%`);
  const nisMax = Math.max(point.threshold * 1.35, point.nis, 1);
  $("nisMeter").style.width = `${Math.min(100, point.nis / nisMax * 100)}%`;
  $("thresholdMark").style.left = `${Math.min(98, point.threshold / nisMax * 100)}%`;
  const badge = $("regionBadge");
  badge.textContent = point.region;
  badge.className = `badge ${point.region}`;

  setText("sensorSat", point.satellites);
  setText("sensorHdop", fmt(point.hdop));
  setText("sensorVelocity", fmt(point.velocity, 3));
  setText("sensorOmega", `${fmt(point.omega * 180 / Math.PI, 1)}°/s`);
  setText("sensorYaw", fmt(point.yaw, 1));
  setText("sensorGyro", `${fmt(point.gyro_z, 2)}°/s`);
  setText("sensorVoltage", fmt(point.voltage, 2));
  setText("sensorMotors", `${fmt(point.motor_l, 2)} / ${fmt(point.motor_r, 2)}`);
  setText("sensorLatency", fmt(point.latency_ms, 0));
  setText("sensorPacket", `${point.queue_depth} / ${point.packet_loss}`);
  setText("latencyNow", `${fmt(point.latency_ms, 0)} ms`);

  drawTrajectory();
  drawSeries($("residualChart"), state.data.points.map(p => Math.hypot(p.ekf_x - p.gps_x, p.ekf_y - p.gps_y)), colors.coral, "m");
  drawSeries($("latencyChart"), state.data.points.map(p => p.latency_ms), colors.teal, "ms");
}

function canvasContext(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function drawTrajectory() {
  const canvas = $("trajectoryCanvas");
  const { ctx, width, height } = canvasContext(canvas);
  const bounds = state.data.bounds;
  const pad = 34;
  const spanX = Math.max(0.5, bounds.max_x - bounds.min_x);
  const spanY = Math.max(0.5, bounds.max_y - bounds.min_y);
  const scale = Math.min((width - 2 * pad) / spanX, (height - 2 * pad) / spanY);
  const usedW = spanX * scale;
  const usedH = spanY * scale;
  const offsetX = (width - usedW) / 2;
  const offsetY = (height - usedH) / 2;
  const map = (x, y) => [
    offsetX + (x - bounds.min_x) * scale,
    height - offsetY - (y - bounds.min_y) * scale,
  ];

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f8faf9";
  ctx.fillRect(0, 0, width, height);
  drawGrid(ctx, width, height, bounds, map);

  const points = state.data.points;
  drawPath(ctx, points, points.length - 1, map, "gps_x", "gps_y", colors.coral, 1, 0.25);
  drawPath(ctx, points, points.length - 1, map, "security_x", "security_y", colors.gold, 1, 0.24);
  drawPath(ctx, points, points.length - 1, map, "ekf_x", "ekf_y", colors.teal, 1, 0.22);
  drawPath(ctx, points, state.index, map, "gps_x", "gps_y", colors.coral, 1.5, 0.82);
  drawPath(ctx, points, state.index, map, "security_x", "security_y", colors.gold, 1.8, 0.92);
  drawPath(ctx, points, state.index, map, "ekf_x", "ekf_y", colors.teal, 2.3, 1);

  const point = current();
  const [gx, gy] = map(point.gps_x, point.gps_y);
  ctx.fillStyle = colors.coral;
  ctx.beginPath(); ctx.arc(gx, gy, 4, 0, Math.PI * 2); ctx.fill();

  const [x, y] = map(point.ekf_x, point.ekf_y);
  drawRover(ctx, x, y, -point.path_heading, colors.teal);
  const scalePixels = Math.min(90, scale);
  $("mapScale").style.width = `${scalePixels}px`;
  $("mapScale").textContent = scale >= 55 ? "1 m" : "2 m";
}

function drawGrid(ctx, width, height, bounds, map) {
  ctx.strokeStyle = colors.grid;
  ctx.fillStyle = colors.muted;
  ctx.lineWidth = 1;
  ctx.font = "10px Segoe UI";
  const step = (bounds.max_x - bounds.min_x) > 7 ? 2 : 1;
  const startX = Math.floor(bounds.min_x / step) * step;
  const startY = Math.floor(bounds.min_y / step) * step;
  for (let value = startX; value <= bounds.max_x; value += step) {
    const [x] = map(value, 0);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
    ctx.fillText(`${value}m`, x + 3, height - 7);
  }
  for (let value = startY; value <= bounds.max_y; value += step) {
    const [, y] = map(0, value);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    ctx.fillText(`${value}m`, 5, y - 4);
  }
}

function drawPath(ctx, points, end, map, xKey, yKey, color, lineWidth, alpha) {
  if (end < 1) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";
  ctx.beginPath();
  for (let i = 0; i <= end; i++) {
    const [x, y] = map(points[i][xKey], points[i][yKey]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

function drawRover(ctx, x, y, angle, color) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.fillStyle = color;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(13, 0); ctx.lineTo(-8, -8); ctx.lineTo(-5, 0); ctx.lineTo(-8, 8); ctx.closePath();
  ctx.fill(); ctx.stroke();
  ctx.restore();
}

function drawSeries(canvas, values, color, unit) {
  const { ctx, width, height } = canvasContext(canvas);
  const pad = { left: 34, right: 8, top: 8, bottom: 20 };
  const visible = values.slice(0, state.index + 1);
  const max = Math.max(...values, 1);
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = colors.grid; ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = pad.top + (height - pad.top - pad.bottom) * i / 3;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
  }
  ctx.fillStyle = colors.muted; ctx.font = "9px Segoe UI";
  ctx.fillText(`${max.toFixed(max < 10 ? 1 : 0)} ${unit}`, 2, pad.top + 4);
  ctx.fillText("0", 20, height - pad.bottom + 3);
  if (visible.length < 2) return;
  ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.beginPath();
  visible.forEach((value, index) => {
    const x = pad.left + (width - pad.left - pad.right) * index / Math.max(1, values.length - 1);
    const y = height - pad.bottom - finite(value) / max * (height - pad.top - pad.bottom);
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function stopPlayback() {
  state.playing = false;
  state.lastFrame = 0;
  state.carry = 0;
  $("playPause").textContent = "▶";
  $("playPause").setAttribute("aria-label", "Play");
}

function togglePlayback() {
  if (!state.data) return;
  state.playing = !state.playing;
  $("playPause").textContent = state.playing ? "❚❚" : "▶";
  $("playPause").setAttribute("aria-label", state.playing ? "Pause" : "Play");
  state.lastFrame = performance.now();
  if (state.playing) requestAnimationFrame(playFrame);
}

function playFrame(now) {
  if (!state.playing || !state.data) return;
  const elapsed = (now - state.lastFrame) / 1000 * state.speed;
  state.lastFrame = now;
  const points = state.data.points;
  let targetTime = points[state.index].t + elapsed + state.carry;
  let moved = false;
  while (state.index < points.length - 1 && points[state.index + 1].t <= targetTime) {
    state.index++;
    moved = true;
  }
  state.carry = targetTime - points[state.index].t;
  if (moved) render();
  if (state.index >= points.length - 1) {
    stopPlayback();
    render();
    return;
  }
  requestAnimationFrame(playFrame);
}

$("playPause").addEventListener("click", togglePlayback);
$("jumpStart").addEventListener("click", () => { stopPlayback(); state.index = 0; render(); });
$("stepBack").addEventListener("click", () => { stopPlayback(); state.index = Math.max(0, state.index - 1); render(); });
$("stepForward").addEventListener("click", () => {
  stopPlayback();
  if (state.data) state.index = Math.min(state.data.points.length - 1, state.index + 1);
  render();
});
$("timeline").addEventListener("input", (event) => { stopPlayback(); state.index = Number(event.target.value); render(); });
$("speedSelect").addEventListener("change", (event) => { state.speed = Number(event.target.value); });
window.addEventListener("resize", render);
window.addEventListener("keydown", (event) => {
  if (event.target.matches("select,input")) return;
  if (event.code === "Space") { event.preventDefault(); togglePlayback(); }
  if (event.code === "ArrowLeft") $("stepBack").click();
  if (event.code === "ArrowRight") $("stepForward").click();
});

initialize();
