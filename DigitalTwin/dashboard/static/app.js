"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  data: null,
  index: 0,
  playing: false,
  speed: 1,
  lastFrame: 0,
  carry: 0,
  mode: "replay",
  liveTimer: null,
  commandSpeed: "medium",
  activeCommand: "stop",
  activeView: "overview",
};
const colors = {
  ink: "#172126",
  grid: "#dfe6e7",
  teal: "#087f79",
  coral: "#dc604c",
  gold: "#a96f12",
  muted: "#819096",
};

function setText(id, value) { const element = $(id); if (element) element.textContent = value; }
function finite(value, fallback = 0) { return Number.isFinite(Number(value)) ? Number(value) : fallback; }
function fmt(value, digits = 2) { return finite(value).toFixed(digits); }
function metric(value, digits = 2, suffix = "") {
  return value === null || value === undefined || !Number.isFinite(Number(value)) ? "--" : `${Number(value).toFixed(digits)}${suffix}`;
}
function human(value) { return String(value || "--").replaceAll("_", " "); }
function percent(value, digits = 0) {
  return value === null || value === undefined || !Number.isFinite(Number(value)) ? "--" : `${(Number(value) * 100).toFixed(digits)}%`;
}
function sourceLabel(value) {
  const parts = String(value || "").replaceAll("\\", "/").split("/");
  return parts[parts.length - 1] || "stream";
}
function deg(rad) { return finite(rad) * 180 / Math.PI; }
function timeLabel(seconds) {
  const value = Math.max(0, finite(seconds));
  const minutes = Math.floor(value / 60);
  return `${String(minutes).padStart(2, "0")}:${(value % 60).toFixed(1).padStart(4, "0")}`;
}

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

async function initialize() {
  try {
    const modePayload = await fetchJson("/api/mode");
    state.mode = modePayload.mode || "replay";
    if (state.mode === "replay") {
      await initializeReplay();
    } else {
      await initializeStream();
    }
  } catch (error) {
    showError(error);
  }
}

async function initializeReplay() {
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
}

async function initializeStream() {
  const select = $("runSelect");
  select.innerHTML = "";
  const option = document.createElement("option");
  option.value = "stream";
  option.textContent = state.mode === "csv" ? "Dummy CSV stream" : "UGV01 live T:147 stream";
  select.appendChild(option);
  select.disabled = true;
  setSystem("Stream starting", "Waiting for T:147 samples", "loading");
  await pollStream();
  state.liveTimer = window.setInterval(pollStream, 500);
}

async function pollStream() {
  try {
    const payload = await fetchJson("/api/stream");
    const previousLength = state.data?.points?.length || 0;
    state.data = normalizePayload(payload);
    if (!state.playing || state.index >= previousLength - 1) {
      state.index = Math.max(0, state.data.points.length - 1);
    }
    $("timeline").max = Math.max(0, state.data.points.length - 1);
    populateRun();
    render();
    const detail = `${state.data.summary.updates || 0} updates | ${sourceLabel(state.data.metadata.source)}`;
    setSystem(payload.running ? "Live prototype running" : "Stream complete", detail, payload.error ? "error" : "ready");
    if (payload.error) setText("errorText", payload.error);
    $("errorView").hidden = !payload.error;
  } catch (error) {
    showError(error);
  }
}

async function loadReplay(runId) {
  stopPlayback();
  setSystem("Loading replay", "Running the accepted log through the digital twin", "loading");
  $("runSelect").disabled = true;
  try {
    state.data = normalizePayload(await fetchJson(`/api/replay?id=${encodeURIComponent(runId)}`));
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

function normalizePayload(payload) {
  if (payload.schema === "ugv01_dashboard_replay_v2") {
    const points = payload.points.map((p) => ({
      ...p,
      twin_x: p.ekf_x,
      twin_y: p.ekf_y,
      twin_theta: p.path_heading || p.ekf_theta || 0,
      firmware_yaw_deg: p.yaw,
      encoder_v: p.velocity,
      omega: p.omega,
      gps_valid: Number.isFinite(Number(p.gps_x)) && Number.isFinite(Number(p.gps_y)),
      gps_agreement_m: Math.hypot(finite(p.ekf_x) - finite(p.gps_x), finite(p.ekf_y) - finite(p.gps_y)),
      gps_heading_agreement_deg: null,
      packet_gap: p.packet_loss || 0,
      condition: p.region || "replay",
    }));
    return {
      ...payload,
      points,
      summary: {
        ...payload.summary,
        gps_agreement_rmse_m: payload.summary.agreement_rmse_m,
        gps_agreement_p95_m: payload.summary.agreement_p95_m,
        gps_agreement_max_m: null,
        gps_heading_mae_deg: null,
        gps_heading_p95_deg: null,
        gps_RPEp_1s_m: null,
        gps_RPEp_5s_m: null,
        gps_RPEp_10s_m: null,
      },
      metadata: {
        label: payload.metadata.label,
        source: payload.metadata.path,
        paper_role: "accepted benign replay",
        runtime_inputs: "T:147 telemetry with GPS",
        twin_model: "legacy replay EKF branch",
        reference_note: "historical GPS replay",
      },
    };
  }
  return payload;
}

function setSystem(title, detail, status) {
  setText("systemState", title);
  setText("systemDetail", detail);
  $("stateDot").className = `state-dot ${status === "ready" ? "" : status}`.trim();
}

function setActiveView(view) {
  state.activeView = view;
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.classList.toggle("selected", button.dataset.view === view);
  });
  document.querySelectorAll("[data-view-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.viewPanel !== view;
  });
  render();
}

function showError(error) {
  setSystem("Dashboard error", "See the message below", "error");
  setText("errorText", error.message || String(error));
  $("errorView").hidden = false;
}

function populateRun() {
  if (!state.data) return;
  const { metadata: meta, summary } = state.data;
  const experiment = meta.experiment || {};
  const condition = [experiment.physical_condition, experiment.wireless_condition].filter(Boolean).join(" / ");
  const trial = experiment.trial == null ? "" : `trial ${experiment.trial}`;
  setText("metaSurface", state.mode.toUpperCase());
  setText("metaSpeed", state.data.policy?.name || meta.twin_model || "--");
  setText("metaRoute", meta.runtime_inputs || "--");
  setText("metaNetwork", condition || meta.reference_note || "--");
  setText("metaTrial", [trial, contractHeadline(state.data.contracts)].filter(Boolean).join(" | "));
  setText("controlMode", state.mode === "live" ? "live" : "dry run");
  $("controlMode").className = `badge ${state.mode === "live" ? "safe" : "warning"}`;
}

function gpsAgreementLabel() {
  const p = current();
  if (!p || !p.gps_valid) return "no fix";
  const d = Math.hypot(finite(p.twin_x) - finite(p.gps_x), finite(p.twin_y) - finite(p.gps_y));
  return `${fmt(d, 2)} m`;
}

function current() {
  return state.data?.points[Math.min(state.index, Math.max(0, state.data.points.length - 1))] || null;
}

function render() {
  if (!state.data || !state.data.points.length) return;
  const point = current();
  $("timeline").value = state.index;
  const total = state.data.points[state.data.points.length - 1].t;
  setText("timelineTime", `${timeLabel(point.t)} / ${timeLabel(total)}`);
  setText("mapTime", timeLabel(point.t));
  setText("mapPosition", `E ${fmt(point.twin_x)} m | N ${fmt(point.twin_y)} m`);
  setText("poseX", fmt(point.twin_x));
  setText("poseY", fmt(point.twin_y));
  setText("poseTheta", fmt(deg(point.twin_theta), 1));
  setText("poseVelocity", fmt(point.encoder_v));

  const gpsDelta = point.gps_valid ? finite(point.gps_agreement_m, Math.hypot(point.twin_x - point.gps_x, point.twin_y - point.gps_y)) : NaN;

  const yawDisagreementDeg = Math.abs(deg(point.yaw_disagreement || 0));
  setText("residualNow", point.gps_valid ? `${fmt(gpsDelta, 2)} m` : "no fix");
  const badge = $("regionBadge");
  badge.textContent = human(point.resource_mode || point.condition);
  badge.className = `badge ${point.resource_mode === "high" ? "blind" : point.resource_mode === "normal" ? "warning" : "safe"}`;

  setText("sensorSat", point.gps_valid ? point.satellites : "--");
  setText("sensorHdop", point.gps_valid ? `${fmt(point.hdop)} / ${fmt(point.gps_age_s * 1000, 0)} ms` : "no fix");
  setText("sensorVelocity", human(point.resource_mode || state.data.policy?.resource_mode));
  setText("sensorOmega", `${metric(point.requested_update_rate_hz || state.data.policy?.requested_update_rate_hz, 1)} / ${metric(state.data.summary.actual_update_rate_hz, 1)} Hz req/actual`);
  setText("sensorYaw", metric(point.aoi_s * 1000, 0));
  setText("sensorGyro", `${metric(state.data.summary.aoi_p95_ms, 0)} / ${metric(state.data.summary.jitter_p95_ms, 0)} ms`);
  setText("sensorVoltage", metric(point.bytes_per_s, 0));
  setText("sensorMotors", `${metric(state.data.summary.evaluation_p95_ms, 2)} ms`);
  setText("sensorLatency", fmt(point.latency_ms, 0));
  setText("sensorPacket", `${point.queue_depth || 0} / ${state.data.summary.packet_loss || 0} / ${state.data.summary.stale_packets || 0}`);
  setText("latencyNow", metric(point.gps_heading_agreement_deg, 1, " deg"));

  renderContracts(point.contracts || state.data.contracts || []);
  renderContractTable(point.contracts || state.data.contracts || []);
  renderQualificationTimelines();
  renderPolicyReasoning();
  renderEvents(state.data.events || []);

  drawTrajectory();
  drawSeries($("residualChart"), state.data.points.map(p => p.gps_agreement_m), colors.coral, "m");
  drawSeries($("latencyChart"), state.data.points.map(p => p.gps_heading_agreement_deg), colors.gold, "deg");
}

function contractHeadline(contracts) {
  if (!contracts?.length) return "contract engine waiting";
  const withdrawn = contracts.filter(c => c.status === "withdrawn").length;
  const unobservable = contracts.filter(c => c.status === "unobservable").length;
  if (withdrawn) return `${withdrawn} service${withdrawn === 1 ? "" : "s"} withdrawn`;
  if (unobservable) return `${unobservable} service${unobservable === 1 ? "" : "s"} unobservable`;
  const qualified = contracts.filter(c => c.status === "qualified").length;
  return `${qualified} / ${contracts.length} services qualified`;
}

function renderContracts(contracts) {
  const container = $("contractCards");
  if (!container) return;
  const qualified = contracts.filter(c => c.status === "qualified").length;
  setText("contractCount", `${qualified} / ${contracts.length || 4}`);
  if (!contracts.length) {
    container.innerHTML = '<div class="contract-empty">Waiting for synchronized GPS evidence</div>';
    return;
  }
  container.innerHTML = contracts.map(contract => {
    const stateClass = String(contract.status || "unobservable").replaceAll("_", "-");
    const position = contract.position_error_m == null ? "--" : `${fmt(contract.position_error_m, 2)} / ${fmt(contract.position_tolerance_m, 2)} m`;
    const heading = contract.heading_error_deg == null ? "--" : `${fmt(contract.heading_error_deg, 1)} / ${fmt(contract.heading_tolerance_deg, 0)} deg`;
    const freshness = contract.aoi_s == null ? "--" : `${fmt(contract.aoi_s * 1000, 0)} / ${fmt(contract.maximum_aoi_s * 1000, 0)} ms`;
    return `<article class="contract-card ${stateClass}">
      <header><strong>${contract.label}</strong><span>${human(contract.status)}</span></header>
      <small>${contract.family === "global_synchronization" ? "Global" : `${contract.horizon_s}s local`} | ${contract.reason}</small>
      <dl><div><dt>Position</dt><dd>${position}</dd></div><div><dt>Heading</dt><dd>${heading}</dd></div><div><dt>AoI</dt><dd>${freshness}</dd></div></dl>
    </article>`;
  }).join("");
}

function combinedMargin(contract) {
  const margins = [];
  if (Number.isFinite(Number(contract.position_margin_m))) margins.push(`${metric(contract.position_margin_m, 2, " m")} pos`);
  if (Number.isFinite(Number(contract.heading_margin_deg))) margins.push(`${metric(contract.heading_margin_deg, 1, " deg")} head`);
  if (Number.isFinite(Number(contract.freshness_margin_s))) margins.push(`${metric(contract.freshness_margin_s * 1000, 0, " ms")} AoI`);
  return margins.length ? margins.join(" | ") : "--";
}

function renderContractTable(contracts) {
  const body = $("contractTableBody");
  if (!body) return;
  const qualified = contracts.filter(c => c.status === "qualified").length;
  const withdrawn = contracts.filter(c => c.status === "withdrawn").length;
  const badge = $("contractDetailState");
  if (badge) {
    badge.textContent = contracts.length ? `${qualified}/${contracts.length} qualified` : "waiting";
    badge.className = `badge ${withdrawn ? "blind" : qualified === contracts.length ? "safe" : "warning"}`;
  }
  if (!contracts.length) {
    body.innerHTML = '<tr><td colspan="6">Waiting for synchronized GPS evidence</td></tr>';
    return;
  }
  body.innerHTML = contracts.map(contract => {
    const horizon = contract.family === "global_synchronization" ? "global" : `${metric(contract.horizon_s, 0, " s")}`;
    const tolerance = `${metric(contract.position_tolerance_m, 2, " m")} / ${metric(contract.heading_tolerance_deg, 0, " deg")}`;
    const aoi = metric(contract.maximum_aoi_s * 1000, 0, " ms");
    const stateClass = String(contract.status || "unobservable").replaceAll("_", "-");
    const satisfaction = contract.satisfaction_fraction == null ? "--" : percent(contract.satisfaction_fraction, 0);
    return `<tr class="${stateClass}">
      <td><strong>${contract.label}</strong><small>${human(contract.reason)} | pass ${satisfaction}</small></td>
      <td>${horizon}</td>
      <td>${tolerance}</td>
      <td>${aoi}</td>
      <td>${combinedMargin(contract)}</td>
      <td><span class="state-chip ${stateClass}">${human(contract.status)}</span></td>
    </tr>`;
  }).join("");
}

function renderQualificationTimelines() {
  const container = $("qualificationTimelines");
  if (!container || !state.data?.points?.length) return;
  const points = state.data.points.slice(Math.max(0, state.index - 119), state.index + 1);
  const latestContracts = current()?.contracts || state.data.contracts || [];
  setText("timelineWindow", `${points.length} samples`);
  if (!latestContracts.length) {
    container.innerHTML = '<div class="contract-empty">Waiting for contract evaluations</div>';
    return;
  }
  container.innerHTML = latestContracts.map(contract => {
    const serviceId = contract.service_id;
    const segments = points.map(point => {
      const match = (point.contracts || []).find(item => item.service_id === serviceId);
      const status = String(match?.status || "unobservable").replaceAll("_", "-");
      return `<i class="${status}" title="${human(match?.status)}"></i>`;
    }).join("");
    return `<div class="timeline-row">
      <span>${contract.label}</span>
      <div class="timeline-track">${segments}</div>
      <b>${human(contract.status)}</b>
    </div>`;
  }).join("");
}

function renderPolicyReasoning() {
  const decision = state.data?.policy?.decision || {};
  const summary = state.data?.summary || {};
  const currentMode = decision.current_mode || state.data?.policy?.resource_mode;
  const desiredMode = decision.desired_mode || currentMode;
  const stateClass = currentMode === "high" ? "blind" : currentMode === "normal" ? "warning" : "safe";
  setText("policyMode", human(currentMode));
  setText("policyRate", metric(decision.update_rate_hz || state.data?.policy?.requested_update_rate_hz, 1, " Hz"));
  setText("policyCost", metric(decision.relative_cost, 2));
  setText("policyReason", `${human(decision.reason)}${desiredMode !== currentMode ? ` -> waiting to switch to ${human(desiredMode)}` : ""}`);
  setText("policyAoi", metric(decision.aoi_s * 1000, 0, " ms"));
  setText("policyNormalTrigger", metric(decision.aoi_normal_trigger_s * 1000, 0, " ms"));
  setText("policyHighTrigger", metric(decision.aoi_high_trigger_s * 1000, 0, " ms"));
  setText("policyStates", statusCounts(decision.contract_statuses || {}));
  setText("policyFreshness", metric(summary.aoi_p95_ms, 0, " ms"));
  setText("policyJitter", metric(summary.jitter_p95_ms, 0, " ms"));
  setText("policyBandwidth", metric(summary.bytes_per_s, 0));
  setText("policyCompute", metric(summary.evaluation_p95_ms, 2, " ms"));
  const badge = $("policyStateBadge");
  if (badge) {
    badge.textContent = human(currentMode);
    badge.className = `badge ${stateClass}`;
  }
}

function statusCounts(statuses) {
  const values = Object.values(statuses);
  if (!values.length) return "--";
  const counts = {};
  values.forEach(value => { counts[value] = (counts[value] || 0) + 1; });
  return Object.entries(counts).map(([key, value]) => `${value} ${human(key)}`).join(" | ");
}

function renderEvents(events) {
  const container = $("eventLog");
  if (!container) return;
  setText("eventCount", events.length);
  const recent = events.slice(-5).reverse();
  if (!recent.length) {
    container.innerHTML = "<span>No transitions yet</span>";
    return;
  }
  container.innerHTML = recent.map(event => `<div><time>${timeLabel(event.t)}</time><strong>${human(event.service_id || event.type)}</strong><span>${human(event.from)} to ${human(event.to)}: ${event.reason || "policy update"}</span></div>`).join("");
}

function conditionClass(condition) {
  const label = String(condition || "");
  if (label.includes("high")) return "blind";
  if (label.includes("turn")) return "warning";
  return "safe";
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
  drawPath(ctx, points, points.length - 1, map, "gps_x", "gps_y", colors.coral, 1, 0.22, true);
  drawPath(ctx, points, points.length - 1, map, "twin_x", "twin_y", colors.teal, 1, 0.18, false);
  drawPath(ctx, points, state.index, map, "gps_x", "gps_y", colors.coral, 1.6, 0.82, true);
  drawPath(ctx, points, state.index, map, "twin_x", "twin_y", colors.teal, 2.4, 1, false);

  const point = current();
  if (point.gps_valid) {
    const [gx, gy] = map(point.gps_x, point.gps_y);
    ctx.fillStyle = colors.coral;
    ctx.beginPath();
    ctx.arc(gx, gy, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  const [x, y] = map(point.twin_x, point.twin_y);
  drawRover(ctx, x, y, -point.twin_theta, colors.teal);
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
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
    ctx.fillText(`${value}m`, x + 3, height - 7);
  }
  for (let value = startY; value <= bounds.max_y; value += step) {
    const [, y] = map(0, value);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
    ctx.fillText(`${value}m`, 5, y - 4);
  }
}

function drawPath(ctx, points, end, map, xKey, yKey, color, lineWidth, alpha, requireValid) {
  if (end < 1) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";
  ctx.beginPath();
  let started = false;
  for (let i = 0; i <= end; i++) {
    if (requireValid && !points[i].gps_valid) continue;
    const xVal = points[i][xKey];
    const yVal = points[i][yKey];
    if (!Number.isFinite(Number(xVal)) || !Number.isFinite(Number(yVal))) continue;
    const [x, y] = map(xVal, yVal);
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  if (started) ctx.stroke();
  ctx.restore();
}

function drawFirmwareHeadingPath(ctx, points, map) {
  const point = current();
  const [x, y] = map(point.twin_x, point.twin_y);
  const angle = -finite(point.firmware_yaw_deg) * Math.PI / 180;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.strokeStyle = colors.gold;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(28, 0);
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
  ctx.moveTo(13, 0);
  ctx.lineTo(-8, -8);
  ctx.lineTo(-5, 0);
  ctx.lineTo(-8, 8);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawSeries(canvas, values, color, unit) {
  const { ctx, width, height } = canvasContext(canvas);
  const pad = { left: 36, right: 8, top: 8, bottom: 20 };
  const numeric = values.map(v => v === null || v === undefined || !Number.isFinite(Number(v)) ? null : Number(v));
  const validValues = numeric.filter(v => v !== null);
  const max = Math.max(...validValues, 1);
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = pad.top + (height - pad.top - pad.bottom) * i / 3;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
  }
  ctx.fillStyle = colors.muted;
  ctx.font = "9px Segoe UI";
  ctx.fillText(`${max.toFixed(max < 10 ? 1 : 0)} ${unit}`, 2, pad.top + 4);
  ctx.fillText("0", 20, height - pad.bottom + 3);
  if (validValues.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.8;
  let started = false;
  ctx.beginPath();
  numeric.slice(0, state.index + 1).forEach((value, index) => {
    if (value === null) {
      started = false;
      return;
    }
    const x = pad.left + (width - pad.left - pad.right) * index / Math.max(1, values.length - 1);
    const y = height - pad.bottom - value / max * (height - pad.top - pad.bottom);
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

function stopPlayback() {
  state.playing = false;
  state.lastFrame = 0;
  state.carry = 0;
  $("playPause").innerHTML = "&#9658;";
  $("playPause").setAttribute("aria-label", "Play");
}

function togglePlayback() {
  if (!state.data || state.mode !== "replay") return;
  state.playing = !state.playing;
  $("playPause").textContent = state.playing ? "Pause" : "Play";
  $("playPause").setAttribute("aria-label", state.playing ? "Pause" : "Play");
  state.lastFrame = performance.now();
  if (state.playing) requestAnimationFrame(playFrame);
}

function playFrame(now) {
  if (!state.playing || !state.data) return;
  const elapsed = (now - state.lastFrame) / 1000 * state.speed;
  state.lastFrame = now;
  const points = state.data.points;
  const targetTime = points[state.index].t + elapsed + state.carry;
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
document.querySelectorAll("[data-speed]").forEach((button) => {
  button.addEventListener("click", () => {
    state.commandSpeed = button.dataset.speed;
    document.querySelectorAll("[data-speed]").forEach((item) => item.classList.toggle("selected", item === button));
  });
});
document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => setActiveView(button.dataset.view));
});

async function sendDrive(command) {
  state.activeCommand = command;
  setText("lastCommand", `${command} @ ${state.commandSpeed}`);
  try {
    const response = await fetchJsonPost("/api/command", { command, speed: state.commandSpeed });
    const sent = response.sent ? "sent" : "dry run";
    const payload = response.payload || {};
    setText("lastCommand", `${command} ${sent} L=${fmt(payload.L, 2)} R=${fmt(payload.R, 2)}`);
  } catch (error) {
    showError(error);
  }
}

async function fetchJsonPost(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

document.querySelectorAll("[data-command]").forEach((button) => {
  const command = button.dataset.command;
  button.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    sendDrive(command);
  });
  button.addEventListener("pointerup", () => {
    if (command !== "stop") sendDrive("stop");
  });
  button.addEventListener("pointerleave", () => {
    if (state.activeCommand === command && command !== "stop") sendDrive("stop");
  });
  button.addEventListener("click", (event) => {
    event.preventDefault();
    if (command === "stop") sendDrive("stop");
  });
});

window.addEventListener("resize", render);
window.addEventListener("keydown", (event) => {
  if (event.target.matches("select,input")) return;
  if (event.code === "Space") { event.preventDefault(); togglePlayback(); }
  if (event.code === "ArrowLeft") $("stepBack").click();
  if (event.code === "ArrowRight") $("stepForward").click();
});

initialize();
