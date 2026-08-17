// MARVIN Mission Control — console client. Renders the operator/HOUSTON/MARVIN transcript, the
// avatar-chip handoff, the comms light-time indicator, and MARVIN's downlink panels.
const $ = (id) => document.getElementById(id);
const transcript = $("transcript"), downlink = $("downlink");
let DELAY = 12, lastOps = null;

// ---- transcript + chips ----------------------------------------------------
function addMsg(who, text, sys) {
  const el = document.createElement("div");
  el.className = "msg " + who + (sys ? " sys" : "");
  const label = { operator: "OPERATOR", houston: "HOUSTON", marvin: "MARVIN", sys: "SYS" }[who] || who;
  el.innerHTML = `<span class="from">${label}</span><span class="body"></span>`;
  el.querySelector(".body").textContent = text;
  transcript.appendChild(el);
  transcript.scrollTop = transcript.scrollHeight;
}
function setChip(who, state) {
  const chip = $("chip-" + who), dot = $("state-" + who);
  dot.className = "statedot" + (state && state !== "idle" ? " " + state : "");
  chip.classList.toggle("busy", state === "thinking" || state === "transmitting" || state === "speaking");
  chip.classList.toggle("dim", state === "relaying" && who === "houston");
}
function comms(dir, seconds) {
  DELAY = seconds || DELAY;
  const bar = $("comms");
  bar.classList.add("active");
  $("comms-txt").textContent = `${dir} ${Math.round(seconds)}s (compressed; real Mars one-way 4–24 min)`;
  const sig = $("signal");
  sig.style.setProperty("--dur", seconds + "s");
  sig.className = "signal"; void sig.offsetWidth;      // restart the animation
  sig.className = "signal " + (dir === "uplink" ? "up" : "down");
  setTimeout(() => { bar.classList.remove("active"); $("comms-txt").textContent = "standby"; }, seconds * 1000);
}

// ---- downlink panels -------------------------------------------------------
function panel(id, title, klass = "") {
  $("empty")?.remove();
  let p = $("panel-" + id);
  if (!p) {
    p = document.createElement("div");
    p.id = "panel-" + id; p.className = "panel " + klass;
    p.innerHTML = `<h3>${title}</h3><div class="content"></div>`;
    downlink.prepend(p);
  }
  return p.querySelector(".content");
}
const riskColor = (r) => `hsl(${Math.round((1 - Math.min(r, 1)) * 130)}, 70%, 52%)`;

function renderBriefing(b) {
  const c = panel("briefing", "Mission Briefing · " + (b.mission_id || ""), "brief");
  const objs = (b.objectives || []).map(o =>
    `<div class="obj">[${o.priority ?? "?"}] <b>${o.type}</b> ${o.target}<br><span class="sci">${o.science_rationale || ""}</span></div>`).join("");
  c.innerHTML = `<dl><dt>intent</dt><dd>${b.operator_intent || ""}</dd>
    <dt>summary</dt><dd>${b.objective_summary || ""}</dd></dl>${objs}
    <div class="adv">▸ advisory (orbital, non-binding): ${(b.route_advisory || {}).note || ""}</div>`;
}
function renderDeviation(d) {
  const c = panel("deviation", "Deviation Report", "deviation");
  c.parentElement.className = "panel deviation";
  c.innerHTML = `<div class="dev-row">Earth advised: <b class="bad">${d.advisory_route} route — ${d.advisory_risk_pct}% entrapment</b></div>
    <div class="dev-row">MARVIN chose: <b class="good">${d.chosen_route} route — ${d.chosen_risk_pct}%, ${d.chosen_length_m} m</b> <span style="color:var(--text3)">(${d.decided_by})</span></div>
    <div class="dev-just">${d.justification || ""}</div>`;
}
function renderRouteCmp(p) {
  const c = panel("cmp", "Route Comparison");
  const rows = p.rows.map(r =>
    `<tr class="${r.safe && r.risk_pct <= 10 ? "" : ""}">
      <td>${r.name}</td><td class="num">${r.length_m} m</td>
      <td class="num ${r.risk_pct >= 50 ? "risk-bad" : "risk-ok"}">${r.risk_pct}%</td>
      <td>${r.safe ? "safe" : "risky"}</td></tr>`).join("");
  c.innerHTML = `<table class="cmp"><tr><th>route</th><th class="num">length</th><th class="num">risk</th><th>verdict</th></tr>${rows}</table>`;
}
function renderPower(p) {
  const c = panel("power", "Power Budget");
  const pct = p.battery_pct, res = p.reserve_pct || 15;
  c.innerHTML = `<div style="font-family:var(--mono);font-size:12px">battery ${pct}% · reserve ${res}%</div>
    <div class="power-bar"><div class="power-fill" style="width:${pct}%"></div>
    <div class="power-reserve" style="left:${res}%"></div></div>`;
}
function addLog(text) {
  const c = panel("log", "Decision Log");
  if (!c.querySelector(".log")) c.innerHTML = `<div class="log"></div>`;
  const line = document.createElement("div"); line.textContent = text;
  c.querySelector(".log").appendChild(line);
  c.querySelector(".log").scrollTop = 1e9;
}
function renderOpsMap(p) {
  lastOps = p;
  const c = panel("ops", "Ops Map · onboard DEM");
  let cv = $("opsmap");
  if (!cv) { c.innerHTML = `<canvas id="opsmap" width="368" height="368"></canvas>`; cv = $("opsmap"); }
  const ctx = cv.getContext("2d"), W = cv.width, H = cv.height, ext = p.extent || 5;
  const toXY = (x, y) => [(x + ext) / (2 * ext) * W, (ext - y) / (2 * ext) * H];
  // terrain heatmap
  const g = p.grid, n = g.length, cw = W / n, ch = H / n;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const v = g[i][j];
    ctx.fillStyle = `rgb(${40 + v * 150}, ${24 + v * 90}, ${18 + v * 55})`;
    ctx.fillRect(j * cw, (n - 1 - i) * ch, cw + 1, ch + 1);   // grid row i = +y, canvas y flips
  }
  // routes colored by risk, chosen thicker
  (p.routes || []).forEach(r => {
    ctx.strokeStyle = riskColor(r.risk); ctx.lineWidth = r.chosen ? 3 : 1.4;
    ctx.setLineDash(r.chosen ? [] : [4, 3]); ctx.beginPath();
    r.waypoints.forEach((w, k) => { const [px, py] = toXY(w[0], w[1]); k ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
    ctx.stroke();
  });
  ctx.setLineDash([]);
  // targets
  (p.targets || []).forEach(t => { const [px, py] = toXY(t.x, t.y);
    ctx.fillStyle = t.done ? "#8d8d8d" : "#42be65"; ctx.beginPath(); ctx.arc(px, py, 4, 0, 7); ctx.fill(); });
  // rover
  const [rx, ry] = toXY(p.pose[0], p.pose[1]);
  ctx.fillStyle = "#ff832b"; ctx.beginPath(); ctx.arc(rx, ry, 5, 0, 7); ctx.fill();
  ctx.strokeStyle = "#ff832b"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(rx, ry);
  ctx.lineTo(rx + Math.cos(-p.pose[2]) * 12, ry + Math.sin(-p.pose[2]) * 12); ctx.stroke();
}

// ---- websocket -------------------------------------------------------------
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    switch (m.type) {
      case "hello": DELAY = m.delay_s; $("backend").textContent = "ground: " + m.houston_backend;
        $("footnote").textContent = "HOUSTON: " + m.houston_backend + " · MARVIN: local granite4.1:3b (flight build)"; break;
      case "chat": addMsg(m.who, m.text); break;
      case "chip": setChip(m.who, m.state); break;
      case "comms": comms(m.dir, m.seconds); break;
      case "briefing": renderBriefing(m.briefing); break;
      case "deviation": renderDeviation(m.payload); addMsg("marvin",
        `Negative on the ${m.payload.advisory_route} approach — ${m.payload.advisory_risk_pct}% entrapment across 20 rollouts. Taking the ${m.payload.chosen_route} route, ${m.payload.chosen_length_m} m at ${m.payload.chosen_risk_pct}%.`); break;
      case "log": addLog(m.text); break;
      case "telemetry": if (lastOps) { lastOps.pose = [m.payload.pose[0], m.payload.pose[1], lastOps.pose[2]]; renderOpsMap(lastOps); }
        break;
      case "panel":
        if (m.payload.type === "ops_map") renderOpsMap(m.payload);
        else if (m.payload.type === "route_comparison") renderRouteCmp(m.payload);
        else if (m.payload.type === "power") renderPower(m.payload); break;
      case "summary": addMsg("marvin", `Objective complete — sample ${m.payload.sampled ? "cached" : "not cached"} via the ${m.payload.route} route. Battery ${m.payload.battery_pct}%. ${m.payload.note}`); break;
      case "frame": {
        const img = $("simview"); img.src = "data:image/jpeg;base64," + m.data;
        img.classList.add("live"); $("sim-empty").style.display = "none"; break;
      }
    }
  };
  ws.onclose = () => { addMsg("sys", "link dropped — reconnecting…", true); setTimeout(connect, 1500); };
  window._ws = ws;
}

document.querySelectorAll(".camtoggle button").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll(".camtoggle button").forEach((x) => x.classList.remove("on"));
    b.classList.add("on");
    if (window._ws && window._ws.readyState === 1)
      window._ws.send(JSON.stringify({ type: "camera", mode: b.dataset.cam }));
  });
});

$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  const t = $("input").value.trim();
  if (!t || !window._ws || window._ws.readyState !== 1) return;
  window._ws.send(JSON.stringify({ type: "operator", text: t }));
  $("input").value = "";
});
connect();
