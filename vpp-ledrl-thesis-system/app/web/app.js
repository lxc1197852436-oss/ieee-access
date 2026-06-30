let latestResult = null;

const $ = (id) => document.getElementById(id);

function requestPayload() {
  const days = Number($("days").value || 7);
  return {
    region: $("region").value || "广东省",
    start: $("start").value || "2025-07-01 00:00:00",
    periods: Math.max(1, days) * 96,
    freq: "15min",
    seed: Number($("seed").value || 42),
    policies: ["rule", "ledrl", "random"],
  };
}

async function runExperiment() {
  const btn = $("runBtn");
  btn.disabled = true;
  btn.textContent = "运行中...";
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestPayload()),
    });
    latestResult = await res.json();
    renderAll();
  } finally {
    btn.disabled = false;
    btn.textContent = "运行实验";
  }
}

function renderAll() {
  if (!latestResult) return;
  const select = $("policySelect");
  select.innerHTML = "";
  latestResult.results.forEach((r, idx) => {
    const option = document.createElement("option");
    option.value = idx;
    option.textContent = r.policy;
    select.appendChild(option);
  });
  select.value = "1";
  renderMetrics();
  renderChart();
  renderEvents();
  renderExplanations();
}

function selectedResult() {
  const idx = Number($("policySelect").value || 0);
  return latestResult.results[idx];
}

function renderMetrics() {
  const metrics = $("metrics");
  metrics.innerHTML = "";
  latestResult.results.forEach((r) => {
    const m = r.metrics;
    const card = document.createElement("div");
    card.className = "metric-card";
    card.innerHTML = `
      <div class="label">${r.policy}</div>
      <div class="value">${formatNumber(m.total_reward_yuan)} 元</div>
      <div class="sub">CVaR 5%: ${formatNumber(m.cvar_5_yuan)} | 终止SOC: ${m.final_soc.toFixed(2)}</div>
    `;
    metrics.appendChild(card);
  });
}

function renderChart() {
  const result = selectedResult();
  const data = result.history;
  const canvas = $("chart");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  drawAxes(ctx, canvas);
  drawLine(ctx, data.map((d) => d.price_yuan_mwh), "#d9480f", 30, 390, "电价");
  drawLine(ctx, data.map((d) => d.soc * 900), "#2155a3", 30, 390, "SOC×900");
  drawBars(ctx, data.map((d) => d.actual_action_mw), "#0f8b8d", 30, 390);

  ctx.fillStyle = "#172033";
  ctx.font = "16px sans-serif";
  ctx.fillText(`${result.policy} 运行轨迹：电价、SOC 与储能动作`, 24, 24);
  ctx.font = "12px sans-serif";
  ctx.fillStyle = "#d9480f";
  ctx.fillText("红线=电价", 24, 412);
  ctx.fillStyle = "#2155a3";
  ctx.fillText("蓝线=SOC×900", 120, 412);
  ctx.fillStyle = "#0f8b8d";
  ctx.fillText("绿色柱=动作MW", 240, 412);
}

function drawAxes(ctx, canvas) {
  ctx.strokeStyle = "#d8deea";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(48, 40);
  ctx.lineTo(48, 390);
  ctx.lineTo(canvas.width - 20, 390);
  ctx.stroke();
}

function drawLine(ctx, values, color, top, bottom) {
  const max = Math.max(...values);
  const min = Math.min(...values);
  const w = $("chart").width - 80;
  const h = bottom - top;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = 50 + (i / Math.max(1, values.length - 1)) * w;
    const y = bottom - ((v - min) / Math.max(1e-6, max - min)) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawBars(ctx, values, color, top, bottom) {
  const w = $("chart").width - 80;
  const zero = (top + bottom) / 2;
  ctx.fillStyle = color;
  values.forEach((v, i) => {
    const x = 50 + (i / Math.max(1, values.length - 1)) * w;
    const barH = Math.max(-90, Math.min(90, v * 42));
    ctx.globalAlpha = 0.35;
    ctx.fillRect(x, zero, Math.max(1, w / values.length), -barH);
    ctx.globalAlpha = 1;
  });
}

function renderEvents() {
  const data = selectedResult().history.filter((d) => d.event_type !== "正常运行").slice(0, 60);
  const box = $("events");
  box.innerHTML = data.map((d) => `
    <div class="item warn">
      <div class="time">${d.timestamp}</div>
      <div class="title">${d.event_type}</div>
      <div class="body">${d.event_text}</div>
    </div>
  `).join("") || '<div class="item"><div class="body">当前场景未触发文本事件。</div></div>';
}

function renderExplanations() {
  const data = selectedResult().explanations;
  const box = $("explanations");
  box.innerHTML = data.map((d) => `
    <div class="item">
      <div class="time">${d.timestamp} · ${d.event_type}</div>
      <div class="title">风险${d.risk_level} · ${d.predicted_price_trend} · ${d.suggested_action}</div>
      <div class="body">${d.explanation}</div>
    </div>
  `).join("");
}

function formatNumber(v) {
  return Number(v).toLocaleString("zh-CN", { maximumFractionDigits: 1 });
}

$("runBtn").addEventListener("click", runExperiment);
$("policySelect").addEventListener("change", () => {
  renderChart();
  renderEvents();
  renderExplanations();
});

runExperiment();

