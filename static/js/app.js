const diseaseForm = document.querySelector("#disease-form");
if (diseaseForm) {
  diseaseForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = document.querySelector("#disease-message");
    message.textContent = "识别中...";
    const response = await fetch("/api/disease/predict", { method: "POST", body: new FormData(diseaseForm) });
    const payload = await response.json();
    message.textContent = payload.message;
    if (!payload.success) return;
    const data = payload.data;
    document.querySelector("#original-image").src = data.original_image_url;
    document.querySelector("#result-image").src = data.result_image_url;
    document.querySelector("#class-name").textContent = data.class_name;
    document.querySelector("#confidence").textContent = data.confidence;
    document.querySelector("#severity").textContent = data.severity?.label || "-";
    document.querySelector("#note").textContent = data.note || `推理引擎：${data.engine}`;
    renderList("#disease-actions", data.actions || []);
  });
}

const yieldForm = document.querySelector("#yield-form");
async function submitYield(useDemo = false) {
  const message = document.querySelector("#yield-message");
  message.textContent = "预测中...";
  const data = useDemo ? { use_demo: true } : Object.fromEntries(new FormData(yieldForm).entries());
  const response = await fetch("/api/yield/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });
  const payload = await response.json();
  message.textContent = payload.message;
  if (!payload.success) return;
  const report = payload.data;
  document.querySelector("#yield-value").textContent = report.prediction.predicted_yield;
  document.querySelector("#yield-engine").textContent = `模型：${report.prediction.engine}`;
  document.querySelector("#yield-risk").textContent = report.risk.label;
  document.querySelector("#yield-risk").dataset.level = report.risk.level;
  document.querySelector("#yield-delta").textContent = `较近期均值 ${formatSigned(report.baseline.delta_percent)}%`;
  document.querySelector("#yield-interval").textContent = `置信区间：${report.confidence_interval.lower} - ${report.confidence_interval.upper} kg/ha（基于近期波动估计）`;
  renderSensitivity(report.sensitivity || []);
  renderList("#yield-recommendations", report.recommendations || []);
}
if (yieldForm) {
  yieldForm.addEventListener("submit", (event) => {
    event.preventDefault();
    submitYield(false);
  });
  document.querySelector("#demo-yield").addEventListener("click", () => submitYield(true));
}

function renderSensitivity(rows) {
  const tbody = document.querySelector("#sensitivity-table");
  if (!tbody) return;
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td>${row.scenario}</td>
      <td>${row.predicted_yield}</td>
      <td>${formatSigned(row.impact)}</td>
    </tr>
  `).join("");
}

function renderList(selector, items) {
  const list = document.querySelector(selector);
  if (!list) return;
  list.innerHTML = items.map((item) => `<li>${item}</li>`).join("");
}

function formatSigned(value) {
  const number = Number(value);
  return `${number > 0 ? "+" : ""}${number.toFixed(2)}`;
}
