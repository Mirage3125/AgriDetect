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
    document.querySelector("#note").textContent = data.note || `推理引擎：${data.engine}`;
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
  document.querySelector("#yield-value").textContent = payload.data.prediction.predicted_yield;
  document.querySelector("#yield-engine").textContent = `模型：${payload.data.prediction.engine}`;
}
if (yieldForm) {
  yieldForm.addEventListener("submit", (event) => {
    event.preventDefault();
    submitYield(false);
  });
  document.querySelector("#demo-yield").addEventListener("click", () => submitYield(true));
}
