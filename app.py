from __future__ import annotations

import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

from src.common.config import ensure_runtime_dirs, load_config, project_path
from src.disease.predict_resnet import predict_resnet
from src.disease.predict_yolo import predict_yolo
from src.disease.utils import allowed_image
from src.yield_prediction.analytics import build_yield_decision_report
from src.yield_prediction.preprocess import load_yield_dataframe


cfg = load_config()
ensure_runtime_dirs(cfg)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = cfg["disease"]["max_upload_mb"] * 1024 * 1024


def api_response(success: bool, message: str, data: dict | None = None, status: int = 200):
    return jsonify({"success": success, "message": message, "data": data or {}}), status


def static_url(path: str | Path) -> str:
    rel = Path(path).resolve().relative_to(project_path("static").resolve()).as_posix()
    return url_for("static", filename=rel)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/disease")
def disease_page():
    return render_template("disease.html")


@app.post("/api/disease/predict")
def disease_predict():
    if "image" not in request.files:
        return api_response(False, "No image file uploaded", status=400)
    file = request.files["image"]
    if not file.filename:
        return api_response(False, "No selected image", status=400)
    if not allowed_image(file.filename):
        return api_response(False, "Only JPG, JPEG and PNG files are supported", status=400)
    method = request.form.get("method", "yolo").lower()
    upload_dir = project_path(cfg["paths"]["uploads"])
    suffix = Path(secure_filename(file.filename)).suffix.lower()
    upload_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    file.save(upload_path)
    try:
        prediction = predict_resnet(upload_path) if method == "resnet" else predict_yolo(upload_path)
        result_image = prediction.get("result_image")
        confidence = round(float(prediction.get("confidence", 0.0)), 4)
        return api_response(True, "Prediction completed", {
            "method": method,
            "engine": prediction.get("engine"),
            "class_name": prediction.get("class_name"),
            "confidence": confidence,
            "severity": disease_severity(confidence),
            "note": prediction.get("note", ""),
            "actions": disease_actions(prediction.get("class_name", "")),
            "original_image_url": static_url(upload_path),
            "result_image_url": static_url(result_image) if result_image else static_url(upload_path),
        })
    except Exception as exc:
        return api_response(False, f"Inference failed: {exc}", status=500)


@app.get("/yield")
def yield_page():
    df = load_yield_dataframe()
    rows = df.tail(12).to_dict("records")
    return render_template("yield.html", rows=rows)


@app.post("/api/yield/predict")
def yield_predict():
    try:
        payload = request.get_json(silent=True) or {}
        use_demo = bool(payload.get("use_demo", False))
        if use_demo:
            report = build_yield_decision_report()
        else:
            row = {feature: float(payload[feature]) for feature in cfg["yield_prediction"]["features"]}
            report = build_yield_decision_report(row)
        df = load_yield_dataframe()
        trend = df.tail(18)[["year", "month", "crop_yield"]].to_dict("records")
        report["trend"] = trend
        return api_response(True, "Yield prediction completed", report)
    except KeyError as exc:
        return api_response(False, f"Missing input field: {exc}", status=400)
    except Exception as exc:
        return api_response(False, f"Prediction failed: {exc}", status=500)


@app.get("/api/health")
def health():
    return api_response(True, "Service is healthy", {
        "torch_available": True,
        "yolo_weights_exists": project_path(cfg["paths"]["yolo_weights"]).exists(),
        "resnet_weights_exists": project_path(cfg["paths"]["resnet_weights"]).exists(),
        "lstm_weights_exists": project_path(cfg["paths"]["lstm_weights"]).exists(),
    })


def disease_severity(confidence: float) -> dict[str, str]:
    if confidence >= 0.75:
        return {"level": "high", "label": "高置信"}
    if confidence >= 0.55:
        return {"level": "medium", "label": "需复核"}
    return {"level": "low", "label": "低置信"}


def disease_actions(class_name: str) -> list[str]:
    action_map = {
        "healthy": ["保持当前水肥管理。", "继续巡田并保留图片样本作为健康基线。"],
        "leaf_spot": ["清理重病叶片，降低田间湿度。", "连续 3-5 天复拍同一区域，观察斑点扩散速度。"],
        "rust": ["优先检查叶背和相邻植株。", "将结果交给农技人员复核后再决定用药。"],
        "blight": ["检查根茎、排水和土壤湿度。", "隔离疑似病株并记录发病位置。"],
    }
    return action_map.get(class_name, ["建议结合人工巡检复核模型输出。"])


if __name__ == "__main__":
    app.run(host=cfg["flask"]["host"], port=cfg["flask"]["port"], debug=cfg["flask"]["debug"])
