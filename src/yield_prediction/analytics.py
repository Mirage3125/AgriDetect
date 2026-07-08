from __future__ import annotations

from copy import deepcopy
from statistics import mean, pstdev
from typing import Any

from src.common.config import load_config
from src.yield_prediction.preprocess import load_yield_dataframe
from src.yield_prediction.predict_lstm import predict_next_yield


def build_recent_rows(input_row: dict[str, float] | None = None) -> list[dict[str, float]]:
    cfg = load_config()
    features = cfg["yield_prediction"]["features"]
    seq_len = cfg["yield_prediction"]["sequence_length"]
    df = load_yield_dataframe()
    recent = df.tail(seq_len)[features].to_dict("records")
    if input_row is not None:
        recent = df.tail(seq_len - 1)[features].to_dict("records")
        recent.append(input_row)
    return recent


def build_yield_decision_report(input_row: dict[str, float] | None = None) -> dict[str, Any]:
    recent_rows = build_recent_rows(input_row)
    prediction = predict_next_yield(recent_rows)
    predicted = float(prediction["predicted_yield"])
    historical = load_yield_dataframe()["crop_yield"].tail(18).astype(float).tolist()
    baseline = mean(historical)
    volatility = pstdev(historical) if len(historical) > 1 else baseline * 0.04
    interval_width = max(volatility * 0.65, predicted * 0.025)
    delta_pct = ((predicted - baseline) / baseline) * 100 if baseline else 0.0

    return {
        "prediction": prediction,
        "risk": _risk_level(delta_pct),
        "confidence_interval": {
            "lower": round(max(0.0, predicted - interval_width), 2),
            "upper": round(predicted + interval_width, 2),
            "basis": "recent historical volatility",
        },
        "baseline": {
            "recent_average": round(baseline, 2),
            "delta_percent": round(delta_pct, 2),
        },
        "sensitivity": _scenario_sensitivity(recent_rows, predicted),
        "recommendations": _recommendations(recent_rows[-1], delta_pct),
    }


def _scenario_sensitivity(recent_rows: list[dict[str, float]], baseline_prediction: float) -> list[dict[str, Any]]:
    scenarios = [
        ("rainfall", 0.1, "降雨增加 10%"),
        ("soil_moisture", 0.1, "土壤湿度增加 10%"),
        ("fertilizer_usage", 0.1, "施肥量增加 10%"),
        ("temperature", -0.08, "温度降低 8%"),
    ]
    impacts: list[dict[str, Any]] = []
    for feature, ratio, label in scenarios:
        rows = deepcopy(recent_rows)
        rows[-1][feature] = float(rows[-1][feature]) * (1 + ratio)
        scenario_prediction = float(predict_next_yield(rows)["predicted_yield"])
        impacts.append({
            "feature": feature,
            "scenario": label,
            "predicted_yield": round(scenario_prediction, 2),
            "impact": round(scenario_prediction - baseline_prediction, 2),
        })
    return sorted(impacts, key=lambda item: abs(item["impact"]), reverse=True)


def _risk_level(delta_pct: float) -> dict[str, str]:
    if delta_pct < -8:
        return {"level": "high", "label": "高风险", "reason": "预测产量显著低于近期均值"}
    if delta_pct < -3:
        return {"level": "medium", "label": "中风险", "reason": "预测产量略低于近期均值"}
    return {"level": "low", "label": "低风险", "reason": "预测产量接近或高于近期均值"}


def _recommendations(row: dict[str, float], delta_pct: float) -> list[str]:
    tips: list[str] = []
    if row.get("soil_moisture", 0) < 35:
        tips.append("土壤湿度偏低，建议优先核查灌溉计划。")
    if row.get("soil_ph", 7) < 5.8 or row.get("soil_ph", 7) > 7.5:
        tips.append("土壤 pH 偏离适宜区间，建议结合测土结果调整。")
    if row.get("temperature", 0) > 32:
        tips.append("温度偏高，建议关注蒸散增强和高温胁迫。")
    if delta_pct < -3:
        tips.append("预测低于近期均值，建议复核气象、水肥和病害风险。")
    return tips or ["当前输入未触发明显风险，建议持续监测关键农情指标。"]
