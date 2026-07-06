from __future__ import annotations

import pickle
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import load_config, project_path
from src.yield_prediction.model import YieldLSTM
from src.yield_prediction.preprocess import build_sequences, load_yield_dataframe


def predict_next_yield(recent_rows: list[dict[str, float]] | None = None) -> dict[str, Any]:
    cfg = load_config()
    features = cfg["yield_prediction"]["features"]
    seq_len = cfg["yield_prediction"]["sequence_length"]
    model_path = project_path(cfg["paths"]["lstm_weights"])
    scaler_path = project_path(cfg["paths"]["lstm_scaler"])

    if recent_rows is None:
        df = load_yield_dataframe()
        recent_rows = df.tail(seq_len)[features].to_dict("records")
    if len(recent_rows) < seq_len:
        raise ValueError(f"Need at least {seq_len} recent rows for LSTM prediction.")

    if model_path.exists() and scaler_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        with scaler_path.open("rb") as file:
            scalers = pickle.load(file)
        model = YieldLSTM(input_size=checkpoint["input_size"], **{
            "hidden_size": checkpoint["train_cfg"]["hidden_size"],
            "num_layers": checkpoint["train_cfg"]["num_layers"],
            "dropout": checkpoint["train_cfg"]["dropout"],
        })
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        array = np.asarray([[row[col] for col in features] for row in recent_rows[-seq_len:]], dtype=np.float32)
        scaled = scalers["feature_scaler"].transform(array)
        with torch.no_grad():
            pred_scaled = model(torch.tensor(scaled).unsqueeze(0)).numpy()
        value = float(scalers["target_scaler"].inverse_transform(pred_scaled)[0, 0])
        return {"engine": "lstm", "predicted_yield": round(value, 2), "metrics": checkpoint.get("metrics", {})}

    df = load_yield_dataframe()
    data = build_sequences(df)
    model = LinearRegression().fit(data["X"].reshape(len(data["X"]), -1), data["y"].ravel())
    sample = np.asarray([[row[col] for col in features] for row in recent_rows[-seq_len:]], dtype=np.float32)
    sample_scaled = data["feature_scaler"].transform(sample).reshape(1, -1)
    pred_scaled = model.predict(sample_scaled).reshape(-1, 1)
    value = float(data["target_scaler"].inverse_transform(pred_scaled)[0, 0])
    return {"engine": "linear-fallback", "predicted_yield": round(value, 2), "metrics": {"note": "LSTM weights not found; used demo fallback."}}


def main() -> None:
    print(predict_next_yield())


if __name__ == "__main__":
    main()
