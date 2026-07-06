from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.common.config import load_config, project_path


def load_yield_dataframe(csv_path: str | Path | None = None) -> pd.DataFrame:
    cfg = load_config()
    path = project_path(csv_path or cfg["paths"]["yield_data"])
    if not path.exists():
        raise FileNotFoundError(f"Yield CSV not found: {path}. Run scripts/prepare_demo_data.py first.")
    df = pd.read_csv(path)
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(limit_direction="both").fillna(df[numeric_cols].median())
    for col in numeric_cols:
        low, high = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(low, high)
    return df


def build_sequences(df: pd.DataFrame, sequence_length: int | None = None) -> dict[str, Any]:
    cfg = load_config()
    features = cfg["yield_prediction"]["features"]
    target = cfg["yield_prediction"]["target"]
    seq_len = sequence_length or cfg["yield_prediction"]["sequence_length"]
    missing = [col for col in features + [target] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in yield data: {missing}")

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    x_scaled = feature_scaler.fit_transform(df[features])
    y_scaled = target_scaler.fit_transform(df[[target]])
    x_seq, y_seq = [], []
    for i in range(seq_len, len(df)):
        x_seq.append(x_scaled[i - seq_len:i])
        y_seq.append(y_scaled[i])
    return {
        "X": np.asarray(x_seq, dtype=np.float32),
        "y": np.asarray(y_seq, dtype=np.float32),
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "features": features,
        "target": target,
    }
