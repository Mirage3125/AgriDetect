from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import load_config, project_path
from src.yield_prediction.model import YieldLSTM
from src.yield_prediction.preprocess import build_sequences, load_yield_dataframe


def train_lstm(csv_path: str | None = None, epochs: int | None = None) -> dict[str, float]:
    cfg = load_config()
    train_cfg = cfg["yield_prediction"]["train"]
    df = load_yield_dataframe(csv_path)
    data = build_sequences(df)
    X, y = data["X"], data["y"]
    if len(X) < 4:
        raise ValueError("Not enough rows to train LSTM. Need more time periods.")
    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
    )
    model = YieldLSTM(
        input_size=X.shape[2],
        hidden_size=train_cfg["hidden_size"],
        num_layers=train_cfg["num_layers"],
        dropout=train_cfg["dropout"],
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    losses: list[float] = []
    for _ in range(epochs or train_cfg["epochs"]):
        total = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss)
        losses.append(total / max(1, len(train_loader)))

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(X_test if len(X_test) else X_train)).numpy()
    y_eval = y_test if len(y_test) else y_train
    pred = data["target_scaler"].inverse_transform(pred_scaled).ravel()
    true = data["target_scaler"].inverse_transform(y_eval).ravel()
    metrics = {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "r2": float(r2_score(true, pred)) if len(true) > 1 else 0.0,
    }

    model_path = project_path(cfg["paths"]["lstm_weights"])
    scaler_path = project_path(cfg["paths"]["lstm_scaler"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_size": X.shape[2], "train_cfg": train_cfg, "metrics": metrics}, model_path)
    with scaler_path.open("wb") as file:
        pickle.dump({"feature_scaler": data["feature_scaler"], "target_scaler": data["target_scaler"]}, file)

    figures = project_path(cfg["paths"]["figures"])
    figures.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(figures / "lstm_training_loss.png")
    plt.close()
    plt.figure(figsize=(7, 4))
    plt.plot(true, label="True")
    plt.plot(pred, label="Predicted")
    plt.legend()
    plt.title("Crop Yield Prediction")
    plt.tight_layout()
    plt.savefig(figures / "yield_prediction_compare.png")
    plt.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    print(train_lstm(args.csv, args.epochs))


if __name__ == "__main__":
    main()
