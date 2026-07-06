from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else ROOT_DIR / path


def ensure_runtime_dirs(config: dict[str, Any] | None = None) -> None:
    cfg = config or load_config()
    for key in ("uploads", "results", "figures"):
        project_path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    for key in ("yolo_weights", "resnet_weights", "lstm_weights", "lstm_scaler"):
        project_path(cfg["paths"][key]).parent.mkdir(parents=True, exist_ok=True)
