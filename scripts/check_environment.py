from __future__ import annotations

import importlib.util
import platform
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import ROOT_DIR


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    print(f"Project root: {ROOT_DIR}")
    print(f"Python: {platform.python_version()}")
    for module in ["torch", "torchvision", "ultralytics", "flask", "pandas", "sklearn"]:
        print(f"{module}: {has_module(module)}")
    if has_module("torch"):
        import torch

        print(f"torch version: {torch.__version__}")
        print(f"cuda available: {torch.cuda.is_available()}")
    patterns = ["*.pt", "*.pth", "*.csv", "*.yaml", "*.yml", "*.py"]
    for pattern in patterns:
        matches = list(Path(ROOT_DIR).rglob(pattern))
        print(f"{pattern}: {len(matches)} files")


if __name__ == "__main__":
    main()
