from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import load_config, project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="YOLO data.yaml path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    data_yaml = project_path(args.data) if args.data else project_path("data/disease/data.yaml")
    weights = project_path(load_config()["paths"]["yolo_weights"])
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data yaml not found: {data_yaml}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Install it only when YOLO training is needed: pip install ultralytics") from exc
    model = YOLO("yolov8n.pt")
    results = model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz)
    save_dir = Path(getattr(results, "save_dir", ""))
    best = save_dir / "weights" / "best.pt"
    if not best.exists() and getattr(model, "trainer", None):
        best = Path(model.trainer.best)
    if best.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, weights)
        print(f"Training finished. Saved YOLO weights to {weights}")
    else:
        print(f"Training finished, but best.pt was not found automatically. Check run output and copy it to {weights}")


if __name__ == "__main__":
    main()
