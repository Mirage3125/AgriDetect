from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import load_config, project_path
from src.disease.utils import draw_demo_detection, heuristic_leaf_prediction


def predict_yolo(image_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    cfg = load_config()
    image = project_path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")
    out_dir = project_path(output_dir or cfg["paths"]["results"])
    out_path = out_dir / f"{image.stem}_yolo_result.jpg"
    classes = cfg["disease"]["class_names"]
    weights = project_path(cfg["paths"]["yolo_weights"])

    if weights.exists():
        try:
            from ultralytics import YOLO

            model = YOLO(str(weights))
            results = model.predict(source=str(image), conf=cfg["disease"]["confidence_threshold"], save=False, verbose=False)
            result = results[0]
            result.save(filename=str(out_path))
            boxes = result.boxes
            if boxes and len(boxes) > 0:
                cls_id = int(boxes.cls[0].item())
                confidence = float(boxes.conf[0].item())
                label = model.names.get(cls_id, classes[cls_id] if cls_id < len(classes) else str(cls_id))
                return {"engine": "yolov8", "class_name": label, "confidence": confidence, "result_image": str(out_path)}
        except Exception:
            pass

    prediction = heuristic_leaf_prediction(image, classes)
    draw_demo_detection(image, out_path, prediction)
    prediction.update({"engine": "heuristic-yolo-fallback", "result_image": str(out_path)})
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    print(predict_yolo(args.image, args.output))


if __name__ == "__main__":
    main()
