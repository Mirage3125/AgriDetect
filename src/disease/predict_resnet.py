from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import load_config, project_path
from src.disease.utils import heuristic_leaf_prediction


def predict_resnet(image_path: str | Path) -> dict[str, Any]:
    cfg = load_config()
    image = project_path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")
    classes = cfg["disease"]["class_names"]
    weights = project_path(cfg["paths"]["resnet_weights"])
    if weights.exists():
        try:
            from torchvision import models, transforms

            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
            model.load_state_dict(torch.load(weights, map_location="cpu"))
            model.eval()
            transform = transforms.Compose([
                transforms.Resize((cfg["disease"]["image_size"], cfg["disease"]["image_size"])),
                transforms.ToTensor(),
            ])
            tensor = transform(Image.open(image).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0]
            confidence, index = torch.max(probs, dim=0)
            label = classes[int(index)]
            return {"engine": "resnet18", "class_name": label, "confidence": float(confidence), "note": ""}
        except Exception:
            pass
    prediction = heuristic_leaf_prediction(image, classes)
    prediction["engine"] = "heuristic-resnet-fallback"
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    print(predict_resnet(args.image))


if __name__ == "__main__":
    main()
