from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageStat

from src.common.config import project_path


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


DISEASE_NOTES = {
    "healthy": "叶片状态较健康，建议继续保持水肥和通风管理。",
    "leaf_spot": "疑似叶斑类病害，建议清理病叶并关注湿度与通风。",
    "rust": "疑似锈病特征，建议尽早隔离观察并按当地农技建议处理。",
    "blight": "疑似枯萎或疫病类症状，建议检查土壤湿度和植株根茎状态。",
}


def allowed_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def inspect_image(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return {"valid": True, "size": image.size, "mode": image.mode}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def heuristic_leaf_prediction(image_path: Path, class_names: list[str]) -> dict[str, Any]:
    with Image.open(image_path).convert("RGB") as image:
        stat = ImageStat.Stat(image.resize((96, 96)))
        red, green, blue = stat.mean
        brightness = sum(stat.mean) / 3
        if green > red * 1.08 and green > blue * 1.08 and brightness > 85:
            label = "healthy"
            confidence = 0.72
        elif red > green * 1.08:
            label = "rust"
            confidence = 0.68
        elif brightness < 90:
            label = "blight"
            confidence = 0.64
        else:
            label = "leaf_spot"
            confidence = 0.61
    if label not in class_names:
        label = class_names[0]
    return {"class_name": label, "confidence": confidence, "note": DISEASE_NOTES.get(label, "")}


def draw_demo_detection(image_path: Path, output_path: Path, prediction: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        width, height = image.size
        margin_x = max(8, width // 8)
        margin_y = max(8, height // 8)
        box = [margin_x, margin_y, width - margin_x, height - margin_y]
        draw.rectangle(box, outline=(38, 132, 82), width=max(3, width // 120))
        text = f"{prediction['class_name']} {prediction['confidence']:.2f}"
        draw.rectangle([box[0], max(0, box[1] - 24), box[0] + 180, box[1]], fill=(38, 132, 82))
        draw.text((box[0] + 6, max(0, box[1] - 20)), text, fill=(255, 255, 255))
        image.save(output_path)
    return output_path


def find_dataset_images(data_dir: str | Path) -> list[Path]:
    root = project_path(data_dir)
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS]


def sample_images(paths: list[Path], count: int = 8) -> list[Path]:
    return random.sample(paths, min(count, len(paths))) if paths else []
