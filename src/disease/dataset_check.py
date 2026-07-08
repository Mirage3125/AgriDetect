from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import load_config, project_path
from src.disease.utils import inspect_image


def check_dataset(data_dir: str | Path | None = None) -> dict[str, Any]:
    cfg = load_config()
    root = project_path(data_dir or cfg["paths"]["disease_data"])
    result: dict[str, Any] = {
        "dataset_dir": str(root),
        "exists": root.exists(),
        "classes": [],
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "missing_labels": [],
        "empty_labels": [],
        "broken_images": [],
    }
    if not root.exists():
        return result

    image_exts = {".jpg", ".jpeg", ".png"}
    for split_name in ("train", "val", "valid", "test"):
        split_candidates = [root / split_name, root / "images" / split_name]
        count = 0
        for split in split_candidates:
            if split.exists():
                count += len([p for p in split.rglob("*") if p.suffix.lower() in image_exts])
        if count:
            key = "val_images" if split_name == "valid" else f"{split_name}_images"
            result[key] = count

    ignored_dirs = {"train", "val", "valid", "test", "images", "labels", "raw_sources"}
    class_root = root / "classification"
    if class_root.exists():
        class_dirs = [p.name for p in class_root.iterdir() if p.is_dir()]
    else:
        class_dirs = [p.name for p in root.iterdir() if p.is_dir() and p.name not in ignored_dirs]
    result["classes"] = sorted(class_dirs)

    for image_path in root.rglob("*"):
        if image_path.suffix.lower() not in image_exts:
            continue
        info = inspect_image(image_path)
        if not info["valid"]:
            result["broken_images"].append(str(image_path))
        label_path = Path(str(image_path).replace(f"{Path('images')}", f"{Path('labels')}")).with_suffix(".txt")
        if "images" in image_path.parts and not label_path.exists():
            result["missing_labels"].append(str(image_path))
        elif label_path.exists() and label_path.stat().st_size == 0:
            result["empty_labels"].append(str(label_path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    report = check_dataset(args.data)
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
