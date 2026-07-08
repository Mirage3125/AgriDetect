from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import ROOT_DIR


DATASET_ID = "susnato/plant_disease_detection_processed"


def convert_split(split: str, output_dir: Path, limit: int | None, category_map: dict[int, int]) -> dict[str, int]:
    dataset = load_dataset(DATASET_ID, split=split, streaming=True)
    dataset = dataset.remove_columns(["pixel_values", "pixel_mask", "labels"])
    yolo_split = "val" if split == "test" else split
    image_dir = output_dir / "images" / yolo_split
    label_dir = output_dir / "labels" / yolo_split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    box_count = 0
    progress = tqdm(total=limit, desc=f"Exporting PlantDoc {split}")
    for item in dataset:
        if limit is not None and count >= limit:
            break
        width = int(item["width"])
        height = int(item["height"])
        boxes = item["objects"]["bbox"]
        categories = item["objects"]["category"]
        if not boxes:
            continue
        image: Image.Image = item["image"].convert("RGB")
        stem = f"{split}_{int(item['image_id']):06d}"
        image.save(image_dir / f"{stem}.jpg", quality=92)
        label_lines: list[str] = []
        for bbox, category in zip(boxes, categories):
            raw_category = int(category)
            if raw_category not in category_map:
                category_map[raw_category] = len(category_map)
            class_id = category_map[raw_category]
            x, y, box_w, box_h = [float(value) for value in bbox]
            x_center = (x + box_w / 2) / width
            y_center = (y + box_h / 2) / height
            label_lines.append(
                f"{class_id} {clip01(x_center):.6f} {clip01(y_center):.6f} {clip01(box_w / width):.6f} {clip01(box_h / height):.6f}"
            )
            box_count += 1
        (label_dir / f"{stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        count += 1
        progress.update(1)
    progress.close()
    return {"images": count, "boxes": box_count}


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def write_data_yaml(output_dir: Path, category_map: dict[int, int]) -> None:
    names = [f"plantdoc_class_{raw}" for raw, _ in sorted(category_map.items(), key=lambda item: item[1])]
    yaml = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names: {names}\n"
    )
    (output_dir / "data.yaml").write_text(yaml, encoding="utf-8")
    (output_dir / "category_map.json").write_text(
        json.dumps({"raw_to_yolo": category_map, "names": names}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/disease/plantdoc_yolo_subset")
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--val-limit", type=int, default=50)
    args = parser.parse_args()
    output_dir = ROOT_DIR / args.output
    category_map: dict[int, int] = {}
    train_stats = convert_split("train", output_dir, args.train_limit, category_map)
    val_stats = convert_split("test", output_dir, args.val_limit, category_map)
    write_data_yaml(output_dir, category_map)
    manifest = {
        "dataset_id": DATASET_ID,
        "output_dir": str(output_dir),
        "train": train_stats,
        "val": val_stats,
        "num_classes": len(category_map),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
