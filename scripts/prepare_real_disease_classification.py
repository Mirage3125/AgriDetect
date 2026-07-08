from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import ROOT_DIR


DATASET_ID = "Hemg/new-plant-diseases-dataset"


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "unknown"


def export_dataset(
    output_dir: Path,
    max_per_class: int | None,
    val_ratio: float,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    features = dataset.features
    class_names = list(features["label"].names)
    saved_counts: dict[str, dict[str, int]] = {
        class_name: {"train": 0, "val": 0, "total": 0}
        for class_name in class_names
    }
    seen_per_class: defaultdict[int, int] = defaultdict(int)
    target_total = None if max_per_class is None else max_per_class * len(class_names)
    progress = tqdm(total=target_total, desc="Exporting disease images")

    for item in dataset:
        label_id = int(item["label"])
        if max_per_class is not None and seen_per_class[label_id] >= max_per_class:
            if all(count >= max_per_class for count in seen_per_class.values()) and len(seen_per_class) == len(class_names):
                break
            continue

        class_name = class_names[label_id]
        split = "val" if rng.random() < val_ratio else "train"
        class_dir = output_dir / split / safe_name(class_name)
        class_dir.mkdir(parents=True, exist_ok=True)
        index = seen_per_class[label_id]
        image_path = class_dir / f"{safe_name(class_name)}_{index:05d}.jpg"
        image: Image.Image = item["image"].convert("RGB")
        image.save(image_path, quality=92)

        seen_per_class[label_id] += 1
        saved_counts[class_name][split] += 1
        saved_counts[class_name]["total"] += 1
        progress.update(1)

    progress.close()
    manifest = {
        "dataset_id": DATASET_ID,
        "output_dir": str(output_dir),
        "max_per_class": max_per_class,
        "val_ratio": val_ratio,
        "classes": class_names,
        "counts": saved_counts,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/disease/real_classification")
    parser.add_argument("--max-per-class", type=int, default=120, help="Use no limit with --full.")
    parser.add_argument("--full", action="store_true", help="Export the full dataset. This downloads about 1.1GB.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    max_per_class = None if args.full else args.max_per_class
    manifest = export_dataset(ROOT_DIR / args.output, max_per_class, args.val_ratio, args.seed)
    total = sum(item["total"] for item in manifest["counts"].values())
    print(f"Exported {total} images to {manifest['output_dir']}")
    print(f"Classes: {len(manifest['classes'])}")


if __name__ == "__main__":
    main()
