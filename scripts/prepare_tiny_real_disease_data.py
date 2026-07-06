from __future__ import annotations

import random
import sys
import urllib.request
from urllib.error import HTTPError, URLError
from pathlib import Path

from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import ROOT_DIR


SOURCES = {
    "healthy": [
        {
            "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Tomato_scanned.jpg",
            "source": "Wikimedia Commons / Wikipedia Potato leaf page",
        },
        {
            "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Potato-leaf.jpg",
            "source": "Wikimedia Commons / Wikipedia Potato leaf page",
        },
    ],
    "late_blight": [
        {
            "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Late_blight_on_potato_leaf_2.jpg",
            "source": "Wikimedia Commons / Wikipedia Phytophthora infestans page",
        }
    ],
    "rust": [
        {
            "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Cedar_apple_rust_heavily_infected_leaf_underside.JPG",
            "source": "Wikimedia Commons / Wikipedia Gymnosporangium juniperi-virginianae page",
        },
        {
            "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Cedar_apple_rust_showing_stomata.jpg",
            "source": "Wikimedia Commons / Wikipedia Gymnosporangium juniperi-virginianae page",
        },
    ],
}


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "AgriDetectTinyDataset/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        output.write_bytes(response.read())


def augment_image(source: Path, output: Path, seed: int) -> None:
    rng = random.Random(seed)
    with Image.open(source).convert("RGB") as image:
        width, height = image.size
        crop_ratio = rng.uniform(0.78, 1.0)
        crop_w, crop_h = int(width * crop_ratio), int(height * crop_ratio)
        left = rng.randint(0, max(0, width - crop_w))
        top = rng.randint(0, max(0, height - crop_h))
        image = image.crop((left, top, left + crop_w, top + crop_h)).resize((224, 224))
        if rng.random() > 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.12))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.15))
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, quality=92)


def create_yolo_label(label_path: Path, class_id: int) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(f"{class_id} 0.5 0.5 0.86 0.86\n", encoding="utf-8")


def main() -> None:
    raw_dir = ROOT_DIR / "data" / "disease" / "raw_sources"
    cls_dir = ROOT_DIR / "data" / "disease" / "classification"
    yolo_dir = ROOT_DIR / "data" / "disease"
    metadata_lines = [
        "# Tiny real disease dataset",
        "",
        "This tiny dataset uses several real public web images as seed images and creates small augmented train/val samples for local demonstration.",
        "It is suitable for smoke tests and resume demos, not for reporting real model accuracy.",
        "",
        "## Sources",
    ]
    class_names = list(SOURCES)
    for class_id, class_name in enumerate(class_names):
        source_paths: list[Path] = []
        for idx, item in enumerate(SOURCES[class_name]):
            suffix = Path(item["url"].split("?")[0]).suffix or ".jpg"
            raw_path = raw_dir / class_name / f"{class_name}_{idx}{suffix}"
            if not raw_path.exists():
                try:
                    download(item["url"], raw_path)
                except (HTTPError, URLError, TimeoutError) as exc:
                    print(f"Skipped unavailable source for {class_name}: {item['url']} ({exc})")
                    continue
            if raw_path.exists():
                source_paths.append(raw_path)
                metadata_lines.append(f"- {class_name}: {item['source']} - {item['url']}")
        if not source_paths:
            raise RuntimeError(f"No source images available for class: {class_name}")

        for split, count in {"train": 8, "val": 3}.items():
            for sample_idx in range(count):
                source = source_paths[sample_idx % len(source_paths)]
                file_name = f"{class_name}_{split}_{sample_idx:02d}.jpg"
                cls_output = cls_dir / class_name / file_name
                yolo_image = yolo_dir / "images" / split / file_name
                augment_image(source, cls_output, seed=class_id * 1000 + sample_idx + (0 if split == "train" else 100))
                augment_image(source, yolo_image, seed=class_id * 1000 + sample_idx + (10 if split == "train" else 110))
                create_yolo_label(yolo_dir / "labels" / split / f"{Path(file_name).stem}.txt", class_id)

    data_yaml = yolo_dir / "data.yaml"
    data_yaml.write_text(
        "path: data/disease\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names: {class_names}\n",
        encoding="utf-8",
    )
    (yolo_dir / "DATASET_SOURCES.md").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    print(f"Created classification dataset: {cls_dir}")
    print(f"Created YOLO tiny dataset: {yolo_dir}")
    print(f"Classes: {class_names}")


if __name__ == "__main__":
    main()
