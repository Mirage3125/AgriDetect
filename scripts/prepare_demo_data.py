from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import ROOT_DIR, load_config, project_path


def create_yield_csv() -> Path:
    cfg = load_config()
    path = project_path(cfg["paths"]["yield_data"])
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    rows = []
    historical = 6100.0
    for idx in range(60):
        year = 2021 + idx // 12
        month = idx % 12 + 1
        temperature = 18 + 10 * np.sin((month - 2) / 12 * 2 * np.pi) + rng.normal(0, 1.0)
        rainfall = 80 + 45 * np.sin((month + 1) / 12 * 2 * np.pi) + rng.normal(0, 8)
        humidity = 62 + 12 * np.sin(month / 12 * 2 * np.pi) + rng.normal(0, 3)
        soil_moisture = 35 + rainfall * 0.08 + rng.normal(0, 1.5)
        soil_ph = 6.5 + rng.normal(0, 0.15)
        fertilizer_usage = 180 + rng.normal(0, 10)
        planting_area = 120 + rng.normal(0, 2)
        crop_yield = historical * 0.55 + temperature * 22 + rainfall * 5 + soil_moisture * 28 + fertilizer_usage * 7 + rng.normal(0, 120)
        rows.append({
            "year": year,
            "month": month,
            "temperature": round(float(temperature), 2),
            "rainfall": round(float(max(10, rainfall)), 2),
            "humidity": round(float(humidity), 2),
            "soil_moisture": round(float(soil_moisture), 2),
            "soil_ph": round(float(soil_ph), 2),
            "fertilizer_usage": round(float(fertilizer_usage), 2),
            "planting_area": round(float(planting_area), 2),
            "historical_yield": round(float(historical), 2),
            "crop_yield": round(float(crop_yield), 2),
        })
        historical = crop_yield
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return path


def create_sample_leaf() -> Path:
    cfg = load_config()
    path = project_path(cfg["paths"]["sample_image"])
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 420), (238, 244, 226))
    draw = ImageDraw.Draw(image)
    draw.ellipse((120, 55, 520, 365), fill=(72, 145, 75), outline=(42, 105, 52), width=5)
    draw.line((320, 70, 320, 360), fill=(226, 232, 180), width=5)
    for box in [(210, 150, 245, 185), (385, 220, 425, 260), (275, 260, 305, 290)]:
        draw.ellipse(box, fill=(156, 91, 45), outline=(102, 61, 34), width=3)
    image.save(path)
    return path


def main() -> None:
    ROOT_DIR.joinpath("outputs").mkdir(exist_ok=True)
    print(f"Created yield data: {create_yield_csv()}")
    print(f"Created sample image: {create_sample_leaf()}")


if __name__ == "__main__":
    main()
