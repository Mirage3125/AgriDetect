from __future__ import annotations

import json

from app import app
from scripts.prepare_demo_data import create_sample_leaf, create_yield_csv


def test_health() -> None:
    client = app.test_client()
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["success"] is True


def test_disease_prediction() -> None:
    image = create_sample_leaf()
    client = app.test_client()
    with image.open("rb") as file:
        response = client.post("/api/disease/predict", data={"method": "yolo", "image": (file, "sample_leaf.png")})
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["data"]["class_name"]


def test_yield_prediction() -> None:
    create_yield_csv()
    client = app.test_client()
    response = client.post("/api/yield/predict", data=json.dumps({"use_demo": True}), content_type="application/json")
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["data"]["prediction"]["predicted_yield"] > 0
