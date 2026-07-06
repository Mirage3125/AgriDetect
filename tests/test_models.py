from __future__ import annotations

from scripts.prepare_demo_data import create_sample_leaf, create_yield_csv
from src.disease.predict_resnet import predict_resnet
from src.disease.predict_yolo import predict_yolo
from src.yield_prediction.predict_lstm import predict_next_yield


def test_model_fallbacks() -> None:
    image = create_sample_leaf()
    create_yield_csv()
    assert predict_yolo(image)["class_name"]
    assert predict_resnet(image)["class_name"]
    assert predict_next_yield()["predicted_yield"] > 0
