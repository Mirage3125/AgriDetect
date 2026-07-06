from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import load_config, project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/disease/classification")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    data_dir = project_path(args.data)
    if not data_dir.exists():
        raise FileNotFoundError(f"Classification dataset not found: {data_dir}")
    try:
        from torchvision import datasets, models, transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for ResNet training.") from exc

    cfg = load_config()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(str(data_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        total_loss, correct = 0.0, 0
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            correct += int((outputs.argmax(1) == labels).sum())
        print(f"epoch={epoch + 1} loss={total_loss / max(1, len(loader)):.4f} acc={correct / len(dataset):.4f}")
    weights = project_path(cfg["paths"]["resnet_weights"])
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    Path(weights.with_suffix(".classes.txt")).write_text("\n".join(dataset.classes), encoding="utf-8")
    print(f"Saved ResNet weights to {weights}")


if __name__ == "__main__":
    main()
