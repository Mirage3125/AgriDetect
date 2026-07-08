from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import load_config, project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/disease/classification")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained ResNet18 weights.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the final classifier layer.")
    args = parser.parse_args()
    data_dir = project_path(args.data)
    if not data_dir.exists():
        raise FileNotFoundError(f"Classification dataset not found: {data_dir}")
    try:
        from torchvision import datasets, models, transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for ResNet training.") from exc

    cfg = load_config()
    train_dir = data_dir / "train" if (data_dir / "train").exists() else data_dir
    val_dir = data_dir / "val" if (data_dir / "val").exists() else None
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    eval_dataset = datasets.ImageFolder(str(val_dir), transform=eval_transform) if val_dir else train_dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    weights_arg = models.ResNet18_Weights.DEFAULT if args.pretrained else None
    model = models.resnet18(weights=weights_arg)
    if args.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=args.lr)
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        total_loss, correct = 0.0, 0
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            correct += int((outputs.argmax(1) == labels).sum())
        train_loss = total_loss / max(1, len(train_loader))
        train_acc = correct / max(1, len(train_dataset))
        val_acc = evaluate(model, eval_loader)["accuracy"]
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"epoch={epoch + 1} loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
    weights = project_path(cfg["paths"]["resnet_weights"])
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    Path(weights.with_suffix(".classes.txt")).write_text("\n".join(train_dataset.classes), encoding="utf-8")
    metrics = evaluate(model, eval_loader, train_dataset.classes)
    metrics["history"] = history
    metrics["data_dir"] = str(data_dir)
    metrics["num_classes"] = len(train_dataset.classes)
    metrics["train_size"] = len(train_dataset)
    metrics["eval_size"] = len(eval_dataset)
    output_dir = project_path("outputs/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resnet18_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    save_confusion_matrix(metrics["confusion_matrix"], train_dataset.classes, project_path("outputs/figures/resnet18_confusion_matrix.png"))
    print(f"Saved ResNet weights to {weights}")
    print(f"Saved metrics to {output_dir / 'resnet18_metrics.json'}")


def evaluate(model: nn.Module, loader: DataLoader, classes: list[str] | None = None) -> dict:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            y_true.extend(labels.tolist())
            y_pred.extend(outputs.argmax(1).tolist())
    labels = list(range(len(classes))) if classes else None
    report = classification_report(y_true, y_pred, target_names=classes, labels=labels, output_dict=True, zero_division=0) if classes else {}
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist() if classes else []
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def save_confusion_matrix(matrix: list[list[int]], classes: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig_size = max(8, min(18, len(classes) * 0.45))
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(np.asarray(matrix), cmap="Greens")
    plt.title("ResNet18 Confusion Matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    plt.yticks(range(len(classes)), classes, fontsize=7)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
