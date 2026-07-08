"""Config-driven training entrypoint -- shared loop for baseline CNN, ResNet18,
and EfficientNet-B0.

Usage:
    python -m src.retinocare.models.train --config configs/train_config.yaml --model resnet18
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score

from src.retinocare.data.dataset import get_dataloaders
from src.retinocare.models.baseline_cnn import BaselineCNN
from src.retinocare.models.resnet_transfer import build_efficientnet_b0, build_resnet18, unfreeze_backbone


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    elif name == "resnet18":
        return build_resnet18(num_classes=num_classes, freeze_backbone=True)
    elif name == "efficientnet_b0":
        return build_efficientnet_b0(num_classes=num_classes, freeze_backbone=True)
    raise ValueError(f"Unknown model name: {name}")


def compute_class_weights(train_dl, num_classes: int, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights, used in CrossEntropyLoss to counter
    the APTOS class imbalance (No DR dominates the dataset)."""
    counts = torch.zeros(num_classes)
    for _, labels in train_dl:
        for label in labels:
            counts[label] += 1
    weights = 1.0 / counts.clamp(min=1)
    weights = weights / weights.sum() * num_classes  # normalize so avg weight ~= 1
    return weights.to(device)


def run_epoch(model, dataloader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, weighted_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--model", type=str, choices=["baseline_cnn", "resnet18", "efficientnet_b0"], required=True
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dl, val_dl, _ = get_dataloaders(config)
    num_classes = 5

    model = build_model(args.model, num_classes).to(device)

    train_cfg = config["training"]
    class_weights = (
        compute_class_weights(train_dl, num_classes, device)
        if train_cfg.get("class_weighting", False)
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    best_val_f1 = 0.0
    epochs_without_improvement = 0
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{args.model}.pt"

    freeze_epochs = config.get("model", {}).get("freeze_backbone_epochs", 0)

    for epoch in range(train_cfg["epochs"]):
        # Stage-2 fine-tuning: unfreeze backbone after the head has stabilized
        if args.model != "baseline_cnn" and epoch == freeze_epochs:
            print(f"Epoch {epoch}: unfreezing backbone for full fine-tuning")
            unfreeze_backbone(model)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=train_cfg["learning_rate"] * 0.1, weight_decay=train_cfg["weight_decay"]
            )

        train_loss, train_f1 = run_epoch(model, train_dl, criterion, optimizer, device, train=True)
        val_loss, val_f1 = run_epoch(model, val_dl, criterion, optimizer, device, train=False)

        print(
            f"Epoch {epoch+1}/{train_cfg['epochs']} | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> new best val_f1={val_f1:.4f}, saved to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1} (no improvement for "
                      f"{train_cfg['early_stopping_patience']} epochs)")
                break

    print(f"Training complete. Best val_f1={best_val_f1:.4f}. Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()