"""Transfer learning models: ResNet18 and EfficientNet-B0, both pretrained on
ImageNet, with a fresh classifier head swapped in for our 5 severity classes."""

import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 5, freeze_backbone: bool = True) -> nn.Module:
    """Loads ImageNet-pretrained ResNet18 and replaces the final FC layer.

    If freeze_backbone=True, all existing layers are frozen (requires_grad=False)
    so only the new head trains in stage 1. Call unfreeze_backbone(model) later
    for stage-2 fine-tuning at a lower learning rate.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # new layer: requires_grad=True by default
    return model


def build_efficientnet_b0(num_classes: int = 5, freeze_backbone: bool = True) -> nn.Module:
    """Same pattern as build_resnet18, for a 3-way model comparison."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """Stage-2 fine-tuning: unfreeze every parameter so the whole network
    can be fine-tuned at a lower learning rate."""
    for param in model.parameters():
        param.requires_grad = True