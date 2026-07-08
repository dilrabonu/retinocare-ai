"""A simple custom CNN -- establishes a performance floor before transfer learning."""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """4 conv blocks (Conv -> BatchNorm -> ReLU -> MaxPool) + global average
    pool + a linear classifier head. Deliberately small: the goal is a
    reference number, not to compete with the transfer-learning models."""

    def __init__(self, num_classes: int = 5):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)        # (B, 256, H/16, W/16)
        x = self.pool(x)            # (B, 256, 1, 1)
        x = torch.flatten(x, 1)     # (B, 256)
        return self.classifier(x)   # (B, num_classes)