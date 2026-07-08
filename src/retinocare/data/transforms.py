"""Augmentation pipelines built on albumentations.

Design notes:
- Augmentation for retinal images must stay conservative: aggressive color
  or geometric distortion can destroy clinically meaningful signal
  (microaneurysms, hemorrhages, exudates are small and subtle).
- Normalization uses ImageNet mean/std since ResNet18 / EfficientNet-B0
  were pretrained on ImageNet -- matching normalization statistics matters
  for transfer learning to work well.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Training augmentation: mild geometric + color jitter, then normalize."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.5, border_mode=0),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> A.Compose:
    """Eval/test pipeline: resize + normalize only -- no randomness, for
    reproducible and comparable metrics across runs."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )