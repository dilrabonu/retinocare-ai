"""PyTorch Dataset/DataLoader classes for the APTOS 2019 diabetic retinopathy dataset."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.retinocare.data.transforms import get_eval_transforms, get_train_transforms

SEVERITY_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


class RetinopathyDataset(Dataset):
    """Loads fundus images and their severity labels (0-4).

    Expects a CSV with columns [image_id, diagnosis], where image_id has no
    file extension (APTOS convention) and images live in image_dir as
    <image_id>.png.
    """

    def __init__(self, csv_path: str | Path, image_dir: str | Path, transform=None):
        self.df = pd.read_csv(csv_path)
        if "image_id" not in self.df.columns or "diagnosis" not in self.df.columns:
            raise ValueError(
                f"Expected columns ['image_id', 'diagnosis'] in {csv_path}, "
                f"got {list(self.df.columns)}"
            )
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.image_dir / f"{row['image_id']}.png"
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        label = int(row["diagnosis"])

        if self.transform is not None:
            image_tensor = self.transform(image=image_np)["image"]
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        return image_tensor, label

    def class_counts(self) -> dict:
        """Returns {class_index: count} -- used to compute class weights for
        the loss function (see train.py)."""
        return self.df["diagnosis"].value_counts().sort_index().to_dict()


def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Builds train/val/test DataLoaders from the config's data paths and batch size."""
    data_cfg = config["data"]
    image_size = data_cfg["image_size"]

    train_ds = RetinopathyDataset(
        csv_path=data_cfg["train_csv"],
        image_dir=data_cfg["image_dir"],
        transform=get_train_transforms(image_size),
    )
    val_ds = RetinopathyDataset(
        csv_path=data_cfg["val_csv"],
        image_dir=data_cfg["image_dir"],
        transform=get_eval_transforms(image_size),
    )
    test_ds = RetinopathyDataset(
        csv_path=data_cfg["test_csv"],
        image_dir=data_cfg["image_dir"],
        transform=get_eval_transforms(image_size),
    )

    common = dict(batch_size=data_cfg["batch_size"], num_workers=data_cfg["num_workers"])
    train_dl = DataLoader(train_ds, shuffle=True, **common)
    val_dl = DataLoader(val_ds, shuffle=False, **common)
    test_dl = DataLoader(test_ds, shuffle=False, **common)

    return train_dl, val_dl, test_dl