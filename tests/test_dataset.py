"""Unit tests for RetinopathyDataset -- run with: pytest tests/test_dataset.py -v"""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.retinocare.data.dataset import RetinopathyDataset
from src.retinocare.data.transforms import get_eval_transforms


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Creates 6 fake images (2 per class, 3 classes) + a matching CSV."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    rows = []
    for i in range(6):
        img = Image.fromarray(np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8))
        img.save(image_dir / f"img{i}.png")
        rows.append({"image_id": f"img{i}", "diagnosis": i % 3})

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return csv_path, image_dir


def test_dataset_length_matches_csv(synthetic_dataset):
    csv_path, image_dir = synthetic_dataset
    ds = RetinopathyDataset(csv_path, image_dir, transform=get_eval_transforms(64))
    assert len(ds) == 6


def test_dataset_returns_tensor_and_label(synthetic_dataset):
    csv_path, image_dir = synthetic_dataset
    ds = RetinopathyDataset(csv_path, image_dir, transform=get_eval_transforms(64))
    image, label = ds[0]

    assert image.shape == (3, 64, 64)
    assert isinstance(label, int)
    assert 0 <= label <= 4


def test_dataset_rejects_missing_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"wrong_col": [1, 2]}).to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="Expected columns"):
        RetinopathyDataset(bad_csv, tmp_path, transform=None)


def test_class_counts(synthetic_dataset):
    csv_path, image_dir = synthetic_dataset
    ds = RetinopathyDataset(csv_path, image_dir, transform=get_eval_transforms(64))
    counts = ds.class_counts()
    assert counts == {0: 2, 1: 2, 2: 2}