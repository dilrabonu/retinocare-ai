"""Evaluation utilities: per-class F1, confusion matrix, calibration checks."""

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def compute_classification_report(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: list | None = None
) -> dict:
    """Returns per-class precision/recall/F1 plus weighted & macro F1.

    Overall accuracy can hide the fact that a model is specifically bad at
    telling Severe from Proliferative DR -- this report surfaces that.
    """
    labels = list(range(len(class_names))) if class_names else None
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return report


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list,
    save_path: str,
) -> None:
    """Saves a confusion matrix heatmap (true label x predicted label)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def expected_calibration_error(probs: np.ndarray, y_true: Sequence[int], n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE): checks whether confidence scores are
    trustworthy -- e.g. does "80% confident" mean "correct 80% of the time"?

    This matters directly for the RAG agent, which phrases its recommendation
    based on the model's confidence score.
    """
    y_true = np.asarray(y_true)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        bin_size = in_bin.sum()
        if bin_size == 0:
            continue
        avg_confidence_in_bin = confidences[in_bin].mean()
        avg_accuracy_in_bin = accuracies[in_bin].mean()
        ece += (bin_size / n) * abs(avg_confidence_in_bin - avg_accuracy_in_bin)

    return float(ece)