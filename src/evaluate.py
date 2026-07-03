"""Model evaluation: metrics, confusion matrix, ROC curves and reports.

Run from the project root::

    python -m src.evaluate --config config.yaml --checkpoint models/best_model.pt

Produces (under ``outputs/``):

* a per-class + macro/weighted classification report (text + JSON),
* a normalised confusion-matrix figure,
* one-vs-rest ROC curves,
* a JSON summary of headline metrics.

On imbalanced data, **macro-F1** and the **per-class recall** in the confusion
matrix matter far more than raw accuracy, so those are emphasised throughout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from .dataset import create_dataloaders
from .model import build_model
from .utils import AAMI_CLASSES, Config, ensure_dir, get_device, get_logger, load_config
from .visualization import plot_confusion_matrix, plot_roc_curves

logger = get_logger(__name__)


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over a loader.

    Returns ``(y_true, y_pred, y_score)`` where ``y_score`` holds the softmax
    probability matrix of shape ``(N, num_classes)``.
    """
    model.eval()
    all_true, all_pred, all_score = [], [], []
    for signals, targets in loader:
        signals = signals.to(device)
        logits = model(signals)
        probs = torch.softmax(logits, dim=1)
        all_score.append(probs.cpu().numpy())
        all_pred.append(probs.argmax(dim=1).cpu().numpy())
        all_true.append(targets.numpy())
    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_score),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return headline scalar metrics as a dict."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert an integer label vector to a one-hot matrix (for ROC curves)."""
    onehot = np.zeros((len(y), num_classes), dtype=np.int64)
    onehot[np.arange(len(y)), y] = 1
    return onehot


def load_checkpoint_model(checkpoint_path: str | Path, config: Config, device: torch.device):
    """Load a trained model from a checkpoint saved by :mod:`src.train`.

    The checkpoint is created and consumed entirely within this project, so it
    is a trusted source. ``weights_only=False`` is required because we also
    persist the config dict alongside the tensors.
    """
    input_length = 2 * config.data["beat_window"]
    model = build_model(config.model, input_length=input_length).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate(config: Config, checkpoint_path: str | Path) -> dict:
    """Evaluate a checkpoint on the held-out test split and save all artefacts.

    Returns the dict of headline metrics.
    """
    device = get_device()
    logger.info("Evaluating on device: %s", device)

    _train, _val, test_loader, _weights = create_dataloaders(config)
    model = load_checkpoint_model(checkpoint_path, config, device)

    y_true, y_pred, y_score = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)

    # Only report classes that actually appear in the test split to avoid
    # sklearn warnings and misleading zero rows.
    present = sorted(set(y_true) | set(y_pred))
    target_names = [AAMI_CLASSES[i] for i in present]

    report_txt = classification_report(
        y_true, y_pred, labels=present, target_names=target_names, zero_division=0, digits=4
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=present,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    logger.info("\n%s", report_txt)
    logger.info("Headline metrics: %s", json.dumps(metrics, indent=2))

    # --- Save artefacts ----------------------------------------------------
    reports_dir = ensure_dir(config.output["reports_dir"])
    cm_dir = ensure_dir(config.output["confusion_matrix_dir"])
    fig_dir = ensure_dir(config.output["figures_dir"])

    (reports_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    with (reports_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump({"headline": metrics, "per_class": report_dict}, fh, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(AAMI_CLASSES))))
    plot_confusion_matrix(
        cm,
        class_names=AAMI_CLASSES,
        normalize=True,
        save_path=cm_dir / "confusion_matrix.png",
    )
    plot_confusion_matrix(
        cm,
        class_names=AAMI_CLASSES,
        normalize=False,
        save_path=cm_dir / "confusion_matrix_counts.png",
    )
    plot_roc_curves(
        _one_hot(y_true, len(AAMI_CLASSES)),
        y_score,
        class_names=AAMI_CLASSES,
        save_path=fig_dir / "roc_curves.png",
    )
    logger.info("Saved evaluation artefacts under %s", Path(config.output["reports_dir"]).parent)
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI args for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained ECG model.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for evaluation."""
    args = parse_args()
    config = load_config(args.config)
    evaluate(config, args.checkpoint)


if __name__ == "__main__":
    main()
