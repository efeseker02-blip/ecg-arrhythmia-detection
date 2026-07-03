"""Reusable, publication-quality plotting utilities.

Every function accepts an optional ``save_path`` and returns the Matplotlib
``Figure`` so plots can be either written to disk (from scripts) or displayed
inline (from notebooks/Streamlit). A consistent style is applied on import.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .utils import AAMI_CLASSES, ensure_dir

# A clean, consistent visual style for every figure in the project.
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Human-readable names for the AAMI super-classes, used in legends/titles.
CLASS_FULL_NAMES: dict[str, str] = {
    "N": "Normal",
    "S": "Supraventricular",
    "V": "Ventricular",
    "F": "Fusion",
    "Q": "Unknown/Paced",
}


def _finalize(fig: Figure, save_path: str | Path | None) -> Figure:
    """Tidy layout and optionally save a figure, returning it either way."""
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_training_history(
    history: list[dict],
    save_path: str | Path | None = None,
) -> Figure:
    """Plot train/val loss, accuracy and validation F1 across epochs."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train", marker="o", ms=3)
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val", marker="o", ms=3)
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()

    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train", marker="o", ms=3)
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="val", marker="o", ms=3)
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()

    axes[2].plot(epochs, [h["val_f1"] for h in history], color="green", marker="o", ms=3)
    axes[2].set(title="Validation macro-F1", xlabel="Epoch", ylabel="F1")

    fig.suptitle("Training History", fontweight="bold")
    return _finalize(fig, save_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    normalize: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot a (optionally row-normalised) confusion matrix as a heatmap.

    Row-normalisation shows per-class recall, which is far more informative than
    raw counts on an imbalanced dataset.
    """
    class_names = class_names or AAMI_CLASSES
    matrix = cm.astype(np.float64)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix" + (" (normalised)" if normalize else ""),
    )
    ax.grid(False)

    # Annotate each cell with its value, choosing a legible text colour.
    threshold = matrix.max() / 2.0
    fmt = ".2f" if normalize else "d"
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j] if normalize else int(cm[i, j])
            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=9,
            )
    return _finalize(fig, save_path)


def plot_roc_curves(
    y_true_onehot: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot one-vs-rest ROC curves (with AUC) for every class."""
    from sklearn.metrics import auc, roc_curve

    class_names = class_names or AAMI_CLASSES
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, name in enumerate(class_names):
        if y_true_onehot[:, i].sum() == 0:
            continue  # class absent from the test set — skip
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} ({CLASS_FULL_NAMES[name]}) — AUC {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="One-vs-Rest ROC Curves",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )
    ax.legend(loc="lower right", fontsize=8)
    return _finalize(fig, save_path)


def plot_class_distribution(
    labels: np.ndarray,
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Bar chart of beat counts per class (log scale — huge imbalance)."""
    class_names = class_names or AAMI_CLASSES
    counts = np.bincount(labels, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(class_names, counts, color="steelblue")
    ax.set(title="Class Distribution", xlabel="AAMI class", ylabel="Beat count (log scale)")
    ax.set_yscale("log")
    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    return _finalize(fig, save_path)


def plot_sample_beats(
    signals: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
    n_per_class: int = 3,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot a grid of example beats, one row per class.

    A visual sanity check that the segmentation is centred on the R-peak and
    that the morphological differences between classes are learnable.
    """
    class_names = class_names or AAMI_CLASSES
    fig, axes = plt.subplots(
        len(class_names), n_per_class, figsize=(3 * n_per_class, 2.2 * len(class_names))
    )
    axes = np.atleast_2d(axes)
    for row, cls_idx in enumerate(range(len(class_names))):
        idxs = np.where(labels == cls_idx)[0]
        for col in range(n_per_class):
            ax = axes[row, col]
            if col < len(idxs):
                ax.plot(signals[idxs[col]], color="crimson", lw=1)
            if col == 0:
                ax.set_ylabel(
                    f"{class_names[cls_idx]}\n{CLASS_FULL_NAMES[class_names[cls_idx]]}",
                    fontsize=9,
                )
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Sample beats per class", fontweight="bold")
    return _finalize(fig, save_path)


def plot_prediction(
    signal: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str] | None = None,
    true_label: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot a single beat next to its predicted class-probability bar chart."""
    class_names = class_names or AAMI_CLASSES
    fig, (ax_sig, ax_prob) = plt.subplots(1, 2, figsize=(11, 4))

    ax_sig.plot(signal, color="crimson", lw=1.2)
    title = "Input beat"
    if true_label is not None:
        title += f" (true: {true_label})"
    ax_sig.set(title=title, xlabel="Sample", ylabel="Normalised amplitude")

    pred_idx = int(np.argmax(probabilities))
    colors = ["seagreen" if i == pred_idx else "lightgray" for i in range(len(class_names))]
    ax_prob.bar(class_names, probabilities, color=colors)
    ax_prob.set(title="Predicted probabilities", xlabel="Class", ylabel="Probability", ylim=(0, 1))
    for i, prob in enumerate(probabilities):
        ax_prob.text(i, prob, f"{prob:.2f}", ha="center", va="bottom", fontsize=9)
    return _finalize(fig, save_path)
