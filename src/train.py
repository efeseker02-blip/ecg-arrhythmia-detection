"""Training loop with early stopping, LR scheduling, checkpoints & TensorBoard.

Run from the project root::

    python -m src.train --config config.yaml

Key training features (all configurable in ``config.yaml``):

* **Class-weighted cross-entropy** to counter the heavy class imbalance.
* **ReduceLROnPlateau** learning-rate scheduling on validation F1.
* **Early stopping** on validation macro-F1 (not accuracy — accuracy is
  misleading on imbalanced data).
* **Checkpointing** of the best model plus a ``last`` checkpoint.
* **TensorBoard** logging of losses, accuracy and F1.
* **GPU/MPS/CPU** auto-selection and full seeding for reproducibility.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import create_dataloaders
from .model import build_model
from .utils import (
    Config,
    count_parameters,
    ensure_dir,
    get_device,
    get_logger,
    load_config,
    set_seed,
    validate_config,
)

logger = get_logger(__name__)


@dataclass
class EpochMetrics:
    """Container for a single epoch's train/val metrics (also logged as JSON)."""

    epoch: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    val_f1: float
    lr: float


class EarlyStopping:
    """Stop training when the monitored metric stops improving.

    Parameters
    ----------
    patience:
        Number of epochs with no improvement to tolerate before stopping.
    mode:
        ``"max"`` (default, e.g. F1) or ``"min"`` (e.g. loss).
    min_delta:
        Minimum change that counts as an improvement.
    """

    def __init__(self, patience: int = 7, mode: str = "max", min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Update state with the latest metric; return ``True`` if it improved."""
        improved = (
            self.best is None
            or (self.mode == "max" and metric > self.best + self.min_delta)
            or (self.mode == "min" and metric < self.best - self.min_delta)
        )
        if improved:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True
        return improved


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
) -> tuple[float, float, float]:
    """Run one pass over ``loader``; train if ``optimizer`` is given, else eval.

    Returns ``(mean_loss, accuracy, macro_f1)``.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    progress = tqdm(loader, desc=desc, leave=False)
    for signals, targets in progress:
        signals = signals.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(signals)
            loss = criterion(logits, targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * signals.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.detach().cpu().numpy())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    mean_loss = total_loss / len(targets)
    accuracy = float((preds == targets).mean())
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    return mean_loss, accuracy, macro_f1


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: EpochMetrics,
    path: Path,
    config: Config,
) -> None:
    """Persist model + optimizer state and the config used to train it."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": asdict(metrics),
            "config": config.raw,
        },
        path,
    )


def train(config: Config) -> Path:
    """Train the model end-to-end and return the path to the best checkpoint."""
    validate_config(config)
    set_seed(config.seed)
    device = get_device()
    logger.info("Using device: %s", device)

    train_loader, val_loader, _test_loader, class_weights = create_dataloaders(config)

    input_length = 2 * config.data["beat_window"]
    model = build_model(config.model, input_length=input_length).to(device)
    logger.info(
        "Building model '%s' — %s trainable parameters",
        config.model.get("name", "ecg_cnn_1d"),
        f"{count_parameters(model):,}",
    )

    # Class-weighted loss to combat imbalance (see compute_class_weights).
    weight_tensor = (
        torch.from_numpy(class_weights).to(device) if config.training["use_class_weights"] else None
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training["learning_rate"],
        weight_decay=config.training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.training["lr_scheduler_factor"],
        patience=config.training["lr_scheduler_patience"],
    )
    stopper = EarlyStopping(patience=config.training["early_stopping_patience"], mode="max")

    ckpt_dir = ensure_dir(config.training["checkpoint_dir"])
    best_path = ckpt_dir / "best_model.pt"
    last_path = ckpt_dir / "last_model.pt"
    writer = SummaryWriter(log_dir=ensure_dir(config.training["tensorboard_dir"]))

    history: list[dict] = []
    best_f1 = -1.0

    for epoch in range(1, config.training["epochs"] + 1):
        train_loss, train_acc, _ = _run_epoch(
            model, train_loader, criterion, device, optimizer, desc=f"Epoch {epoch} [train]"
        )
        val_loss, val_acc, val_f1 = _run_epoch(
            model, val_loader, criterion, device, None, desc=f"Epoch {epoch} [val]"
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_f1)

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            val_f1=val_f1,
            lr=current_lr,
        )
        history.append(asdict(metrics))

        # TensorBoard scalars.
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("val_macro_f1", val_f1, epoch)
        writer.add_scalar("learning_rate", current_lr, epoch)

        logger.info(
            "Epoch %02d | train_loss %.4f acc %.3f | val_loss %.4f acc %.3f f1 %.3f | lr %.2e",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_f1,
            current_lr,
        )

        improved = stopper.step(val_f1)
        save_checkpoint(model, optimizer, epoch, metrics, last_path, config)
        if improved and val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, optimizer, epoch, metrics, best_path, config)
            logger.info("  ↳ New best model (val_f1=%.4f) saved to %s", val_f1, best_path)

        if stopper.should_stop:
            logger.info("Early stopping at epoch %d (best val_f1=%.4f)", epoch, best_f1)
            break

    writer.close()

    # Persist the training history (JSON) and render the loss/accuracy/F1 curves
    # to a figure so the artefact referenced in the README auto-populates.
    history_path = ensure_dir(config.output["reports_dir"])
    with (history_path / "training_history.json").open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    if history:
        from .visualization import plot_training_history

        fig_path = Path(ensure_dir(config.output["figures_dir"])) / "training_history.png"
        plot_training_history(history, save_path=fig_path)
        logger.info("Saved training-history figure to %s", fig_path)
    logger.info("Training complete. Best val macro-F1: %.4f", best_f1)
    return best_path


def parse_args() -> argparse.Namespace:
    """Parse command-line overrides for the most common hyperparameters."""
    parser = argparse.ArgumentParser(description="Train the ECG arrhythmia CNN.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker processes (0 = load in the main process).",
    )
    parser.add_argument(
        "--split-by",
        type=str,
        default=None,
        choices=["record", "beat"],
        help="Split strategy: 'record' (patient-wise) or 'beat' (per-beat).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point: load config, apply overrides, train."""
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config.training["epochs"] = args.epochs
    if args.batch_size is not None:
        config.training["batch_size"] = args.batch_size
    if args.lr is not None:
        config.training["learning_rate"] = args.lr
    if args.seed is not None:
        config.raw["seed"] = args.seed
    if args.num_workers is not None:
        config.training["num_workers"] = args.num_workers
    if args.split_by is not None:
        config.data["split_by"] = args.split_by
    train(config)


if __name__ == "__main__":
    main()
