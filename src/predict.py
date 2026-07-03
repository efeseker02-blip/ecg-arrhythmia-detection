"""Single-record / single-beat inference.

Examples
--------
Predict the class of every beat in a MIT-BIH record and print a summary::

    python -m src.predict --record 100 --checkpoint models/best_model.pt

Predict a single beat from a saved ``.npy`` waveform and plot the result::

    python -m src.predict --beat my_beat.npy --plot

The prediction returns, per beat, the **predicted class**, a **confidence
score** (max softmax probability) and the full **probability distribution**.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .evaluate import load_checkpoint_model
from .preprocessing import bandpass_filter, normalize_beat, process_record
from .utils import AAMI_CLASSES, Config, get_device, get_logger, load_config
from .visualization import CLASS_FULL_NAMES

logger = get_logger(__name__)


@dataclass
class Prediction:
    """The result of classifying a single beat."""

    predicted_class: str
    confidence: float
    probabilities: dict[str, float]

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        dist = ", ".join(f"{c}:{p:.3f}" for c, p in self.probabilities.items())
        return (
            f"{self.predicted_class} ({CLASS_FULL_NAMES[self.predicted_class]}) "
            f"| confidence {self.confidence:.3f} | [{dist}]"
        )


def _predict_batch(model: torch.nn.Module, beats: np.ndarray, device: torch.device) -> np.ndarray:
    """Return the softmax probability matrix for a batch of ``(N, L)`` beats."""
    tensor = torch.from_numpy(beats.astype(np.float32)).unsqueeze(1).to(device)  # (N, 1, L)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    return probs.cpu().numpy()


def predict_beat(
    beat: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    *,
    already_normalized: bool = False,
    normalization: str = "zscore",
) -> Prediction:
    """Classify a single 1-D beat waveform.

    Parameters
    ----------
    beat:
        1-D array of the beat samples.
    model:
        A trained :class:`~src.model.ECGCNN`.
    device:
        Torch device.
    already_normalized:
        Set ``True`` if ``beat`` was already normalised (skip re-normalising).
    normalization:
        Normalisation method to apply when ``already_normalized`` is ``False``.
    """
    if not already_normalized:
        beat = normalize_beat(beat, method=normalization)
    probs = _predict_batch(model, beat[None, :], device)[0]
    idx = int(np.argmax(probs))
    return Prediction(
        predicted_class=AAMI_CLASSES[idx],
        confidence=float(probs[idx]),
        probabilities={AAMI_CLASSES[i]: float(probs[i]) for i in range(len(AAMI_CLASSES))},
    )


def predict_record(
    record_name: str,
    config: Config,
    model: torch.nn.Module,
    device: torch.device,
) -> list[Prediction]:
    """Classify every beat in a MIT-BIH record via the full preprocessing path."""
    beats = process_record(
        record_name,
        config.data["raw_dir"],
        channel=config.data["channel"],
        beat_window=config.data["beat_window"],
        bandpass_low=config.preprocessing["bandpass_low"],
        bandpass_high=config.preprocessing["bandpass_high"],
        filter_order=config.preprocessing["filter_order"],
        fs=config.data["sampling_rate"],
        normalization=config.preprocessing["normalization"],
    )
    signals = np.stack([b.signal for b in beats])
    probs = _predict_batch(model, signals, device)
    predictions = []
    for row in probs:
        idx = int(np.argmax(row))
        predictions.append(
            Prediction(
                predicted_class=AAMI_CLASSES[idx],
                confidence=float(row[idx]),
                probabilities={AAMI_CLASSES[i]: float(row[i]) for i in range(len(AAMI_CLASSES))},
            )
        )
    return predictions


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for prediction."""
    parser = argparse.ArgumentParser(description="Predict ECG heartbeat class(es).")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt")
    parser.add_argument("--record", type=str, default=None, help="MIT-BIH record id, e.g. 100")
    parser.add_argument("--beat", type=str, default=None, help="Path to a 1-D .npy beat waveform")
    parser.add_argument("--plot", action="store_true", help="Plot the single-beat prediction")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for single-beat or whole-record prediction."""
    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    model = load_checkpoint_model(args.checkpoint, config, device)

    if args.beat:
        beat = np.load(args.beat)
        # If the raw waveform is longer than one beat window, band-pass first.
        if beat.size > 2 * config.data["beat_window"]:
            beat = bandpass_filter(beat, fs=config.data["sampling_rate"])
        prediction = predict_beat(beat, model, device)
        print(prediction)
        if args.plot:
            from .visualization import plot_prediction

            probs = np.array([prediction.probabilities[c] for c in AAMI_CLASSES])
            out = Path(config.output["figures_dir"]) / "single_beat_prediction.png"
            plot_prediction(normalize_beat(beat), probs, save_path=out)
            print(f"Saved plot to {out}")
        return

    if args.record:
        predictions = predict_record(args.record, config, model, device)
        counts = Counter(p.predicted_class for p in predictions)
        mean_conf = float(np.mean([p.confidence for p in predictions]))
        print(f"\nRecord {args.record}: {len(predictions)} beats classified")
        print(f"Mean confidence: {mean_conf:.3f}")
        print("Predicted class distribution:")
        for cls in AAMI_CLASSES:
            if counts[cls]:
                print(f"  {cls} ({CLASS_FULL_NAMES[cls]}): {counts[cls]}")
        return

    raise SystemExit("Provide either --record <id> or --beat <path.npy>.")


if __name__ == "__main__":
    main()
