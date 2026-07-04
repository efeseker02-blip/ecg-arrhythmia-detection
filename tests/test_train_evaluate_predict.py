"""Integration-style tests for training utilities, evaluation and inference.

These exercise the pieces that tie the pipeline together — early stopping, the
metric helpers, the checkpoint save/load round-trip, the prediction contract,
batched inference and dataset caching/splitting — all using synthetic data so
no real MIT-BIH database is required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    ECGBeatDataset,
    build_dataset,
    compute_class_weights,
    create_dataloaders,
)
from src.evaluate import collect_predictions, compute_metrics, load_checkpoint_model
from src.model import build_model
from src.predict import Prediction, predict_beat
from src.train import EarlyStopping, EpochMetrics, save_checkpoint
from src.utils import AAMI_CLASSES


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def synthetic_cache_config(tiny_config, tmp_path):
    """Write a small synthetic beats.npz cache and point the config at it."""
    rng = np.random.default_rng(0)
    length = 2 * tiny_config.data["beat_window"]
    signals, labels, records = [], [], []
    n_records = 8
    for r in range(n_records):
        for _ in range(60):
            cls = int(rng.integers(0, 5))
            beat = rng.standard_normal(length).astype(np.float32)
            signals.append(beat)
            labels.append(cls)
            records.append(str(r))
    processed = tmp_path / "processed"
    processed.mkdir()
    np.savez_compressed(
        processed / "beats.npz",
        signals=np.stack(signals).astype(np.float32),
        labels=np.asarray(labels, np.int64),
        records=np.asarray(records),
    )
    tiny_config.data["processed_dir"] = str(processed)
    return tiny_config


# --------------------------------------------------------------------------- #
# EarlyStopping (finding: stopping/patience behaviour untested)               #
# --------------------------------------------------------------------------- #
class TestEarlyStopping:
    def test_improvement_resets_counter(self):
        stopper = EarlyStopping(patience=2, mode="max")
        assert stopper.step(0.5) is True  # first value always "improves"
        assert stopper.step(0.6) is True
        assert stopper.num_bad_epochs == 0
        assert stopper.should_stop is False

    def test_stops_after_patience(self):
        stopper = EarlyStopping(patience=2, mode="max")
        stopper.step(0.8)  # best
        assert stopper.step(0.8) is False  # bad 1
        assert stopper.should_stop is False
        assert stopper.step(0.8) is False  # bad 2 -> stop
        assert stopper.should_stop is True

    def test_min_delta_requires_meaningful_gain(self):
        stopper = EarlyStopping(patience=1, mode="max", min_delta=0.01)
        stopper.step(0.80)
        # A 0.005 gain is below min_delta, so it does NOT count as improvement.
        assert stopper.step(0.805) is False

    def test_min_mode(self):
        stopper = EarlyStopping(patience=5, mode="min")
        assert stopper.step(1.0) is True
        assert stopper.step(0.5) is True  # lower loss improves
        assert stopper.step(0.9) is False  # higher loss does not


# --------------------------------------------------------------------------- #
# compute_metrics (finding: no test of the returned metric dict)              #
# --------------------------------------------------------------------------- #
class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 2, 3, 4, 0, 1])
        metrics = compute_metrics(y, y.copy())
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1_macro"] == pytest.approx(1.0)
        assert metrics["recall_macro"] == pytest.approx(1.0)

    def test_expected_keys(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        metrics = compute_metrics(y_true, y_pred)
        assert set(metrics) == {
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
        }
        assert all(0.0 <= v <= 1.0 for v in metrics.values())


# --------------------------------------------------------------------------- #
# Checkpoint round-trip (finding: save -> load untested)                      #
# --------------------------------------------------------------------------- #
class TestCheckpointRoundTrip:
    def test_reload_reproduces_outputs(self, tiny_config, tmp_path):
        device = torch.device("cpu")
        model = build_model(tiny_config.model, input_length=360).to(device).eval()
        x = torch.randn(3, 1, 360)
        with torch.no_grad():
            before = model(x)

        optimizer = torch.optim.Adam(model.parameters())
        metrics = EpochMetrics(1, 0.1, 0.1, 0.9, 0.9, 0.9, 1e-3)
        path = tmp_path / "ckpt.pt"
        save_checkpoint(model, optimizer, 1, metrics, path, tiny_config)
        assert path.exists()

        reloaded = load_checkpoint_model(path, tiny_config, device)
        with torch.no_grad():
            after = reloaded(x)
        assert torch.allclose(before, after, atol=1e-6)


# --------------------------------------------------------------------------- #
# Prediction contract (finding: predict_beat output untested)                 #
# --------------------------------------------------------------------------- #
class TestPredictBeat:
    def test_output_contract(self):
        device = torch.device("cpu")
        model = build_model({"num_classes": 5, "conv_channels": [8, 16]}, input_length=360)
        beat = np.random.randn(360).astype(np.float32)
        pred = predict_beat(beat, model, device)
        assert isinstance(pred, Prediction)
        assert pred.predicted_class in AAMI_CLASSES
        # Probabilities form a valid distribution...
        total = sum(pred.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-5)
        # ...and confidence is the max probability, matching the predicted class.
        assert pred.confidence == pytest.approx(max(pred.probabilities.values()))
        assert pred.probabilities[pred.predicted_class] == pytest.approx(pred.confidence)

    def test_eval_mode_makes_predictions_deterministic(self):
        """A model left in train() mode must still give stable predictions,
        because inference forces eval()."""
        device = torch.device("cpu")
        model = build_model({"num_classes": 5, "conv_channels": [8, 16]}, input_length=360)
        model.train()  # deliberately hostile starting state
        beat = np.random.randn(360).astype(np.float32)
        p1 = predict_beat(beat, model, device)
        p2 = predict_beat(beat, model, device)
        assert p1.confidence == pytest.approx(p2.confidence)


# --------------------------------------------------------------------------- #
# collect_predictions (finding: shapes/alignment untested)                    #
# --------------------------------------------------------------------------- #
class TestCollectPredictions:
    def test_shapes_and_alignment(self):
        device = torch.device("cpu")
        model = build_model({"num_classes": 5, "conv_channels": [8, 16]}, input_length=360)
        signals = np.random.randn(20, 360).astype(np.float32)
        labels = np.random.randint(0, 5, size=20)
        loader = DataLoader(ECGBeatDataset(signals, labels), batch_size=8)
        y_true, y_pred, y_score = collect_predictions(model, loader, device)
        assert y_true.shape == (20,)
        assert y_pred.shape == (20,)
        assert y_score.shape == (20, 5)
        # y_true must be returned in loader order (unshuffled).
        assert np.array_equal(y_true, labels)
        # Each score row is a probability distribution; argmax matches y_pred.
        assert np.allclose(y_score.sum(axis=1), 1.0, atol=1e-5)
        assert np.array_equal(y_score.argmax(axis=1), y_pred)


# --------------------------------------------------------------------------- #
# Dataset caching & split integration (finding: untested)                     #
# --------------------------------------------------------------------------- #
class TestDatasetIntegration:
    def test_build_dataset_uses_existing_cache(self, synthetic_cache_config):
        # An existing cache is returned as-is without needing raw records.
        path = build_dataset(synthetic_cache_config)
        assert path.exists()
        assert path.name == "beats.npz"

    def test_create_dataloaders_record_split(self, synthetic_cache_config):
        train, val, test, weights = create_dataloaders(synthetic_cache_config, split_by="record")
        for loader in (train, val, test):
            assert len(loader.dataset) > 0
        total = sum(len(loader.dataset) for loader in (train, val, test))
        assert total == 8 * 60  # every beat lands in exactly one split
        assert weights.shape == (5,)

    def test_beat_split_survives_singleton_class(self, tiny_config, tmp_path):
        """The stratified beat split must not crash when a class has <2 beats."""
        length = 2 * tiny_config.data["beat_window"]
        rng = np.random.default_rng(1)
        signals = rng.standard_normal((100, length)).astype(np.float32)
        labels = np.array([0] * 60 + [1] * 39 + [2] * 1)  # class 2 is a singleton
        records = np.array([str(i % 4) for i in range(100)])
        processed = tmp_path / "proc"
        processed.mkdir()
        np.savez_compressed(
            processed / "beats.npz", signals=signals, labels=labels, records=records
        )
        tiny_config.data["processed_dir"] = str(processed)
        # Should fall back to an unstratified split rather than raising.
        train, val, test, _ = create_dataloaders(tiny_config, split_by="beat")
        assert len(train.dataset) > 0 and len(val.dataset) > 0 and len(test.dataset) > 0


# --------------------------------------------------------------------------- #
# Class-weight fix (finding: absent class must not get the largest weight)    #
# --------------------------------------------------------------------------- #
class TestClassWeightFix:
    def test_absent_classes_get_zero_weight(self):
        # Realistic imbalanced train counts with class 4 (Q) absent.
        labels = np.array([0] * 700 + [1] * 25 + [2] * 65 + [3] * 7)  # no class 4
        weights = compute_class_weights(labels, num_classes=5)
        assert weights[4] == 0.0  # absent class contributes nothing
        # The absent class can never carry the largest weight.
        assert weights.argmax() != 4
        # Present classes still have mean 1.
        present = weights[weights > 0]
        assert present.mean() == pytest.approx(1.0, abs=1e-5)
