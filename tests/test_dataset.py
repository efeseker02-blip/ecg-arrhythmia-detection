"""Unit tests for dataset construction, splitting and class weighting."""

from __future__ import annotations

import numpy as np
import torch

from src.dataset import ECGBeatDataset, _split_indices, compute_class_weights


class TestECGBeatDataset:
    def test_item_shape_and_channel_dim(self) -> None:
        signals = np.random.randn(10, 360).astype(np.float32)
        labels = np.random.randint(0, 5, size=10)
        ds = ECGBeatDataset(signals, labels)
        tensor, label = ds[0]
        assert tensor.shape == (1, 360)  # (channel, length)
        assert isinstance(label, int)

    def test_len(self) -> None:
        ds = ECGBeatDataset(np.zeros((7, 360), np.float32), np.zeros(7, int))
        assert len(ds) == 7

    def test_augment_only_when_enabled(self) -> None:
        signals = np.random.randn(4, 360).astype(np.float32)
        labels = np.zeros(4, int)
        clean = ECGBeatDataset(signals, labels, augment=False)
        # Without augmentation the tensor equals the raw signal exactly.
        assert torch.allclose(clean[0][0].squeeze(0), torch.from_numpy(signals[0]))


class TestSplitIndices:
    def test_no_record_leakage(self) -> None:
        """A record must appear in exactly one split (patient-wise splitting)."""
        records = np.array([str(r) for r in range(10) for _ in range(20)])
        labels = np.random.randint(0, 5, size=len(records))
        train, val, test = _split_indices(records, labels, 0.2, 0.2, seed=0)

        train_recs = set(records[train])
        val_recs = set(records[val])
        test_recs = set(records[test])
        assert train_recs.isdisjoint(val_recs)
        assert train_recs.isdisjoint(test_recs)
        assert val_recs.isdisjoint(test_recs)

    def test_covers_all_beats(self) -> None:
        records = np.array([str(r) for r in range(10) for _ in range(20)])
        labels = np.random.randint(0, 5, size=len(records))
        train, val, test = _split_indices(records, labels, 0.2, 0.2, seed=0)
        assert len(train) + len(val) + len(test) == len(records)


class TestClassWeights:
    def test_inverse_frequency(self) -> None:
        """Rarer classes get larger weights."""
        labels = np.array([0] * 90 + [1] * 10)  # class 1 is 9x rarer
        weights = compute_class_weights(labels, num_classes=2)
        assert weights[1] > weights[0]

    def test_mean_is_one(self) -> None:
        labels = np.array([0] * 50 + [1] * 30 + [2] * 20)
        weights = compute_class_weights(labels, num_classes=3)
        assert weights.mean() == np.float32(1.0) or abs(weights.mean() - 1.0) < 1e-5

    def test_absent_class_no_divide_by_zero(self) -> None:
        labels = np.array([0, 0, 1, 1])  # classes 2,3,4 absent
        weights = compute_class_weights(labels, num_classes=5)
        assert np.all(np.isfinite(weights))
