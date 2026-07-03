"""Shared pytest fixtures.

Provides synthetic ECG-like signals and a tiny config so the whole test suite
runs in seconds without needing the real MIT-BIH database on disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils import Config


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded RNG for deterministic tests."""
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_ecg(rng: np.random.Generator) -> np.ndarray:
    """A 10-second synthetic ECG: baseline wander + periodic QRS-like spikes.

    Not physiologically exact, but enough to exercise filtering and
    segmentation deterministically.
    """
    fs = 360
    duration = 10
    t = np.arange(fs * duration) / fs
    baseline = 0.4 * np.sin(2 * np.pi * 0.2 * t)  # 0.2 Hz wander (should be removed)
    signal = baseline + 0.02 * rng.standard_normal(t.shape)
    # Add a Gaussian "R-peak" roughly once per second.
    for centre in range(fs // 2, len(t), fs):
        idx = np.arange(len(t))
        signal += 1.5 * np.exp(-0.5 * ((idx - centre) / 4.0) ** 2)
    return signal.astype(np.float64)


@pytest.fixture
def r_peaks() -> np.ndarray:
    """R-peak sample indices matching ``synthetic_ecg``."""
    fs = 360
    return np.arange(fs // 2, fs * 10, fs, dtype=np.int64)


@pytest.fixture
def symbols(r_peaks: np.ndarray) -> np.ndarray:
    """Alternating normal/ventricular annotation symbols for the peaks."""
    syms = np.array(["N", "V"] * len(r_peaks), dtype="<U2")[: len(r_peaks)]
    return syms


@pytest.fixture
def tiny_config() -> Config:
    """A minimal in-memory config for dataset/model tests."""
    return Config(
        raw={
            "seed": 0,
            "data": {
                "raw_dir": "data/mitdb",
                "processed_dir": "data/processed",
                "sampling_rate": 360,
                "beat_window": 180,
                "channel": 0,
                "val_size": 0.2,
                "test_size": 0.2,
                "excluded_records": [102, 104, 107, 217],
            },
            "preprocessing": {
                "bandpass_low": 0.5,
                "bandpass_high": 40.0,
                "filter_order": 4,
                "normalization": "zscore",
            },
            "model": {
                "in_channels": 1,
                "num_classes": 5,
                "conv_channels": [8, 16],
                "kernel_size": 7,
                "dropout": 0.3,
                "fc_hidden": 32,
            },
            "training": {
                "epochs": 1,
                "batch_size": 16,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "lr_scheduler_patience": 3,
                "lr_scheduler_factor": 0.5,
                "early_stopping_patience": 5,
                "use_class_weights": True,
                "num_workers": 0,
                "checkpoint_dir": "models",
                "tensorboard_dir": "outputs/tensorboard",
            },
            "output": {
                "figures_dir": "outputs/figures",
                "confusion_matrix_dir": "outputs/confusion_matrix",
                "reports_dir": "outputs/reports",
            },
        }
    )
