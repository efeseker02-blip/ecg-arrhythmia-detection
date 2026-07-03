"""Shared utilities: configuration loading, seeding, logging and devices.

Keeping these cross-cutting concerns in one place means every entry point
(``train``, ``evaluate``, ``predict``) behaves identically with respect to
reproducibility and logging.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# The five AAMI EC57 super-classes, in a fixed order so that integer labels are
# stable across the whole project (index 0 == "N", 1 == "S", ...).
AAMI_CLASSES: list[str] = ["N", "S", "V", "F", "Q"]

# Mapping from raw MIT-BIH annotation symbols to AAMI super-classes.  Symbols
# not present here (e.g. rhythm/noise annotations) are ignored during
# segmentation because they do not mark an individual heartbeat.
MITBIH_TO_AAMI: dict[str, str] = {
    # --- Normal (N) ---
    "N": "N",  # Normal beat
    "L": "N",  # Left bundle branch block beat
    "R": "N",  # Right bundle branch block beat
    "e": "N",  # Atrial escape beat
    "j": "N",  # Nodal (junctional) escape beat
    # --- Supraventricular ectopic (S) ---
    "A": "S",  # Atrial premature beat
    "a": "S",  # Aberrated atrial premature beat
    "J": "S",  # Nodal (junctional) premature beat
    "S": "S",  # Supraventricular premature beat
    # --- Ventricular ectopic (V) ---
    "V": "V",  # Premature ventricular contraction
    "E": "V",  # Ventricular escape beat
    # --- Fusion (F) ---
    "F": "F",  # Fusion of ventricular and normal beat
    # --- Unknown / paced (Q) ---
    "/": "Q",  # Paced beat
    "f": "Q",  # Fusion of paced and normal beat
    "Q": "Q",  # Unclassifiable beat
}


@dataclass
class Config:
    """A thin, attribute-access wrapper around the parsed YAML config.

    Using a dataclass (rather than passing raw dicts around) gives us
    autocomplete, a single source of truth, and a place to store derived paths.
    """

    raw: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # ------------------------------------------------------------------ #
    # Convenience accessors for the most-used nested values.             #
    # ------------------------------------------------------------------ #
    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def preprocessing(self) -> dict[str, Any]:
        return self.raw["preprocessing"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def output(self) -> dict[str, Any]:
        return self.raw["output"]


def load_config(path: str | Path = "config.yaml") -> Config:
    """Parse the YAML configuration file into a :class:`Config`.

    Parameters
    ----------
    path:
        Path to the YAML file. Defaults to ``config.yaml`` in the CWD.

    Returns
    -------
    Config
        The parsed configuration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw)


def set_seed(seed: int = 42) -> None:
    """Seed every source of randomness for reproducible experiments.

    Seeds Python's ``random``, NumPy, and PyTorch (CPU + CUDA), and forces
    cuDNN into deterministic mode. Note that full determinism can slightly
    reduce GPU throughput — acceptable for a reproducible research project.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the best available compute device.

    Prefers CUDA, then Apple-Silicon ``mps``, then falls back to CPU. This lets
    the same code run unchanged on a laptop, a Mac, or a CUDA workstation.
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name: str = "ecg", level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a consistent, readable format.

    Idempotent: repeated calls with the same name won't stack duplicate
    handlers (a common cause of doubled log lines).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if missing and return it as a ``Path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
