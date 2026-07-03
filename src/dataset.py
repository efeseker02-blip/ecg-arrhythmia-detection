"""PyTorch dataset construction, caching and train/val/test splitting.

The expensive work (downloading, filtering, segmenting all 44 records) is done
once in :func:`build_dataset` and cached to a single ``.npz`` file. Training
then loads that cache instantly. Splitting is **record-wise** by default so
that beats from the same patient never leak across the train/test boundary —
the honest way to estimate how the model generalises to *unseen patients*.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .preprocessing import augment_beat, process_record
from .utils import AAMI_CLASSES, Config, get_logger

logger = get_logger(__name__)

# All 48 MIT-BIH records. The four paced records (102, 104, 107, 217) are
# excluded via config; the rest are the standard 44-record evaluation set.
ALL_RECORDS: list[str] = [
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "106",
    "107",
    "108",
    "109",
    "111",
    "112",
    "113",
    "114",
    "115",
    "116",
    "117",
    "118",
    "119",
    "121",
    "122",
    "123",
    "124",
    "200",
    "201",
    "202",
    "203",
    "205",
    "207",
    "208",
    "209",
    "210",
    "212",
    "213",
    "214",
    "215",
    "217",
    "219",
    "220",
    "221",
    "222",
    "223",
    "228",
    "230",
    "231",
    "232",
    "233",
    "234",
]

# Integer <-> class-name lookup, derived once from the canonical order.
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(AAMI_CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: c for c, i in CLASS_TO_IDX.items()}


class ECGBeatDataset(Dataset):
    """In-memory dataset of segmented ECG beats.

    Parameters
    ----------
    signals:
        Float array of shape ``(N, L)`` — one normalised beat per row.
    labels:
        Integer array of shape ``(N,)`` with AAMI class indices.
    augment:
        If ``True``, apply on-the-fly augmentation (train split only).
    seed:
        Seed for the augmentation RNG.
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        *,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        assert len(signals) == len(labels), "signals/labels length mismatch"
        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        signal = self.signals[idx]
        if self.augment:
            signal = augment_beat(signal, self._rng)
        # Add the channel dimension expected by ``nn.Conv1d``: (1, L).
        tensor = torch.from_numpy(np.ascontiguousarray(signal)).unsqueeze(0)
        return tensor, int(self.labels[idx])


def build_dataset(config: Config, force: bool = False) -> Path:
    """Process every included record and cache the result to ``.npz``.

    Parameters
    ----------
    config:
        The project configuration.
    force:
        If ``True``, rebuild even when a cache already exists.

    Returns
    -------
    Path
        Path to the cached ``.npz`` file containing ``signals``, ``labels``
        and ``records`` arrays.
    """
    processed_dir = Path(config.data["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_path = processed_dir / "beats.npz"

    if cache_path.exists() and not force:
        logger.info("Using cached dataset at %s", cache_path)
        return cache_path

    excluded = {str(r) for r in config.data.get("excluded_records", [])}
    records = [r for r in ALL_RECORDS if r not in excluded]
    logger.info("Building dataset from %d records ...", len(records))

    all_signals: list[np.ndarray] = []
    all_labels: list[int] = []
    all_records: list[str] = []

    for record in records:
        beats = process_record(
            record,
            config.data["raw_dir"],
            channel=config.data["channel"],
            beat_window=config.data["beat_window"],
            bandpass_low=config.preprocessing["bandpass_low"],
            bandpass_high=config.preprocessing["bandpass_high"],
            filter_order=config.preprocessing["filter_order"],
            fs=config.data["sampling_rate"],
            normalization=config.preprocessing["normalization"],
        )
        for beat in beats:
            all_signals.append(beat.signal)
            all_labels.append(CLASS_TO_IDX[beat.label])
            all_records.append(beat.record)

    signals = np.stack(all_signals).astype(np.float32)
    labels = np.asarray(all_labels, dtype=np.int64)
    records_arr = np.asarray(all_records)

    np.savez_compressed(cache_path, signals=signals, labels=labels, records=records_arr)
    logger.info("Cached %d beats to %s", len(signals), cache_path)
    return cache_path


def _split_indices(
    records_arr: np.ndarray,
    labels: np.ndarray,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split beats into train/val/test by **record** (patient-wise).

    Grouping by record prevents patient leakage: no heartbeat from a test
    patient is ever seen during training. Records are assigned to splits with a
    stratification-free but reproducible shuffle; because different patients
    have very different class mixes, we fall back gracefully when a split would
    be empty.
    """
    unique_records = np.unique(records_arr)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_records)

    n = len(unique_records)
    n_test = max(1, int(round(test_size * n)))
    n_val = max(1, int(round(val_size * n)))
    test_records = set(unique_records[:n_test])
    val_records = set(unique_records[n_test : n_test + n_val])

    train_idx, val_idx, test_idx = [], [], []
    for i, rec in enumerate(records_arr):
        if rec in test_records:
            test_idx.append(i)
        elif rec in val_records:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def create_dataloaders(
    config: Config,
    *,
    split_by: str = "record",
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """Build train/val/test :class:`DataLoader`s from the cached dataset.

    Parameters
    ----------
    config:
        Project configuration.
    split_by:
        ``"record"`` for patient-wise splitting (recommended, no leakage) or
        ``"beat"`` for a stratified per-beat split (higher but optimistic
        scores).

    Returns
    -------
    train_loader, val_loader, test_loader, class_weights
        The three loaders plus an array of inverse-frequency class weights
        (float32) for the loss function.
    """
    cache_path = build_dataset(config)
    # The cache is produced by build_dataset() in this same project (trusted),
    # and every stored array has an explicit numeric or fixed-width unicode
    # dtype — so we can (and do) load with allow_pickle=False for safety.
    data = np.load(cache_path, allow_pickle=False)
    signals, labels, records_arr = data["signals"], data["labels"], data["records"]

    if split_by == "record":
        train_idx, val_idx, test_idx = _split_indices(
            records_arr,
            labels,
            config.data["val_size"],
            config.data["test_size"],
            config.seed,
        )
    elif split_by == "beat":
        idx = np.arange(len(labels))
        train_idx, temp_idx = train_test_split(
            idx,
            test_size=config.data["val_size"] + config.data["test_size"],
            stratify=labels,
            random_state=config.seed,
        )
        rel_test = config.data["test_size"] / (config.data["val_size"] + config.data["test_size"])
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=rel_test, stratify=labels[temp_idx], random_state=config.seed
        )
    else:
        raise ValueError(f"Unknown split_by: {split_by!r}")

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    train_ds = ECGBeatDataset(signals[train_idx], labels[train_idx], augment=True, seed=config.seed)
    val_ds = ECGBeatDataset(signals[val_idx], labels[val_idx], augment=False)
    test_ds = ECGBeatDataset(signals[test_idx], labels[test_idx], augment=False)

    batch_size = config.training["batch_size"]
    num_workers = config.training["num_workers"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_weights = compute_class_weights(labels[train_idx], num_classes=len(AAMI_CLASSES))
    return train_loader, val_loader, test_loader, class_weights


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Return inverse-frequency class weights, normalised to mean 1.

    The MIT-BIH dataset is severely imbalanced (~90% normal beats). Weighting
    the cross-entropy loss by the inverse class frequency stops the network
    from collapsing to the trivial "everything is normal" solution.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid divide-by-zero for absent classes
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()  # normalise so the average weight is 1
    return weights.astype(np.float32)
