"""ECG signal loading, filtering, beat segmentation and normalisation.

Biomedical background
---------------------
An electrocardiogram (ECG) records the heart's electrical activity. Each
heartbeat produces a characteristic **PQRST** waveform; the tall, sharp
**R-peak** is the easiest fiducial point to detect and is what the MIT-BIH
cardiologists annotated. Arrhythmias are abnormalities in the *rhythm* or
*morphology* of these beats — e.g. a premature ventricular contraction (PVC)
has a widened, bizarre QRS complex.

This module turns a continuous ECG recording into a set of fixed-length,
per-beat windows suitable for a neural network:

1.  **Load** the raw signal + expert annotations (``load_record``).
2.  **Denoise** with a band-pass filter (``bandpass_filter``) to remove
    baseline wander and high-frequency muscle noise.
3.  **Segment** the signal into individual beats centred on each annotated
    R-peak (``segment_beats``).
4.  **Normalise** each beat so the network sees amplitude-invariant morphology
    (``normalize_beat``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from .utils import MITBIH_TO_AAMI, get_logger

logger = get_logger(__name__)


@dataclass
class Beat:
    """A single segmented heartbeat.

    Attributes
    ----------
    signal:
        The 1-D filtered, normalised waveform of length ``2 * beat_window``.
    label:
        The AAMI super-class string ("N", "S", "V", "F" or "Q").
    record:
        The MIT-BIH record id the beat came from (useful for record-wise splits
        and for tracing predictions back to a patient).
    r_peak:
        The sample index of the R-peak within the original recording.
    """

    signal: np.ndarray
    label: str
    record: str
    r_peak: int


# --------------------------------------------------------------------------- #
# 1. Loading                                                                   #
# --------------------------------------------------------------------------- #
def load_record(
    record_name: str,
    raw_dir: str | Path,
    channel: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one MIT-BIH record's signal and beat annotations via WFDB.

    Parameters
    ----------
    record_name:
        Numeric record id, e.g. ``"100"``.
    raw_dir:
        Directory containing the ``.dat``/``.hea``/``.atr`` files.
    channel:
        Which lead to extract (MIT-BIH records are 2-lead).

    Returns
    -------
    signal:
        1-D float array of raw millivolt samples.
    r_peaks:
        Integer sample indices of every annotated beat.
    symbols:
        Array of raw annotation symbol strings, aligned with ``r_peaks``.

    Notes
    -----
    ``wfdb`` is imported lazily so that unit tests covering the pure-NumPy
    functions (filtering, segmentation, normalisation) do not require the
    dependency to be installed.
    """
    import wfdb  # local import: keeps the heavy dep optional for tests

    path = str(Path(raw_dir) / str(record_name))
    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, "atr")

    signal = record.p_signal[:, channel].astype(np.float64)
    r_peaks = np.asarray(annotation.sample, dtype=np.int64)
    symbols = np.asarray(annotation.symbol, dtype="<U2")
    return signal, r_peaks, symbols


def download_database(
    raw_dir: str | Path,
    records: list[str] | None = None,
) -> None:
    """Download the MIT-BIH Arrhythmia Database into ``raw_dir`` via WFDB.

    If ``records`` is ``None`` the full 48-record database is fetched. This is
    a thin wrapper around ``wfdb.dl_database`` that creates the target
    directory and logs progress.
    """
    import wfdb

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MIT-BIH Arrhythmia Database to %s ...", raw_dir)
    # 'mitdb' is the PhysioNet slug for the MIT-BIH Arrhythmia Database.
    # wfdb expects the sentinel string "all" (not None) to fetch every record.
    wfdb.dl_database("mitdb", str(raw_dir), records=records if records else "all")
    logger.info("Download complete.")


# --------------------------------------------------------------------------- #
# 2. Denoising                                                                 #
# --------------------------------------------------------------------------- #
def bandpass_filter(
    signal: np.ndarray,
    low: float = 0.5,
    high: float = 40.0,
    fs: int = 360,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter to an ECG signal.

    The passband (default 0.5–40 Hz) is chosen to:

    * reject **baseline wander** (< 0.5 Hz, caused by respiration/electrode
      motion), and
    * reject **high-frequency noise** (> 40 Hz, mostly EMG/muscle and mains),

    while preserving the diagnostically important QRS morphology (~10–25 Hz).

    ``filtfilt`` applies the filter forwards and backwards so there is **no
    phase distortion** — important because a shifted R-peak would corrupt the
    subsequent beat segmentation.

    Parameters
    ----------
    signal:
        1-D raw ECG samples.
    low, high:
        Passband edges in Hz.
    fs:
        Sampling frequency in Hz.
    order:
        Butterworth filter order.

    Returns
    -------
    np.ndarray
        The filtered signal, same shape as the input.
    """
    nyquist = 0.5 * fs
    low_norm = low / nyquist
    high_norm = high / nyquist
    b, a = butter(order, [low_norm, high_norm], btype="band")
    # padlen guards against very short signals in the unit tests.
    padlen = min(3 * max(len(a), len(b)), len(signal) - 1)
    return filtfilt(b, a, signal, padlen=max(padlen, 0))


# --------------------------------------------------------------------------- #
# 3. Segmentation                                                              #
# --------------------------------------------------------------------------- #
def segment_beats(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    symbols: np.ndarray,
    record_name: str,
    beat_window: int = 180,
) -> list[Beat]:
    """Cut a continuous recording into fixed-length beats around each R-peak.

    Each beat spans ``[r_peak - beat_window, r_peak + beat_window)`` samples,
    i.e. ``2 * beat_window`` samples total (1 second at 360 Hz for the default
    window). Beats whose window would fall outside the recording are dropped,
    as are annotation symbols that don't map to an AAMI class (rhythm/noise
    markers).

    Parameters
    ----------
    signal:
        The (ideally already filtered) 1-D ECG signal.
    r_peaks:
        Sample indices of annotated R-peaks.
    symbols:
        Raw MIT-BIH annotation symbols aligned with ``r_peaks``.
    record_name:
        Record id, stored on each :class:`Beat` for traceability.
    beat_window:
        Half-window length in samples.

    Returns
    -------
    list[Beat]
        The extracted beats.
    """
    beats: list[Beat] = []
    n = len(signal)
    for r_peak, symbol in zip(r_peaks, symbols, strict=True):
        aami = MITBIH_TO_AAMI.get(str(symbol))
        if aami is None:
            continue  # not a classifiable heartbeat symbol
        start = int(r_peak) - beat_window
        end = int(r_peak) + beat_window
        if start < 0 or end > n:
            continue  # window falls off the edge of the recording
        window = signal[start:end].astype(np.float32)
        beats.append(Beat(signal=window, label=aami, record=str(record_name), r_peak=int(r_peak)))
    return beats


# --------------------------------------------------------------------------- #
# 4. Normalisation                                                             #
# --------------------------------------------------------------------------- #
def normalize_beat(beat: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalise a single beat to make the network amplitude-invariant.

    Different patients and electrode placements produce different absolute
    voltages; what matters diagnostically is the *shape* of the waveform.
    Per-beat normalisation removes that nuisance amplitude variation.

    Parameters
    ----------
    beat:
        1-D beat waveform.
    method:
        ``"zscore"`` (subtract mean, divide by std) or ``"minmax"`` (scale to
        ``[0, 1]``).

    Returns
    -------
    np.ndarray
        The normalised beat (float32).
    """
    beat = beat.astype(np.float32)
    if method == "zscore":
        mean = beat.mean()
        std = beat.std()
        # Guard against a flat (all-equal) segment producing a divide-by-zero.
        return (beat - mean) / (std + 1e-8)
    if method == "minmax":
        lo, hi = beat.min(), beat.max()
        return (beat - lo) / (hi - lo + 1e-8)
    raise ValueError(f"Unknown normalization method: {method!r}")


# --------------------------------------------------------------------------- #
# 5. Data augmentation                                                         #
# --------------------------------------------------------------------------- #
def augment_beat(
    beat: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = 0.05,
    max_shift: int = 10,
) -> np.ndarray:
    """Lightly augment a beat with Gaussian noise and a small time shift.

    These physiologically plausible perturbations enlarge the effective
    training set and improve robustness to real-world sensor noise and
    imperfect R-peak localisation. Augmentation should only ever be applied to
    the **training** split.

    Parameters
    ----------
    beat:
        1-D normalised beat.
    rng:
        A seeded NumPy random generator (keeps augmentation reproducible).
    noise_std:
        Standard deviation of the additive Gaussian noise.
    max_shift:
        Maximum absolute circular time-shift in samples.

    Returns
    -------
    np.ndarray
        The augmented beat, same length as the input.
    """
    shift = int(rng.integers(-max_shift, max_shift + 1))
    shifted = np.roll(beat, shift)
    noise = rng.normal(0.0, noise_std, size=beat.shape).astype(np.float32)
    return (shifted + noise).astype(np.float32)


# --------------------------------------------------------------------------- #
# End-to-end record processing                                                 #
# --------------------------------------------------------------------------- #
def process_record(
    record_name: str,
    raw_dir: str | Path,
    *,
    channel: int = 0,
    beat_window: int = 180,
    bandpass_low: float = 0.5,
    bandpass_high: float = 40.0,
    filter_order: int = 4,
    fs: int = 360,
    normalization: str = "zscore",
) -> list[Beat]:
    """Run the full load → filter → segment → normalise pipeline for a record.

    Returns the list of normalised :class:`Beat` objects for the record. This
    is the single function :mod:`src.dataset` calls per record when building
    the processed dataset.
    """
    signal, r_peaks, symbols = load_record(record_name, raw_dir, channel=channel)
    filtered = bandpass_filter(
        signal, low=bandpass_low, high=bandpass_high, fs=fs, order=filter_order
    )
    beats = segment_beats(filtered, r_peaks, symbols, record_name, beat_window=beat_window)
    for beat in beats:
        beat.signal = normalize_beat(beat.signal, method=normalization)
    logger.info("Record %s: extracted %d beats", record_name, len(beats))
    return beats
