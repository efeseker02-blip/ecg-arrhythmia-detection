"""Unit tests for the signal-processing pipeline."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src.preprocessing import (
    augment_beat,
    bandpass_filter,
    download_database,
    normalize_beat,
    segment_beats,
)


class TestDownloadDatabase:
    """The downloader must ask WFDB for the whole database correctly."""

    def _fake_wfdb(self, calls: dict) -> types.ModuleType:
        fake = types.ModuleType("wfdb")

        def dl_database(db, dl_dir, records="all", **kwargs):  # noqa: ANN001
            calls["db"] = db
            calls["records"] = records

        fake.dl_database = dl_database
        return fake

    def test_none_records_requests_all(self, monkeypatch, tmp_path) -> None:
        # Regression: passing records=None used to reach wfdb as None and crash
        # with "'NoneType' object is not iterable"; it must become the "all"
        # sentinel that wfdb expects.
        calls: dict = {}
        monkeypatch.setitem(sys.modules, "wfdb", self._fake_wfdb(calls))
        download_database(tmp_path, records=None)
        assert calls["records"] == "all"
        assert calls["db"] == "mitdb"

    def test_explicit_records_passed_through(self, monkeypatch, tmp_path) -> None:
        calls: dict = {}
        monkeypatch.setitem(sys.modules, "wfdb", self._fake_wfdb(calls))
        download_database(tmp_path, records=["100", "101"])
        assert calls["records"] == ["100", "101"]


class TestBandpassFilter:
    def test_output_shape_preserved(self, synthetic_ecg: np.ndarray) -> None:
        filtered = bandpass_filter(synthetic_ecg, fs=360)
        assert filtered.shape == synthetic_ecg.shape

    def test_removes_baseline_wander(self, synthetic_ecg: np.ndarray) -> None:
        """A 0.2 Hz wander lies below the 0.5 Hz cutoff and should be attenuated."""
        filtered = bandpass_filter(synthetic_ecg, low=0.5, high=40.0, fs=360)
        # After removing the slow drift, the mean should be ~0 and the overall
        # variance should drop (the large-amplitude wander is gone).
        assert abs(filtered.mean()) < abs(synthetic_ecg.mean()) + 1e-6
        assert filtered.var() < synthetic_ecg.var()

    def test_is_finite(self, synthetic_ecg: np.ndarray) -> None:
        assert np.all(np.isfinite(bandpass_filter(synthetic_ecg, fs=360)))


class TestSegmentBeats:
    def test_beat_length_and_count(self, synthetic_ecg, r_peaks, symbols) -> None:
        window = 180
        beats = segment_beats(synthetic_ecg, r_peaks, symbols, "test", beat_window=window)
        assert len(beats) > 0
        for beat in beats:
            assert beat.signal.shape == (2 * window,)
            assert beat.label in {"N", "V"}
            assert beat.record == "test"

    def test_drops_edge_beats(self, synthetic_ecg) -> None:
        """A peak too close to the signal edge cannot form a full window."""
        r_peaks = np.array([5, len(synthetic_ecg) - 5])  # both within 180 of an edge
        symbols = np.array(["N", "N"], dtype="<U2")
        beats = segment_beats(synthetic_ecg, r_peaks, symbols, "edge", beat_window=180)
        assert beats == []

    def test_ignores_non_beat_symbols(self, synthetic_ecg, r_peaks) -> None:
        """Rhythm/noise symbols not in the AAMI map are skipped."""
        symbols = np.array(["+"] * len(r_peaks), dtype="<U2")  # '+' == rhythm change
        beats = segment_beats(synthetic_ecg, r_peaks, symbols, "x", beat_window=180)
        assert beats == []


class TestNormalizeBeat:
    def test_zscore_zero_mean_unit_std(self, rng) -> None:
        beat = rng.standard_normal(360).astype(np.float32) * 5 + 3
        norm = normalize_beat(beat, method="zscore")
        assert norm.mean() == pytest.approx(0.0, abs=1e-4)
        assert norm.std() == pytest.approx(1.0, abs=1e-3)

    def test_minmax_range(self, rng) -> None:
        beat = rng.standard_normal(360).astype(np.float32)
        norm = normalize_beat(beat, method="minmax")
        assert norm.min() == pytest.approx(0.0, abs=1e-6)
        assert norm.max() == pytest.approx(1.0, abs=1e-6)

    def test_flat_signal_no_nan(self) -> None:
        """A constant beat must not produce NaNs (divide-by-zero guard)."""
        beat = np.ones(360, dtype=np.float32)
        assert np.all(np.isfinite(normalize_beat(beat, method="zscore")))
        assert np.all(np.isfinite(normalize_beat(beat, method="minmax")))

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_beat(np.zeros(10), method="bogus")


class TestAugmentBeat:
    def test_shape_preserved(self, rng) -> None:
        beat = rng.standard_normal(360).astype(np.float32)
        assert augment_beat(beat, rng).shape == beat.shape

    def test_changes_signal(self, rng) -> None:
        beat = rng.standard_normal(360).astype(np.float32)
        aug = augment_beat(beat, rng, noise_std=0.1, max_shift=5)
        assert not np.allclose(aug, beat)

    def test_deterministic_with_seed(self) -> None:
        beat = np.arange(360, dtype=np.float32)
        a = augment_beat(beat, np.random.default_rng(1))
        b = augment_beat(beat, np.random.default_rng(1))
        assert np.allclose(a, b)
