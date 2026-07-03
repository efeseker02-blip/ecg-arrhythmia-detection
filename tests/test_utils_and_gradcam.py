"""Tests for utility helpers and Grad-CAM explainability."""

from __future__ import annotations

import numpy as np
import torch

from src.gradcam import GradCAM1D
from src.model import ECGCNN
from src.utils import (
    AAMI_CLASSES,
    MITBIH_TO_AAMI,
    get_device,
    set_seed,
)


class TestUtils:
    def test_aami_classes_order_stable(self) -> None:
        assert AAMI_CLASSES == ["N", "S", "V", "F", "Q"]

    def test_symbol_mapping_covers_common_symbols(self) -> None:
        # A few well-known mappings from the MIT-BIH annotation manual.
        assert MITBIH_TO_AAMI["N"] == "N"
        assert MITBIH_TO_AAMI["V"] == "V"  # PVC
        assert MITBIH_TO_AAMI["A"] == "S"  # atrial premature
        assert MITBIH_TO_AAMI["/"] == "Q"  # paced

    def test_set_seed_reproducible(self) -> None:
        set_seed(123)
        a = torch.randn(5)
        set_seed(123)
        b = torch.randn(5)
        assert torch.allclose(a, b)

    def test_get_device_returns_device(self) -> None:
        assert isinstance(get_device(), torch.device)


class TestGradCAM:
    def test_saliency_shape_and_range(self) -> None:
        model = ECGCNN(conv_channels=[8, 16])
        cam = GradCAM1D(model)
        beat = torch.randn(1, 1, 360)
        saliency = cam(beat)
        assert saliency.shape == (360,)
        assert saliency.min() >= 0.0 and saliency.max() <= 1.0 + 1e-6
        assert np.all(np.isfinite(saliency))

    def test_explains_specific_class(self) -> None:
        model = ECGCNN(num_classes=5, conv_channels=[8, 16])
        cam = GradCAM1D(model)
        saliency = cam(torch.randn(1, 1, 360), class_idx=2)
        assert saliency.shape == (360,)
