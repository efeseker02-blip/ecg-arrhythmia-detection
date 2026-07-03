"""Unit tests for the CNN model and its building blocks."""

from __future__ import annotations

import torch

from src.model import ECGCNN, ConvBlock, build_model
from src.utils import count_parameters


class TestConvBlock:
    def test_halves_temporal_length(self) -> None:
        block = ConvBlock(1, 8, kernel_size=7)
        x = torch.randn(4, 1, 360)
        out = block(x)
        # MaxPool(2) halves the length; channels grow to 8.
        assert out.shape == (4, 8, 180)


class TestECGCNN:
    def test_forward_output_shape(self) -> None:
        model = ECGCNN(num_classes=5, conv_channels=[8, 16])
        x = torch.randn(8, 1, 360)
        logits = model(x)
        assert logits.shape == (8, 5)

    def test_logits_not_normalised(self) -> None:
        """forward() must return raw logits, not probabilities (sum != 1)."""
        model = ECGCNN(num_classes=5, conv_channels=[8, 16])
        logits = model(torch.randn(2, 1, 360))
        row_sums = logits.exp().sum(dim=1)  # would be 1 only if softmaxed
        assert not torch.allclose(row_sums, torch.ones(2), atol=1e-3)

    def test_predict_proba_sums_to_one(self) -> None:
        model = ECGCNN(num_classes=5, conv_channels=[8, 16])
        probs = model.predict_proba(torch.randn(3, 1, 360))
        assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_configurable_depth(self) -> None:
        """More conv blocks => more parameters."""
        small = ECGCNN(conv_channels=[8])
        large = ECGCNN(conv_channels=[8, 16, 32])
        assert count_parameters(large) > count_parameters(small)

    def test_robust_to_input_length(self) -> None:
        """AdaptiveAvgPool should let any beat length flow through."""
        model = ECGCNN(conv_channels=[8, 16], input_length=260)
        for length in (200, 260, 360):
            out = model(torch.randn(2, 1, length))
            assert out.shape == (2, 5)

    def test_backward_pass(self) -> None:
        """Gradients should flow to the first conv layer."""
        model = ECGCNN(conv_channels=[8, 16])
        logits = model(torch.randn(4, 1, 360))
        loss = logits.sum()
        loss.backward()
        first_conv = next(model.features.parameters())
        assert first_conv.grad is not None
        assert torch.isfinite(first_conv.grad).all()


class TestBuildModel:
    def test_build_from_config(self, tiny_config) -> None:
        model = build_model(tiny_config.model, input_length=360)
        assert isinstance(model, ECGCNN)
        out = model(torch.randn(2, 1, 360))
        assert out.shape == (2, tiny_config.model["num_classes"])
