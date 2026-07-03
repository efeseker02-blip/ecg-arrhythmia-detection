"""The configurable 1-D convolutional neural network for beat classification.

Why a 1-D CNN?
--------------
An ECG beat is a 1-D time series, and arrhythmias are defined by *local*
morphological features — the width of the QRS complex, the presence/absence of
a P-wave, ST-segment shape. Convolutional filters are ideal for learning these
translation-invariant local patterns directly from the raw waveform, without
hand-engineered features. Stacking conv blocks builds a hierarchy: early layers
learn edges/slopes, deeper layers learn whole-complex shapes.

The architecture follows the classic ``Conv → BatchNorm → ReLU → MaxPool →
Dropout`` block, repeated with growing channel width, then a small MLP head with
a softmax output over the five AAMI classes.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """One convolutional block: Conv1d → BatchNorm → ReLU → MaxPool → Dropout.

    * **Conv1d** learns local waveform features.
    * **BatchNorm** stabilises and speeds up training by normalising
      activations, and adds a mild regularising effect.
    * **ReLU** introduces non-linearity.
    * **MaxPool** halves the temporal resolution, giving translation tolerance
      and enlarging the receptive field of later layers.
    * **Dropout** randomly zeros activations to reduce over-fitting.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # "same" padding keeps the length unchanged before pooling.
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.block(x)


class ECGCNN(nn.Module):
    """A configurable 1-D CNN for ECG heartbeat classification.

    Parameters
    ----------
    in_channels:
        Number of input leads (1 for single-lead beats).
    num_classes:
        Number of output classes (5 AAMI super-classes).
    conv_channels:
        Output channel width of each successive :class:`ConvBlock`.
    kernel_size:
        Convolution kernel size (odd number recommended for symmetric padding).
    dropout:
        Dropout probability used in every block and in the classifier head.
    fc_hidden:
        Width of the hidden fully-connected layer.
    input_length:
        Length of each input beat in samples (default 360 = ``2 * beat_window``).
        Used to size the classifier via an ``AdaptiveAvgPool`` so the model is
        robust to changes in ``beat_window``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        conv_channels: list[int] | None = None,
        kernel_size: int = 7,
        dropout: float = 0.3,
        fc_hidden: int = 128,
        input_length: int = 360,
    ) -> None:
        super().__init__()
        conv_channels = conv_channels or [32, 64, 128]
        self.input_length = input_length

        blocks: list[nn.Module] = []
        prev = in_channels
        for out_ch in conv_channels:
            blocks.append(ConvBlock(prev, out_ch, kernel_size=kernel_size, dropout=dropout))
            prev = out_ch
        self.features = nn.Sequential(*blocks)

        # Global average pooling collapses the whole temporal dimension to a
        # single value per channel. This decouples the classifier's input width
        # from ``input_length``/``beat_window`` (so the model stays configurable
        # without manual shape arithmetic) and, because any length is divisible
        # by 1, it runs on every backend — including Apple-Silicon ``mps``, which
        # rejects non-divisible adaptive-pool sizes.
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        flattened = conv_channels[-1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )
        # NOTE: no softmax here — ``nn.CrossEntropyLoss`` expects raw logits and
        # applies log-softmax internally. Softmax is applied only at inference
        # time (see ``predict_proba``).

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-initialise conv/linear weights for stable ReLU training."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits of shape ``(batch, num_classes)``."""
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax class probabilities (inference convenience)."""
        self.eval()
        return torch.softmax(self.forward(x), dim=1)


def build_model(model_cfg: dict, input_length: int = 360) -> ECGCNN:
    """Instantiate an :class:`ECGCNN` from the ``model`` section of the config."""
    return ECGCNN(
        in_channels=model_cfg.get("in_channels", 1),
        num_classes=model_cfg.get("num_classes", 5),
        conv_channels=model_cfg.get("conv_channels", [32, 64, 128]),
        kernel_size=model_cfg.get("kernel_size", 7),
        dropout=model_cfg.get("dropout", 0.3),
        fc_hidden=model_cfg.get("fc_hidden", 128),
        input_length=input_length,
    )
