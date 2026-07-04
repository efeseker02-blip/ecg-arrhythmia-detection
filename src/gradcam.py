"""Grad-CAM explainability for the 1-D ECG CNN.

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which parts of
the input the network relied on for its decision. For an ECG, this answers a
clinically meaningful question: *did the model look at the QRS complex — the
physiologically relevant region — or is it latching onto an artefact?*

How it works
------------
1.  Forward-pass a beat and capture the activations of the last conv layer.
2.  Back-propagate the target class score to get gradients w.r.t. those
    activations.
3.  Global-average-pool the gradients to get one importance weight per channel.
4.  Weight-and-sum the activation maps, apply ReLU, and upsample to the input
    length to obtain a per-sample saliency curve.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class GradCAM1D:
    """Grad-CAM for 1-D convolutional models.

    Parameters
    ----------
    model:
        A trained model containing at least one ``nn.Conv1d`` layer.
    target_layer:
        The convolutional layer to attach hooks to. Defaults to the last
        ``Conv1d`` found in ``model``.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or self._find_last_conv(model)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        # Hooks capture activations (forward) and gradients (backward). We keep
        # the handles so they can be removed — otherwise repeatedly constructing
        # a GradCAM1D on the same (e.g. Streamlit-cached) model would stack
        # duplicate hooks that fire on every pass and never get released.
        self._handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def remove(self) -> None:
        """Detach all hooks from the target layer."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> GradCAM1D:
        return self

    def __exit__(self, *_exc) -> None:
        self.remove()

    @staticmethod
    def _find_last_conv(model: nn.Module) -> nn.Module:
        """Return the last ``nn.Conv1d`` module in the network."""
        conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv1d):
                conv = module
        if conv is None:
            raise ValueError("No Conv1d layer found in model for Grad-CAM.")
        return conv

    def _save_activation(self, _module, _inp, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    def __call__(self, beat: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Return a per-sample saliency map in ``[0, 1]`` for one beat.

        Parameters
        ----------
        beat:
            Tensor of shape ``(1, 1, L)`` (single beat, single channel).
        class_idx:
            Class to explain. Defaults to the model's predicted class.
        """
        logits = self.model(beat)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # weights: importance of each channel = mean gradient over time.
        weights = self._gradients.mean(dim=2, keepdim=True)  # (1, C, 1)
        cam = (weights * self._activations).sum(dim=1).squeeze(0)  # (L',)
        cam = torch.relu(cam)

        # Upsample the low-resolution CAM back to the input length.
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = torch.nn.functional.interpolate(
            cam, size=beat.shape[-1], mode="linear", align_corners=False
        ).squeeze()

        cam = cam.cpu().numpy()
        # Normalise to [0, 1] for visualisation.
        cam = cam - cam.min()
        denom = cam.max() if cam.max() > 0 else 1.0
        return cam / denom


def plot_gradcam(
    beat: np.ndarray,
    saliency: np.ndarray,
    predicted_class: str,
    save_path=None,
):
    """Overlay a Grad-CAM saliency curve on top of the input beat.

    The beat is drawn as a line coloured by saliency (hotter = more important),
    so the QRS region should light up for a well-behaved model.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from .utils import ensure_dir

    x = np.arange(len(beat))
    points = np.array([x, beat]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(9, 4))
    lc = LineCollection(segments, cmap="inferno", array=saliency[:-1], linewidth=2.5)
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(beat.min() - 0.5, beat.max() + 0.5)
    ax.set(
        title=f"Grad-CAM explanation — predicted: {predicted_class}",
        xlabel="Sample",
        ylabel="Normalised amplitude",
    )
    fig.colorbar(lc, ax=ax, label="Saliency (importance)")
    fig.tight_layout()
    if save_path is not None:
        from pathlib import Path

        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
