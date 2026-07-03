"""Interactive Streamlit dashboard for the ECG arrhythmia classifier.

Run with::

    streamlit run app/streamlit_app.py

Features
--------
* Load a trained checkpoint (or run in "demo" mode with an untrained model).
* Classify a synthetic or uploaded beat, or browse beats from the processed
  cache.
* Visualise the predicted probability distribution and a Grad-CAM explanation.

The app degrades gracefully: if no checkpoint or data cache is present it still
runs, so reviewers can explore the UI immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

# Make ``src`` importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import load_checkpoint_model  # noqa: E402
from src.gradcam import GradCAM1D, plot_gradcam  # noqa: E402
from src.model import build_model  # noqa: E402
from src.predict import predict_beat  # noqa: E402
from src.utils import AAMI_CLASSES, get_device, load_config  # noqa: E402
from src.visualization import CLASS_FULL_NAMES, plot_prediction  # noqa: E402

st.set_page_config(page_title="ECG Arrhythmia Detection", page_icon="🫀", layout="wide")


@st.cache_resource
def _load(config_path: str, checkpoint_path: str):
    """Load config + model once and cache across reruns."""
    config = load_config(config_path)
    device = get_device()
    input_length = 2 * config.data["beat_window"]
    if Path(checkpoint_path).exists():
        model = load_checkpoint_model(checkpoint_path, config, device)
        trained = True
    else:
        # Demo mode: an untrained model so the UI still works out-of-the-box.
        model = build_model(config.model, input_length=input_length).to(device).eval()
        trained = False
    return config, model, device, trained


def _synthetic_beat(class_idx: int, length: int, seed: int = 0) -> np.ndarray:
    """Generate a simple synthetic beat for demoing without the real DB."""
    rng = np.random.default_rng(seed + class_idx)
    t = np.linspace(0, 1, length)
    beat = np.exp(-0.5 * ((t - 0.5) / (0.03 + 0.01 * class_idx)) ** 2) * (1 + 0.3 * class_idx)
    beat = beat + 0.05 * rng.standard_normal(length)
    return ((beat - beat.mean()) / (beat.std() + 1e-8)).astype(np.float32)


def main() -> None:
    """Render the dashboard."""
    st.title("🫀 ECG Arrhythmia Detection")
    st.caption(
        "1-D CNN heartbeat classifier trained on the MIT-BIH Arrhythmia Database. "
        "**Educational demo — not a medical device.**"
    )

    with st.sidebar:
        st.header("Configuration")
        config_path = st.text_input("Config path", "config.yaml")
        checkpoint_path = st.text_input("Checkpoint path", "models/best_model.pt")
        source = st.radio("Beat source", ["Synthetic demo", "Upload .npy"])

    try:
        config, model, device, trained = _load(config_path, checkpoint_path)
    except FileNotFoundError as exc:
        st.error(f"Could not load configuration: {exc}")
        return

    length = 2 * config.data["beat_window"]
    if not trained:
        st.warning(
            "No trained checkpoint found — running in **demo mode** with an "
            "untrained model. Train one with `python -m src.train` for real "
            "predictions."
        )

    # --- Obtain a beat ----------------------------------------------------
    beat: np.ndarray | None = None
    if source == "Synthetic demo":
        cls_name = st.sidebar.selectbox(
            "Synthetic class morphology",
            AAMI_CLASSES,
            format_func=lambda c: f"{c} — {CLASS_FULL_NAMES[c]}",
        )
        seed = st.sidebar.number_input("Random seed", 0, 9999, 0)
        beat = _synthetic_beat(AAMI_CLASSES.index(cls_name), length, int(seed))
    else:
        uploaded = st.sidebar.file_uploader("Upload a 1-D beat (.npy)", type="npy")
        if uploaded is not None:
            beat = np.load(uploaded).astype(np.float32).ravel()
            if beat.size != length:
                st.error(f"Expected a beat of length {length}, got {beat.size}.")
                beat = None

    if beat is None:
        st.info("Choose or upload a beat in the sidebar to run a prediction.")
        return

    # --- Predict ----------------------------------------------------------
    prediction = predict_beat(beat, model, device, already_normalized=True)
    probs = np.array([prediction.probabilities[c] for c in AAMI_CLASSES])

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted class", prediction.predicted_class)
    col2.metric("Full name", CLASS_FULL_NAMES[prediction.predicted_class])
    col3.metric("Confidence", f"{prediction.confidence:.1%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Beat & probability distribution")
        fig = plot_prediction(beat, probs)
        st.pyplot(fig)
    with right:
        st.subheader("Grad-CAM explanation")
        cam = GradCAM1D(model)
        beat_tensor = torch.from_numpy(beat).float().view(1, 1, -1).to(device)
        saliency = cam(beat_tensor)
        fig_cam = plot_gradcam(beat, saliency, prediction.predicted_class)
        st.pyplot(fig_cam)
        st.caption(
            "Hotter colours mark the samples the model relied on most. A trained "
            "model should focus on the QRS complex."
        )


if __name__ == "__main__":
    main()
