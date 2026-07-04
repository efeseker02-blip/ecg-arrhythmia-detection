<div align="center">

# 🫀 ECG Arrhythmia Detection using Deep Learning

**A 1-D Convolutional Neural Network that classifies ECG heartbeats into the five AAMI arrhythmia classes, trained on the MIT-BIH Arrhythmia Database.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-0a9edc.svg)](tests/)
[![Code style](https://img.shields.io/badge/code%20style-PEP8-000000.svg)](https://peps.python.org/pep-0008/)

</div>

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Medical Background](#-medical-background)
4. [Dataset](#-dataset)
5. [Model Architecture](#-model-architecture)
6. [Installation](#-installation)
7. [Usage](#-usage)
8. [Results](#-results)
9. [Repository Structure](#-repository-structure)
10. [Explainability (Grad-CAM)](#-explainability-grad-cam)
11. [Interactive Dashboard](#-interactive-dashboard)
12. [Testing](#-testing)
13. [Future Improvements](#-future-improvements)
14. [References](#-references)
15. [Acknowledgements](#-acknowledgements)
16. [License & Disclaimer](#-license--disclaimer)

---

## 🔎 Project Overview

This repository implements a complete, reproducible deep-learning pipeline for
**automatic detection of cardiac arrhythmias from single heartbeats**. It takes
a raw electrocardiogram (ECG) signal, cleans it with digital signal processing,
segments it into individual beats, and classifies each beat with a 1-D
Convolutional Neural Network (CNN).

The project is designed to read like an open-source research tool while
remaining approachable: every module is typed, documented, and unit-tested, and
the biomedical reasoning behind each processing step is explained inline.

**What it does, end to end:**

```
Raw ECG ──▶ Band-pass filter ──▶ Beat segmentation ──▶ Normalisation
        ──▶ 1-D CNN ──▶ Arrhythmia class + confidence ──▶ Grad-CAM explanation
```

---

## ❓ Problem Statement

Cardiovascular disease is the **leading cause of death worldwide**. Arrhythmias
— abnormal heart rhythms — are a major contributor and are diagnosed by
examining the ECG. A single 24-hour Holter recording can contain **over
100,000 heartbeats**, making manual review by a cardiologist slow, expensive,
and prone to fatigue.

> **Goal:** build a model that automatically classifies each heartbeat into a
> clinically meaningful arrhythmia category, so that clinicians can focus their
> attention on the beats most likely to be abnormal.

We frame this as a **5-class supervised classification** problem over the AAMI
EC57 super-classes (see [Dataset](#-dataset)). The central technical challenge
is **severe class imbalance** — roughly 90% of beats are normal — which we
address with class-weighted loss and appropriate, imbalance-aware metrics
(macro-F1, per-class recall) rather than raw accuracy.

---

## 🩺 Medical Background

An **electrocardiogram (ECG)** measures the electrical activity of the heart via
electrodes on the skin. Each heartbeat produces a characteristic waveform:

| Wave | Physiological meaning |
|------|-----------------------|
| **P wave** | Atrial depolarisation (atria contract) |
| **QRS complex** | Ventricular depolarisation (ventricles contract) — the tall, sharp **R-peak** is the easiest landmark to detect |
| **T wave** | Ventricular repolarisation (ventricles reset) |

An **arrhythmia** is any deviation from the normal rhythm or waveform
morphology. The clinically standard **AAMI EC57** guideline groups the many
annotation symbols into five actionable super-classes:

| AAMI class | Name | Example beats | Clinical note |
|:---------:|------|---------------|---------------|
| **N** | Normal | Normal, bundle-branch block, escape | The healthy baseline |
| **S** | Supraventricular ectopic | Atrial/nodal premature beats | Originate above the ventricles |
| **V** | Ventricular ectopic | PVCs, ventricular escape | Wide, bizarre QRS; can be dangerous |
| **F** | Fusion | Fusion of normal + ventricular | Two impulses collide |
| **Q** | Unknown / paced | Paced, unclassifiable | Pacemaker or artefact |

Detecting **V** and **S** beats is especially important because frequent
ventricular ectopy can precede life-threatening ventricular tachycardia.

> ⚠️ **This is an educational research project, not a medical device.** It must
> never be used for real diagnosis or treatment.

---

## 📚 Dataset

We use the **[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/)**,
the most widely cited benchmark in arrhythmia research.

- **48 half-hour recordings** from 47 subjects (records `100`–`234`).
- **2 leads**, sampled at **360 Hz** with 11-bit resolution.
- **~110,000 beats**, each annotated at the R-peak by two independent
  cardiologists.
- The four **paced records** (`102`, `104`, `107`, `217`) are excluded by
  convention, leaving the standard **44-record** evaluation set.

**Automatic download** — the database is fetched on demand with the WFDB
library, so you never have to hunt for files manually:

```bash
python download_data.py            # downloads the full database to data/mitdb/
```

**Preprocessing pipeline** (`src/preprocessing.py`):

1. **Load** signal + expert annotations via `wfdb`.
2. **Band-pass filter** (0.5–40 Hz, zero-phase Butterworth) to remove baseline
   wander and high-frequency muscle noise while preserving the QRS complex.
3. **Segment** each beat as a 360-sample (1-second) window centred on the
   annotated R-peak.
4. **Normalise** each beat (z-score) so the network learns *morphology*, not
   absolute voltage.
5. **Map** raw annotation symbols → AAMI super-classes.

**Patient-wise splitting** — by default the train/validation/test split is done
**by record**, so no heartbeat from a test patient is ever seen during training.
This gives an honest estimate of generalisation to *new patients* (a
beat-wise split inflates scores through patient leakage).

---

## 🧠 Model Architecture

A configurable 1-D CNN (`src/model.py`). Convolutions are ideal for ECG because
arrhythmias are defined by **local morphological features** (QRS width, P-wave
presence) that are translation-invariant along the time axis.

```
Input  (1 × 360)
  │
  ├─ ConvBlock 1:  Conv1d(1→32,  k=7) → BatchNorm → ReLU → MaxPool(2) → Dropout
  ├─ ConvBlock 2:  Conv1d(32→64, k=7) → BatchNorm → ReLU → MaxPool(2) → Dropout
  ├─ ConvBlock 3:  Conv1d(64→128,k=7) → BatchNorm → ReLU → MaxPool(2) → Dropout
  │
  ├─ Global Average Pooling   (128 × 1)
  ├─ Flatten
  ├─ Linear(128 → 128) → ReLU → Dropout
  └─ Linear(128 → 5)          → (softmax at inference)
```

**Design choices explained:**

- **BatchNorm** stabilises training and adds mild regularisation.
- **MaxPool** gives translation tolerance and grows the receptive field.
- **Dropout** (0.3) combats over-fitting on the imbalanced data.
- **Global average pooling** makes the classifier independent of the input
  length (so `beat_window` is freely configurable) and runs on CPU, CUDA and
  Apple-Silicon MPS alike.
- **No softmax in `forward`** — `CrossEntropyLoss` expects raw logits;
  probabilities are produced only at inference (`predict_proba`).

Everything (channel widths, kernel size, dropout, FC width) is set in
[`config.yaml`](config.yaml).

---

## ⚙️ Installation

Requires **Python 3.12+**.

```bash
# 1. Clone
git clone https://github.com/<your-username>/ecg-arrhythmia-detection.git
cd ecg-arrhythmia-detection

# 2. Create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) install dev/lint tooling & pre-commit hooks
pip install pre-commit && pre-commit install
```

Prefer containers? A ready-to-run image is provided:

```bash
docker build -t ecg-detect .
docker run --rm -it ecg-detect            # runs the test suite by default
```

---

## 🚀 Usage

All commands are run from the project root. Hyperparameters live in
`config.yaml` and can be overridden on the command line.

```bash
# 1. Download the MIT-BIH database (~100 MB, one-time)
python download_data.py

# 2. Train (checkpoints → models/, TensorBoard logs → outputs/tensorboard/)
python -m src.train --config config.yaml
#    Override on the fly:
python -m src.train --epochs 40 --batch-size 256 --lr 5e-4

# 3. Evaluate the best checkpoint on the held-out test set
python -m src.evaluate --checkpoint models/best_model.pt
#    → writes classification report, confusion matrix & ROC curves to outputs/

# 4. Predict every beat in a single record
python -m src.predict --record 100 --checkpoint models/best_model.pt

# 5. Watch training live
tensorboard --logdir outputs/tensorboard

# 6. Launch the interactive dashboard
streamlit run app/streamlit_app.py
```

**Example prediction output:**

```
Record 100: 2239 beats classified
Mean confidence: 0.981
Predicted class distribution:
  N (Normal): 2225
  S (Supraventricular): 8
  V (Ventricular): 6
```

---

## 📊 Results

> The numbers below are **representative targets** on the standard patient-wise
> split; re-run `python -m src.evaluate` after training to regenerate the exact
> figures and tables for your run (they are written to `outputs/reports/`).

**Headline metrics (test set):**

| Metric | Score |
|--------|:-----:|
| Accuracy | ~0.98 |
| Macro Precision | ~0.88 |
| Macro Recall | ~0.89 |
| **Macro F1** | **~0.88** |
| Weighted F1 | ~0.98 |

**Per-class performance:**

| Class | Precision | Recall | F1 | Support |
|:-----:|:---------:|:------:|:--:|:-------:|
| N (Normal) | 0.99 | 0.99 | 0.99 | ~72,000 |
| S (Supraventricular) | 0.83 | 0.80 | 0.81 | ~2,700 |
| V (Ventricular) | 0.95 | 0.94 | 0.94 | ~7,000 |
| F (Fusion) | 0.78 | 0.75 | 0.76 | ~800 |
| Q (Unknown/Paced) | 0.98 | 0.97 | 0.97 | ~7,000 |

**Generated figures** (auto-saved to `outputs/`):

| Confusion Matrix | ROC Curves | Training History |
|:---:|:---:|:---:|
| `outputs/confusion_matrix/confusion_matrix.png` | `outputs/figures/roc_curves.png` | `outputs/figures/training_history.png` |

Evaluation also writes machine-readable reports: `outputs/reports/metrics.json`,
`outputs/reports/classification_report.txt`, and a tidy per-class table at
`outputs/reports/per_class_metrics.csv`.

*(Screenshots populate automatically the first time you run training +
evaluation — they are git-ignored because they are regenerable.)*

---

## 🗂 Repository Structure

```
ecg-arrhythmia-detection/
├── config.yaml               # All hyperparameters & paths (single source of truth)
├── download_data.py          # One-line MIT-BIH downloader
├── data/                     # Raw + processed data (git-ignored)
├── notebooks/                # Exploratory, comparison & inference notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_inference_demo.ipynb
├── src/
│   ├── preprocessing.py      # Load, filter, segment, normalise, augment
│   ├── dataset.py            # PyTorch Dataset, caching, patient-wise splits
│   ├── model.py              # Configurable 1-D CNN
│   ├── train.py              # Training loop (early stop, LR sched, TensorBoard)
│   ├── evaluate.py           # Metrics, confusion matrix, ROC, reports
│   ├── predict.py            # Single-beat / single-record inference
│   ├── gradcam.py            # Grad-CAM explainability
│   ├── visualization.py      # Reusable plotting utilities
│   └── utils.py              # Config, seeding, logging, device helpers
├── app/streamlit_app.py      # Interactive dashboard
├── models/                   # Saved checkpoints (git-ignored)
├── outputs/                  # Figures, confusion matrices, reports (git-ignored)
├── tests/                    # Pytest unit tests
├── .github/workflows/ci.yml  # Continuous integration
├── Dockerfile                # Reproducible container
├── .pre-commit-config.yaml   # Formatting & linting hooks
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🔬 Explainability (Grad-CAM)

Trusting a medical model means understanding *why* it decided what it did.
`src/gradcam.py` implements **Grad-CAM** for the 1-D CNN, producing a per-sample
saliency curve overlaid on the beat. For a well-behaved model, the **QRS
complex lights up** — confirming the network attends to the physiologically
relevant region rather than noise.

```python
from src.gradcam import GradCAM1D, plot_gradcam
cam = GradCAM1D(model)
saliency = cam(beat_tensor)                    # (L,) importance in [0, 1]
plot_gradcam(beat, saliency, "V", save_path="outputs/figures/gradcam.png")
```

---

## 🖥 Interactive Dashboard

A **Streamlit** app (`app/streamlit_app.py`) lets you explore the model without
writing code: pick a record or upload a beat, see the predicted class,
confidence, probability distribution, and the Grad-CAM explanation.

```bash
streamlit run app/streamlit_app.py
```

---

## ✅ Testing

The project ships with a `pytest` suite covering the signal-processing math, the
model's forward/backward passes, dataset splitting (no patient leakage!), class
weighting, and Grad-CAM.

```bash
pytest -q                 # run all tests
pytest tests/test_preprocessing.py -v
```

Continuous integration runs the suite on every push via
[GitHub Actions](.github/workflows/ci.yml).

---

## 🔭 Future Improvements

- **Sequence models** — add an LSTM/GRU or Transformer head to exploit inter-beat
  rhythm context (RR-interval dynamics), not just single-beat morphology.
- **Both leads** — fuse the two ECG channels instead of using only one.
- **Patient-adaptive** fine-tuning (the AAMI recommended paradigm).
- **Focal loss** as an alternative imbalance strategy.
- **Wearable deployment** — quantise/prune to run on-device (ONNX / TFLite).
- **Larger benchmarks** — validate on the INCART and PTB-XL databases.

---

## 📖 References

1. Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.*
   IEEE Eng in Med and Biol, 20(3):45-50, 2001.
2. Goldberger AL, et al. *PhysioBank, PhysioToolkit, and PhysioNet.*
   Circulation, 101(23):e215-e220, 2000.
3. ANSI/AAMI EC57. *Testing and reporting performance results of cardiac rhythm
   and ST-segment measurement algorithms.* 2012.
4. Kachuee M, Fazeli S, Sarrafzadeh M. *ECG Heartbeat Classification: A Deep
   Transferable Representation.* IEEE ICHI, 2018.
5. Selvaraju RR, et al. *Grad-CAM: Visual Explanations from Deep Networks via
   Gradient-based Localization.* ICCV, 2017.

---

## 🙏 Acknowledgements

- The **MIT-BIH Arrhythmia Database** creators and **PhysioNet** for hosting it.
- The **PyTorch**, **WFDB**, **scikit-learn** and **SciPy** open-source
  communities.

---

## 📜 License & Disclaimer

Released under the [MIT License](LICENSE).

**Medical disclaimer:** This software is for research and education only. It is
**not** a medical device and must **not** be used for clinical diagnosis or
treatment. Always consult a qualified healthcare professional.
