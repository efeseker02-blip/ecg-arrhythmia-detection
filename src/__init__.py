"""ECG Arrhythmia Detection — deep-learning heartbeat classification package.

Modules
-------
utils          : configuration, seeding, logging and device helpers.
preprocessing  : ECG loading, filtering, beat segmentation and normalisation.
dataset        : PyTorch ``Dataset``/``DataLoader`` construction and splits.
model          : the configurable 1-D CNN architecture.
train          : the training loop (early stopping, LR scheduling, TensorBoard).
evaluate       : metrics, confusion matrix, ROC curves and reports.
predict        : single-record / single-beat inference.
visualization  : reusable plotting utilities.
"""

__version__ = "1.0.0"
