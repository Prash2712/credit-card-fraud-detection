"""
src package initializer.

This package contains all core modules required for the
Credit Card Fraud Detection project, including:

- data_loader:      Loading raw datasets
- preprocess:       Scaling, sampling (SMOTE), transformations
- train:            Model training pipeline
- evaluate:         Evaluation pipeline and metrics
- utils:            Utility functions (plots, helpers)

Import modules directly from src for convenience.
"""

from .data_loader import load_data
from .preprocess import scale_features, apply_smote
from .train import train_model
from .evaluate import evaluate_model
from . import utils

__all__ = [
    "load_data",
    "scale_features",
    "apply_smote",
    "train_model",
    "evaluate_model",
    "utils",
]
