import numpy as np
import pandas as pd

from src.preprocess import apply_smote, scale_features


def test_scale_features_uses_training_fit_only():
    X_train = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [10.0, 11.0, 12.0]})
    X_test = pd.DataFrame({"a": [100.0], "b": [110.0]})

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    assert X_train_scaled.shape == (3, 2)
    assert X_test_scaled.shape == (1, 2)
    assert np.allclose(scaler.mean_, np.array([1.0, 11.0]))


def test_smote_balances_minority_class():
    X_train = np.arange(40, dtype=float).reshape(20, 2)
    y_train = np.array([0] * 15 + [1] * 5)

    X_resampled, y_resampled = apply_smote(X_train, y_train)

    values, counts = np.unique(y_resampled, return_counts=True)
    class_counts = dict(zip(values.tolist(), counts.tolist()))

    assert X_resampled.shape[0] == len(y_resampled)
    assert class_counts[0] == class_counts[1]
