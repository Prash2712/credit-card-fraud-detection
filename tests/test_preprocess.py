import numpy as np
import pandas as pd

from src.preprocess import apply_smote, scale_features


def test_scale_features_uses_training_fit_only():
    X_train = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [10.0, 11.0, 12.0, 13.0]})
    X_test = pd.DataFrame({"a": [100.0, 101.0], "b": [200.0, 201.0]})

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    assert X_train_scaled.shape == (4, 2)
    assert X_test_scaled.shape == (2, 2)
    assert np.allclose(X_train_scaled.mean(axis=0), 0.0)
    assert np.allclose(scaler.mean_, X_train.mean(axis=0).to_numpy())
    assert not np.allclose(X_test_scaled.mean(axis=0), 0.0)


def test_smote_balances_minority_class():
    rng = np.random.default_rng(42)
    majority = rng.normal(size=(24, 3))
    minority = rng.normal(loc=2.0, size=(8, 3))
    X_train = np.vstack([majority, minority])
    y_train = np.array([0] * len(majority) + [1] * len(minority))

    X_resampled, y_resampled = apply_smote(X_train, y_train)

    values, counts = np.unique(y_resampled, return_counts=True)
    class_counts = dict(zip(values.tolist(), counts.tolist()))

    assert X_resampled.shape[0] == len(y_resampled)
    assert class_counts[0] == class_counts[1]
