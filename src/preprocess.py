import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def scale_features(X_train, X_test):
    """
    Standard scale numerical features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train):
    """
    Apply SMOTE oversampling to handle class imbalance.
    """
    sm = SMOTE(sampling_strategy="minority", random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    return X_resampled, y_resampled
