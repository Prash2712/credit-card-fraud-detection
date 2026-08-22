import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data_loader import load_data
from preprocess import apply_smote, scale_features


def train_model():
    """Train the fraud classifier and evaluate it on a held-out test split."""
    df = load_data("data/raw/creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    X_resampled, y_resampled = apply_smote(X_train_scaled, y_train)

    model = XGBClassifier(
        scale_pos_weight=10,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_resampled, y_resampled)

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    report = classification_report(y_test, predictions, output_dict=True)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "classification_report": report,
        "test_rows": int(len(y_test)),
        "fraud_rows_in_test": int(y_test.sum()),
        "random_state": 42,
    }

    model_dir = Path("models")
    results_dir = Path("results")
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "xgboost_fraud_detector.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    (results_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(classification_report(y_test, predictions))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Average precision: {metrics['average_precision']:.4f}")
    print("Saved model artifacts to models/ and hold-out metrics to results/test_metrics.json")


if __name__ == "__main__":
    train_model()
