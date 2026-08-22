import joblib
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from data_loader import load_data


def evaluate_model():
    """Evaluate persisted artifacts on the same deterministic hold-out definition used in training."""
    df = load_data("data/raw/creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = joblib.load("models/xgboost_fraud_detector.pkl")
    scaler = joblib.load("models/scaler.pkl")

    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    print(classification_report(y_test, predictions))
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(f"Average precision: {average_precision_score(y_test, probabilities):.4f}")


if __name__ == "__main__":
    evaluate_model()
