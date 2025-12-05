import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from data_loader import load_data
from preprocess import scale_features

def evaluate_model():
    # Load dataset
    df = load_data("data/raw/creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Load saved artifacts
    model = joblib.load("models/xgboost_fraud_detector.pkl")
    scaler = joblib.load("models/scaler.pkl")

    # Scale features
    X_scaled = scaler.transform(X)

    # Predictions
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    # Metrics
    print("Classification Report:\n", classification_report(y, preds))
    print("ROC-AUC Score:", roc_auc_score(y, probs))

if __name__ == "__main__":
    evaluate_model()
