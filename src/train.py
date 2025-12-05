import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from data_loader import load_data
from preprocess import scale_features, apply_smote
import joblib

def train_model():
    # --- Load Data ---
    df = load_data("data/raw/creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Scale Data ---
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # --- Apply SMOTE ---
    X_resampled, y_resampled = apply_smote(X_train_scaled, y_train)

    # --- Train Model (XGBoost Best) ---
    model = XGBClassifier(
        scale_pos_weight=10,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        random_state=42
    )

    model.fit(X_resampled, y_resampled)

    # --- Evaluate ---
    preds = model.predict(X_test_scaled)
    print("Classification Report:\n", classification_report(y_test, preds))

    # --- Save Model + Scaler ---
    joblib.dump(model, "models/xgboost_fraud_detector.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("\nModel training complete. Files saved in /models/")

if __name__ == "__main__":
    train_model()
