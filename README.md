# Credit Card Fraud Detection 🚨💳

A complete end-to-end machine learning pipeline designed to detect fraudulent credit card transactions. This project focuses on handling extreme class imbalance, maximizing recall for the minority fraud class, and building reproducible, industry-grade ML workflows.

---

## 🧠 Why This Project Matters  
Credit card fraud is one of the most critical challenges in financial security, costing businesses billions annually. Standard rule-based systems often fail due to rapidly evolving fraud patterns and heavily imbalanced data.

This project demonstrates:
- Real-world **class imbalance handling**
- Clean **ML engineering structure**
- Practical fraud detection workflow
- A foundation for deploying real-time fraud detection systems

---

## 📂 Project Structure

```text
.
├── data/
│   ├── raw/              <- Download dataset manually and place here
│   └── processed/
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── models/
├── results/
├── requirements.txt
├── environment.yml
├── LICENSE
└── README.md



Dataset:

Name: Credit Card Fraud Detection

Source: https://www.kaggle.com/mlg-ulb/creditcardfraud

Files included: None (dataset not committed due to licensing)

Place the dataset here after download:

data/raw/creditcard.csv

🛠 Installation & Setup
1. Clone the repository:
git clone https://github.com/Prash2712/credit-card-fraud-detection.git
cd credit-card-fraud-detection

2. Create environment (choose one):
Option A — Conda (recommended):
conda env create -f environment.yml
conda activate fraud-env

Option B — Python venv:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. Launch Jupyter Notebook:
jupyter notebook

🚀 How to Run This Project
Stage	Description	Run
EDA	Visualizing data, imbalance, correlations	notebooks/01_exploration.ipynb
Feature Engineering	Scaling, sampling, PCA	notebooks/02_feature_engineering.ipynb
Train Model	Classical ML models + sampling	src/train.py or 03_model_training.ipynb
Evaluate	ROC/AUC, F1, confusion matrix	src/evaluate.py or 04_evaluation.ipynb
🧪 Models Used
✔ Logistic Regression (Baseline)

Class weighting

Fast and interpretable

✔ Random Forest

Handles non-linearity

Works well with unbalanced data

✔ XGBoost

Often best performance

Can focus on minority class

Handles imbalance with scale_pos_weight

⭐ Future Additions

Isolation Forest

Autoencoders for anomaly detection

Neural networks for tabular data

Real-time fraud scoring API (AWS Lambda)

| Model               | Precision | Recall (Fraud) | F1-Score | ROC-AUC |
| ------------------- | --------- | -------------- | -------- | ------- |
| Logistic Regression | 0.63      | 0.78           | 0.70     | 0.94    |
| Random Forest       | 0.91      | 0.89           | 0.90     | 0.99    |
| XGBoost             | 0.96      | 0.93           | 0.94     | 0.99    |

🧠 Key Skills Demonstrated

Imbalanced data handling (SMOTE, class weighting, undersampling)

Exploratory Data Analysis (EDA)

Feature Engineering for tabular data

Model training & hyperparameter tuning

Evaluation using ROC-AUC, confusion matrix, recall

Clean production-style project structure

Reproducibility (environment.yml + requirements.txt)

Good coding practices and modular Python scripts

🔮 Future Enhancements

Add SHAP explainability

Build API for real-time fraud scoring

Add MLOps pipeline (CI/CD, model registry)

Deploy using AWS Lambda + API Gateway

Try autoencoders for anomaly detection

Author

Prasanth Balisetty
Graduate Data Scientist
GitHub: https://github.com/Prash2712

LinkedIn: https://www.linkedin.com/in/prasanth-chowdary-33322a234/
