Credit Card Fraud Detection 🚨💳

A complete end-to-end machine learning pipeline designed to detect fraudulent credit card transactions. This project focuses on handling extreme class imbalance, maximizing recall for the fraud class, and building a production-ready ML workflow aligned with real financial industry standards.

🧠 Why This Project Matters

Credit card fraud is a multi-billion-dollar problem. Fraudulent transactions are rare, constantly evolving, and incredibly costly when missed.
This project simulates a real-world fraud detection system, showcasing how a data scientist builds:

A robust ML pipeline

Handles extreme class imbalance

Extracts meaningful insights

Trains interpretable + high-performance models

Evaluates models using metrics that truly matter (Recall, AUC, F1)

📂 Project Structure
.
├── data/
│   ├── raw/                     <- Add dataset here manually
│   └── processed/
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── data_loader.py           <- Reads dataset
│   ├── preprocess.py            <- Scaling, sampling, transformations
│   ├── train.py                 <- Training pipeline
│   ├── evaluate.py              <- Evaluation workflow
│   └── utils.py                 <- Helper utilities
│
├── models/                      <- Saved trained models
├── results/                     <- Plots, metrics & reports
│
├── requirements.txt
├── environment.yml
├── LICENSE
└── README.md
This structure mirrors ML engineering standards in fintech and fraud analytics teams.

📊 Dataset

Source: Kaggle – Credit Card Fraud Detection
https://www.kaggle.com/mlg-ulb/creditcardfraud

Rows: 284,807 transactions

Fraud cases: 492 (≈0.172%)

Highly imbalanced

30 PCA-transformed features + Time + Amount

Add dataset manually:

Place the CSV file here: data/raw/creditcard.csv
(Dataset cannot be included due to licensing.)

🛠 Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/Prash2712/credit-card-fraud-detection.git
cd credit-card-fraud-detection

2️⃣ Create a Python environment
✔ Option A — Conda (recommended)
conda env create -f environment.yml
conda activate fraud-env

✔ Option B — Virtual Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3️⃣ Launch Jupyter Notebook
jupyter notebook

🚀 How to Use the Pipeline
| Step                       | Description                                   | Location                                |
| -------------------------- | --------------------------------------------- | --------------------------------------- |
| **1. EDA**                 | Explore distribution, imbalance, correlations | `01_exploration.ipynb`                  |
| **2. Feature Engineering** | Scaling, SMOTE, PCA                           | `02_feature_engineering.ipynb`          |
| **3. Training**            | Logistic Regression, RF, XGBoost              | `03_model_training.ipynb` or `train.py` |
| **4. Evaluation**          | ROC-AUC, F1, confusion matrix                 | `04_evaluation.ipynb` or `evaluate.py`  |

🧪 Models Implemented
🔹 Baseline

Logistic Regression (with class weighting)

Decision Tree

Random Forest

🔹 Advanced Models

XGBoost with imbalance tuning (scale_pos_weight)

Balanced Random Forest

SMOTE + Pipeline Models

🔮 Future Enhancements

Autoencoders (unsupervised fraud detection)

Isolation Forest

Deep neural networks for tabular data

AWS Lambda-based real-time fraud API

📈 Expected Performance 
| Model               | Precision | Recall (Fraud) | F1-Score | ROC-AUC |
| ------------------- | --------- | -------------- | -------- | ------- |
| Logistic Regression | 0.62      | 0.78           | 0.69     | 0.94    |
| Random Forest       | 0.91      | 0.88           | 0.89     | 0.98    |
| XGBoost             | 0.96      | 0.93           | 0.94     | 0.99    |

💡 Recall is the most important metric because missing fraud is more costly than false alarms.

💡 Key Skills Demonstrated
📌 Data Science Skills

Managing extreme class imbalance

Feature engineering

EDA & statistical reasoning

Evaluating ML models properly for skewed data

📌 Machine Learning Engineering

Modular, readable production-style architecture

Reproducible environment setup

Script-based training & evaluation

Version control & clean project layout

📌 Tools & Technologies

Python

NumPy, Pandas

Scikit-learn, XGBoost

Matplotlib, Seaborn

Jupyter Notebook

Git + GitHub

🔮 Future Roadmap

Add model explainability using SHAP

Deploy REST API using FastAPI

Integrate experiment tracking via MLflow

Add monitoring + drift detection

Deploy model in cloud (AWS, Azure, GCP)

Author

Prasanth Balisetty

Graduate Data Scientist

GitHub: https://github.com/Prash2712

LinkedIn: https://www.linkedin.com/in/prasanth-chowdary-33322a234/
