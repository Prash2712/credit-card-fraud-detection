# Credit Card Fraud Detection

Fraud data is a good place to make an apparently small evaluation mistake look impressive. The positive class is tiny, so overall accuracy is almost useless, resampling before a train/test split leaks information, and ROC-AUC can look healthy while the precision-recall trade-off is still poor.

This repo keeps the pipeline deliberately small and concentrates on those boundaries.

## Current workflow

```text
creditcard.csv
  -> stratified 80/20 hold-out
  -> scaler fitted on training rows
  -> SMOTE on training rows only
  -> XGBoost classifier
  -> untouched test distribution
  -> classification report + ROC-AUC + Average Precision
  -> model/scaler + JSON metrics
```

The source dataset is the well-known ULB credit-card fraud dataset commonly distributed through Kaggle. It is not committed here. Put the CSV at:

```text
data/raw/creditcard.csv
```

The expected target is `Class`, with `1` for a fraudulent transaction.

## Why the order matters

The split happens before scaling and SMOTE. The scaler sees only training rows, and synthetic minority examples are created only inside the training partition.

The test set therefore keeps the original prevalence. That is important: balancing the test set would make precision much easier to interpret incorrectly.

## Metrics

The training script writes the observed test results to:

```text
results/test_metrics.json
```

I keep both ROC-AUC and Average Precision. ROC-AUC is useful for ranking quality across thresholds; Average Precision is usually more revealing when positives are extremely rare.

The README does not contain an “expected 99%” score. If the experiment has not been run on the exact data/code version, it is not a result.

## Run

```bash
git clone https://github.com/Prash2712/credit-card-fraud-detection.git
cd credit-card-fraud-detection

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

Training creates:

```text
models/xgboost_fraud_detector.pkl
models/scaler.pkl
results/test_metrics.json
```

`src/evaluate.py` reconstructs the same deterministic hold-out rather than evaluating the saved model against the full dataset.

## Code

```text
src/data_loader.py   input checks
src/preprocess.py    scaling + train-only SMOTE
src/train.py         split, fit, evaluation and artifacts
src/evaluate.py      repeatable held-out evaluation
src/utils.py         small shared helpers
```

## What is missing before I would call this a fraud decision system

A classifier score is only one part of an operational fraud process. This repo does not yet include:

- probability calibration
- a cost/benefit threshold tied to investigation capacity
- point-in-time or temporal validation
- drift / prevalence monitoring
- delayed-label handling
- case-management or analyst feedback

Those are more important next steps than adding a fifth classifier to a comparison table.

## One modelling choice I would revisit

The current implementation uses both SMOTE and `scale_pos_weight`. That is useful for exploring imbalance handling, but I would compare these strategies separately under the same untouched test set before keeping both. The selection criterion should be a business-relevant precision/recall or expected-cost target, not whichever setup maximises one headline metric.

## Checks

The repo has GitHub Actions for source checks, with executable preprocessing tests being added as the pipeline is tightened.

**Prasanth Balisetty**  
[LinkedIn](https://www.linkedin.com/in/prasanth-chowdary-33322a234/)