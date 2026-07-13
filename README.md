# Credit Card Fraud Detection

ML pipeline on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 fraud (0.17%). SQL feature engineering → baseline vs. XGBoost → SHAP explainability.

## Pipeline

SQLite + SQL feature engineering → EDA → Logistic Regression baseline → XGBoost → SHAP

## Results

| Model | PR-AUC | Fraud Precision | Fraud Recall |
|---|---|---|---|
| Logistic Regression | 0.754 | 0.06 | 0.92 |
| **XGBoost** | **0.869** | **0.85** | **0.84** |

+15.3% PR-AUC over baseline. XGBoost catches 82/98 fraud cases with only 14 false alarms (vs. baseline's 1,434). PR-AUC is used instead of accuracy since a "predict all legit" model would score 99.83% accuracy while catching zero fraud.

Class imbalance handled via class weighting (`scale_pos_weight`), not synthetic oversampling.

## SQL feature engineering

Two engineered features built as SQL window functions: transaction velocity (rolling 1-hour count/avg via `RANGE BETWEEN ... PRECEDING`) and a high-amount flag (`PERCENT_RANK()`). An earlier correlated-subquery version was O(n²) and never finished on 284K rows; rewriting as window functions over a single pass brought it to ~5.7 seconds.

## Explainability

SHAP (TreeExplainer) confirms V14 and V4 as the top predictors — consistent with the independent correlation analysis from EDA. Includes a per-transaction waterfall plot showing exactly which features drove one real fraud prediction.

## Structure

```
data/        creditcard.csv, fraud.db, model_features.csv (not committed)
docs/        EDA + SHAP chart outputs
sql/         schema + feature engineering views
src/         load_data.py, eda.py, train_model.py
```

## Setup

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
# download creditcard.csv from Kaggle into data/
python src/load_data.py
python src/eda.py
python src/train_model.py
```

## Stack

Python, pandas, scikit-learn, XGBoost, SHAP, SQLite, matplotlib/seaborn

