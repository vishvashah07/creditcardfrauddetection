# Credit Card Fraud Detection

An end-to-end machine learning pipeline to detect fraudulent credit card transactions, built on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## Problem Statement

Credit card fraud detection is a real-world imbalanced classification problem — only 0.17% of transactions are fraudulent. A naive model that predicts "legit" every time would score 99.83% accuracy while catching zero fraud. This project is built around that challenge: handling class imbalance correctly, choosing the right evaluation metric, and explaining model decisions.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Storage | SQLite |
| Feature Engineering | SQL (window functions) |
| EDA & Modeling | Python, pandas, scikit-learn, XGBoost, SHAP |
| Visualization | matplotlib, seaborn, Plotly |
| Dashboard | Streamlit |

---

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions (Sept 2013, European cardholders)
- **Fraud cases:** 492 (0.172%)
- **Features:** V1–V28 (PCA-anonymized), Time, Amount, Class

---

## Project Structure

```
fraud-detection-project/
├── data/                        # Raw + processed data (not committed — see Setup)
├── docs/                        # EDA and SHAP chart outputs (PNG)
├── sql/
│   ├── 01_schema.sql            # Database schema and indexes
│   └── 02_feature_engineering.sql  # Engineered feature views
├── src/
│   ├── load_data.py             # Loads CSV into SQLite, runs SQL views
│   ├── eda.py                   # Exploratory analysis, saves charts
│   ├── train_model.py           # Trains models, evaluates, runs SHAP
│   └── dashboard.py             # Streamlit EDA dashboard
├── requirements.txt
└── README.md
```

---

## Pipeline

```
creditcard.csv
      │
      ▼
load_data.py  ──►  fraud.db  (SQLite database with engineered feature views)
      │
      ▼
eda.py  ──►  docs/*.png  +  data/model_features.csv
      │
      ▼
train_model.py  ──►  model results + docs/shap_*.png
      │
      ▼
dashboard.py  ──►  live Streamlit EDA dashboard (reads from fraud.db)
```

---

## SQL Feature Engineering

Two fraud-relevant features engineered directly in SQL before any Python modeling:

**1. Transaction Velocity** — rolling count and average amount of transactions in the trailing 1-hour window, computed using a SQL window function:
```sql
COUNT(*) OVER (
    ORDER BY time_seconds
    RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
) AS txn_count_last_hour
```

**2. High-Amount Flag** — marks transactions in the top 1% by amount using `PERCENT_RANK()`.

> An earlier version used correlated subqueries (one `SELECT COUNT(*)` per row), which was O(n²) and never finished on 284,807 rows. Rewriting as window functions over a single sorted pass reduced runtime from "never completes" to **5.7 seconds**.

---

## Modeling

Class imbalance handled via **class weighting** (`scale_pos_weight` in XGBoost), not synthetic oversampling — keeping every training example a real transaction.

**Evaluation metric: Precision-Recall AUC** — accuracy is not used since a "predict all legit" model scores 99.83% while catching zero fraud.

| Model | PR-AUC | Fraud Precision | Fraud Recall | False Alarms |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.754 | 0.06 | 0.92 | 1,434 |
| **XGBoost** | **0.869** | **0.85** | **0.84** | **14** |

XGBoost delivers a **+15.3% improvement in PR-AUC** over the baseline, catching 82 of 98 fraud cases with only 14 false alarms — compared to the baseline's 1,434 false alarms for a slightly higher recall.

---

## Explainability (SHAP)

SHAP (TreeExplainer) is used to verify model decisions are driven by real patterns:

- **Global feature importance** — V14 and V4 are the two strongest predictors, consistent with the independent correlation analysis from EDA (two methods agreeing increases confidence the result is real, not noise)
- **Per-transaction waterfall plot** — shows exactly which features pushed one specific real fraud case from a neutral baseline to a confident fraud prediction

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/vishvashah07/creditcardfrauddetection.git
cd creditcardfrauddetection

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Download creditcard.csv and place it in data/

# 5. Run the pipeline in order
python src/load_data.py       # builds data/fraud.db
python src/eda.py             # EDA charts → docs/, model_features.csv → data/
python src/train_model.py     # trains models, SHAP charts → docs/

# 6. Launch the dashboard
streamlit run src/dashboard.py
```

---

## EDA Highlights

| Finding | Insight |
|---|---|
| 0.172% fraud rate | Accuracy is useless as a metric |
| Fraud peaks at hour 2 AM (1.75% fraud rate) | Temporal signal worth engineering as a feature |
| V14 and V17 strongest negative correlators | Confirmed later by SHAP as top model features |
| Fraud transactions have lower median amount | Fraudsters tend toward smaller, less conspicuous amounts |

---

## Possible Extensions

- Threshold tuning based on a defined business cost ratio (missed fraud vs. false alarm cost)
- Hyperparameter tuning with Optuna or GridSearchCV
- Model persistence with joblib + FastAPI serving endpoint for real-time predictions
- Power BI dashboard connected via ODBC for stakeholder reporting