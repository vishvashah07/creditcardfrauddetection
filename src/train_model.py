"""
train_model.py
---------------
Trains fraud detection models on the engineered feature set produced by
src/eda.py (data/model_features.csv), which itself came from the SQL
feature views in sql/02_feature_engineering.sql.

Day 3 plan (built incrementally):
  1. Load data + train/test split            <-- this step
  2. Baseline: Logistic Regression
  3. Main model: XGBoost with class weighting
  4. Evaluation: PR-AUC, confusion matrix, classification report
  5. SHAP explainability

Run from the project root:
    python src/train_model.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "model_features.csv"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42  # fixed seed so results are reproducible — important to mention in interviews


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run src/eda.py first — it saves "
            f"this file as its last step."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def split_data(df: pd.DataFrame):
    # Drop transaction_id (not a real feature, just a row identifier)
    feature_cols = [c for c in df.columns if c not in ("transaction_id", "class")]
    X = df[feature_cols]
    y = df["class"]

    # stratify=y is essential here: with only 492 fraud cases out of 284,807,
    # a non-stratified split could easily put very few (or disproportionately
    # many) fraud cases in the test set by chance, making evaluation unreliable.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"\nTrain set: {X_train.shape[0]:,} rows  |  fraud cases: {y_train.sum()} ({y_train.mean()*100:.3f}%)")
    print(f"Test set:  {X_test.shape[0]:,} rows  |  fraud cases: {y_test.sum()} ({y_test.mean()*100:.3f}%)")

    return X_train, X_test, y_train, y_test


def train_baseline(X_train, X_test, y_train, y_test):
    """
    Logistic Regression baseline.

    Why a baseline matters: it's the model you'd reach for in five
    minutes with no tuning. If XGBoost can't beat this meaningfully,
    the added complexity isn't justified — a real point worth making
    in an interview.

    Logistic Regression is sensitive to feature scale, so we standardize
    the inputs first (mean 0, std 1). Tree-based models like XGBoost
    don't need this, which is why scaling only happens here.

    class_weight='balanced' tells sklearn to automatically upweight the
    rare class (fraud) in the loss function, roughly in inverse
    proportion to its frequency — our chosen way of handling the
    284,315 : 492 imbalance without resampling the data.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Baseline — Logistic Regression")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)
    print(f"\nPR-AUC (Precision-Recall AUC): {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, scaler, pr_auc


def train_xgboost(X_train, X_test, y_train, y_test):
    """
    XGBoost — the main model for this project.

    Unlike Logistic Regression, tree-based models don't need scaled
    features, so we pass X_train/X_test in their original form.

    scale_pos_weight is XGBoost's version of class weighting: it
    upweights the positive class (fraud) in the loss function. The
    standard recommendation is (# negative / # positive) — here that's
    roughly 284,315 / 492 ~= 578, meaning the model is told a missed
    fraud case costs ~578x more than a false alarm on a legit one.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Main Model — XGBoost")
    print("=" * 60)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",  # optimize directly for PR-AUC during training
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)
    print(f"\nPR-AUC (Precision-Recall AUC): {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, pr_auc


def explain_with_shap(model, X_test, y_test):
    """
    SHAP (SHapley Additive exPlanations) explains *why* the model made
    each prediction, not just whether it was right. For fraud detection
    specifically, this matters beyond curiosity: a real bank legally
    needs to justify to a customer (or a regulator) why their card was
    flagged, and "the model said so" isn't an acceptable answer.

    We use a sample of the test set (not all 56,962 rows) because SHAP's
    exact computation cost grows with both rows and trees, and a sample
    of a few thousand is plenty to see stable, representative patterns.
    """
    print("\n" + "=" * 60)
    print("STEP 4: SHAP Explainability")
    print("=" * 60)

    # Sample for speed; TreeExplainer is fast but no need to run on all 56,962 rows
    sample_size = min(2000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # Global feature importance: which features matter most, on average,
    # across all predictions
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: docs/shap_summary.png")

    # Bar version: simpler, just average impact magnitude per feature —
    # easier to read at a glance for a non-technical audience
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "shap_importance_bar.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: docs/shap_importance_bar.png")

    # Force plot for one actual fraud case caught by the model — shows
    # exactly which features pushed that single prediction toward "fraud"
    fraud_indices = y_test[y_test == 1].index
    sample_fraud_idx = [i for i in fraud_indices if i in X_sample.index]
    if sample_fraud_idx:
        idx = sample_fraud_idx[0]
        row_position = X_sample.index.get_loc(idx)
        plt.figure()
        shap.plots.waterfall(shap_values[row_position], show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(DOCS_DIR / "shap_single_fraud_case.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved: docs/shap_single_fraud_case.png (explains transaction_id present in this sample)")
    else:
        print("No fraud cases landed in the SHAP sample — skipping single-case plot.")

    print("\nSHAP analysis complete.")


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print("\nStep 1 complete: data loaded and split.")

    baseline_model, scaler, baseline_pr_auc = train_baseline(X_train, X_test, y_train, y_test)
    print("\nStep 2 complete: baseline trained.")

    xgb_model, xgb_pr_auc = train_xgboost(X_train, X_test, y_train, y_test)
    print("\nStep 3 complete: XGBoost trained.")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"Logistic Regression PR-AUC: {baseline_pr_auc:.4f}")
    print(f"XGBoost PR-AUC:             {xgb_pr_auc:.4f}")
    improvement = ((xgb_pr_auc - baseline_pr_auc) / baseline_pr_auc) * 100
    print(f"Improvement: {improvement:+.1f}%")

    explain_with_shap(xgb_model, X_test, y_test)
    print("\nStep 4 complete: SHAP explainability done.")


if __name__ == "__main__":
    main()