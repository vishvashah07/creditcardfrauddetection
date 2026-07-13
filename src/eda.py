"""
eda.py
------
Exploratory Data Analysis for the Credit Card Fraud Detection project.

Connects to the SQLite database built in Day 1 (data/fraud.db), runs a
series of analyses, prints findings to the console, and saves every
chart as a PNG file in docs/ so they can be viewed afterward or used
in a portfolio writeup / README.

Run from the project root:
    python src/eda.py
"""

import sqlite3
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves files, doesn't try to pop up windows
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fraud.db"
DOCS_DIR = PROJECT_ROOT / "docs"
DATA_DIR = PROJECT_ROOT / "data"

DOCS_DIR.mkdir(exist_ok=True)


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DB_PATH}. Run src/load_data.py first to build the database."
        )

    conn = sqlite3.connect(DB_PATH)
    print(f"Connected to: {DB_PATH}")

    # ------------------------------------------------------------------
    # 1. Load data via the SQL feature view
    # ------------------------------------------------------------------
    section("1. Loading data from v_model_features view")
    df = pd.read_sql_query("SELECT * FROM v_model_features", conn)
    print(f"Shape: {df.shape}")
    print(df.head())

    missing = df.isnull().sum().sum()
    print(f"\nTotal missing values: {missing}")

    # ------------------------------------------------------------------
    # 2. Class imbalance
    # ------------------------------------------------------------------
    section("2. Class Imbalance")
    class_counts = df["class"].value_counts()
    class_pct = df["class"].value_counts(normalize=True) * 100
    print("Counts:\n", class_counts)
    print("\nPercentages:\n", class_pct.round(4))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=df, x="class", ax=ax[0])
    ax[0].set_title("Transaction Count by Class")
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(["Legit (0)", "Fraud (1)"])

    ax[1].pie(class_counts, labels=["Legit", "Fraud"], autopct="%1.3f%%",
              colors=["#4C72B0", "#C44E52"])
    ax[1].set_title("Class Distribution")
    plt.tight_layout()
    fig.savefig(DOCS_DIR / "class_imbalance.png", dpi=120)
    plt.close(fig)
    print(f"Saved: docs/class_imbalance.png")

    # ------------------------------------------------------------------
    # 3. Transaction amount: fraud vs legit
    # ------------------------------------------------------------------
    section("3. Transaction Amount: Fraud vs Legit")
    amount_stats = pd.read_sql_query("""
        SELECT class,
               COUNT(*) AS num_transactions,
               ROUND(AVG(amount), 2) AS avg_amount,
               ROUND(MIN(amount), 2) AS min_amount,
               ROUND(MAX(amount), 2) AS max_amount
        FROM transactions
        GROUP BY class
    """, conn)
    print(amount_stats)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x="class", y="amount", ax=ax[0])
    ax[0].set_yscale("log")
    ax[0].set_title("Transaction Amount by Class (log scale)")
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(["Legit", "Fraud"])

    sns.histplot(data=df[df["amount"] < 500], x="amount", hue="class",
                 bins=50, ax=ax[1], stat="density", common_norm=False)
    ax[1].set_title("Amount Distribution (under $500), normalized")
    plt.tight_layout()
    fig.savefig(DOCS_DIR / "amount_distribution.png", dpi=120)
    plt.close(fig)
    print(f"Saved: docs/amount_distribution.png")

    # ------------------------------------------------------------------
    # 4. Hour-of-day fraud pattern
    # ------------------------------------------------------------------
    section("4. Fraud Rate by Hour of Day")
    hourly = pd.read_sql_query("SELECT * FROM v_powerbi_summary ORDER BY hour_of_day, class", conn)

    pivot = hourly.pivot(index="hour_of_day", columns="class", values="num_transactions").fillna(0)
    pivot.columns = ["legit", "fraud"]
    pivot["fraud_rate_pct"] = (pivot["fraud"] / (pivot["legit"] + pivot["fraud"])) * 100
    print(pivot.sort_values("fraud_rate_pct", ascending=False).head())

    fig, ax = plt.subplots(figsize=(12, 5))
    ax2 = ax.twinx()
    ax.bar(pivot.index, pivot["legit"], alpha=0.3, label="Legit volume", color="steelblue")
    ax2.plot(pivot.index, pivot["fraud_rate_pct"], color="red", marker="o", label="Fraud rate %")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Legit transaction volume")
    ax2.set_ylabel("Fraud rate (%)")
    ax.set_title("Transaction Volume vs Fraud Rate by Hour of Day")
    fig.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(DOCS_DIR / "fraud_rate_by_hour.png", dpi=120)
    plt.close(fig)
    print(f"Saved: docs/fraud_rate_by_hour.png")

    # ------------------------------------------------------------------
    # 5. Engineered feature sanity check
    # ------------------------------------------------------------------
    section("5. Engineered SQL Features: Fraud vs Legit Averages")
    feature_check = df.groupby("class")[
        ["txn_count_last_hour", "avg_amount_last_hour", "is_high_amount"]
    ].mean().round(2)
    print(feature_check)

    # ------------------------------------------------------------------
    # 6. Correlation of PCA features with fraud label
    # ------------------------------------------------------------------
    section("6. Correlation of V1-V28 with Fraud Label")
    v_cols = [c for c in df.columns if c.startswith("v") and c[1:].isdigit()]
    corr_with_class = df[v_cols + ["class"]].corr()["class"].drop("class").sort_values()
    print(corr_with_class)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in corr_with_class]
    corr_with_class.plot(kind="barh", color=colors, ax=ax)
    ax.set_title("Correlation of PCA Features with Fraud Label")
    ax.set_xlabel("Correlation coefficient")
    plt.tight_layout()
    fig.savefig(DOCS_DIR / "feature_correlation.png", dpi=120)
    plt.close(fig)
    print(f"Saved: docs/feature_correlation.png")

    # ------------------------------------------------------------------
    # 7. Save cleaned feature set for modeling
    # ------------------------------------------------------------------
    section("7. Saving model-ready feature set")
    out_path = DATA_DIR / "model_features.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} — shape {df.shape}")

    conn.close()

    section("DONE — Summary of Findings")
    print("""
- Severe class imbalance: ~0.17% fraud -> accuracy is not a usable metric,
  use Precision-Recall AUC instead.
- Fraud transactions show a different amount distribution than legit ones.
- Fraud rate varies by hour of day - a real, explainable temporal signal.
- Several PCA components (V-features) show strong correlation with fraud.
- Engineered SQL features (velocity, high-amount flag) differ between classes.

Charts saved in docs/:
  - class_imbalance.png
  - amount_distribution.png
  - fraud_rate_by_hour.png
  - feature_correlation.png

Next step: src/train_model.py (XGBoost + Logistic Regression, PR-AUC, SHAP)
""")


if __name__ == "__main__":
    main()
