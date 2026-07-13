"""
dashboard.py
------------
Streamlit dashboard for the Credit Card Fraud Detection project.
EDA only — class imbalance, transaction amount patterns, time-of-day
fraud rate, and feature correlation, all pulled live from fraud.db.

Run from the project root:
    streamlit run src/dashboard.py
"""

import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fraud.db"

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")


@st.cache_data
def load_summary_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM v_powerbi_summary", conn)
    conn.close()
    return df


@st.cache_data
def load_transactions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT amount, class FROM transactions", conn)
    conn.close()
    return df


@st.cache_data
def load_model_features():
    path = PROJECT_ROOT / "data" / "model_features.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def render_eda_tab():
    st.write("284,807 transactions, 492 fraud (0.17%) — a severely imbalanced classification problem.")

    txns = load_transactions()
    summary = load_summary_data()

    # ---- Row 1: Class imbalance + Amount distribution ----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Imbalance")
        class_counts = txns["class"].value_counts().rename({0: "Legit", 1: "Fraud"})
        fig = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            color=class_counts.index,
            color_discrete_map={"Legit": "#4C72B0", "Fraud": "#C44E52"},
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Fraud rate", f"{(txns['class'].mean() * 100):.3f}%")

    with col2:
        st.subheader("Transaction Amount: Fraud vs Legit")
        txns_capped = txns[txns["amount"] < 500].copy()
        txns_capped["class_label"] = txns_capped["class"].map({0: "Legit", 1: "Fraud"})
        fig = px.histogram(
            txns_capped, x="amount", color="class_label",
            histnorm="probability density", barmode="overlay", nbins=50,
            color_discrete_map={"Legit": "#4C72B0", "Fraud": "#C44E52"},
        )
        fig.update_layout(xaxis_title="Amount (capped at $500)", yaxis_title="Density")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Row 2: Hourly fraud rate ----
    st.subheader("Fraud Rate by Hour of Day")
    pivot = summary.pivot(index="hour_of_day", columns="class", values="num_transactions").fillna(0)
    pivot.columns = ["legit", "fraud"]
    pivot["fraud_rate_pct"] = (pivot["fraud"] / (pivot["legit"] + pivot["fraud"])) * 100
    pivot = pivot.reset_index()

    fig = go.Figure()
    fig.add_bar(x=pivot["hour_of_day"], y=pivot["legit"], name="Legit volume",
                marker_color="lightsteelblue", opacity=0.6, yaxis="y1")
    fig.add_scatter(x=pivot["hour_of_day"], y=pivot["fraud_rate_pct"], name="Fraud rate %",
                     mode="lines+markers", marker_color="#C44E52", yaxis="y2")
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis=dict(title="Legit transaction volume"),
        yaxis2=dict(title="Fraud rate (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)
    top_hour = pivot.sort_values("fraud_rate_pct", ascending=False).iloc[0]
    st.caption(f"Highest fraud rate: hour {int(top_hour['hour_of_day'])} ({top_hour['fraud_rate_pct']:.2f}%)")

    # ---- Row 3: Feature correlation ----
    st.subheader("Correlation of PCA Features with Fraud Label")
    df = load_model_features()
    if df is not None:
        v_cols = [c for c in df.columns if c.startswith("v") and c[1:].isdigit()]
        corr = df[v_cols + ["class"]].corr()["class"].drop("class").sort_values()
        fig = px.bar(
            x=corr.values, y=corr.index, orientation="h",
            color=corr.values, color_continuous_scale=["#C44E52", "#4C72B0"],
        )
        fig.update_layout(xaxis_title="Correlation coefficient", yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run src/eda.py first to generate model_features.csv for this chart.")


def main():
    st.title("Credit Card Fraud Detection")

    if not DB_PATH.exists():
        st.error(f"Could not find {DB_PATH}. Run src/load_data.py first.")
        return

    render_eda_tab()


if __name__ == "__main__":
    main()