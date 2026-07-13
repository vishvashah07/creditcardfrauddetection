"""
load_data.py
------------
Loads the Kaggle 'creditcard.csv' file into a SQLite database using the
schema defined in sql/01_schema.sql, then runs the feature engineering
SQL to create views used later by both the ML notebook and Power BI.

Usage:
    python src/load_data.py --csv data/creditcard.csv --db data/fraud.db
"""

import argparse
import sqlite3
import pandas as pd
from pathlib import Path


def load_schema(conn: sqlite3.Connection, schema_path: Path) -> None:
    with open(schema_path, "r") as f:
        conn.executescript(f.read())
    conn.commit()
    print(f"Schema loaded from {schema_path}")


def load_csv_into_db(conn: sqlite3.Connection, csv_path: Path) -> None:
    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Normalize column names to match schema (Kaggle file has 'Time', 'Amount', 'Class', 'V1'...'V28')
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"time": "time_seconds"})

    print(f"Loaded {len(df):,} rows. Inserting into database...")
    df.to_sql("transactions", conn, if_exists="append", index=False)
    conn.commit()
    print("Data inserted successfully.")


def load_feature_views(conn: sqlite3.Connection, sql_path: Path) -> None:
    with open(sql_path, "r") as f:
        conn.executescript(f.read())
    conn.commit()
    print(f"Feature views created from {sql_path}")


def sanity_check(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("SELECT class, COUNT(*) FROM transactions GROUP BY class")
    print("\nClass distribution check:")
    for row in cur.fetchall():
        label = "Fraud" if row[0] == 1 else "Legit"
        print(f"  {label} (class={row[0]}): {row[1]:,} transactions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/creditcard.csv")
    parser.add_argument("--db", default="data/fraud.db")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / args.csv
    db_path = project_root / args.db
    schema_path = project_root / "sql" / "01_schema.sql"
    features_path = project_root / "sql" / "02_feature_engineering.sql"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Download creditcard.csv from Kaggle "
            f"and place it in the data/ folder."
        )

    conn = sqlite3.connect(db_path)
    try:
        load_schema(conn, schema_path)
        load_csv_into_db(conn, csv_path)
        load_feature_views(conn, features_path)
        sanity_check(conn)
    finally:
        conn.close()

    print(f"\nDatabase ready at: {db_path}")


if __name__ == "__main__":
    main()
