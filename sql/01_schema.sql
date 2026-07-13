-- ============================================================
-- Credit Card Fraud Detection - Database Schema
-- ============================================================
-- Dataset: Kaggle "Credit Card Fraud Detection" (mlg-ulb)
-- 284,807 transactions | 492 frauds (0.172%) | Sept 2013, European cardholders
--
-- Note on columns:
-- V1-V28 are PCA-anonymized features (original features hidden for
-- confidentiality reasons by the dataset provider). 'Time' is seconds
-- elapsed since the first transaction. 'Amount' is transaction amount.
-- 'Class' = 1 for fraud, 0 for legit.
-- ============================================================

DROP TABLE IF EXISTS transactions;

CREATE TABLE transactions (
    transaction_id  INTEGER PRIMARY KEY AUTOINCREMENT,  -- surrogate key (synthetic, not in raw data)
    time_seconds     REAL NOT NULL,                       -- seconds since first transaction in dataset
    v1  REAL, v2  REAL, v3  REAL, v4  REAL, v5  REAL,
    v6  REAL, v7  REAL, v8  REAL, v9  REAL, v10 REAL,
    v11 REAL, v12 REAL, v13 REAL, v14 REAL, v15 REAL,
    v16 REAL, v17 REAL, v18 REAL, v19 REAL, v20 REAL,
    v21 REAL, v22 REAL, v23 REAL, v24 REAL, v25 REAL,
    v26 REAL, v27 REAL, v28 REAL,
    amount           REAL NOT NULL,
    class            INTEGER NOT NULL CHECK (class IN (0,1))  -- 1 = fraud, 0 = legit
);

-- Index on class for fast filtering of fraud vs legit (used constantly in EDA)
CREATE INDEX idx_transactions_class ON transactions(class);

-- Index on time for window-based queries (transaction velocity, time-of-day patterns)
CREATE INDEX idx_transactions_time ON transactions(time_seconds);
