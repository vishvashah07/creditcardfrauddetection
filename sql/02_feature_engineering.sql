-- ============================================================
-- Credit Card Fraud Detection - Feature Engineering & EDA Queries
-- ============================================================
-- These queries do two jobs:
--   1. Generate engineered features for the ML model (saved as a view)
--   2. Power exploratory analysis that also feeds the Power BI dashboard
-- ============================================================

-- Drop any existing views first. CREATE VIEW IF NOT EXISTS is a no-op
-- if a view by that name already exists -- which silently keeps an old
-- (and in this project's history, once-broken/slow) definition around
-- even after this file is edited. Dropping first guarantees the views
-- below always match what's written in this file.
DROP VIEW IF EXISTS v_model_features;
DROP VIEW IF EXISTS v_transaction_velocity;
DROP VIEW IF EXISTS v_high_amount_flag;
DROP VIEW IF EXISTS v_powerbi_summary;


-- ------------------------------------------------------------
-- 1. Class balance check (the famous 0.17% fraud rate)
-- ------------------------------------------------------------
SELECT
    class,
    COUNT(*) AS num_transactions,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM transactions), 4) AS pct_of_total
FROM transactions
GROUP BY class;


-- ------------------------------------------------------------
-- 2. Transaction amount stats: fraud vs legit
-- ------------------------------------------------------------
SELECT
    class,
    COUNT(*)            AS num_transactions,
    ROUND(AVG(amount), 2)  AS avg_amount,
    ROUND(MIN(amount), 2)  AS min_amount,
    ROUND(MAX(amount), 2)  AS max_amount,
    -- Standard deviation isn't built into SQLite, computed manually if needed in Python instead
    ROUND(SUM(amount), 2)  AS total_amount
FROM transactions
GROUP BY class;


-- ------------------------------------------------------------
-- 3. Time-bucketed view: hour-of-day pattern
-- ------------------------------------------------------------
-- 'time_seconds' is seconds elapsed since first transaction, spanning ~2 days.
-- We bucket into hour-of-day (mod 24) to find risky hours.
SELECT
    CAST(time_seconds / 3600 AS INTEGER) % 24 AS hour_of_day,
    class,
    COUNT(*) AS num_transactions
FROM transactions
GROUP BY hour_of_day, class
ORDER BY hour_of_day, class;


-- ------------------------------------------------------------
-- 4-6. Final feature view for ML: all engineered features in ONE view
-- ------------------------------------------------------------
-- This combines what used to be three separate views (transaction
-- velocity, high-amount flag, and a join across them) into a single
-- view. All window functions run over 'transactions' directly in one
-- pass; joining separately materialized views on transaction_id caused
-- SQLite's planner to re-run per-row lookups across views (visible as
-- repeated "CORRELATED SCALAR SUBQUERY" / per-row rowid searches in
-- EXPLAIN QUERY PLAN), which made the join effectively O(n^2).
--
-- Feature 1: txn_count_last_hour / avg_amount_last_hour
--   "Transaction velocity" -- is this moment unusually busy compared to
--   the preceding hour? A classic fraud signal. Computed as a rolling
--   window over time_seconds (no card/account ID exists in this
--   anonymized dataset, so this approximates velocity at the
--   transaction-stream level rather than per-card).
--
-- Feature 2: is_high_amount
--   Flags transactions in the top 1% by amount, via PERCENT_RANK()
--   computed once over the whole table.
CREATE VIEW IF NOT EXISTS v_model_features AS
SELECT
    transaction_id,
    time_seconds,
    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
    v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
    v21, v22, v23, v24, v25, v26, v27, v28,
    amount,
    COUNT(*) OVER (
        ORDER BY time_seconds
        RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
    ) AS txn_count_last_hour,
    ROUND(AVG(amount) OVER (
        ORDER BY time_seconds
        RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
    ), 2) AS avg_amount_last_hour,
    CASE
        WHEN PERCENT_RANK() OVER (ORDER BY amount) >= 0.99 THEN 1
        ELSE 0
    END AS is_high_amount,
    class
FROM transactions;


-- Standalone views below are kept for readability / ad-hoc exploration
-- (e.g. "what does just the velocity feature look like on its own?"),
-- but the model and EDA scripts query v_model_features above, not these.

CREATE VIEW IF NOT EXISTS v_transaction_velocity AS
SELECT
    transaction_id,
    time_seconds,
    amount,
    class,
    COUNT(*) OVER (
        ORDER BY time_seconds
        RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
    ) AS txn_count_last_hour,
    ROUND(AVG(amount) OVER (
        ORDER BY time_seconds
        RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
    ), 2) AS avg_amount_last_hour
FROM transactions;

CREATE VIEW IF NOT EXISTS v_high_amount_flag AS
SELECT
    transaction_id,
    amount,
    class,
    CASE WHEN amount_percentile >= 0.99 THEN 1 ELSE 0 END AS is_high_amount
FROM (
    SELECT
        transaction_id,
        amount,
        class,
        PERCENT_RANK() OVER (ORDER BY amount) AS amount_percentile
    FROM transactions
);


-- ------------------------------------------------------------
-- 7. Query for Power BI: daily/hourly fraud summary (lightweight, pre-aggregated)
-- ------------------------------------------------------------
-- Power BI will connect directly to this view rather than the raw 284K-row
-- table, so the dashboard stays fast.
CREATE VIEW IF NOT EXISTS v_powerbi_summary AS
SELECT
    CAST(time_seconds / 3600 AS INTEGER) % 24 AS hour_of_day,
    class,
    COUNT(*) AS num_transactions,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(AVG(amount), 2) AS avg_amount
FROM transactions
GROUP BY hour_of_day, class;
