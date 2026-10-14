"""Unit tests for MLRanker, Spearman Rank IC, and top decile selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.ml_ranker import MLRanker


def test_ml_ranker_synthetic_learning_and_ic() -> None:
    # Generate synthetic dataset with 100 stocks across 3 dates
    # Feature mom_12m is intentionally strongly correlated with forward target
    rng = np.random.default_rng(42)
    n = 300

    dates = ["2023-01-31"] * 100 + ["2023-02-28"] * 100 + ["2023-03-31"] * 100
    tickers = [f"STK_{i % 100:03d}" for i in range(n)]

    mom_12m = rng.standard_normal(n)
    noise = rng.standard_normal(n) * 0.2
    # target has high positive correlation with mom_12m
    target = 0.8 * mom_12m + noise

    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": tickers,
            "mom_1m": rng.standard_normal(n),
            "mom_12m": mom_12m,
            "beta_252d": np.ones(n),
            "ivol_252d": np.ones(n) * 0.2,
            "pe_ratio": np.ones(n) * 15.0,
            "best_pe_ratio": np.ones(n) * 14.0,
            "px_to_book": np.ones(n) * 2.0,
            "fcf_yield": np.ones(n) * 0.05,
            "roe": np.ones(n) * 0.12,
            "prof_margin": np.ones(n) * 0.10,
            "oper_margin": np.ones(n) * 0.15,
            "debt_to_equity": np.ones(n) * 0.5,
            "quick_ratio": np.ones(n) * 1.2,
            "free_cash_flow": np.ones(n) * 1e6,
        }
    )

    ranker = MLRanker(model_type="lightgbm", random_state=42)
    ranker.fit(df, target)

    # 1. Overall Rank IC must be positive and strong
    ic = ranker.compute_rank_ic(df, target)
    assert ic > 0.50

    # 2. Mean cross-sectional IC
    df["prediction"] = ranker.predict(df)
    df["target"] = target
    mean_ic = MLRanker.compute_mean_cross_sectional_ic(df)
    assert mean_ic > 0.50


def test_ml_ranker_select_top_decile_cardinality() -> None:
    # 500 stocks on a single evaluation date
    n = 500
    tickers = [f"T_{i:03d}" for i in range(n)]
    df_t = pd.DataFrame(
        {
            "date": ["2024-01-31"] * n,
            "ticker": tickers,
            "mom_1m": np.linspace(-1.0, 1.0, n),
            "mom_12m": np.linspace(-2.0, 2.0, n),
            "beta_252d": np.ones(n),
            "ivol_252d": np.ones(n) * 0.2,
            "pe_ratio": np.ones(n) * 15.0,
            "best_pe_ratio": np.ones(n) * 14.0,
            "px_to_book": np.ones(n) * 2.0,
            "fcf_yield": np.ones(n) * 0.05,
            "roe": np.ones(n) * 0.12,
            "prof_margin": np.ones(n) * 0.10,
            "oper_margin": np.ones(n) * 0.15,
            "debt_to_equity": np.ones(n) * 0.5,
            "quick_ratio": np.ones(n) * 1.2,
            "free_cash_flow": np.ones(n) * 1e6,
        }
    )

    ranker = MLRanker(model_type="lightgbm", random_state=42)
    # Target linearly proportional to mom_12m
    ranker.fit(df_t, df_t["mom_12m"])

    # 10% of 500 stocks = 50 stocks
    top_stocks = ranker.select_top_decile(df_t, top_pct=0.10, min_n=35, max_n=60)
    assert len(top_stocks) == 50
    # Stocks with highest mom_12m should be in the top decile
    assert "T_499" in top_stocks
    assert "T_000" not in top_stocks
