"""Unit tests for performance metrics, drawdown analytics, and overlay attribution."""

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.backtest.metrics import (
    build_full_performance_report,
    calculate_drawdown_series,
    calculate_performance_metrics,
)


def test_drawdown_series_calculation() -> None:
    """Test peak-to-trough drawdown calculation and max duration."""
    # NAV series: 100 -> 120 -> 90 -> 110
    # Peak at 120. At 90, dd = (90 - 120) / 120 = -0.25 (-25%)
    # At 110, dd = (110 - 120) / 120 = -0.0833
    nav = pd.Series([100.0, 120.0, 90.0, 110.0])
    max_dd, dd_series, max_dur = calculate_drawdown_series(nav)

    assert pytest.approx(max_dd, rel=1e-5) == -0.25
    assert pytest.approx(dd_series.iloc[2], rel=1e-5) == -0.25
    assert max_dur == 2


def test_calculate_performance_metrics() -> None:
    """Test annualized metrics on synthetic daily returns."""
    # 252 days of constant 0.05% daily return
    daily_ret = np.full(252, 0.0005)
    metrics = calculate_performance_metrics(daily_ret, risk_free_rate=0.0, annualization_days=252)

    expected_total = (1.0 + 0.0005) ** 252 - 1.0
    assert pytest.approx(metrics.total_return, rel=1e-5) == expected_total
    assert pytest.approx(metrics.cagr, rel=1e-5) == expected_total
    assert pytest.approx(metrics.annualized_volatility, abs=1e-6) == 0.0
    assert pytest.approx(metrics.max_drawdown, abs=1e-6) == 0.0
    assert metrics.win_rate == 1.0


def test_full_performance_report_construction() -> None:
    """Test end-to-end report and attribution on synthetic history ledger."""
    n_days = 100
    dates = pd.bdate_range(start="2024-01-01", periods=n_days)

    # Synthetic daily unhedged return: small positive with volatility
    np.random.seed(42)
    unhedged_ret = np.random.normal(loc=0.0004, scale=0.01, size=n_days)
    # Synthetic futures PnL mitigating 20% of drawdown
    fut_pnl = -0.5 * np.minimum(unhedged_ret, 0.0) * 10_000_000.0
    hedged_ret = unhedged_ret + (fut_pnl / 10_000_000.0)

    nav_hedged = 10_000_000.0 * np.cumprod(1.0 + hedged_ret)

    df = pd.DataFrame(
        {
            "date": dates,
            "unhedged_return": unhedged_ret,
            "hedged_return": hedged_ret,
            "nav_hedged": nav_hedged,
            "futures_daily_pnl_eur": fut_pnl,
            "futures_cost_eur": np.full(n_days, 10.0),
            "transaction_cost_eur": np.full(n_days, 25.0),
            "na_sleeve_return": unhedged_ret * 1.05,
            "eu_sleeve_return": unhedged_ret * 0.95,
            "na_weight": np.full(n_days, 0.60),
            "eu_weight": np.full(n_days, 0.40),
        }
    )

    report = build_full_performance_report(df, total_rebalances=4)

    assert report.total_rebalances == 4
    assert report.total_transaction_costs_eur == 25.0 * n_days
    assert report.overlay_attribution.hedged_cagr is not None
    assert report.sleeve_na.region_bloc == "NA"
    assert report.sleeve_eu.region_bloc == "EU"
