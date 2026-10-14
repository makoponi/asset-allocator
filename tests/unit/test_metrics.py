"""Unit tests for performance and risk metrics calculations."""

from __future__ import annotations

import numpy as np
import pytest

from src.execution.metrics import compute_drawdown_series, compute_performance_metrics


def test_compute_drawdown_series_known_values() -> None:
    # NAV sequence: 100 -> 120 -> 90 -> 110 -> 150 -> 120
    nav = [100.0, 120.0, 90.0, 110.0, 150.0, 120.0]
    dd_series, max_dd = compute_drawdown_series(nav)

    assert len(dd_series) == len(nav)
    # Peak at 120: drop to 90 is (90 - 120) / 120 = -0.25 (-25%)
    # Peak at 150: drop to 120 is (120 - 150) / 150 = -0.20 (-20%)
    # Max DD must be -0.25
    assert max_dd == pytest.approx(-0.25)
    assert dd_series[0] == pytest.approx(0.0)
    assert dd_series[1] == pytest.approx(0.0)
    assert dd_series[2] == pytest.approx(-0.25)
    assert dd_series[4] == pytest.approx(0.0)
    assert dd_series[5] == pytest.approx(-0.20)


def test_performance_metrics_positive_drift() -> None:
    # 252 days with constant daily return of +0.05% (~13.4% annual)
    daily_ret = 0.0005
    n_days = 252
    rets = [daily_ret] * n_days
    bench_rets = [0.0003] * n_days  # benchmark lower return

    metrics = compute_performance_metrics(
        daily_returns=rets,
        benchmark_returns=bench_rets,
        risk_free_rate=0.0,
        annual_turnover=1.5,
    )

    expected_total_ret = (1.0 + daily_ret) ** n_days - 1.0
    assert metrics.total_return == pytest.approx(expected_total_ret, abs=1e-4)
    assert metrics.cagr == pytest.approx(expected_total_ret, abs=1e-4)
    assert metrics.max_drawdown == pytest.approx(0.0, abs=1e-6)
    assert metrics.annualized_turnover == pytest.approx(1.5)


def test_performance_metrics_sharpe_and_beta() -> None:
    # Synthetic random walks with known covariance
    rng = np.random.default_rng(42)
    n = 504  # 2 years

    market = rng.normal(0.0004, 0.01, n)
    # Portfolio = 1.2 * market + idiosyncratic noise
    noise = rng.normal(0.0, 0.005, n)
    portfolio = 1.2 * market + noise

    metrics = compute_performance_metrics(
        daily_returns=portfolio,
        benchmark_returns=market,
        risk_free_rate=0.0,
    )

    assert metrics.beta is not None
    assert metrics.beta == pytest.approx(1.2, abs=0.1)
    assert metrics.annualized_volatility > 0.10
    assert metrics.sharpe_ratio != 0.0
    assert metrics.sortino_ratio != 0.0
    assert metrics.calmar_ratio >= 0.0
