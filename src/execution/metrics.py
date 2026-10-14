"""Quantitative performance and risk metrics engine.

Computes:
- Total Return, CAGR
- Annualized Volatility
- Sharpe Ratio, Sortino Ratio
- Maximum Drawdown, Calmar Ratio
- Beta, Jensen's Alpha, Tracking Error, Information Ratio
- Annualized Turnover
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Comprehensive portfolio performance and risk metrics container."""

    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    beta: float | None = None
    alpha: float | None = None
    tracking_error: float | None = None
    information_ratio: float | None = None
    annualized_turnover: float = 0.0


def compute_drawdown_series(
    nav_series: Sequence[float] | np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], float]:
    """Calculate the drawdown series and maximum drawdown from a NAV curve."""
    nav = np.asarray(nav_series, dtype=float)
    if len(nav) == 0:
        return np.array([]), 0.0

    running_max = np.maximum.accumulate(nav)
    running_max = np.where(running_max <= 0.0, 1e-12, running_max)
    drawdowns = (nav - running_max) / running_max
    max_dd = float(np.min(drawdowns))
    return drawdowns, max_dd


def compute_performance_metrics(
    daily_returns: Sequence[float] | np.ndarray[Any, Any],
    benchmark_returns: Sequence[float] | np.ndarray[Any, Any] | None = None,
    risk_free_rate: float = 0.0,
    annual_turnover: float = 0.0,
) -> PerformanceMetrics:
    """Calculate standardized financial risk and return metrics from daily percentage returns.

    Parameters
    ----------
    daily_returns : Sequence[float]
        Series of daily portfolio returns.
    benchmark_returns : Sequence[float], optional
        Synchronized daily returns of the benchmark (e.g. STOXX 600).
    risk_free_rate : float
        Annualized risk-free rate (default: 0.0).
    annual_turnover : float
        Estimated or realized annualized one-way turnover.

    Returns
    -------
    PerformanceMetrics
        Object containing all evaluated metrics.
    """
    rets = np.asarray(daily_returns, dtype=float)
    n_days = len(rets)

    if n_days < 2:
        return PerformanceMetrics(
            total_return=0.0,
            cagr=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
        )

    # 1. Total return and CAGR
    nav = np.cumprod(1.0 + rets)
    total_ret = float(nav[-1] - 1.0)
    years = n_days / 252.0
    cagr = float((nav[-1]) ** (1.0 / years) - 1.0) if years > 0 and nav[-1] > 0 else total_ret

    # 2. Annualized Volatility
    vol_daily = float(np.std(rets, ddof=1))
    vol_ann = vol_daily * math.sqrt(252.0)

    # 3. Sharpe Ratio
    rf_daily = risk_free_rate / 252.0
    excess_daily = rets - rf_daily
    mean_excess_ann = float(np.mean(excess_daily)) * 252.0
    sharpe = mean_excess_ann / vol_ann if vol_ann > 1e-12 else 0.0

    # 4. Sortino Ratio (downside deviation below risk-free)
    downside_rets = np.minimum(0.0, excess_daily)
    downside_vol_daily = float(np.sqrt(np.mean(downside_rets**2)))
    downside_vol_ann = downside_vol_daily * math.sqrt(252.0)
    sortino = mean_excess_ann / downside_vol_ann if downside_vol_ann > 1e-12 else 0.0

    # 5. Drawdown and Calmar
    _dd_series, max_dd = compute_drawdown_series(nav)
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0

    # 6. Benchmark-relative metrics (Alpha, Beta, Tracking Error, Information Ratio)
    beta = None
    alpha = None
    tracking_error = None
    info_ratio = None

    if benchmark_returns is not None and len(benchmark_returns) == n_days:
        b_rets = np.asarray(benchmark_returns, dtype=float)
        var_b = float(np.var(b_rets, ddof=1))

        if var_b > 1e-12:
            cov_pb = float(np.cov(rets, b_rets)[0, 1])
            beta_val = cov_pb / var_b
            beta = float(beta_val)

            # Benchmark CAGR
            b_nav = np.cumprod(1.0 + b_rets)
            b_cagr = (
                float((b_nav[-1]) ** (1.0 / years) - 1.0)
                if years > 0 and b_nav[-1] > 0
                else float(b_nav[-1] - 1.0)
            )

            # Jensen's Alpha (annualized)
            alpha = float(cagr - (risk_free_rate + beta_val * (b_cagr - risk_free_rate)))

            # Tracking error and Information Ratio
            diff = rets - b_rets
            diff_vol_ann = float(np.std(diff, ddof=1)) * math.sqrt(252.0)
            tracking_error = float(diff_vol_ann)
            info_ratio = float((cagr - b_cagr) / diff_vol_ann) if diff_vol_ann > 1e-12 else 0.0

    return PerformanceMetrics(
        total_return=total_ret,
        cagr=cagr,
        annualized_volatility=vol_ann,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        beta=beta,
        alpha=alpha,
        tracking_error=tracking_error,
        information_ratio=info_ratio,
        annualized_turnover=annual_turnover,
    )
