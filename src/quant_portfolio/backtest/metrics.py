"""Quantitative risk/return metrics, drawdown analytics, and alpha decomposition."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant_portfolio.config import BacktestConfig, PortfolioConfig

logger = logging.getLogger(__name__)


class PerformanceMetrics(BaseModel):
    """Core annualized risk and performance metrics."""

    total_return: float
    cagr: float
    annualized_volatility: float = Field(ge=0.0)
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float = Field(le=0.0)
    calmar_ratio: float
    win_rate: float = Field(ge=0.0, le=1.0)
    profit_factor: float = Field(ge=0.0)


class SleeveMetrics(BaseModel):
    """Regional sleeve performance metrics."""

    region_bloc: str
    total_return: float
    cagr: float
    annualized_volatility: float = Field(ge=0.0)
    sharpe_ratio: float
    average_weight: float = Field(ge=0.0, le=1.0)


class HedgeOverlayAttribution(BaseModel):
    """Attribution of futures hedging overlay vs unhedged baseline."""

    unhedged_cagr: float
    hedged_cagr: float
    cagr_delta: float
    unhedged_volatility: float = Field(ge=0.0)
    hedged_volatility: float = Field(ge=0.0)
    volatility_reduction: float
    unhedged_max_dd: float = Field(le=0.0)
    hedged_max_dd: float = Field(le=0.0)
    max_dd_mitigation: float
    total_futures_pnl_eur: float
    total_futures_cost_eur: float
    net_futures_contribution_eur: float


class FullPerformanceReport(BaseModel):
    """Complete quantitative performance dossier for Walk-Forward simulation."""

    unhedged_metrics: PerformanceMetrics
    hedged_metrics: PerformanceMetrics
    overlay_attribution: HedgeOverlayAttribution
    sleeve_na: SleeveMetrics
    sleeve_eu: SleeveMetrics
    total_rebalances: int
    total_transaction_costs_eur: float
    max_drawdown_duration_days: int


def calculate_drawdown_series(nav_series: pd.Series) -> tuple[float, pd.Series, int]:
    """Compute peak-to-trough drawdowns, maximum drawdown, and max duration.

    Parameters
    ----------
    nav_series : pd.Series
        Series of daily portfolio NAV or cumulative values.

    Returns
    -------
    tuple[float, pd.Series, int]
        (max_drawdown, drawdown_series, max_drawdown_duration_days)
    """
    if nav_series.empty:
        return 0.0, pd.Series(dtype=np.float64), 0

    values = nav_series.to_numpy(dtype=np.float64)
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max

    max_dd = float(np.min(drawdowns))

    # Calculate max duration under previous peak
    duration = 0
    max_duration = 0
    for dd in drawdowns:
        if dd < -1e-6:
            duration += 1
            if duration > max_duration:
                max_duration = duration
        else:
            duration = 0

    dd_series = pd.Series(drawdowns, index=nav_series.index, name="drawdown")
    return max_dd, dd_series, max_duration


def calculate_performance_metrics(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_days: int = 252,
) -> PerformanceMetrics:
    """Calculate annualized performance, risk, and tail ratios from daily returns.

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Daily percentage returns.
    risk_free_rate : float
        Annual risk-free rate (e.g. 0.02 for 2%).
    annualization_days : int
        Business days per year (default: 252).

    Returns
    -------
    PerformanceMetrics
        Standardized quantitative metrics.
    """
    arr = np.asarray(returns, dtype=np.float64)
    # Remove NaNs
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return PerformanceMetrics(
            total_return=0.0,
            cagr=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            win_rate=0.0,
            profit_factor=0.0,
        )

    n_days = len(arr)
    cum_return = float(np.prod(1.0 + arr) - 1.0)

    # CAGR: (1 + R)^(252 / N) - 1
    if 1.0 + cum_return > 0 and n_days > 0:
        cagr = float((1.0 + cum_return) ** (annualization_days / n_days) - 1.0)
    else:
        cagr = -1.0

    # Daily std & annualized vol
    daily_std = float(np.std(arr, ddof=1)) if n_days > 1 else 0.0
    ann_vol = daily_std * np.sqrt(annualization_days) if daily_std > 1e-12 else 0.0

    # Sharpe ratio
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / annualization_days) - 1.0
    excess_returns = arr - daily_rf
    mean_excess = float(np.mean(excess_returns))
    sharpe = (mean_excess / daily_std * np.sqrt(annualization_days)) if daily_std > 1e-12 else 0.0

    # Downside deviation for Sortino
    downside_diff = np.minimum(excess_returns, 0.0)
    downside_variance = float(np.mean(downside_diff**2))
    downside_std = np.sqrt(downside_variance) * np.sqrt(annualization_days)
    sortino = ((cagr - risk_free_rate) / downside_std) if downside_std > 1e-12 else 0.0

    # Max Drawdown & Calmar
    nav = np.cumprod(1.0 + arr)
    max_dd, _, _ = calculate_drawdown_series(pd.Series(nav))
    calmar = (cagr / abs(max_dd)) if abs(max_dd) > 1e-6 else 0.0

    # Win rate & Profit factor
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    win_rate = float(len(wins) / n_days) if n_days > 0 else 0.0

    sum_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
    sum_losses = float(np.sum(np.abs(losses))) if len(losses) > 0 else 0.0
    profit_factor = sum_wins / sum_losses if sum_losses > 1e-12 else (99.0 if sum_wins > 0 else 0.0)

    return PerformanceMetrics(
        total_return=cum_return,
        cagr=cagr,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
    )


def compute_sleeve_metrics(
    daily_sleeve_returns: pd.Series,
    average_weight: float,
    region_bloc: str,
    annualization_days: int = 252,
) -> SleeveMetrics:
    """Compute performance metrics for an isolated regional sleeve."""
    metrics = calculate_performance_metrics(
        returns=daily_sleeve_returns,
        annualization_days=annualization_days,
    )
    return SleeveMetrics(
        region_bloc=region_bloc,
        total_return=metrics.total_return,
        cagr=metrics.cagr,
        annualized_volatility=metrics.annualized_volatility,
        sharpe_ratio=metrics.sharpe_ratio,
        average_weight=average_weight,
    )


def build_full_performance_report(
    daily_history_df: pd.DataFrame,
    total_rebalances: int = 0,
    config: PortfolioConfig | None = None,
) -> FullPerformanceReport:
    """Construct complete performance report from backtest daily ledger.

    Expected columns in daily_history_df:
    - `unhedged_return`: daily return of long-only equity portfolio
    - `hedged_return`: daily return of portfolio including futures overlay
    - `nav_hedged`: daily hedged portfolio NAV
    - `futures_daily_pnl_eur`: daily P&L from futures contracts
    - `transaction_cost_eur`: daily transaction costs deducted
    - Optional: `na_sleeve_return`, `eu_sleeve_return`, `na_weight`, `eu_weight`
    """
    cfg = config or PortfolioConfig()
    bt_cfg: BacktestConfig = cfg.backtest or BacktestConfig()
    ann_days = bt_cfg.annualization_days
    rf = bt_cfg.risk_free_rate

    unhedged_ret = daily_history_df["unhedged_return"]
    hedged_ret = daily_history_df["hedged_return"]

    m_unhedged = calculate_performance_metrics(
        unhedged_ret, risk_free_rate=rf, annualization_days=ann_days
    )
    m_hedged = calculate_performance_metrics(
        hedged_ret, risk_free_rate=rf, annualization_days=ann_days
    )

    _, _, max_dur = calculate_drawdown_series(daily_history_df["nav_hedged"])

    total_costs = float(daily_history_df.get("transaction_cost_eur", pd.Series([0.0])).sum())
    total_fut_pnl = float(daily_history_df.get("futures_daily_pnl_eur", pd.Series([0.0])).sum())
    fut_costs = float(daily_history_df.get("futures_cost_eur", pd.Series([0.0])).sum())

    attribution = HedgeOverlayAttribution(
        unhedged_cagr=m_unhedged.cagr,
        hedged_cagr=m_hedged.cagr,
        cagr_delta=m_hedged.cagr - m_unhedged.cagr,
        unhedged_volatility=m_unhedged.annualized_volatility,
        hedged_volatility=m_hedged.annualized_volatility,
        volatility_reduction=m_unhedged.annualized_volatility - m_hedged.annualized_volatility,
        unhedged_max_dd=m_unhedged.max_drawdown,
        hedged_max_dd=m_hedged.max_drawdown,
        max_dd_mitigation=abs(m_unhedged.max_drawdown) - abs(m_hedged.max_drawdown),
        total_futures_pnl_eur=total_fut_pnl,
        total_futures_cost_eur=fut_costs,
        net_futures_contribution_eur=total_fut_pnl - fut_costs,
    )

    # Regional sleeves
    na_ret = daily_history_df.get("na_sleeve_return", unhedged_ret)
    eu_ret = daily_history_df.get("eu_sleeve_return", unhedged_ret)
    avg_na_w = float(daily_history_df.get("na_weight", pd.Series([0.60])).mean())
    avg_eu_w = float(daily_history_df.get("eu_weight", pd.Series([0.40])).mean())

    sleeve_na = compute_sleeve_metrics(na_ret, avg_na_w, "NA", annualization_days=ann_days)
    sleeve_eu = compute_sleeve_metrics(eu_ret, avg_eu_w, "EU", annualization_days=ann_days)

    return FullPerformanceReport(
        unhedged_metrics=m_unhedged,
        hedged_metrics=m_hedged,
        overlay_attribution=attribution,
        sleeve_na=sleeve_na,
        sleeve_eu=sleeve_eu,
        total_rebalances=total_rebalances,
        total_transaction_costs_eur=total_costs,
        max_drawdown_duration_days=max_dur,
    )
