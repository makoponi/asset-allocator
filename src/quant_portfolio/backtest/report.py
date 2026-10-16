"""Text reporting, ASCII tables (CP1252-safe), and CSV export for backtest results."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from quant_portfolio.backtest.metrics import FullPerformanceReport

logger = logging.getLogger(__name__)


def format_performance_summary_table(report: FullPerformanceReport) -> str:
    """Format quantitative performance report into ASCII table safe for Windows CP1252 console."""
    u = report.unhedged_metrics
    h = report.hedged_metrics
    att = report.overlay_attribution

    u_tot = f"{u.total_return * 100:.2f}%"
    h_tot = f"{h.total_return * 100:.2f}%"
    u_cagr = f"{u.cagr * 100:.2f}%"
    h_cagr = f"{h.cagr * 100:.2f}%"
    u_vol = f"{u.annualized_volatility * 100:.2f}%"
    h_vol = f"{h.annualized_volatility * 100:.2f}%"
    u_sh = f"{u.sharpe_ratio:.2f}"
    h_sh = f"{h.sharpe_ratio:.2f}"
    u_so = f"{u.sortino_ratio:.2f}"
    h_so = f"{h.sortino_ratio:.2f}"
    u_mdd = f"{u.max_drawdown * 100:.2f}%"
    h_mdd = f"{h.max_drawdown * 100:.2f}%"
    u_cal = f"{u.calmar_ratio:.2f}"
    h_cal = f"{h.calmar_ratio:.2f}"
    u_win = f"{u.win_rate * 100:.2f}%"
    h_win = f"{h.win_rate * 100:.2f}%"
    u_pf = f"{u.profit_factor:.2f}"
    h_pf = f"{h.profit_factor:.2f}"

    lines: list[str] = [
        "=" * 80,
        "                     TRANSATLANTIC QUANT PORTFOLIO - BACKTEST DOSSIER           ",
        "=" * 80,
        f" Total Rebalancing Events:      {report.total_rebalances}",
        f" Total Transaction Costs:       {report.total_transaction_costs_eur:,.2f} EUR",
        f" Max Drawdown Duration:         {report.max_drawdown_duration_days} business days",
        "-" * 80,
        f" {'Metric':<30} | {'Unhedged (Long-Only)':<22} | {'Hedged (Overlay)':<20}",
        "-" * 80,
        f" {'Total Return':<30} | {u_tot:>22} | {h_tot:>20}",
        f" {'CAGR (Annualized Return)':<30} | {u_cagr:>22} | {h_cagr:>20}",
        f" {'Annualized Volatility':<30} | {u_vol:>22} | {h_vol:>20}",
        f" {'Sharpe Ratio (Rf=0%)':<30} | {u_sh:>22} | {h_sh:>20}",
        f" {'Sortino Ratio':<30} | {u_so:>22} | {h_so:>20}",
        f" {'Max Drawdown':<30} | {u_mdd:>22} | {h_mdd:>20}",
        f" {'Calmar Ratio':<30} | {u_cal:>22} | {h_cal:>20}",
        f" {'Win Rate':<30} | {u_win:>22} | {h_win:>20}",
        f" {'Profit Factor':<30} | {u_pf:>22} | {h_pf:>20}",
        "-" * 80,
        "                           FUTURES HEDGING ATTRIBUTION                          ",
        "-" * 80,
        f" Volatility Reduction:          {att.volatility_reduction * 100:>+.2f}%",
        f" Max Drawdown Mitigation:       {att.max_dd_mitigation * 100:>+.2f}%",
        f" Gross Futures P&L:             {att.total_futures_pnl_eur:>+15,.2f} EUR",
        f" Futures Execution/Roll Costs:  {att.total_futures_cost_eur:>15,.2f} EUR",
        f" Net Overlay Contribution:      {att.net_futures_contribution_eur:>+15,.2f} EUR",
        "-" * 80,
        "                           REGIONAL SLEEVE BREAKDOWN                            ",
        "-" * 80,
        f" NA Sleeve (Avg Wgt: {report.sleeve_na.average_weight * 100:.1f}%): "
        f"CAGR = {report.sleeve_na.cagr * 100:.2f}%, "
        f"Vol = {report.sleeve_na.annualized_volatility * 100:.2f}%, "
        f"Sharpe = {report.sleeve_na.sharpe_ratio:.2f}",
        f" EU Sleeve (Avg Wgt: {report.sleeve_eu.average_weight * 100:.1f}%): "
        f"CAGR = {report.sleeve_eu.cagr * 100:.2f}%, "
        f"Vol = {report.sleeve_eu.annualized_volatility * 100:.2f}%, "
        f"Sharpe = {report.sleeve_eu.sharpe_ratio:.2f}",
        "=" * 80,
    ]
    return "\n".join(lines)


def export_report_to_dataframes(
    report: FullPerformanceReport,
    daily_history_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Convert performance report and history ledger into exportable DataFrames."""
    u = report.unhedged_metrics
    h = report.hedged_metrics
    att = report.overlay_attribution

    summary_records = [
        {"metric": "Total Return", "unhedged": u.total_return, "hedged": h.total_return},
        {"metric": "CAGR", "unhedged": u.cagr, "hedged": h.cagr},
        {
            "metric": "Annualized Volatility",
            "unhedged": u.annualized_volatility,
            "hedged": h.annualized_volatility,
        },
        {"metric": "Sharpe Ratio", "unhedged": u.sharpe_ratio, "hedged": h.sharpe_ratio},
        {"metric": "Sortino Ratio", "unhedged": u.sortino_ratio, "hedged": h.sortino_ratio},
        {"metric": "Max Drawdown", "unhedged": u.max_drawdown, "hedged": h.max_drawdown},
        {"metric": "Calmar Ratio", "unhedged": u.calmar_ratio, "hedged": h.calmar_ratio},
        {"metric": "Win Rate", "unhedged": u.win_rate, "hedged": h.win_rate},
        {"metric": "Profit Factor", "unhedged": u.profit_factor, "hedged": h.profit_factor},
    ]
    summary_df = pd.DataFrame(summary_records)

    attribution_records = [
        {"item": "Volatility Reduction", "value": att.volatility_reduction},
        {"item": "Max Drawdown Mitigation", "value": att.max_dd_mitigation},
        {"item": "Gross Futures PnL EUR", "value": att.total_futures_pnl_eur},
        {"item": "Futures Costs EUR", "value": att.total_futures_cost_eur},
        {"item": "Net Futures Contribution EUR", "value": att.net_futures_contribution_eur},
    ]
    attribution_df = pd.DataFrame(attribution_records)

    sleeve_records = [
        {
            "region_bloc": "NA",
            "cagr": report.sleeve_na.cagr,
            "volatility": report.sleeve_na.annualized_volatility,
            "sharpe": report.sleeve_na.sharpe_ratio,
            "average_weight": report.sleeve_na.average_weight,
        },
        {
            "region_bloc": "EU",
            "cagr": report.sleeve_eu.cagr,
            "volatility": report.sleeve_eu.annualized_volatility,
            "sharpe": report.sleeve_eu.sharpe_ratio,
            "average_weight": report.sleeve_eu.average_weight,
        },
    ]
    sleeves_df = pd.DataFrame(sleeve_records)

    result: dict[str, pd.DataFrame] = {
        "summary": summary_df,
        "attribution": attribution_df,
        "sleeves": sleeves_df,
    }

    if daily_history_df is not None and not daily_history_df.empty:
        result["daily_history"] = daily_history_df.copy()

    return result
