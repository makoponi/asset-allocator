"""Unit tests for ASCII report formatting and DataFrame export."""

import numpy as np
import pandas as pd

from quant_portfolio.backtest.metrics import build_full_performance_report
from quant_portfolio.backtest.report import (
    export_report_to_dataframes,
    format_performance_summary_table,
)


def _build_mock_report() -> tuple[object, pd.DataFrame]:
    """Create a synthetic report and ledger for testing."""
    n_days = 50
    dates = pd.bdate_range(start="2024-01-01", periods=n_days)
    unhedged_ret = np.full(n_days, 0.0005)
    hedged_ret = np.full(n_days, 0.0004)
    nav_hedged = 10_000_000.0 * np.cumprod(1.0 + hedged_ret)

    df = pd.DataFrame(
        {
            "date": dates,
            "unhedged_return": unhedged_ret,
            "hedged_return": hedged_ret,
            "nav_hedged": nav_hedged,
            "futures_daily_pnl_eur": np.zeros(n_days),
            "futures_cost_eur": np.full(n_days, 5.0),
            "transaction_cost_eur": np.full(n_days, 15.0),
            "na_sleeve_return": unhedged_ret,
            "eu_sleeve_return": unhedged_ret,
            "na_weight": np.full(n_days, 0.60),
            "eu_weight": np.full(n_days, 0.40),
        }
    )
    report = build_full_performance_report(df, total_rebalances=2)
    return report, df


def test_format_performance_summary_table() -> None:
    """Ensure summary table is created and can be encoded in CP1252 without UnicodeEncodeError."""
    report, _ = _build_mock_report()
    table_str = format_performance_summary_table(report)  # type: ignore[arg-type]

    assert "TRANSATLANTIC QUANT PORTFOLIO" in table_str
    assert "FUTURES HEDGING ATTRIBUTION" in table_str
    assert "REGIONAL SLEEVE BREAKDOWN" in table_str

    # Must be encodable in standard Windows CP1252
    encoded = table_str.encode("cp1252")
    assert len(encoded) > 0


def test_export_report_to_dataframes() -> None:
    """Verify conversion of report into exportable DataFrames."""
    report, df = _build_mock_report()
    dfs = export_report_to_dataframes(report, daily_history_df=df)  # type: ignore[arg-type]

    assert "summary" in dfs
    assert "attribution" in dfs
    assert "sleeves" in dfs
    assert "daily_history" in dfs

    assert len(dfs["summary"]) == 9
    assert len(dfs["attribution"]) == 5
    assert len(dfs["sleeves"]) == 2
    assert len(dfs["daily_history"]) == 50
