"""Unit tests for Phase 7 CLI commands: backtest and rebalance."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from quant_portfolio.cli import cmd_backtest


def test_cli_backtest_help_flag() -> None:
    """Verify that backtest --help exits with code 0 and displays parameters."""
    res = subprocess.run(
        [sys.executable, "-m", "quant_portfolio.cli", "backtest", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "--start" in res.stdout
    assert "--capital" in res.stdout
    assert "--no-hedge" in res.stdout


def test_cli_rebalance_help_flag() -> None:
    """Verify that rebalance --help exits with code 0 and displays parameters."""
    res = subprocess.run(
        [sys.executable, "-m", "quant_portfolio.cli", "rebalance", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "--date" in res.stdout
    assert "--capital" in res.stdout
    assert "--holdings" in res.stdout


@patch("quant_portfolio.cli.WalkForwardBacktestEngine")
def test_cmd_backtest_execution(mock_engine_cls: MagicMock, tmp_path: Path) -> None:
    """Verify cmd_backtest orchestrates engine, formatting, and file export."""
    mock_engine = MagicMock()
    mock_engine_cls.return_value = mock_engine

    mock_daily = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "unhedged_return": 0.001,
                "hedged_return": 0.001,
                "nav_unhedged": 10_010_000.0,
                "nav_hedged": 10_010_000.0,
            }
        ]
    )
    mock_report = MagicMock()
    mock_rebal = pd.DataFrame([{"rebalance_date": "2024-01-02", "n_candidates": 60}])

    mock_engine.run_backtest.return_value = (mock_daily, mock_report, mock_rebal)

    with (
        patch(
            "quant_portfolio.cli.format_performance_summary_table",
            return_value="SUMMARY_TABLE",
        ),
        patch("quant_portfolio.cli.export_report_to_dataframes") as mock_export,
    ):
        mock_export.return_value = {
            "summary": pd.DataFrame([{"metric": "CAGR", "unhedged": 0.1, "hedged": 0.1}]),
            "attribution": pd.DataFrame([{"item": "Futures", "value": 0.0}]),
            "sleeves": pd.DataFrame([{"region_bloc": "NA", "cagr": 0.1}]),
        }
        cmd_backtest(
            start_date="2024-01-02",
            end_date="2024-01-10",
            capital=10_000_000.0,
            method="hrp",
            output_dir=str(tmp_path),
        )

    mock_engine.run_backtest.assert_called_once()
    assert (tmp_path / "backtest_summary.csv").exists()
    assert (tmp_path / "portfolio_daily_history.csv").exists()
    assert (tmp_path / "rebalance_log.csv").exists()
