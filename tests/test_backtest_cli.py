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


@patch("quant_portfolio.cli.AlphaPipeline")
@patch("quant_portfolio.cli.PortfolioAllocationPipeline")
@patch("quant_portfolio.cli.TurnoverBufferFilter")
@patch("quant_portfolio.cli.RegimeHedgingPipeline")
def test_cmd_rebalance_execution(
    mock_regime_cls: MagicMock,
    mock_tb_cls: MagicMock,
    mock_alloc_cls: MagicMock,
    mock_alpha_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Verify cmd_rebalance scores universe, filters turnover buffer, and generates orders."""
    from quant_portfolio.cli import cmd_rebalance

    mock_alpha = MagicMock()
    mock_alpha_cls.return_value = mock_alpha
    mock_alpha.score_and_rank_universe.return_value = pd.DataFrame(
        [
            {"ticker": "AAPL US", "region_bloc": "NA", "score": 0.1, "regional_decile": 1},
            {"ticker": "SAP GY", "region_bloc": "EU", "score": 0.08, "regional_decile": 1},
        ]
    )

    mock_tb = MagicMock()
    mock_tb_cls.return_value = mock_tb
    mock_tb.filter_candidates_with_buffer.return_value = pd.DataFrame(
        [
            {"ticker": "AAPL US", "region_bloc": "NA", "score": 0.1},
            {"ticker": "SAP GY", "region_bloc": "EU", "score": 0.08},
        ]
    )
    mock_tb.generate_delta_orders.return_value = pd.DataFrame(
        [
            {
                "ticker": "AAPL US",
                "region_bloc": "NA",
                "action": "BUY",
                "weight_delta": 0.6,
                "estimated_trade_value_eur": 600_000.0,
            }
        ]
    )

    mock_alloc = MagicMock()
    mock_alloc_cls.return_value = mock_alloc
    mock_risk_summary = MagicMock()
    mock_risk_summary.n_stocks = 2
    mock_risk_summary.n_na = 1
    mock_risk_summary.n_eu = 1
    mock_risk_summary.na_weight = 0.60
    mock_risk_summary.eu_weight = 0.40
    mock_risk_summary.annualized_volatility = 0.15
    mock_risk_summary.cvar_95 = 0.03
    mock_alloc.allocate_candidates.return_value = (
        pd.DataFrame([{"ticker": "AAPL US", "weight": 0.6, "region_bloc": "NA"}]),
        mock_risk_summary,
    )

    mock_regime = MagicMock()
    mock_regime_cls.return_value = mock_regime
    mock_hedge_summary = MagicMock()
    mock_hedge_summary.portfolio_effective_hedge_ratio = 0.5
    mock_hedge_summary.total_hedged_notional_eur = 300_000.0
    mock_hedge_summary.na_sleeve.contract_symbol = "ES"
    mock_hedge_summary.na_sleeve.contracts_rounded = 1
    mock_hedge_summary.na_sleeve.regime_name = "Alert"
    mock_hedge_summary.na_sleeve.hedged_notional_eur = 300_000.0
    mock_hedge_summary.eu_sleeve.contract_symbol = "FXXP"
    mock_hedge_summary.eu_sleeve.contracts_rounded = 0
    mock_hedge_summary.eu_sleeve.regime_name = "Calm"
    mock_hedge_summary.eu_sleeve.hedged_notional_eur = 0.0
    mock_regime.execute_hedging_pipeline.return_value = mock_hedge_summary

    out_file = tmp_path / "orders.csv"
    cmd_rebalance(
        as_of_date="2024-01-02",
        capital=1_000_000.0,
        output_path=str(out_file),
    )

    mock_alpha.score_and_rank_universe.assert_called_once_with(as_of_date="2024-01-02")
    mock_tb.filter_candidates_with_buffer.assert_called_once()
    assert out_file.exists()
