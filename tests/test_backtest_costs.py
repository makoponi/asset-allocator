"""Unit tests for transaction cost modeling in transatlantic backtest."""

import pandas as pd
import pytest

from quant_portfolio.backtest.costs import TransactionCostModel


def test_equity_costs_calculation() -> None:
    """Test 5 bps on NA equities and 10 bps on EU equities."""
    model = TransactionCostModel()

    # 1,000,000 EUR in NA trade -> 5 bps = 500 EUR
    # 500,000 EUR in EU trade -> 10 bps = 500 EUR
    trades_df = pd.DataFrame(
        [
            {"ticker": "AAPL US", "region_bloc": "NA", "traded_value_eur": 1_000_000.0},
            {"ticker": "ASML NA", "region_bloc": "EU", "traded_value_eur": 500_000.0},
        ]
    )

    total_cost, enriched = model.calculate_equity_costs(trades_df)
    assert pytest.approx(total_cost, rel=1e-6) == 1000.0
    assert pytest.approx(enriched.loc[0, "cost_eur"], rel=1e-6) == 500.0
    assert pytest.approx(enriched.loc[1, "cost_eur"], rel=1e-6) == 500.0


def test_equity_costs_via_weight_delta() -> None:
    """Test cost derivation from weight_delta and portfolio_value_eur."""
    model = TransactionCostModel()
    # 2% of 10M = 200k (NA @ 5bps = 100 EUR)
    # 1% of 10M = 100k (EU @ 10bps = 100 EUR)
    trades_df = pd.DataFrame(
        [
            {"ticker": "MSFT US", "region_bloc": "NA", "weight_delta": 0.02},
            {"ticker": "MC FP", "region_bloc": "EU", "weight_delta": -0.01},
        ]
    )

    total_cost, enriched = model.calculate_equity_costs(trades_df, portfolio_value_eur=10_000_000.0)
    assert pytest.approx(total_cost, rel=1e-6) == 200.0
    assert pytest.approx(enriched.loc[0, "cost_eur"], rel=1e-6) == 100.0
    assert pytest.approx(enriched.loc[1, "cost_eur"], rel=1e-6) == 100.0


def test_futures_costs_calculation() -> None:
    """Test 1.5 bps on futures traded notional."""
    model = TransactionCostModel()

    # ES notional: 268,518 EUR. 5 contracts = 1,342,590 EUR * 0.00015 = 201.3885 EUR
    # FXXP notional: 26,000 EUR. 10 contracts = 260,000 EUR * 0.00015 = 39.0 EUR
    deltas = {"ES": 5, "FXXP": -10}
    notionals = {"ES": 268_518.52, "FXXP": 26_000.0}

    total_cost, by_symbol = model.calculate_futures_costs(deltas, notionals)
    expected_es = 5 * 268_518.52 * 0.00015
    expected_fxxp = 10 * 26_000.0 * 0.00015

    assert pytest.approx(by_symbol["ES"], rel=1e-5) == expected_es
    assert pytest.approx(by_symbol["FXXP"], rel=1e-5) == expected_fxxp
    assert pytest.approx(total_cost, rel=1e-5) == expected_es + expected_fxxp


def test_total_rebalance_costs_breakdown() -> None:
    """Test full breakdown across NA equity, EU equity, and futures overlay."""
    model = TransactionCostModel()
    trades_df = pd.DataFrame(
        [
            {"ticker": "NVDA US", "region_bloc": "NA", "traded_value_eur": 2_000_000.0},  # 1000 EUR
            {"ticker": "SAP GY", "region_bloc": "EU", "traded_value_eur": 1_000_000.0},  # 1000 EUR
        ]
    )
    deltas = {"ES": 2}
    notionals = {"ES": 250_000.0}  # 500,000 * 0.00015 = 75 EUR

    breakdown = model.calculate_total_rebalance_costs(
        equity_trades_df=trades_df,
        portfolio_value_eur=10_000_000.0,
        futures_delta=deltas,
        futures_notionals_eur=notionals,
    )

    assert pytest.approx(breakdown.na_equity_cost_eur, rel=1e-5) == 1000.0
    assert pytest.approx(breakdown.eu_equity_cost_eur, rel=1e-5) == 1000.0
    assert pytest.approx(breakdown.futures_cost_eur, rel=1e-5) == 75.0
    assert pytest.approx(breakdown.total_cost_eur, rel=1e-5) == 2075.0
