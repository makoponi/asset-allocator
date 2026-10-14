"""Unit tests for BacktestEngine simulation workflow and accounting."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import pytest

from src.execution.backtest_engine import BacktestEngine
from src.features.technical_engine import TechnicalEngine
from src.ingestion.storage import (
    FundamentalPiTRecord,
    MarketStore,
    PriceEODRecord,
    UniverseMemberRecord,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_backtest_engine_simulation_accounting(tmp_path: Path) -> None:
    store = MarketStore(db_path=tmp_path / "test_backtest.db")

    rebalance_dates = [
        date(2023, 1, 31),
        date(2023, 2, 28),
        date(2023, 3, 31),
    ]

    tickers = [f"STK_{i:02d}" for i in range(15)]
    sectors = ["Technology", "Financials", "Healthcare"]

    # 1. Register universe members and fundamentals
    member_records = []
    fund_records = []
    for d in rebalance_dates:
        for i, t in enumerate(tickers):
            member_records.append(
                UniverseMemberRecord(
                    date=d.isoformat(),
                    ticker=t,
                    gics_sector=sectors[i % len(sectors)],
                    is_active=1,
                )
            )

    for i, t in enumerate(tickers):
        fund_records.append(
            FundamentalPiTRecord(
                as_of_date="2022-12-31",
                filing_date="2022-12-31",
                ticker=t,
                metric_name="PE_RATIO",
                metric_value=12.0 + (i % 5),
            )
        )
        fund_records.append(
            FundamentalPiTRecord(
                as_of_date="2022-12-31",
                filing_date="2023-01-15",
                ticker=t,
                metric_name="RETURN_COM_EQY",
                metric_value=15.0 + (i % 5),
            )
        )

    store.insert_universe_members(member_records)
    store.insert_fundamentals_pit(fund_records)

    # 2. Build 300 business days of prices
    prices: list[PriceEODRecord] = []
    start_d = date(2022, 1, 1)
    end_d = date(2023, 4, 30)
    curr = start_d

    import numpy as np

    rng = np.random.default_rng(42)
    px_map = {t: 100.0 + i for i, t in enumerate(tickers)}
    while curr <= end_d:
        if curr.weekday() < 5:
            for t in tickers:
                base_ret = 0.0005 if int(t.split("_")[1]) > 5 else -0.0002
                noise = float(rng.normal(0.0, 0.005))
                ret = base_ret + noise
                px_next = max(1.0, px_map[t] * (1.0 + ret))
                px_map[t] = px_next
                prices.append(
                    PriceEODRecord(
                        date=curr,
                        ticker=t,
                        px_last=px_next,
                        volume=50000.0,
                        ret_1d=ret,
                    )
                )
        curr += timedelta(days=1)

    tech_engine = TechnicalEngine(prices)

    engine = BacktestEngine(
        market_store=store,
        tech_engine=tech_engine,
        price_records=prices,
        retrain_freq_months=1,
        cost_bps=10.0,  # 10 bps
        top_n=5,
        max_weight=0.25,
    )

    result = engine.run_simulation(
        start_date=date(2023, 1, 31),
        end_date=date(2023, 3, 31),
        initial_capital=100.0,
    )

    # 3. Assertions
    df_hist = result.history_df
    assert len(df_hist) > 40
    assert "portfolio_nav" in df_hist.columns
    assert "benchmark_nav" in df_hist.columns

    # Verify initial NAV begins at approximately initial_capital (minus initial turnover fee)
    init_nav = df_hist.iloc[0]["portfolio_nav"]
    assert init_nav == pytest.approx(100.0, abs=0.2)

    # Verify metrics
    p_met = result.metrics
    assert p_met.total_return != 0.0
    assert p_met.annualized_volatility > 0.0
    assert p_met.max_drawdown <= 0.0

    # Rebalance records: should have recorded 3 rebalances
    assert len(result.monthly_rebalance_records) == 3
    for rec in result.monthly_rebalance_records:
        assert rec["turnover"] > 0.0
        assert rec["cost_penalty"] >= 0.0
        assert rec["num_stocks"] <= 5
