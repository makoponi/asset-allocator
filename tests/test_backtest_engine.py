"""Unit tests for the Walk-Forward Backtesting Engine."""

from unittest.mock import MagicMock

import pandas as pd

from quant_portfolio.backtest.engine import WalkForwardBacktestEngine
from quant_portfolio.config import PortfolioConfig
from quant_portfolio.hedging.engine import DualSleeveHedgeSummary
from quant_portfolio.risk.pipeline import PortfolioRiskSummary


def test_walk_forward_engine_simulation_run(tmp_path: object) -> None:
    """Test full Walk-Forward backtest loop on synthetic multi-day data."""
    # Setup mock stores
    mock_sqlite = MagicMock()
    mock_parquet = MagicMock()

    cfg = PortfolioConfig()
    cfg.backtest.rebalance_freq_days = 5  # Rebalance every 5 business days
    cfg.backtest.initial_capital_eur = 1_000_000.0

    # Mock allocation pipeline
    mock_alloc_pipeline = MagicMock()
    mock_alloc_pipeline.allocate_candidates.return_value = (
        pd.DataFrame(
            [
                {"ticker": "AAPL US", "weight": 0.60, "region_bloc": "NA"},
                {"ticker": "ASML NA", "weight": 0.40, "region_bloc": "EU"},
            ]
        ),
        PortfolioRiskSummary(
            as_of_date="2024-01-02",
            method="hrp",
            n_stocks=2,
            n_na=1,
            n_eu=1,
            na_weight=0.60,
            eu_weight=0.40,
            annualized_volatility=0.15,
            cvar_95=0.20,
        ),
    )

    from quant_portfolio.hedging.engine import SleeveHedgePlan

    na_plan = SleeveHedgePlan(
        region_bloc="NA",
        sleeve_weight=0.60,
        sleeve_value_eur=600_000.0,
        sleeve_beta=1.0,
        regime_state=1,
        regime_name="Alert",
        hedge_ratio=0.5,
        hedged_notional_eur=300_000.0,
        contract_symbol="ES",
        contract_price=5000.0,
        contract_multiplier=50.0,
        contract_currency="USD",
        contract_value_eur=231_481.48,
        contracts_exact=1.29,
        contracts_rounded=1,
    )

    eu_plan = SleeveHedgePlan(
        region_bloc="EU",
        sleeve_weight=0.40,
        sleeve_value_eur=400_000.0,
        sleeve_beta=1.0,
        regime_state=0,
        regime_name="Calm",
        hedge_ratio=0.0,
        hedged_notional_eur=0.0,
        contract_symbol="FXXP",
        contract_price=500.0,
        contract_multiplier=50.0,
        contract_currency="EUR",
        contract_value_eur=25_000.0,
        contracts_exact=0.0,
        contracts_rounded=0,
    )

    # Mock regime hedging pipeline
    mock_regime_pipeline = MagicMock()
    mock_regime_pipeline.execute_hedging_pipeline.return_value = DualSleeveHedgeSummary(
        as_of_date="2024-01-02",
        total_portfolio_value_eur=1_000_000.0,
        total_hedged_notional_eur=231_481.48,
        portfolio_effective_hedge_ratio=0.2315,
        na_sleeve=na_plan,
        eu_sleeve=eu_plan,
    )

    engine = WalkForwardBacktestEngine(
        sqlite_store=mock_sqlite,
        parquet_store=mock_parquet,
        config=cfg,
        allocation_pipeline=mock_alloc_pipeline,
        regime_pipeline=mock_regime_pipeline,
    )

    # Custom candidate ranked universe loader
    def mock_universe_loader(as_of_date: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "ticker": "AAPL US",
                    "region_bloc": "NA",
                    "score": 2.0,
                    "regional_decile": 1,
                    "regional_rank": 1,
                },
                {
                    "ticker": "ASML NA",
                    "region_bloc": "EU",
                    "score": 1.8,
                    "regional_decile": 1,
                    "regional_rank": 1,
                },
            ]
        )

    # Custom futures prices loader
    def mock_futures_loader(as_of_date: str) -> dict[str, float]:
        return {"ES": 5000.0, "MES": 5000.0, "FXXP": 500.0, "FESX": 4500.0}

    # Run 10 business days backtest: 2024-01-02 to 2024-01-15
    daily_df, report, rebal_df = engine.run_backtest(
        start_date="2024-01-02",
        end_date="2024-01-15",
        allocation_method="hrp",
        initial_capital_eur=1_000_000.0,
        enable_hedging=True,
        rebalance_freq_days=5,
        ranked_universe_loader=mock_universe_loader,
        futures_prices_loader=mock_futures_loader,
    )

    assert not daily_df.empty
    assert len(daily_df) == 10
    assert len(rebal_df) == 2  # Day 1 and Day 6 rebalance

    # Verify NAV preservation and accounting continuity
    from typing import cast

    assert cast("float", daily_df.loc[0, "nav_unhedged"]) > 0
    assert cast("float", daily_df.loc[0, "nav_hedged"]) > 0

    # Verify report components
    assert report.total_rebalances == 2
    assert report.hedged_metrics.total_return is not None
    assert report.overlay_attribution is not None
    assert report.sleeve_na.region_bloc == "NA"
    assert report.sleeve_eu.region_bloc == "EU"
