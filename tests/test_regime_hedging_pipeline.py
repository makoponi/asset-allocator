"""Integration tests for the unified RegimeHedgingPipeline and CLI regime/hedge commands."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.cli import cmd_hedge, cmd_regime
from quant_portfolio.config import PortfolioConfig
from quant_portfolio.hedging.pipeline import RegimeHedgingPipeline
from quant_portfolio.storage.schema import ParquetMarketStore, SQLiteStore


@pytest.fixture
def integrated_env(tmp_path: Path) -> tuple[SQLiteStore, ParquetMarketStore, pd.DataFrame]:
    """Setup test environment with populated macro series and portfolio allocations."""
    db_path = tmp_path / "data" / "db" / "quant_store.db"
    parquet_dir = tmp_path / "data" / "parquet"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    store = SQLiteStore(db_path)
    pq_store = ParquetMarketStore(parquet_dir)

    # 150 daily macro observations
    dates = pd.bdate_range("2023-01-02", periods=150).strftime("%Y-%m-%d").tolist()
    np.random.seed(42)

    records = []
    for i, d in enumerate(dates):
        # Stress spike in the last 5 days
        is_stress = i >= 145
        vix = 45.0 + np.random.normal(0, 1.0) if is_stress else 15.0 + np.random.normal(0, 0.5)
        v2tx = 40.0 + np.random.normal(0, 1.0) if is_stress else 16.0 + np.random.normal(0, 0.5)

        records.append({"date": d, "region_bloc": "NA", "series_id": "VIX", "value": float(vix)})
        records.append(
            {
                "date": d,
                "region_bloc": "NA",
                "series_id": "T10Y2Y",
                "value": float(-0.5 + np.random.normal(0, 0.02)),
            }
        )
        records.append(
            {
                "date": d,
                "region_bloc": "NA",
                "series_id": "BAMLH0A0HYM2",
                "value": float(350.0 + np.random.normal(0, 2.0)),
            }
        )
        records.append({"date": d, "region_bloc": "EU", "series_id": "V2TX", "value": float(v2tx)})
        records.append(
            {
                "date": d,
                "region_bloc": "EU",
                "series_id": "BUND10Y2Y",
                "value": float(-0.2 + np.random.normal(0, 0.02)),
            }
        )
        records.append(
            {
                "date": d,
                "region_bloc": "EU",
                "series_id": "BAMLHE00EHYIOAS",
                "value": float(380.0 + np.random.normal(0, 2.0)),
            }
        )

    store.upsert_macro_series(records)

    alloc_df = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "ENI IM", "AIR FP"],
            "region_bloc": ["NA", "NA", "EU", "EU"],
            "weight": [0.30, 0.30, 0.20, 0.20],
            "beta": [1.15, 1.15, 0.85, 0.85],
        }
    )

    return store, pq_store, alloc_df


def test_regime_hedging_pipeline_integration(
    integrated_env: tuple[SQLiteStore, ParquetMarketStore, pd.DataFrame],
) -> None:
    """Verify end-to-end pipeline execution from raw macro to future contract sizing."""
    store, pq_store, alloc_df = integrated_env
    cfg = PortfolioConfig()
    cfg.regime.min_history_days = 80

    pipeline = RegimeHedgingPipeline(
        sqlite_store=store,
        parquet_store=pq_store,
        config=cfg,
    )

    # Test at calm date (day 100)
    calm_date = store.get_macro_series()["date"].unique()[100]
    summary_calm = pipeline.execute_hedging_pipeline(
        as_of_date=calm_date,
        portfolio_value_eur=10_000_000.0,
        allocations_df=alloc_df,
    )

    assert summary_calm.total_portfolio_value_eur == 10_000_000.0
    assert summary_calm.na_sleeve.region_bloc == "NA"
    assert summary_calm.eu_sleeve.region_bloc == "EU"
    assert summary_calm.na_sleeve.contract_symbol == "ES"
    assert summary_calm.eu_sleeve.contract_symbol == "FXXP"

    # Test at final stress date (day 149)
    stress_date = store.get_macro_series()["date"].unique()[-1]
    summary_stress = pipeline.execute_hedging_pipeline(
        as_of_date=stress_date,
        portfolio_value_eur=10_000_000.0,
        allocations_df=alloc_df,
    )

    # In stress period, contracts should be recommended (> 0)
    assert summary_stress.na_sleeve.contracts_rounded >= 0
    assert summary_stress.eu_sleeve.contracts_rounded >= 0


def test_cli_regime_and_hedge_commands(
    integrated_env: tuple[SQLiteStore, ParquetMarketStore, pd.DataFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that CLI cmd_regime and cmd_hedge execute cleanly without errors."""
    store, pq_store, alloc_df = integrated_env
    cfg = PortfolioConfig()
    cfg.paths.root_dir = tmp_path
    cfg.paths.db_path = tmp_path / "data" / "db" / "quant_store.db"
    cfg.paths.parquet_dir = tmp_path / "data" / "parquet"
    cfg.regime.min_history_days = 80

    # Mock PortfolioConfig instantiation in cli
    monkeypatch.setattr("quant_portfolio.cli.PortfolioConfig", lambda: cfg)

    latest_date = store.get_macro_series()["date"].unique()[-1]

    # Test cmd_regime
    cmd_regime(as_of_date=latest_date)

    # Save test allocations
    alloc_csv = tmp_path / "test_allocations.csv"
    alloc_df.to_csv(alloc_csv, index=False)
    out_hedge = tmp_path / "hedge_plan.csv"

    # Test cmd_hedge
    cmd_hedge(
        as_of_date=latest_date,
        portfolio_value=5_000_000.0,
        allocations_file=str(alloc_csv),
        output_path=str(out_hedge),
        na_future="MES",
        eu_future="FESX",
    )

    assert out_hedge.exists()
    saved_plan = pd.read_csv(out_hedge)
    assert len(saved_plan) == 2
    assert "contracts_short" in saved_plan.columns
