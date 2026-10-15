"""Unit and integration tests for PortfolioAllocationPipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.risk.pipeline import PortfolioAllocationPipeline
from quant_portfolio.storage.schema import ParquetMarketStore


@pytest.fixture
def mock_transatlantic_market_store(tmp_path: Path) -> ParquetMarketStore:
    """Populate temporary parquet store with synthetic price history for 35 NA and 25 EU assets."""
    p_dir = tmp_path / "parquet"
    store = ParquetMarketStore(p_dir)

    dates = pd.bdate_range(start="2023-01-01", end="2024-01-31").strftime("%Y-%m-%d").tolist()
    records = []
    np.random.seed(42)

    tickers_na = [f"NA_STOCK_{i:02d} US" for i in range(35)]
    tickers_eu = [f"EU_STOCK_{i:02d} GY" for i in range(25)]
    all_tickers = [(t, "NA") for t in tickers_na] + [(t, "EU") for t in tickers_eu]

    for t, reg in all_tickers:
        px = 100.0
        for d in dates:
            step = np.random.normal(0.0004, 0.015)
            px *= 1.0 + step
            records.append(
                {
                    "date": d,
                    "ticker": t,
                    "region_bloc": reg,
                    "px_last_eur": px,
                    "tot_return_idx_eur": px,
                    "volume_eur": 1_000_000.0,
                }
            )

    df = pd.DataFrame(records)
    store.write_daily_market(df)
    return store


@pytest.fixture
def sample_candidates_df() -> pd.DataFrame:
    """Sample candidate assets dataframe from Phase 4 sleeve selector."""
    tickers_na = [f"NA_STOCK_{i:02d} US" for i in range(35)]
    tickers_eu = [f"EU_STOCK_{i:02d} GY" for i in range(25)]

    records = []
    for t in tickers_na:
        records.append({"ticker": t, "region_bloc": "NA", "score": 0.05})
    for t in tickers_eu:
        records.append({"ticker": t, "region_bloc": "EU", "score": 0.03})

    return pd.DataFrame(records)


def test_pipeline_hrp_allocation(
    mock_transatlantic_market_store: ParquetMarketStore,
    sample_candidates_df: pd.DataFrame,
) -> None:
    """Verify that HRP pipeline end-to-end fulfills all transatlantic constraints."""
    pipeline = PortfolioAllocationPipeline(parquet_store=mock_transatlantic_market_store)
    alloc_df, summary = pipeline.allocate_candidates(
        candidates_df=sample_candidates_df,
        as_of_date="2024-01-31",
        method="hrp",
    )

    assert not alloc_df.empty
    assert summary.n_stocks == 60
    assert summary.n_na == 35
    assert summary.n_eu == 25
    assert 0.50 - 1e-5 <= summary.na_weight <= 0.65 + 1e-5
    assert 0.35 - 1e-5 <= summary.eu_weight <= 0.50 + 1e-5
    assert summary.annualized_volatility > 0.0
    assert "mctr" in alloc_df.columns


def test_pipeline_cvar_allocation(
    mock_transatlantic_market_store: ParquetMarketStore,
    sample_candidates_df: pd.DataFrame,
) -> None:
    """Verify that CVaR pipeline end-to-end fulfills all transatlantic constraints."""
    pipeline = PortfolioAllocationPipeline(parquet_store=mock_transatlantic_market_store)
    alloc_df, summary = pipeline.allocate_candidates(
        candidates_df=sample_candidates_df,
        as_of_date="2024-01-31",
        method="cvar",
    )

    assert not alloc_df.empty
    assert summary.n_stocks == 60
    assert 0.50 - 1e-5 <= summary.na_weight <= 0.65 + 1e-5
    assert 0.35 - 1e-5 <= summary.eu_weight <= 0.50 + 1e-5
    assert summary.cvar_95 >= 0.0


def test_pipeline_ensemble_allocation(
    mock_transatlantic_market_store: ParquetMarketStore,
    sample_candidates_df: pd.DataFrame,
) -> None:
    """Verify that 50/50 ensemble allocation blends weights and respects constraints."""
    pipeline = PortfolioAllocationPipeline(parquet_store=mock_transatlantic_market_store)
    alloc_df, summary = pipeline.allocate_candidates(
        candidates_df=sample_candidates_df,
        as_of_date="2024-01-31",
        method="ensemble",
    )

    assert not alloc_df.empty
    weights = alloc_df["weight"].to_numpy()
    assert abs(float(np.sum(weights)) - 1.0) < 1e-5
    assert 0.50 - 1e-5 <= summary.na_weight <= 0.65 + 1e-5
    assert 0.35 - 1e-5 <= summary.eu_weight <= 0.50 + 1e-5


def test_pipeline_empty_candidates_handled(
    mock_transatlantic_market_store: ParquetMarketStore,
) -> None:
    """Verify handling when candidates DataFrame is empty."""
    pipeline = PortfolioAllocationPipeline(parquet_store=mock_transatlantic_market_store)
    alloc_df, summary = pipeline.allocate_candidates(
        candidates_df=pd.DataFrame(),
        as_of_date="2024-01-31",
    )
    assert alloc_df.empty
    assert summary.n_stocks == 0
