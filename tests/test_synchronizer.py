"""Unit tests for MultiDaySynchronizer."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.config import PortfolioConfig, RiskConfig
from quant_portfolio.risk.synchronizer import MultiDaySynchronizer
from quant_portfolio.storage.schema import ParquetMarketStore


@pytest.fixture
def mock_market_store(tmp_path: Path) -> ParquetMarketStore:
    """Populate temporary parquet store with synthetic daily data for US and EU stocks."""
    p_dir = tmp_path / "parquet"
    store = ParquetMarketStore(p_dir)

    dates = pd.bdate_range(start="2023-01-01", end="2024-01-31").strftime("%Y-%m-%d").tolist()
    records = []
    np.random.seed(42)

    tickers = ["AAPL US", "MSFT US", "SAP GY", "ASML NA"]
    for t in tickers:
        reg = "NA" if "US" in t else "EU"
        base_px = 100.0
        for d in dates:
            # Introduce random holidays
            if reg == "NA" and d == "2023-07-04":
                continue  # US Independence Day
            if reg == "EU" and d == "2023-05-01":
                continue  # Labour day in EU

            step = np.random.normal(0.0005, 0.015)
            base_px *= 1.0 + step
            records.append(
                {
                    "date": d,
                    "ticker": t,
                    "region_bloc": reg,
                    "px_last_eur": base_px,
                    "tot_return_idx_eur": base_px,
                    "volume_eur": 1_000_000.0,
                }
            )

    df = pd.DataFrame(records)
    store.write_daily_market(df)
    return store


def test_aligned_price_matrix_handles_holidays(mock_market_store: ParquetMarketStore) -> None:
    """Verify that forward-filling reconciles exchange holiday differences."""
    sync = MultiDaySynchronizer(parquet_store=mock_market_store)
    tickers = ["AAPL US", "SAP GY"]
    prices = sync.get_aligned_price_matrix(
        tickers=tickers, as_of_date="2024-01-31", lookback_days=100
    )

    assert not prices.empty
    assert "AAPL US" in prices.columns
    assert "SAP GY" in prices.columns
    assert not prices.isna().any().any()


def test_compute_synchronized_returns_3day(mock_market_store: ParquetMarketStore) -> None:
    """Verify 3-day non-overlapping return calculation length and properties."""
    sync = MultiDaySynchronizer(parquet_store=mock_market_store)
    tickers = ["AAPL US", "MSFT US", "SAP GY", "ASML NA"]
    prices = sync.get_aligned_price_matrix(
        tickers=tickers, as_of_date="2024-01-31", lookback_days=30
    )

    returns = sync.compute_synchronized_returns(prices, resampling_days=3)
    assert not returns.empty
    # Expect ~ len(prices)/3 returns
    expected_len = (len(prices) - 1) // 3
    assert abs(len(returns) - expected_len) <= 1
    assert set(returns.columns) == set(tickers)


def test_empty_or_insufficient_history(mock_market_store: ParquetMarketStore) -> None:
    """Verify safe behavior when tickers are not found or history is insufficient."""
    cfg = PortfolioConfig(risk=RiskConfig(resampling_days=3))
    sync = MultiDaySynchronizer(parquet_store=mock_market_store, config=cfg)
    empty_returns = sync.extract_synchronized_returns(
        tickers=["NON_EXISTENT"], as_of_date="2024-01-31"
    )
    assert empty_returns.empty
