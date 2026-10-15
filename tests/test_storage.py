"""Unit tests for storage schema, SQLite operations, Parquet market data, and deduplication."""

from pathlib import Path

import pandas as pd
import pytest

from quant_portfolio.config import PortfolioConfig, StoragePaths
from quant_portfolio.storage.schema import (
    ParquetMarketStore,
    SQLiteStore,
    deduplicate_universe_by_isin,
)


def test_storage_paths_resolution(temp_dir: Path) -> None:
    """Test path resolution relative to a base directory."""
    paths = StoragePaths().resolve_paths(base=temp_dir)
    assert paths.root_dir == temp_dir
    assert paths.db_path == temp_dir / "data" / "db" / "quant_store.db"
    assert paths.parquet_dir == temp_dir / "data" / "parquet"


def test_portfolio_config_defaults() -> None:
    """Test default values of portfolio configuration."""
    cfg = PortfolioConfig()
    assert cfg.base_currency == "EUR"
    assert cfg.financials_gics_sector == 40
    assert cfg.allocation.min_stock_weight == 0.005
    assert cfg.allocation.max_stock_weight == 0.035
    assert cfg.allocation.na_weight_min == 0.50
    assert cfg.allocation.na_weight_max == 0.65
    assert cfg.regime.inertia_days == 3
    assert cfg.regime.stress_open_prob == 0.70


def test_universe_pit_upsert_idempotence(
    sqlite_store: SQLiteStore,
    sample_universe_df: pd.DataFrame,
) -> None:
    """Verify that multiple upserts of universe_pit do not create duplicates."""
    count1 = sqlite_store.upsert_universe_pit(sample_universe_df)
    assert count1 == 4

    # Fetch rows
    df_read1 = sqlite_store.get_universe_pit()
    assert len(df_read1) == 4

    # Upsert again with the exact same data
    count2 = sqlite_store.upsert_universe_pit(sample_universe_df)
    assert count2 == 4
    df_read2 = sqlite_store.get_universe_pit()
    assert len(df_read2) == 4

    # Upsert with an updated sector for AAPL
    updated_df = sample_universe_df.copy()
    updated_df.loc[updated_df["ticker"] == "AAPL US", "source_index"] = "SPX Rebalanced"
    sqlite_store.upsert_universe_pit(updated_df)

    df_read3 = sqlite_store.get_universe_pit()
    assert len(df_read3) == 4
    aapl_row = df_read3[df_read3["ticker"] == "AAPL US"].iloc[0]
    assert aapl_row["source_index"] == "SPX Rebalanced"


def test_universe_pit_financials_filter(
    sqlite_store: SQLiteStore,
    sample_universe_df: pd.DataFrame,
) -> None:
    """Verify filtering of Financials (GICS Sector 40)."""
    sqlite_store.upsert_universe_pit(sample_universe_df)

    all_stocks = sqlite_store.get_universe_pit()
    assert len(all_stocks) == 4

    non_financials = sqlite_store.get_universe_pit(exclude_financials=True)
    assert len(non_financials) == 3
    assert "JPM US" not in non_financials["ticker"].values


def test_fundamentals_pit_look_ahead_bias_prevention(sqlite_store: SQLiteStore) -> None:
    """Verify that filing_date > as_of_date is strictly rejected to prevent look-ahead bias."""
    # Valid record: filing_date <= as_of_date
    valid_record = pd.DataFrame(
        [
            {
                "as_of_date": "2024-03-31",
                "filing_date": "2024-02-15",
                "ticker": "AAPL US",
                "metric": "PE_RATIO",
                "value": 28.5,
            }
        ]
    )
    sqlite_store.upsert_fundamentals_pit(valid_record)
    read_df = sqlite_store.get_fundamentals_pit()
    assert len(read_df) == 1
    assert read_df.iloc[0]["value"] == 28.5

    # Look-ahead violation: filing_date > as_of_date
    invalid_record = pd.DataFrame(
        [
            {
                "as_of_date": "2024-03-31",
                "filing_date": "2024-04-10",  # Future date relative to as_of_date!
                "ticker": "MSFT US",
                "metric": "PE_RATIO",
                "value": 35.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="Point-in-Time violation"):
        sqlite_store.upsert_fundamentals_pit(invalid_record)

    # Verify invalid record was not inserted
    msft_read = sqlite_store.get_fundamentals_pit(ticker="MSFT US")
    assert msft_read.empty


def test_fundamentals_pit_upsert_idempotence(sqlite_store: SQLiteStore) -> None:
    """Verify idempotence of fundamentals_pit upserts."""
    records = pd.DataFrame(
        [
            {
                "as_of_date": "2024-03-31",
                "filing_date": "2024-03-15",
                "ticker": "SAP GY",
                "metric": "OPER_MARGIN",
                "value": 0.22,
            }
        ]
    )
    sqlite_store.upsert_fundamentals_pit(records)
    sqlite_store.upsert_fundamentals_pit(records)

    res = sqlite_store.get_fundamentals_pit(ticker="SAP GY")
    assert len(res) == 1
    assert res.iloc[0]["value"] == 0.22

    # Update value
    records_update = records.copy()
    records_update["value"] = 0.24
    sqlite_store.upsert_fundamentals_pit(records_update)

    res_updated = sqlite_store.get_fundamentals_pit(ticker="SAP GY")
    assert len(res_updated) == 1
    assert res_updated.iloc[0]["value"] == 0.24


def test_macro_series_idempotence(sqlite_store: SQLiteStore) -> None:
    """Verify macro series storage and upsert idempotence."""
    macro_data = pd.DataFrame(
        [
            {"date": "2024-01-15", "region_bloc": "NA", "series_id": "VIX", "value": 13.5},
            {"date": "2024-01-15", "region_bloc": "EU", "series_id": "V2TX", "value": 14.2},
        ]
    )
    sqlite_store.upsert_macro_series(macro_data)
    sqlite_store.upsert_macro_series(macro_data)

    df_macro = sqlite_store.get_macro_series(start_date="2024-01-15", end_date="2024-01-15")
    assert len(df_macro) == 2

    # Update VIX value
    vix_update = pd.DataFrame(
        [{"date": "2024-01-15", "region_bloc": "NA", "series_id": "VIX", "value": 15.0}]
    )
    sqlite_store.upsert_macro_series(vix_update)
    vix_res = sqlite_store.get_macro_series(region_bloc="NA", series_id="VIX")
    assert len(vix_res) == 1
    assert vix_res.iloc[0]["value"] == 15.0


def test_isin_deduplication_by_liquidity(sample_universe_df: pd.DataFrame) -> None:
    """Verify deduplication of cross-listed securities keeping the most liquid line."""
    # Add volume_eur to simulate liquidity
    df = sample_universe_df.copy()
    volumes = {
        "AAPL US": 100_000_000.0,
        "JPM US": 50_000_000.0,
        "SAP GY": 80_000_000.0,  # High liquidity European primary
        "SAP US": 5_000_000.0,  # Lower liquidity US ADR
    }
    df["volume_eur"] = df["ticker"].map(volumes)

    deduped = deduplicate_universe_by_isin(df, liquidity_col="volume_eur")

    # SAP has 2 lines (SAP GY and SAP US), should keep SAP GY
    assert len(deduped) == 3
    tickers = set(deduped["ticker"].values)
    assert "SAP GY" in tickers
    assert "SAP US" not in tickers


def test_parquet_market_store_write_read_and_idempotence(parquet_store: ParquetMarketStore) -> None:
    """Verify yearly partitioning, reading, and idempotent write in ParquetMarketStore."""
    market_data = pd.DataFrame(
        [
            {
                "date": "2023-12-29",
                "ticker": "AAPL US",
                "region_bloc": "NA",
                "px_last_eur": 175.5,
                "tot_return_idx_eur": 500.0,
                "volume_eur": 120_000_000.0,
            },
            {
                "date": "2024-01-02",
                "ticker": "AAPL US",
                "region_bloc": "NA",
                "px_last_eur": 176.0,
                "tot_return_idx_eur": 501.5,
                "volume_eur": 95_000_000.0,
            },
            {
                "date": "2024-01-02",
                "ticker": "SAP GY",
                "region_bloc": "EU",
                "px_last_eur": 140.0,
                "tot_return_idx_eur": 300.0,
                "volume_eur": 45_000_000.0,
            },
        ]
    )

    # First write
    parquet_store.write_daily_market(market_data)

    # Check files created
    p2023 = parquet_store._year_file_path(2023)
    p2024 = parquet_store._year_file_path(2024)
    assert p2023.exists()
    assert p2024.exists()

    # Read back all
    df_all = parquet_store.read_daily_market()
    assert len(df_all) == 3

    # Filter by region
    df_eu = parquet_store.read_daily_market(region_bloc="EU")
    assert len(df_eu) == 1
    assert df_eu.iloc[0]["ticker"] == "SAP GY"

    # Filter by date range
    df_2024 = parquet_store.read_daily_market(start_date="2024-01-01", end_date="2024-01-31")
    assert len(df_2024) == 2

    # Idempotent re-write with an update on SAP GY price
    market_update = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "ticker": "SAP GY",
                "region_bloc": "EU",
                "px_last_eur": 142.5,  # updated price
                "tot_return_idx_eur": 305.0,
                "volume_eur": 50_000_000.0,
            }
        ]
    )
    parquet_store.write_daily_market(market_update)

    df_2024_after = parquet_store.read_daily_market(start_date="2024-01-01", end_date="2024-01-31")
    assert len(df_2024_after) == 2  # No duplicate row created!
    sap_row = df_2024_after[df_2024_after["ticker"] == "SAP GY"].iloc[0]
    assert sap_row["px_last_eur"] == 142.5

    # Check get_existing_tickers
    existing_tickers = parquet_store.get_existing_tickers()
    assert existing_tickers == {"AAPL US", "SAP GY"}
