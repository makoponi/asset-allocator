"""Pytest fixtures for synthetic transatlantic test data."""

from collections.abc import Generator
from pathlib import Path

import pandas as pd
import pytest

from quant_portfolio.storage.schema import ParquetMarketStore, SQLiteStore


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory path."""
    return tmp_path


@pytest.fixture
def sqlite_store(tmp_path: Path) -> Generator[SQLiteStore, None, None]:
    """Provide an isolated temporary SQLiteStore instance."""
    db_file = tmp_path / "test_store.db"
    store = SQLiteStore(db_file)
    yield store


@pytest.fixture
def parquet_store(tmp_path: Path) -> Generator[ParquetMarketStore, None, None]:
    """Provide an isolated temporary ParquetMarketStore instance."""
    p_dir = tmp_path / "parquet"
    store = ParquetMarketStore(p_dir)
    yield store


@pytest.fixture
def sample_universe_df() -> pd.DataFrame:
    """Sample transatlantic universe records with financials and cross-listings."""
    return pd.DataFrame(
        [
            {
                "as_of_date": "2024-01-31",
                "ticker": "AAPL US",
                "isin": "US0378331005",
                "region_bloc": "NA",
                "source_index": "SPX Index",
                "gics_sector": 45,
                "local_currency": "USD",
                "is_financial": False,
            },
            {
                "as_of_date": "2024-01-31",
                "ticker": "JPM US",
                "isin": "US46625H1005",
                "region_bloc": "NA",
                "source_index": "SPX Index",
                "gics_sector": 40,  # Financials
                "local_currency": "USD",
                "is_financial": True,
            },
            {
                "as_of_date": "2024-01-31",
                "ticker": "SAP GY",
                "isin": "DE0007164600",
                "region_bloc": "EU",
                "source_index": "SXXP Index",
                "gics_sector": 45,
                "local_currency": "EUR",
                "is_financial": False,
            },
            {
                "as_of_date": "2024-01-31",
                "ticker": "SAP US",  # Cross-listed ADR
                "isin": "DE0007164600",
                "region_bloc": "NA",
                "source_index": "SPX Index",
                "gics_sector": 45,
                "local_currency": "USD",
                "is_financial": False,
            },
        ]
    )
