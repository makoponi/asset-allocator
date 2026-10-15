"""Unit tests for BQL universe export parsing, normalization, and SQLite upsert."""

from pathlib import Path

import pandas as pd

from quant_portfolio.ingestion.bql_parser import (
    BQLUniverseIngestionEngine,
    normalize_universe_records,
    parse_bql_universe_file,
)
from quant_portfolio.storage.schema import SQLiteStore


def test_normalize_universe_records_standard() -> None:
    """Verify normalization with standard column names and delisted prefix preservation."""
    raw_df = pd.DataFrame(
        [
            {
                "date": "2023-12-31",
                "ticker": "AAPL US",
                "isin": "US0378331005",
                "benchmark": "SPX Index",
                "gics_sector": "45",
                "crncy": "usd",
            },
            {
                "date": "2023-12-31",
                "ticker": "JPM US",
                "isin": "US46625H1005",
                "benchmark": "SPX Index",
                "gics_sector": 40,  # Financials
                "crncy": "USD",
            },
            {
                "date": "2023-12-31",
                "ticker": "1CNHI IM",  # Delisted numerical prefix preserved!
                "isin": "NL0010545661",
                "benchmark": "SXXP Index",
                "gics_sector": 20,
                "crncy": "EUR",
            },
            {
                "date": "2023-12-31",
                "ticker": "RY CT",
                "isin": "CA7800871021",
                "benchmark": "SPTSX60 Index",
                "gics_sector": 40,
                "crncy": "CAD",
            },
        ]
    )

    norm = normalize_universe_records(raw_df)
    assert len(norm) == 4
    assert set(norm.columns) == {
        "as_of_date",
        "ticker",
        "isin",
        "region_bloc",
        "source_index",
        "gics_sector",
        "local_currency",
        "is_financial",
    }

    # Delisted ticker is preserved exactly
    assert "1CNHI IM" in norm["ticker"].values

    # Financials flag
    assert bool(norm.loc[norm["ticker"] == "JPM US", "is_financial"].iloc[0]) is True
    assert bool(norm.loc[norm["ticker"] == "RY CT", "is_financial"].iloc[0]) is True
    assert bool(norm.loc[norm["ticker"] == "AAPL US", "is_financial"].iloc[0]) is False
    assert bool(norm.loc[norm["ticker"] == "1CNHI IM", "is_financial"].iloc[0]) is False

    # Regions
    assert norm.loc[norm["ticker"] == "AAPL US", "region_bloc"].iloc[0] == "NA"
    assert norm.loc[norm["ticker"] == "RY CT", "region_bloc"].iloc[0] == "NA"
    assert norm.loc[norm["ticker"] == "1CNHI IM", "region_bloc"].iloc[0] == "EU"


def test_normalize_empty_dataframe() -> None:
    """Verify empty DataFrame normalization returns empty schema."""
    norm = normalize_universe_records(pd.DataFrame())
    assert norm.empty
    assert "is_financial" in norm.columns


def test_parse_bql_universe_file_formats(tmp_path: Path) -> None:
    """Verify reading Parquet and CSV files."""
    df = pd.DataFrame(
        [
            {
                "as_of_date": "2024-01-31",
                "ticker": "MSFT US",
                "isin": "US5949181045",
                "source_index": "SPX Index",
                "gics_sector": 45,
                "local_currency": "USD",
            }
        ]
    )
    csv_file = tmp_path / "export.csv"
    parquet_file = tmp_path / "export.parquet"

    df.to_csv(csv_file, index=False)
    df.to_parquet(parquet_file, index=False)

    df_csv = parse_bql_universe_file(csv_file)
    df_parquet = parse_bql_universe_file(parquet_file)

    assert len(df_csv) == 1
    assert len(df_parquet) == 1
    assert df_csv.iloc[0]["ticker"] == "MSFT US"
    assert df_parquet.iloc[0]["ticker"] == "MSFT US"


def test_bql_universe_ingestion_engine(
    sqlite_store: SQLiteStore,
    tmp_path: Path,
) -> None:
    """Verify end-to-end ingestion engine from drop-zone into SQLiteStore."""
    raw_dir = tmp_path / "raw_bql_exports"
    raw_dir.mkdir(parents=True)

    # File 1: S&P 500 members
    df_spx = pd.DataFrame(
        [
            {
                "date": "2024-01-31",
                "id": "AAPL US",
                "id_isin": "US0378331005",
                "index": "SPX Index",
                "gics": 45,
                "crncy": "USD",
            },
            {
                "date": "2024-01-31",
                "id": "JPM US",
                "id_isin": "US46625H1005",
                "index": "SPX Index",
                "gics": 40,
                "crncy": "USD",
            },
        ]
    )
    df_spx.to_csv(raw_dir / "spx_members_2024.csv", index=False)

    # File 2: STOXX 600 members with delisted line
    df_sxxp = pd.DataFrame(
        [
            {
                "date": "2024-01-31",
                "id": "SAP GY",
                "id_isin": "DE0007164600",
                "index": "SXXP Index",
                "gics": 45,
                "crncy": "EUR",
            },
            {
                "date": "2024-01-31",
                "id": "1COV GY",
                "id_isin": "DE0006062144",
                "index": "SXXP Index",
                "gics": 15,
                "crncy": "EUR",
            },
        ]
    )
    df_sxxp.to_parquet(raw_dir / "sxxp_members_2024.parquet", index=False)

    engine = BQLUniverseIngestionEngine(sqlite_store, raw_dir)
    results = engine.ingest_drop_zone()

    assert len(results) == 2
    assert results["spx_members_2024.csv"] == 2
    assert results["sxxp_members_2024.parquet"] == 2

    # Query SQLite
    all_stocks = sqlite_store.get_universe_pit()
    assert len(all_stocks) == 4

    # Non-financials
    non_fin = sqlite_store.get_universe_pit(exclude_financials=True)
    assert len(non_fin) == 3
    assert "JPM US" not in non_fin["ticker"].values
    assert "1COV GY" in non_fin["ticker"].values
