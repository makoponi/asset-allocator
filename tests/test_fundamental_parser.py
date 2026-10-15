from pathlib import Path

import pandas as pd

from quant_portfolio.ingestion.bql_parser import (
    BQLFundamentalsIngestionEngine,
    normalize_fundamentals_records,
)
from quant_portfolio.storage.schema import SQLiteStore


def test_normalize_fundamentals_records_long_format() -> None:
    """Verify normalization of long-format fundamental data."""
    raw = pd.DataFrame(
        [
            {
                "date": "2024-03-31",
                "ticker": "AAPL US",
                "field": "pe_ratio",
                "value": "28.5",
                "filing_date": "2024-02-15",
            },
            {
                "date": "2024-03-31",
                "ticker": "AAPL US",
                "field": "roe",
                "value": 0.45,
                "filing_date": "2024-02-15",
            },
            {
                "date": "2024-03-31",
                "ticker": "MSFT US",
                "field": "unknown_metric",
                "value": 100.0,
                "filing_date": "2024-02-15",
            },
        ]
    )

    norm = normalize_fundamentals_records(raw)
    assert len(norm) == 2
    assert set(norm.columns) == {"as_of_date", "filing_date", "ticker", "metric", "value"}
    assert set(norm["metric"].unique()) == {"PE_RATIO", "RETURN_COM_EQY"}
    assert norm.loc[norm["metric"] == "PE_RATIO", "value"].iloc[0] == 28.5
    assert norm.loc[norm["metric"] == "RETURN_COM_EQY", "value"].iloc[0] == 0.45


def test_normalize_fundamentals_records_wide_format() -> None:
    """Verify normalization of wide-format fundamental data."""
    raw = pd.DataFrame(
        [
            {
                "as_of_date": "2024-01-31",
                "ticker": "SAP GY",
                "pe_ratio": 25.0,
                "px_to_book": 4.2,
                "operating_margin": 0.22,
            },
            {
                "as_of_date": "2024-01-31",
                "ticker": "ASML NA",
                "pe_ratio": 35.5,
                "px_to_book": None,
                "operating_margin": 0.31,
            },
        ]
    )

    norm = normalize_fundamentals_records(raw)
    assert len(norm) == 5  # 3 for SAP + 2 for ASML
    sap_metrics = set(norm[norm["ticker"] == "SAP GY"]["metric"].unique())
    assert sap_metrics == {"PE_RATIO", "PX_TO_BOOK_RATIO", "OPER_MARGIN"}
    assert norm["filing_date"].tolist() == norm["as_of_date"].tolist()


def test_normalize_fundamentals_look_ahead_bias_prevention() -> None:
    """Verify that records with filing_date > as_of_date are rejected."""
    raw = pd.DataFrame(
        [
            {
                "as_of_date": "2024-03-31",
                "filing_date": "2024-04-15",  # Future!
                "ticker": "AAPL US",
                "metric": "PE_RATIO",
                "value": 28.5,
            },
            {
                "as_of_date": "2024-03-31",
                "filing_date": "2024-03-01",  # Past (valid)
                "ticker": "AAPL US",
                "metric": "EV_TO_T12M_EBITDA",
                "value": 18.2,
            },
        ]
    )

    norm = normalize_fundamentals_records(raw)
    assert len(norm) == 1
    assert norm.iloc[0]["metric"] == "EV_TO_T12M_EBITDA"


def test_normalize_empty_fundamentals_dataframe() -> None:
    """Verify handling of empty DataFrame."""
    empty = pd.DataFrame()
    norm = normalize_fundamentals_records(empty)
    assert norm.empty
    assert list(norm.columns) == ["as_of_date", "filing_date", "ticker", "metric", "value"]


def test_bql_fundamentals_ingestion_engine(tmp_path: Path, sqlite_store: SQLiteStore) -> None:
    """Verify end-to-end ingestion of drop-zone fundamental files."""
    drop_dir = tmp_path / "drop_zone"
    drop_dir.mkdir(parents=True)

    # 1. Create a CSV export
    csv_file = drop_dir / "fundamental_export_spx.csv"
    csv_df = pd.DataFrame(
        [
            {
                "as_of_date": "2024-01-31",
                "ticker": "AAPL US",
                "PE_RATIO": 29.0,
                "FREE_CASH_FLOW_YIELD": 0.045,
            },
            {
                "as_of_date": "2024-01-31",
                "ticker": "NVDA US",
                "PE_RATIO": 65.0,
                "FREE_CASH_FLOW_YIELD": 0.025,
            },
        ]
    )
    csv_df.to_csv(csv_file, index=False)

    # 2. Create a Parquet export
    parquet_file = drop_dir / "fundamental_export_sxxp.parquet"
    parquet_df = pd.DataFrame(
        [
            {
                "date": "2024-01-31",
                "security": "MC FP",
                "field": "RETURN_COM_EQY",
                "value": 0.28,
                "filing_date": "2024-01-20",
            }
        ]
    )
    parquet_df.to_parquet(parquet_file, index=False)

    engine = BQLFundamentalsIngestionEngine(store=sqlite_store, raw_dir=drop_dir)
    results = engine.ingest_drop_zone(glob_pattern="fundamental_*.*")

    assert results[csv_file.name] == 4  # 2 tickers * 2 metrics
    assert results[parquet_file.name] == 1

    stored = sqlite_store.get_fundamentals_pit()
    assert len(stored) == 5
    assert set(stored["ticker"].unique()) == {"AAPL US", "NVDA US", "MC FP"}
