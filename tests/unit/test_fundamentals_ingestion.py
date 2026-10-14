"""Unit tests for Point-in-Time fundamental ingestion and parsing logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from src.ingestion.fundamentals_ingestion import (
    MARKET_MULTIPLES,
    FundamentalIngestionManager,
    parse_bdh_fundamental_rows,
)
from src.ingestion.storage import FundamentalPiTRecord, MarketStore

if TYPE_CHECKING:
    from pathlib import Path


class MockIterRowsDf:
    """Mock container mimicking narwhals / polars DataFrame with .iter_rows()."""

    def __init__(self, rows: list[list[Any]]) -> None:
        self._rows = rows

    def iter_rows(self) -> list[list[Any]]:
        return self._rows


@pytest.fixture
def temp_store(tmp_path: Path) -> MarketStore:
    return MarketStore(db_path=tmp_path / "test_fund_store.db")


def test_parse_bdh_fundamental_rows_iter_rows() -> None:
    # 2 rows: 1 market multiple (0 lag) and 1 accounting metric (90d lag)
    mock_df = MockIterRowsDf(
        [
            ["MC FP Equity", "2023-06-30", "PE_RATIO", "25.5"],
            ["MC FP Equity", "2023-06-30", "PROF_MARGIN", "18.2"],
            ["MC FP Equity", "2023-06-30", "QUICK_RATIO", "None"],  # should be skipped
        ]
    )

    records = parse_bdh_fundamental_rows(mock_df, market_multiples=MARKET_MULTIPLES, lag_days=90)
    assert len(records) == 2

    r_pe = next(r for r in records if r.metric_name == "PE_RATIO")
    assert r_pe.ticker == "MC FP"
    assert r_pe.as_of_date == "2023-06-30"
    assert r_pe.filing_date == "2023-06-30"  # 0-lag for market multiple
    assert r_pe.metric_value == pytest.approx(25.5)

    r_prof = next(r for r in records if r.metric_name == "PROF_MARGIN")
    assert r_prof.ticker == "MC FP"
    assert r_prof.as_of_date == "2023-06-30"
    assert r_prof.filing_date == "2023-09-28"  # 2023-06-30 + 90 days = 2023-09-28
    assert r_prof.metric_value == pytest.approx(18.2)


def test_parse_bdh_fundamental_rows_pandas_multiindex() -> None:
    # Test pandas DataFrame with MultiIndex columns (ticker, field)
    cols = pd.MultiIndex.from_tuples(
        [
            ("SAP GY Equity", "PX_TO_BOOK_RATIO"),
            ("SAP GY Equity", "TOT_DEBT_TO_TOT_EQY"),
        ]
    )
    idx = pd.to_datetime(["2022-12-31"])
    df = pd.DataFrame([[4.2, 45.0]], index=idx, columns=cols)

    records = parse_bdh_fundamental_rows(df, market_multiples=MARKET_MULTIPLES, lag_days=90)
    assert len(records) == 2

    r_pb = next(r for r in records if r.metric_name == "PX_TO_BOOK_RATIO")
    assert r_pb.ticker == "SAP GY"
    assert r_pb.as_of_date == "2022-12-31"
    assert r_pb.filing_date == "2022-12-31"
    assert r_pb.metric_value == pytest.approx(4.2)

    r_debt = next(r for r in records if r.metric_name == "TOT_DEBT_TO_TOT_EQY")
    assert r_debt.ticker == "SAP GY"
    assert r_debt.as_of_date == "2022-12-31"
    assert r_debt.filing_date == "2023-03-31"  # 2022-12-31 + 90 days = 2023-03-31
    assert r_debt.metric_value == pytest.approx(45.0)


def test_fundamental_manager_resume_and_missing_tickers(temp_store: MarketStore) -> None:
    manager = FundamentalIngestionManager(market_store=temp_store)

    # Initial state: empty
    assert len(manager.get_already_ingested_tickers()) == 0

    all_tickers = ["MC FP", "SAP GY", "NESN SW", "ASML NA"]
    assert manager.get_missing_tickers(all_tickers) == all_tickers

    # Insert records for MC FP and SAP GY
    records = [
        FundamentalPiTRecord(
            as_of_date="2023-06-30",
            filing_date="2023-06-30",
            ticker="MC FP",
            metric_name="PE_RATIO",
            metric_value=24.0,
        ),
        FundamentalPiTRecord(
            as_of_date="2023-06-30",
            filing_date="2023-09-28",
            ticker="SAP GY",
            metric_name="RETURN_COM_EQY",
            metric_value=15.0,
        ),
    ]
    temp_store.insert_fundamentals_pit(records)

    done = manager.get_already_ingested_tickers()
    assert done == {"MC FP", "SAP GY"}

    missing = manager.get_missing_tickers(all_tickers)
    assert missing == ["NESN SW", "ASML NA"]


def test_fundamental_manager_idempotent_upsert(temp_store: MarketStore) -> None:
    r1 = FundamentalPiTRecord(
        as_of_date="2023-01-31",
        filing_date="2023-01-31",
        ticker="ALV GY",
        metric_name="PE_RATIO",
        metric_value=10.5,
    )
    written1 = temp_store.insert_fundamentals_pit([r1])
    assert written1 == 1

    # Insert updated value for same key
    r2 = FundamentalPiTRecord(
        as_of_date="2023-01-31",
        filing_date="2023-01-31",
        ticker="ALV GY",
        metric_name="PE_RATIO",
        metric_value=11.2,
    )
    written2 = temp_store.insert_fundamentals_pit([r2])
    assert written2 == 1

    queried = temp_store.get_fundamentals_pit(ticker="ALV GY", as_of_date="2023-01-31")
    assert len(queried) == 1
    assert queried[0].metric_value == pytest.approx(11.2)
