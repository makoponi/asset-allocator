"""Tests for SQLite and Parquet storage layer idempotency, types, and schema compliance."""

from datetime import date
from typing import TYPE_CHECKING

import pytest

from src.ingestion.storage import (
    FundamentalPiTRecord,
    MacroSeriesRecord,
    MarketStore,
    PriceEODRecord,
    PriceStore,
    UniverseMemberRecord,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def temp_market_store(tmp_path: Path) -> MarketStore:
    db_file = tmp_path / "test_market_store.db"
    return MarketStore(db_path=db_file)


@pytest.fixture
def temp_price_store(tmp_path: Path) -> PriceStore:
    parquet_dir = tmp_path / "processed_test"
    return PriceStore(base_dir=parquet_dir)


def test_universe_members_idempotency(temp_market_store: MarketStore) -> None:
    records = [
        UniverseMemberRecord(
            date="2025-01-31", ticker="MC FP", gics_sector="Consumer Discretionary", is_active=1
        ),
        UniverseMemberRecord(
            date="2025-01-31", ticker="SAP GY", gics_sector="Information Technology", is_active=1
        ),
    ]

    # First insertion
    inserted = temp_market_store.insert_universe_members(records)
    assert inserted == 2

    # Second insertion of exact same records (idempotent upsert)
    inserted_again = temp_market_store.insert_universe_members(records)
    assert inserted_again == 2

    stored = temp_market_store.get_universe_members(as_of_date="2025-01-31")
    assert len(stored) == 2
    assert {r.ticker for r in stored} == {"MC FP", "SAP GY"}


def test_universe_members_updates_active_flag(temp_market_store: MarketStore) -> None:
    rec_active = UniverseMemberRecord(
        date="2025-01-31", ticker="WDI GY", gics_sector="Financials", is_active=1
    )
    temp_market_store.insert_universe_members([rec_active])

    # Deactivation event
    rec_deactivated = UniverseMemberRecord(
        date="2025-01-31", ticker="WDI GY", gics_sector="Financials", is_active=0
    )
    temp_market_store.insert_universe_members([rec_deactivated])

    active_only = temp_market_store.get_universe_members(
        as_of_date="2025-01-31", is_active_only=True
    )
    assert len(active_only) == 0

    all_members = temp_market_store.get_universe_members(
        as_of_date="2025-01-31", is_active_only=False
    )
    assert len(all_members) == 1
    assert all_members[0].is_active == 0


def test_fundamentals_pit_idempotency(temp_market_store: MarketStore) -> None:
    records = [
        FundamentalPiTRecord(
            as_of_date="2024-12-31",
            filing_date="2025-02-15",
            ticker="NESN SW",
            metric_name="PE_RATIO",
            metric_value=21.4,
        ),
        FundamentalPiTRecord(
            as_of_date="2024-12-31",
            filing_date="2025-02-15",
            ticker="NESN SW",
            metric_name="RETURN_COM_EQY",
            metric_value=0.245,
        ),
    ]

    temp_market_store.insert_fundamentals_pit(records)

    # Updated metric value on later filing revision
    revised = [
        FundamentalPiTRecord(
            as_of_date="2024-12-31",
            filing_date="2025-02-20",
            ticker="NESN SW",
            metric_name="PE_RATIO",
            metric_value=21.8,
        )
    ]
    temp_market_store.insert_fundamentals_pit(revised)

    results = temp_market_store.get_fundamentals_pit(ticker="NESN SW", as_of_date="2024-12-31")
    assert len(results) == 2
    pe_record = next(r for r in results if r.metric_name == "PE_RATIO")
    assert pe_record.metric_value == 21.8
    assert pe_record.filing_date == "2025-02-20"


def test_macro_series_idempotency(temp_market_store: MarketStore) -> None:
    records = [
        MacroSeriesRecord(date="2025-01-10", series_id="V2TX", value=16.85),
        MacroSeriesRecord(date="2025-01-11", series_id="V2TX", value=17.10),
    ]
    temp_market_store.insert_macro_series(records)

    # Re-insert with revision
    revision = [MacroSeriesRecord(date="2025-01-11", series_id="V2TX", value=17.15)]
    temp_market_store.insert_macro_series(revision)

    stored = temp_market_store.get_macro_series(series_id="V2TX")
    assert len(stored) == 2
    assert stored[1].value == 17.15


def test_parquet_price_store_partitioning_and_idempotence(temp_price_store: PriceStore) -> None:
    records_2024 = [
        PriceEODRecord(
            date=date(2024, 12, 30),
            ticker="ASML NA",
            px_last=680.5,
            volume=120000.0,
            ret_1d=0.012,
        ),
        PriceEODRecord(
            date=date(2024, 12, 31),
            ticker="ASML NA",
            px_last=685.0,
            volume=98000.0,
            ret_1d=0.0066,
        ),
    ]
    records_2025 = [
        PriceEODRecord(
            date=date(2025, 1, 2),
            ticker="ASML NA",
            px_last=690.0,
            volume=115000.0,
            ret_1d=0.0073,
        ),
    ]

    # Write multiple years
    temp_price_store.write_prices([*records_2024, *records_2025])

    # Check partitions created
    partition_dirs = list(temp_price_store.base_dir.glob("year=*"))
    assert len(partition_dirs) == 2
    partition_years = {p.name for p in partition_dirs}
    assert partition_years == {"year=2024", "year=2025"}

    # Rewrite duplicate + revised record in 2024
    revised_2024 = [
        PriceEODRecord(
            date=date(2024, 12, 31),
            ticker="ASML NA",
            px_last=685.2,
            volume=98100.0,
            ret_1d=0.0069,
        ),
    ]
    temp_price_store.write_prices(revised_2024)

    # Read back all prices
    all_prices = temp_price_store.read_prices(tickers=["ASML NA"])
    assert len(all_prices) == 3
    dec31 = next(p for p in all_prices if p.date == date(2024, 12, 31))
    assert dec31.px_last == 685.2

    # Test date range filter
    filtered = temp_price_store.read_prices(
        tickers=["ASML NA"], start_date=date(2025, 1, 1), end_date=date(2025, 1, 31)
    )
    assert len(filtered) == 1
    assert filtered[0].date == date(2025, 1, 2)
