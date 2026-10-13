"""Unit and integration tests for STOXX Europe 600 constituent ingestion."""

from typing import TYPE_CHECKING

import pytest

from src.ingestion.bql_client import ConstituentData, MockBloombergClient
from src.ingestion.storage import MarketStore
from src.ingestion.universe_ingestion import (
    UniverseIngestionManager,
    get_monthly_business_end_dates,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def temp_store(tmp_path: Path) -> MarketStore:
    return MarketStore(db_path=tmp_path / "test_universe.db")


def test_get_monthly_business_end_dates() -> None:
    # Test calendar for 2024
    dates_2024 = get_monthly_business_end_dates(2024, 2024)
    assert len(dates_2024) == 12

    # February 2024 was leap year (29 days, Thursday)
    assert dates_2024[1] == "2024-02-29"

    # March 2024: 31st was Sunday, 29th was Good Friday (weekday Friday: 2024-03-29)
    assert dates_2024[2] == "2024-03-29"

    # August 2024: 31st was Saturday -> 2024-08-30 (Friday)
    assert dates_2024[7] == "2024-08-30"


def test_constituent_ingestion_idempotence_and_caching(temp_store: MarketStore) -> None:
    mock_client = MockBloombergClient()
    constituents = [
        ConstituentData(ticker="NESN SW", gics_sector="Consumer Staples", is_active=True),
        ConstituentData(ticker="ASML NA", gics_sector="Information Technology", is_active=True),
        ConstituentData(ticker="NOVN SW", gics_sector="Health Care", is_active=True),
    ]
    mock_client.set_constituents_for_date("2024-01-31", constituents)

    manager = UniverseIngestionManager(bql_client=mock_client, market_store=temp_store)

    # First ingestion -> inserts 3 records
    inserted = manager.ingest_constituents_for_date("2024-01-31")
    assert inserted == 3

    # Check store
    stored = temp_store.get_universe_members(as_of_date="2024-01-31")
    assert len(stored) == 3

    # Second ingestion -> date already exists in store, must skip without touching client
    mock_client.set_constituents_for_date("2024-01-31", [])
    inserted_cached = manager.ingest_constituents_for_date("2024-01-31")
    assert inserted_cached == 0

    # Data remains intact
    stored_again = temp_store.get_universe_members(as_of_date="2024-01-31")
    assert len(stored_again) == 3


def test_survivorship_bias_handling_delisting_tracking(temp_store: MarketStore) -> None:
    """Verifies that historic constituents (like Wirecard WDI GY) are preserved Point-in-Time.

    Wirecard was part of the index until June 2020, then replaced by Delivery Hero.
    """
    mock_client = MockBloombergClient()

    # May 2020: Wirecard is active in SXXP
    may_constituents = [
        ConstituentData(ticker="SAP GY", gics_sector="Information Technology", is_active=True),
        ConstituentData(ticker="WDI GY", gics_sector="Information Technology", is_active=True),
    ]
    mock_client.set_constituents_for_date("2020-05-29", may_constituents)

    # July 2020: Wirecard removed, replaced by Delivery Hero (DHER GY)
    july_constituents = [
        ConstituentData(ticker="SAP GY", gics_sector="Information Technology", is_active=True),
        ConstituentData(ticker="DHER GY", gics_sector="Consumer Discretionary", is_active=True),
    ]
    mock_client.set_constituents_for_date("2020-07-31", july_constituents)

    manager = UniverseIngestionManager(bql_client=mock_client, market_store=temp_store)

    manager.ingest_constituents_for_date("2020-05-29")
    manager.ingest_constituents_for_date("2020-07-31")

    # In May 2020, WDI GY was active
    may_members = temp_store.get_universe_members(as_of_date="2020-05-29")
    assert any(m.ticker == "WDI GY" for m in may_members)
    assert not any(m.ticker == "DHER GY" for m in may_members)

    # In July 2020, WDI GY is NOT in the active universe, but DHER GY is
    july_members = temp_store.get_universe_members(as_of_date="2020-07-31")
    assert not any(m.ticker == "WDI GY" for m in july_members)
    assert any(m.ticker == "DHER GY" for m in july_members)

    # Total distinct unique historical tickers includes both WDI GY and DHER GY
    unique_tickers = manager.get_unique_historical_tickers()
    assert "WDI GY" in unique_tickers
    assert "DHER GY" in unique_tickers
    assert "SAP GY" in unique_tickers
    assert len(unique_tickers) == 3


def test_ingest_universe_range(temp_store: MarketStore) -> None:
    mock_client = MockBloombergClient()

    # Setup dates for Q1 2024
    dates = ["2024-01-31", "2024-02-29", "2024-03-29"]
    for d in dates:
        mock_client.set_constituents_for_date(
            d,
            [
                ConstituentData(ticker="MC FP", gics_sector="Consumer Discretionary"),
                ConstituentData(ticker="OR FP", gics_sector="Consumer Staples"),
            ],
        )

    manager = UniverseIngestionManager(bql_client=mock_client, market_store=temp_store)

    results = manager.ingest_universe_range(
        start_year=2024, end_year=2024, cutoff_date="2024-03-31"
    )

    # Should have ingested January, February, and March
    assert results.get("2024-01-31") == 2
    assert results.get("2024-02-29") == 2
    assert results.get("2024-03-29") == 2

    # Second run should result in 0 queries (all cached)
    results_second = manager.ingest_universe_range(
        start_year=2024, end_year=2024, cutoff_date="2024-03-31"
    )
    assert len(results_second) == 0
