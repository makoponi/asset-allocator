"""Unit tests for BQL client protocol and mock client implementations."""

from datetime import date

import pytest

from src.ingestion.bql_client import (
    ConstituentData,
    HistoricalBar,
    MockBloombergClient,
    RealBloombergClient,
)


def test_mock_bloomberg_client_constituents() -> None:
    client = MockBloombergClient()
    assert client.get_index_constituents("SXXP Index", "2024-01-31") == []

    expected = [
        ConstituentData(ticker="MC FP", gics_sector="Consumer Discretionary"),
        ConstituentData(ticker="SAP GY", gics_sector="Information Technology"),
    ]
    client.set_constituents_for_date("2024-01-31", expected)

    retrieved = client.get_index_constituents("SXXP Index", "2024-01-31")
    assert retrieved == expected
    assert client.get_index_constituents("SXXP Index", "2024-02-29") == []


def test_mock_bloomberg_client_historical_bars() -> None:
    bars = [
        HistoricalBar(date=date(2024, 1, 2), ticker="MC FP", px_last=730.0, volume=50000.0),
        HistoricalBar(date=date(2024, 1, 3), ticker="MC FP", px_last=735.0, volume=48000.0),
        HistoricalBar(date=date(2024, 1, 4), ticker="MC FP", px_last=740.0, volume=51000.0),
        HistoricalBar(date=date(2024, 1, 2), ticker="SAP GY", px_last=140.0, volume=80000.0),
    ]
    client = MockBloombergClient(bars=bars)

    # Filter by ticker and date range
    mc_bars = client.get_historical_bars(
        tickers=["MC FP"], start_date=date(2024, 1, 2), end_date=date(2024, 1, 3)
    )
    assert len(mc_bars) == 2
    assert [b.date for b in mc_bars] == [date(2024, 1, 2), date(2024, 1, 3)]

    # Multiple tickers
    both_bars = client.get_historical_bars(
        tickers=["MC FP", "SAP GY"], start_date=date(2024, 1, 2), end_date=date(2024, 1, 2)
    )
    assert len(both_bars) == 2


def test_real_bloomberg_client_offline_behavior() -> None:
    client = RealBloombergClient()
    # When blpapi is missing, raises RuntimeError
    with pytest.raises(
        RuntimeError, match="blpapi is not installed or Bloomberg Terminal is not running"
    ):
        client.get_index_constituents("SXXP Index", "2024-01-31")

    with pytest.raises(
        RuntimeError, match="blpapi is not installed or Bloomberg Terminal is not running"
    ):
        client.get_historical_bars(["MC FP"], date(2024, 1, 1), date(2024, 1, 2))
