"""Unit tests for BloombergFundamentalsDownloader with mocked xbbg client."""

from typing import Any

import pandas as pd

from quant_portfolio.ingestion.xbbg_fundamentals import BloombergFundamentalsDownloader
from quant_portfolio.storage.schema import SQLiteStore


def test_bloomberg_fundamentals_downloader_mocked_batches(sqlite_store: SQLiteStore) -> None:
    """Verify batching and upserting of fundamental metrics via mocked bdp."""
    calls: list[list[str]] = []

    def mock_bdp(
        securities: list[str], flds: list[str] | None = None, **kwargs: Any
    ) -> pd.DataFrame:
        calls.append(securities)
        data: list[dict[str, Any]] = []
        for s in securities:
            data.append(
                {
                    "ticker": s,
                    "PE_RATIO": 25.0,
                    "PX_TO_BOOK_RATIO": 3.5,
                }
            )
        return pd.DataFrame(data).set_index("ticker")

    downloader = BloombergFundamentalsDownloader(
        store=sqlite_store,
        bdp_func=mock_bdp,
    )

    tickers = ["AAPL US", "MSFT US", "NVDA US"]
    as_of = "2024-01-31"

    # Batch size 2 should result in 2 calls for 3 tickers
    count = downloader.fetch_and_sync_fundamentals(
        tickers=tickers,
        as_of_date=as_of,
        batch_size=2,
        throttle=0.0,
        skip_existing=True,
    )

    assert len(calls) == 2
    assert count == 6  # 3 tickers * 2 metrics

    stored = sqlite_store.get_fundamentals_pit(as_of_date=as_of)
    assert len(stored) == 6
    assert set(stored["ticker"].unique()) == {"AAPL US", "MSFT US", "NVDA US"}

    # Second call with skip_existing=True should perform 0 calls
    calls.clear()
    count_cached = downloader.fetch_and_sync_fundamentals(
        tickers=tickers,
        as_of_date=as_of,
        skip_existing=True,
    )
    assert count_cached == 0
    assert len(calls) == 0


def test_bloomberg_fundamentals_downloader_historical_bdh(sqlite_store: SQLiteStore) -> None:
    """Verify batching and upserting of historical fundamental metrics via mocked bdh."""
    calls: list[list[str]] = []

    def mock_bdh(
        tickers: list[str],
        flds: list[str],
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        calls.append(tickers)
        rows: list[dict[str, Any]] = []
        for t in tickers:
            rows.append(
                {
                    "ticker": t,
                    "date": "2023-12-31",
                    "field": "PE_RATIO",
                    "value": 22.5,
                }
            )
            rows.append(
                {
                    "ticker": t,
                    "date": "2023-12-31",
                    "field": "RETURN_COM_EQY",
                    "value": 0.35,
                }
            )
        return pd.DataFrame(rows)

    downloader = BloombergFundamentalsDownloader(
        store=sqlite_store,
        bdh_func=mock_bdh,
    )

    tickers = ["SAP GY", "ASML NA", "MC FP"]
    count = downloader.fetch_and_sync_historical_fundamentals(
        tickers=tickers,
        start_date="2023-01-01",
        end_date="2023-12-31",
        batch_size=2,
        throttle=0.0,
        skip_existing=False,
    )

    assert len(calls) == 2  # 3 tickers in batches of 2
    assert count == 6  # 3 tickers * 2 fields

    # PE_RATIO is available immediately on 2023-12-31 (price-based)
    stored_immediate = sqlite_store.get_fundamentals_pit(as_of_date="2023-12-31")
    assert len(stored_immediate) == 3
    assert set(stored_immediate["metric"].unique()) == {"PE_RATIO"}

    # RETURN_COM_EQY is available after the 60-day reporting lag (2024-02-29 in leap year 2024)
    stored_lagged = sqlite_store.get_fundamentals_pit(as_of_date="2024-02-29")
    assert len(stored_lagged) == 3
    assert set(stored_lagged["metric"].unique()) == {"RETURN_COM_EQY"}
    assert set(stored_lagged["ticker"].unique()) == {"SAP GY", "ASML NA", "MC FP"}
