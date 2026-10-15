"""Unit tests for xbbg historical index constituent extractor and metadata caching."""

from typing import Any

import pandas as pd

from quant_portfolio.ingestion.xbbg_universe import (
    BloombergUniverseExtractor,
    _extract_ticker_list_from_bds,
    clean_bloomberg_member_ticker,
)
from quant_portfolio.storage.schema import SQLiteStore


def test_clean_bloomberg_member_ticker() -> None:
    """Verify stripping of ' Equity' suffix while preserving exchange and numerical prefixes."""
    assert clean_bloomberg_member_ticker("AAPL US Equity") == "AAPL US"
    assert clean_bloomberg_member_ticker("1CNHI IM Equity") == "1CNHI IM"
    assert clean_bloomberg_member_ticker("SAP GY") == "SAP GY"
    assert clean_bloomberg_member_ticker("  RY CT Equity  ") == "RY CT"


def test_extract_ticker_list_from_bds() -> None:
    """Verify extraction across different Bloomberg column header conventions."""
    df1 = pd.DataFrame({"Index Member": ["AAPL US Equity", "MSFT US Equity"]})
    assert _extract_ticker_list_from_bds(df1) == ["AAPL US", "MSFT US"]

    df2 = pd.DataFrame({"member_ticker_and_exchange_code": ["SAP GY Equity", "1COV GY Equity"]})
    assert _extract_ticker_list_from_bds(df2) == ["1COV GY", "SAP GY"]


def test_fetch_historical_index_members_with_mock(sqlite_store: SQLiteStore) -> None:
    """Verify bds call with END_DATE_OVERRIDE date parameter."""
    bds_calls: list[dict[str, Any]] = []

    def mock_bds(index_ticker: str, field: str, **kwargs: Any) -> pd.DataFrame:
        bds_calls.append({"index": index_ticker, "field": field, "kwargs": kwargs})
        return pd.DataFrame({"Index Member": ["AAPL US Equity", "JPM US Equity"]})

    extractor = BloombergUniverseExtractor(store=sqlite_store, bds_fn=mock_bds)
    members = extractor.fetch_historical_index_members("SPX Index", as_of_date="2024-01-31")

    assert members == ["AAPL US", "JPM US"]
    assert len(bds_calls) == 1
    assert bds_calls[0]["index"] == "SPX Index"
    assert bds_calls[0]["field"] == "INDX_MWEIGHT_HIST"
    assert bds_calls[0]["kwargs"]["END_DATE_OVERRIDE"] == "20240131"


def test_fetch_security_metadata_caching_protects_quotas(sqlite_store: SQLiteStore) -> None:
    """Verify that cached securities in SQLite are NOT queried again via BDP."""
    # Pre-populate cache with AAPL US
    sqlite_store.upsert_security_metadata_cache(
        [
            {
                "ticker": "AAPL US",
                "isin": "US0378331005",
                "gics_sector": 45,
                "local_currency": "USD",
                "name": "Apple Inc",
            }
        ]
    )

    bdp_queried_tickers: list[str] = []

    def mock_bdp(tickers: list[str], flds: list[str]) -> pd.DataFrame:
        bdp_queried_tickers.extend(tickers)
        # Return data for MSFT US
        return pd.DataFrame(
            [
                {
                    "ID_ISIN": "US5949181045",
                    "GICS_SECTOR": 45,
                    "CRNCY": "USD",
                    "NAME": "Microsoft Corp",
                }
            ],
            index=["MSFT US Equity"],
        )

    extractor = BloombergUniverseExtractor(store=sqlite_store, bdp_fn=mock_bdp)
    # Request both AAPL US (cached) and MSFT US (missing)
    meta = extractor.fetch_security_metadata(["AAPL US", "MSFT US"])

    # BDP was ONLY called for MSFT US Equity! AAPL US was served from SQLite cache!
    assert "AAPL US Equity" not in bdp_queried_tickers
    assert any("MSFT US" in t for t in bdp_queried_tickers)

    # Result contains both
    assert meta["AAPL US"]["isin"] == "US0378331005"
    assert meta["MSFT US"]["isin"] == "US5949181045"

    # SQLite cache now has MSFT US as well
    cache_check = sqlite_store.get_security_metadata_cache(["MSFT US"])
    assert "MSFT US" in cache_check


def test_sync_index_universe_date(sqlite_store: SQLiteStore) -> None:
    """Verify end-to-end extraction and upsert for a single date."""

    def mock_bds(index_ticker: str, field: str, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Index Member": ["AAPL US Equity", "JPM US Equity"],
                "Percent Weight": [7.0, 1.5],
            }
        )

    def mock_bdp(tickers: list[str], flds: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "ID_ISIN": "US0378331005",
                    "GICS_SECTOR": 45,
                    "CRNCY": "USD",
                    "NAME": "Apple Inc",
                },
                {
                    "ID_ISIN": "US46625H1005",
                    "GICS_SECTOR": 40,  # Financials!
                    "CRNCY": "USD",
                    "NAME": "JPMorgan Chase",
                },
            ],
            index=["AAPL US Equity", "JPM US Equity"],
        )

    extractor = BloombergUniverseExtractor(
        store=sqlite_store,
        bds_fn=mock_bds,
        bdp_fn=mock_bdp,
    )
    count = extractor.sync_index_universe_date("SPX Index", as_of_date="2024-01-31")
    assert count == 2

    # Query SQLite
    stocks = sqlite_store.get_universe_pit(as_of_date="2024-01-31")
    assert len(stocks) == 2

    aapl = stocks[stocks["ticker"] == "AAPL US"].iloc[0]
    assert aapl["isin"] == "US0378331005"
    assert bool(aapl["is_financial"]) is False

    jpm = stocks[stocks["ticker"] == "JPM US"].iloc[0]
    assert jpm["isin"] == "US46625H1005"
    assert bool(jpm["is_financial"]) is True


def test_fetch_security_metadata_handles_nan_and_delisted_tickers(
    sqlite_store: SQLiteStore,
) -> None:
    """Verify BDP parsing handles NaN GICS sectors, null ISINs, and missing names without error."""
    import numpy as np

    def mock_bdp_nan(tickers: list[str], flds: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "ID_ISIN": np.nan,
                    "GICS_SECTOR": np.nan,  # delisted or historical ticker
                    "CRNCY": np.nan,
                    "NAME": None,
                },
                {
                    "ID_ISIN": "US0378331005",
                    "GICS_SECTOR": "45.0",  # string float
                    "CRNCY": "usd",
                    "NAME": "Apple Inc",
                },
            ],
            index=["1284849D UN Equity", "AAPL US Equity"],
        )

    extractor = BloombergUniverseExtractor(store=sqlite_store, bdp_fn=mock_bdp_nan)
    meta = extractor.fetch_security_metadata(["1284849D UN", "AAPL US"])

    # 1284849D UN was converted safely
    assert meta["1284849D UN"]["isin"] == ""
    assert meta["1284849D UN"]["gics_sector"] == 0
    assert meta["1284849D UN"]["local_currency"] == "EUR"
    assert meta["1284849D UN"]["name"] is None

    # AAPL US
    assert meta["AAPL US"]["isin"] == "US0378331005"
    assert meta["AAPL US"]["gics_sector"] == 45
    assert meta["AAPL US"]["local_currency"] == "USD"
