"""Unit tests for xbbg series downloader, 50-ticker batching, and output formatting."""

import time
from typing import TYPE_CHECKING, Any

import pandas as pd

from quant_portfolio.ingestion.xbbg_client import (
    BloombergSeriesDownloader,
    chunk_list,
    format_xbbg_bdh_output,
)

if TYPE_CHECKING:
    from quant_portfolio.config import RegionBloc


def test_chunk_list() -> None:
    """Verify chunking sequence into maximum size chunks."""
    items = list(range(105))
    chunks = chunk_list(items, chunk_size=50)
    assert len(chunks) == 3
    assert len(chunks[0]) == 50
    assert len(chunks[1]) == 50
    assert len(chunks[2]) == 5


def test_format_xbbg_bdh_output_multiindex() -> None:
    """Verify conversion of MultiIndex xbbg output into daily market schema."""
    dates = pd.date_range("2024-01-02", periods=2, freq="D")
    cols = pd.MultiIndex.from_tuples(
        [
            ("AAPL US", "PX_LAST"),
            ("AAPL US", "TOT_RETURN_INDEX_GROSS_DVDS"),
            ("AAPL US", "PX_VOLUME"),
            ("SAP GY", "PX_LAST"),
            ("SAP GY", "TOT_RETURN_INDEX_GROSS_DVDS"),
            ("SAP GY", "PX_VOLUME"),
        ]
    )
    data = [
        [180.5, 520.0, 50000000.0, 140.0, 310.0, 20000000.0],
        [182.0, 524.5, 45000000.0, 141.5, 313.2, 18000000.0],
    ]
    raw_df = pd.DataFrame(data, index=dates, columns=cols)
    region_map: dict[str, RegionBloc] = {"AAPL US": "NA", "SAP GY": "EU"}

    formatted = format_xbbg_bdh_output(raw_df, region_map)
    assert len(formatted) == 4
    assert set(formatted.columns) == {
        "date",
        "ticker",
        "region_bloc",
        "px_last_eur",
        "tot_return_idx_eur",
        "volume_eur",
    }

    aapl_row = formatted[
        (formatted["ticker"] == "AAPL US") & (formatted["date"] == "2024-01-02")
    ].iloc[0]
    assert aapl_row["region_bloc"] == "NA"
    assert aapl_row["px_last_eur"] == 180.5
    assert aapl_row["tot_return_idx_eur"] == 520.0
    assert aapl_row["volume_eur"] == 50000000.0

    sap_row = formatted[
        (formatted["ticker"] == "SAP GY") & (formatted["date"] == "2024-01-02")
    ].iloc[0]
    assert sap_row["region_bloc"] == "EU"
    assert sap_row["px_last_eur"] == 140.0


def test_bloomberg_series_downloader_mocked_batches() -> None:
    """Verify batching of 120 tickers into 3 calls with Currency='EUR'."""
    call_log: list[dict[str, Any]] = []

    def mock_bdh(
        tickers: list[str],
        flds: list[str],
        start_date: str,
        end_date: str,
        Currency: str,
    ) -> pd.DataFrame:
        call_log.append(
            {
                "tickers": tickers,
                "flds": flds,
                "start_date": start_date,
                "end_date": end_date,
                "currency": Currency,
            }
        )
        dates = pd.date_range("2024-01-02", periods=1)
        cols = pd.MultiIndex.from_tuples(
            [(t, "PX_LAST") for t in tickers]
            + [(t, "TOT_RETURN_INDEX_GROSS_DVDS") for t in tickers]
            + [(t, "PX_VOLUME") for t in tickers]
        )
        vals = [[100.0] * len(tickers) + [200.0] * len(tickers) + [1000000.0] * len(tickers)]
        return pd.DataFrame(vals, index=dates, columns=cols)

    # 120 synthetic tickers
    tickers = [f"TICK_{i} US" for i in range(120)]
    region_map: dict[str, RegionBloc] = {t: "NA" for t in tickers}

    t0 = time.perf_counter()
    downloader = BloombergSeriesDownloader(
        batch_size=50,
        throttle_seconds=0.05,  # fast test throttle
        bdh_fn=mock_bdh,
    )
    res = downloader.fetch_all_series(
        tickers=tickers,
        start_date="2024-01-01",
        end_date="2024-01-05",
        ticker_region_map=region_map,
        currency="EUR",
    )
    elapsed = time.perf_counter() - t0

    # Check batching: 120 tickers with batch_size=50 -> 3 batches (50, 50, 20)
    assert len(call_log) == 3
    assert len(call_log[0]["tickers"]) == 50
    assert len(call_log[1]["tickers"]) == 50
    assert len(call_log[2]["tickers"]) == 20
    assert call_log[0]["currency"] == "EUR"

    # Throttle occurred between batch 1-2 and 2-3 (at least 2 * 0.05s = 0.10s)
    assert elapsed >= 0.08

    # Result contains all 120 tickers
    assert len(res) == 120
    assert set(res["ticker"].unique()) == set(tickers)
