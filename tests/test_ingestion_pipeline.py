"""Unit tests for the end-to-end transatlantic ingestion pipeline."""

from pathlib import Path
from typing import Any

import pandas as pd

from quant_portfolio.config import PortfolioConfig, StoragePaths
from quant_portfolio.ingestion.pipeline import TransatlanticIngestionPipeline
from quant_portfolio.ingestion.xbbg_client import BloombergSeriesDownloader
from quant_portfolio.storage.schema import ParquetMarketStore, SQLiteStore


def test_transatlantic_ingestion_pipeline_end_to_end(tmp_path: Path) -> None:
    """Verify full Phase 2 workflow from BQL raw drop-zone to SQLite and Parquet store."""
    db_file = tmp_path / "test_quant.db"
    parquet_dir = tmp_path / "parquet"
    raw_dir = tmp_path / "raw_bql_exports"
    raw_dir.mkdir(parents=True)

    sqlite_store = SQLiteStore(db_file)
    parquet_store = ParquetMarketStore(parquet_dir)

    # 1. Create synthetic BQL export files in drop-zone
    # SPX file
    df_spx = pd.DataFrame(
        [
            {
                "date": "2024-01-31",
                "ticker": "AAPL US",
                "isin": "US0378331005",
                "benchmark": "SPX Index",
                "gics_sector": 45,
                "local_currency": "USD",
            },
            {
                "date": "2024-01-31",
                "ticker": "JPM US",  # Financials -> must be excluded from active market series!
                "isin": "US46625H1005",
                "benchmark": "SPX Index",
                "gics_sector": 40,
                "local_currency": "USD",
            },
        ]
    )
    df_spx.to_csv(raw_dir / "spx_members.csv", index=False)

    # SXXP file
    df_sxxp = pd.DataFrame(
        [
            {
                "date": "2024-01-31",
                "ticker": "SAP GY",
                "isin": "DE0007164600",
                "benchmark": "SXXP Index",
                "gics_sector": 45,
                "local_currency": "EUR",
            }
        ]
    )
    df_sxxp.to_parquet(raw_dir / "sxxp_members.parquet", index=False)

    # 2. Mock bdh downloader
    downloaded_tickers: list[str] = []

    def mock_bdh(
        tickers: list[str],
        flds: list[str],
        start_date: str,
        end_date: str,
        Currency: str,
    ) -> pd.DataFrame:
        downloaded_tickers.extend(tickers)
        dates = pd.date_range("2024-01-02", periods=2)
        cols = pd.MultiIndex.from_tuples(
            [(t, "PX_LAST") for t in tickers]
            + [(t, "TOT_RETURN_INDEX_GROSS_DVDS") for t in tickers]
            + [(t, "PX_VOLUME") for t in tickers]
        )
        vals = [
            [150.0] * len(tickers) + [450.0] * len(tickers) + [500000.0] * len(tickers),
            [152.0] * len(tickers) + [455.0] * len(tickers) + [600000.0] * len(tickers),
        ]
        return pd.DataFrame(vals, index=dates, columns=cols)

    downloader = BloombergSeriesDownloader(
        batch_size=50,
        throttle_seconds=0.0,
        bdh_fn=mock_bdh,
    )

    # 3. Configure pipeline
    cfg = PortfolioConfig(
        paths=StoragePaths(
            root_dir=tmp_path,
            data_dir=tmp_path / "data",
            raw_bql_dir=raw_dir,
            db_path=db_file,
            parquet_dir=parquet_dir,
        )
    )
    pipeline = TransatlanticIngestionPipeline(
        config=cfg,
        sqlite_store=sqlite_store,
        parquet_store=parquet_store,
        downloader=downloader,
    )

    # 4. Execute pipeline
    summary = pipeline.run_full_pipeline(start_date="2024-01-01", end_date="2024-01-31")

    # 5. Assertions
    assert summary.universe_records_upserted == 3
    assert summary.active_stocks_total == 2  # JPM US excluded!
    assert summary.active_stocks_na == 1  # AAPL US
    assert summary.active_stocks_eu == 1  # SAP GY
    assert summary.market_rows_written > 0

    # Ensure JPM US was NEVER requested via Bloomberg BDH
    assert "JPM US" not in downloaded_tickers
    assert "JPM US Equity" not in downloaded_tickers
    assert "AAPL US Equity" in downloaded_tickers
    assert "SAP GY Equity" in downloaded_tickers

    # 6. Read back from Parquet store
    stored_market = parquet_store.read_daily_market()
    assert len(stored_market) > 0
    assert set(stored_market["ticker"].unique()) == {"AAPL US", "SAP GY"}
    assert "JPM US" not in stored_market["ticker"].values


def test_download_and_sync_skip_existing(tmp_path: Path) -> None:
    """Verify skip_existing skips already stored tickers in Parquet."""
    db_file = tmp_path / "test_skip.db"
    parquet_dir = tmp_path / "parquet_skip"
    store = SQLiteStore(db_file)
    p_store = ParquetMarketStore(parquet_dir)

    # Insert 2 non-financial tickers
    records = [
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
            "ticker": "MSFT US",
            "isin": "US5949181045",
            "region_bloc": "NA",
            "source_index": "SPX Index",
            "gics_sector": 45,
            "local_currency": "USD",
            "is_financial": False,
        },
    ]
    store.upsert_universe_pit(records)

    calls: list[list[str]] = []

    def mock_bdh(tickers: list[str], **kwargs: Any) -> pd.DataFrame:
        calls.append(tickers)
        dates = pd.date_range("2024-01-02", periods=1)
        cols = pd.MultiIndex.from_tuples(
            [(t, "PX_LAST") for t in tickers]
            + [(t, "TOT_RETURN_INDEX_GROSS_DVDS") for t in tickers]
            + [(t, "PX_VOLUME") for t in tickers]
        )
        vals = [[150.0] * len(tickers) + [450.0] * len(tickers) + [500000.0] * len(tickers)]
        return pd.DataFrame(vals, index=dates, columns=cols)

    downloader = BloombergSeriesDownloader(batch_size=50, throttle_seconds=0.0, bdh_fn=mock_bdh)
    pipeline = TransatlanticIngestionPipeline(
        sqlite_store=store,
        parquet_store=p_store,
        downloader=downloader,
    )

    # First run: downloads both AAPL US and MSFT US
    rows1 = pipeline.download_and_sync_market_series(start_date="2024-01-01", end_date="2024-01-05")
    assert rows1 == 2
    assert len(calls) == 1
    assert "AAPL US Equity" in calls[0] and "MSFT US Equity" in calls[0]

    # Second run with skip_existing=True -> should make 0 calls
    calls.clear()
    rows2 = pipeline.download_and_sync_market_series(
        start_date="2024-01-01",
        end_date="2024-01-05",
        skip_existing=True,
    )
    assert rows2 == 0
    assert len(calls) == 0


def test_pipeline_fetch_and_sync_historical_universe_monthly(tmp_path: Path) -> None:
    """Verify monthly universe extraction via xbbg and checkpoint skipping."""
    from typing import Any

    from quant_portfolio.ingestion.xbbg_universe import BloombergUniverseExtractor

    db_file = tmp_path / "test_hist_quant.db"
    store = SQLiteStore(db_file)

    bds_calls: list[str] = []

    def mock_bds(index_ticker: str, field: str, **kwargs: Any) -> pd.DataFrame:
        date_str = str(kwargs.get("END_DATE_OVERRIDE"))
        bds_calls.append(f"{index_ticker}:{date_str}")
        return pd.DataFrame({"Index Member": ["AAPL US Equity", "MSFT US Equity"]})

    def mock_bdp(tickers: list[str], flds: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"ID_ISIN": f"ISIN_{t[:4]}", "GICS_SECTOR": 45, "CRNCY": "USD", "NAME": t}
                for t in tickers
            ],
            index=tickers,
        )

    extractor = BloombergUniverseExtractor(store=store, bds_fn=mock_bds, bdp_fn=mock_bdp)
    pipeline = TransatlanticIngestionPipeline(
        sqlite_store=store,
        universe_extractor=extractor,
    )

    # 1. Run for Jan and Feb 2024
    res1 = pipeline.fetch_and_sync_historical_universe(
        start_date="2024-01-01",
        end_date="2024-02-29",
        freq="BME",
        indices=("SPX Index",),
    )
    assert res1["SPX Index"] == 4  # 2 dates * 2 members = 4 records
    assert len(bds_calls) == 2  # 2 calls: Jan 31 and Feb 29

    # 2. Run again with skip_existing=True -> should skip both dates and make 0 calls!
    bds_calls.clear()
    res2 = pipeline.fetch_and_sync_historical_universe(
        start_date="2024-01-01",
        end_date="2024-02-29",
        freq="BME",
        indices=("SPX Index",),
        skip_existing=True,
    )
    assert res2["SPX Index"] == 0
    assert len(bds_calls) == 0  # Zero BDS calls made! Checkpoint worked!
