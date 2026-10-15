"""End-to-end ingestion pipeline for Phase 2 transatlantic equity universe and market data."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_portfolio.config import PortfolioConfig, RegionBloc
from quant_portfolio.ingestion.bql_parser import BQLUniverseIngestionEngine
from quant_portfolio.ingestion.bql_templates import CORE_UNIVERSE_INDICES
from quant_portfolio.ingestion.calendar_alignment import align_market_panel
from quant_portfolio.ingestion.xbbg_client import (
    BloombergSeriesDownloader,
    chunk_list,
    format_xbbg_bdh_output,
)
from quant_portfolio.ingestion.xbbg_universe import BloombergUniverseExtractor
from quant_portfolio.storage.schema import ParquetMarketStore, SQLiteStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionSummary:
    """Summary statistics for an ingestion run."""

    universe_records_upserted: int
    active_stocks_total: int
    active_stocks_na: int
    active_stocks_eu: int
    market_rows_written: int


class TransatlanticIngestionPipeline:
    """Pipeline coordinator for Phase 2 universe ingestion and market data synchronization."""

    def __init__(
        self,
        config: PortfolioConfig | None = None,
        sqlite_store: SQLiteStore | None = None,
        parquet_store: ParquetMarketStore | None = None,
        downloader: BloombergSeriesDownloader | None = None,
        universe_extractor: BloombergUniverseExtractor | None = None,
    ) -> None:
        self.config = config or PortfolioConfig()
        resolved_paths = self.config.paths.resolve_paths()
        self.sqlite_store = sqlite_store or SQLiteStore(resolved_paths.db_path)
        self.parquet_store = parquet_store or ParquetMarketStore(resolved_paths.parquet_dir)
        self.raw_bql_dir = resolved_paths.raw_bql_dir
        self.downloader = downloader or BloombergSeriesDownloader()
        self.universe_extractor = universe_extractor or BloombergUniverseExtractor(
            store=self.sqlite_store
        )

    def ingest_universe_from_drop_zone(
        self,
        raw_dir: Path | str | None = None,
        glob_pattern: str = "*.*",
    ) -> int:
        """Parse raw BQL constituent exports and populate universe_pit in SQLite."""
        target_dir = Path(raw_dir) if raw_dir is not None else self.raw_bql_dir
        engine = BQLUniverseIngestionEngine(store=self.sqlite_store, raw_dir=target_dir)
        results = engine.ingest_drop_zone(glob_pattern=glob_pattern, dedup_by_isin=True)
        total_files = len(results)
        total_upserted = sum(count for count in results.values() if count > 0)
        logger.info(
            "Ingested %d universe records from %d drop-zone files",
            total_upserted,
            total_files,
        )
        return total_upserted

    def is_universe_date_synced(self, index_ticker: str, as_of_date: str) -> bool:
        """Check whether constituents for an index on a date already exist in SQLite."""
        query = "SELECT COUNT(*) FROM universe_pit WHERE source_index = ? AND as_of_date = ?"
        with self.sqlite_store.get_connection() as conn:
            cursor = conn.execute(query, (index_ticker, as_of_date))
            count = cursor.fetchone()[0]
        return bool(count > 0)

    def fetch_and_sync_historical_universe(
        self,
        start_date: str = "2010-01-01",
        end_date: str = "2026-12-31",
        freq: str = "BME",
        indices: tuple[str, ...] = CORE_UNIVERSE_INDICES,
        skip_existing: bool = True,
    ) -> dict[str, int]:
        """Fetch historical monthly index constituents via xbbg with checkpointing."""
        # Generate date sequence (e.g. business month-ends)
        dt_index = pd.date_range(start=start_date, end=end_date, freq=freq)
        dates = [d.strftime("%Y-%m-%d") for d in dt_index]
        logger.info(
            "Beginning universe sync for %d indices across %d dates (%s to %s)",
            len(indices),
            len(dates),
            start_date,
            end_date,
        )

        counts_by_index: dict[str, int] = {idx: 0 for idx in indices}

        for as_of_date in dates:
            for index_ticker in indices:
                if skip_existing and self.is_universe_date_synced(index_ticker, as_of_date):
                    logger.debug("Skipping %s at %s (already in DB)", index_ticker, as_of_date)
                    continue

                try:
                    count = self.universe_extractor.sync_index_universe_date(
                        index_ticker=index_ticker,
                        as_of_date=as_of_date,
                    )
                    counts_by_index[index_ticker] += count
                    logger.info("Synced %s at %s: %d members", index_ticker, as_of_date, count)
                except Exception as exc:
                    logger.error("Failed syncing %s at %s: %s", index_ticker, as_of_date, exc)

        return counts_by_index

    def download_and_sync_market_series(
        self,
        start_date: str = "2010-01-01",
        end_date: str = "2026-12-31",
        currency: str = "EUR",
        max_ffill_days: int = 3,
        batch_size: int = 50,
        skip_existing: bool = False,
        limit_tickers: int | None = None,
    ) -> int:
        """Query non-financial active universe, fetch series in EUR via xbbg, align and persist."""
        universe_df = self.sqlite_store.get_universe_pit(exclude_financials=True)
        if universe_df.empty:
            logger.warning("No active non-financial stocks found in universe_pit.")
            return 0

        # Unique tickers and region mapping
        unique_tickers: list[str] = sorted(universe_df["ticker"].unique())
        region_map: dict[str, RegionBloc] = {}
        for _, row in universe_df.drop_duplicates(subset=["ticker"]).iterrows():
            reg = str(row["region_bloc"])
            if reg in ("NA", "EU"):
                region_map[str(row["ticker"])] = reg  # type: ignore[assignment]

        if skip_existing:
            already_synced = self.parquet_store.get_existing_tickers()
            before_count = len(unique_tickers)
            unique_tickers = [t for t in unique_tickers if t not in already_synced]
            skipped = before_count - len(unique_tickers)
            if skipped > 0:
                logger.info("Skipping %d tickers already stored in Parquet.", skipped)

        if limit_tickers is not None and limit_tickers > 0:
            unique_tickers = unique_tickers[:limit_tickers]

        if not unique_tickers:
            logger.info("No tickers to download (all already synced or empty).")
            return 0

        logger.info(
            "Fetching EUR market series for %d non-financial stocks (%s to %s)",
            len(unique_tickers),
            start_date,
            end_date,
        )

        batches = chunk_list(unique_tickers, batch_size)
        total_batches = len(batches)
        total_rows_written = 0

        for i, batch in enumerate(batches, start=1):
            try:
                batch_raw = self.downloader.fetch_series_batch(
                    tickers=batch,
                    start_date=start_date,
                    end_date=end_date,
                    currency=currency,
                )
                if not batch_raw.empty:
                    formatted = format_xbbg_bdh_output(batch_raw, region_map)
                    if not formatted.empty:
                        aligned = align_market_panel(formatted, max_ffill_days=max_ffill_days)
                        rows_written = self.parquet_store.write_daily_market(aligned)
                        total_rows_written += rows_written
                        print(
                            f"[{i}/{total_batches}] Synced {len(batch)} tickers "
                            f"({rows_written} rows persisted) | Total: {total_rows_written}"
                        )
                        logger.info(
                            "[%d/%d] Synced %d tickers (%d observations persisted) | Total: %d",
                            i,
                            total_batches,
                            len(batch),
                            rows_written,
                            total_rows_written,
                        )
                    else:
                        logger.warning(
                            "[%d/%d] No valid series formatted for batch %s",
                            i,
                            total_batches,
                            batch,
                        )
                else:
                    logger.warning(
                        "[%d/%d] Empty BDH response for batch %s", i, total_batches, batch
                    )
            except Exception as exc:
                logger.error("[%d/%d] Failed syncing batch %s: %s", i, total_batches, batch, exc)

            if i < total_batches and self.downloader.throttle_seconds > 0:
                time.sleep(self.downloader.throttle_seconds)

        return total_rows_written

    def run_full_pipeline(
        self,
        start_date: str = "2010-01-01",
        end_date: str = "2026-12-31",
    ) -> IngestionSummary:
        """Run complete Phase 2 workflow: drop-zone ingestion -> xbbg download -> Parquet store."""
        upserted = self.ingest_universe_from_drop_zone()

        universe_active = self.sqlite_store.get_universe_pit(exclude_financials=True)
        total_active = len(universe_active["ticker"].unique()) if not universe_active.empty else 0
        na_active = (
            len(universe_active[universe_active["region_bloc"] == "NA"]["ticker"].unique())
            if not universe_active.empty
            else 0
        )
        eu_active = (
            len(universe_active[universe_active["region_bloc"] == "EU"]["ticker"].unique())
            if not universe_active.empty
            else 0
        )

        market_rows = self.download_and_sync_market_series(start_date=start_date, end_date=end_date)

        return IngestionSummary(
            universe_records_upserted=upserted,
            active_stocks_total=total_active,
            active_stocks_na=na_active,
            active_stocks_eu=eu_active,
            market_rows_written=market_rows,
        )
