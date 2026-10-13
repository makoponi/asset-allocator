"""Pipeline e CLI per lo scarico controllato a batch delle serie storiche di prezzo e volume (EOD).

Salvataggio partizionato per anno in Parquet (data/processed/year=YYYY/prices.parquet)
con calcolo dei rendimenti giornalieri ret_1d e deduplicazione totale.
"""

from __future__ import annotations

import argparse
import time
from datetime import date
from typing import TYPE_CHECKING

from src.ingestion.bql_client import RealBloombergClient
from src.ingestion.storage import MarketStore, PriceEODRecord, PriceStore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.ingestion.bql_client import BQLClientProtocol


def compute_daily_returns(
    bars_by_ticker: dict[str, list[tuple[date, float, float]]],
) -> list[PriceEODRecord]:
    """Sort bars chronologically per ticker and calculate daily percentage returns ret_1d."""
    records: list[PriceEODRecord] = []
    for ticker, bars in bars_by_ticker.items():
        sorted_bars = sorted(bars, key=lambda x: x[0])
        prev_px: float | None = None
        for d_val, px, vol in sorted_bars:
            ret = px / prev_px - 1.0 if prev_px is not None and prev_px > 0 else 0.0
            prev_px = px
            records.append(
                PriceEODRecord(
                    date=d_val,
                    ticker=ticker,
                    px_last=px,
                    volume=vol,
                    ret_1d=ret,
                )
            )
    return records


class PriceIngestionManager:
    """Manages batch historical price and volume ingestion with Parquet partitioning."""

    def __init__(
        self,
        bql_client: BQLClientProtocol,
        market_store: MarketStore | None = None,
        price_store: PriceStore | None = None,
    ) -> None:
        self.bql_client = bql_client
        self.market_store = market_store or MarketStore()
        self.price_store = price_store or PriceStore()

    def get_all_universe_tickers(self) -> list[str]:
        """Fetch distinct tickers from SQLite market_store."""
        members = self.market_store.get_universe_members(is_active_only=False)
        return sorted({m.ticker for m in members})

    def ingest_ticker_batch(
        self,
        tickers: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> int:
        """Download bars for a list of tickers and persist to Parquet with ret_1d."""
        if not tickers:
            return 0

        bars = self.bql_client.get_historical_bars(tickers, start_date, end_date)
        if not bars:
            return 0

        # Group by ticker
        bars_by_ticker: dict[str, list[tuple[date, float, float]]] = {}
        for b in bars:
            bars_by_ticker.setdefault(b.ticker, []).append((b.date, b.px_last, b.volume))

        records = compute_daily_returns(bars_by_ticker)
        return self.price_store.write_prices(records)


def run_price_batch_ingestion(
    start_date_str: str,
    end_date_str: str,
    batch_size: int = 50,
    max_tickers: int | None = None,
    throttle_sec: float = 0.5,
) -> None:
    start_d = date.fromisoformat(start_date_str)
    end_d = date.fromisoformat(end_date_str)

    store = MarketStore()
    p_store = PriceStore()
    client = RealBloombergClient()
    manager = PriceIngestionManager(bql_client=client, market_store=store, price_store=p_store)

    all_tickers = manager.get_all_universe_tickers()
    if max_tickers is not None:
        all_tickers = all_tickers[:max_tickers]

    total_tickers = len(all_tickers)
    chunks = [all_tickers[i : i + batch_size] for i in range(0, total_tickers, batch_size)]

    print("==================================================")
    print(f"Ingestione Prezzi EOD: {start_date_str} -> {end_date_str}")
    print(f"Totale Ticker da scaricare: {total_tickers}")
    print(f"Dimensione Batch: {batch_size} (Totale Chunk: {len(chunks)})")
    print("==================================================")

    t_start = time.time()
    total_written = 0
    for idx, chunk in enumerate(chunks, 1):
        t0 = time.time()
        written = manager.ingest_ticker_batch(chunk, start_d, end_d)
        elapsed = time.time() - t0
        total_written += written
        print(
            f"[{idx}/{len(chunks)}] Scaricati {len(chunk)} titoli -> "
            f"{written} barre scritte ({elapsed:.2f}s)"
        )
        if throttle_sec > 0:
            time.sleep(throttle_sec)

    total_time = time.time() - t_start
    print("--------------------------------------------------")
    print(
        f"Completato! Scritte {total_written} barre in {total_time:.1f}s "
        f"({total_time / total_tickers:.2f}s per titolo)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestione serie storiche prezzi e volumi EOD.")
    parser.add_argument("--start-date", type=str, required=True, help="Data inizio YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="Data fine YYYY-MM-DD")
    parser.add_argument("--batch-size", type=int, default=50, help="Dimensione chunk ticker")
    parser.add_argument("--max-tickers", type=int, default=None, help="Limite massimo titoli")
    parser.add_argument("--throttle", type=float, default=0.5, help="Sleep in secondi tra batch")
    args = parser.parse_args()

    run_price_batch_ingestion(
        args.start_date,
        args.end_date,
        batch_size=args.batch_size,
        max_tickers=args.max_tickers,
        throttle_sec=args.throttle,
    )


if __name__ == "__main__":
    main()
