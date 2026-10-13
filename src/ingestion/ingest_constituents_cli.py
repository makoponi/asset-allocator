"""Script CLI per lo scarico controllato a blocchi quinquennali dei costituenti STOXX 600."""

from __future__ import annotations

import argparse
import time
from datetime import date

from src.ingestion.bql_client import RealBloombergClient
from src.ingestion.storage import MarketStore
from src.ingestion.universe_ingestion import (
    UniverseIngestionManager,
    get_monthly_business_end_dates,
)


def run_block_ingestion(start_year: int, end_year: int, cutoff_date: str | None = None) -> None:
    store = MarketStore()
    client = RealBloombergClient()
    manager = UniverseIngestionManager(bql_client=client, market_store=store)

    candidate_dates = get_monthly_business_end_dates(start_year, end_year)
    if cutoff_date is not None:
        candidate_dates = [d for d in candidate_dates if d <= cutoff_date]

    missing = manager.get_missing_dates(candidate_dates)
    total_dates = len(candidate_dates)
    already_cached = total_dates - len(missing)

    print("==================================================")
    print(f"Blocco Quinquennale: {start_year} - {end_year}")
    print(f"Totale date mensili: {total_dates}")
    print(f"Gia presenti in cache: {already_cached}")
    print(f"Da scaricare da Bloomberg: {len(missing)}")
    print("==================================================")

    if not missing:
        print("Tutte le date del blocco sono gia presenti nel database. Nessuna azione necessaria.")
        return

    start_time = time.time()
    for idx, d in enumerate(missing, 1):
        t0 = time.time()
        count = manager.ingest_constituents_for_date(d)
        elapsed = time.time() - t0
        print(f"[{idx}/{len(missing)}] {d} -> Inseriti {count} titoli (tempo: {elapsed:.1f}s)")

    total_elapsed = time.time() - start_time
    print("--------------------------------------------------")
    print(f"Completato blocco {start_year}-{end_year} in {total_elapsed:.1f}s.")

    # Statistiche di sintesi
    all_members = store.get_universe_members(is_active_only=False)
    distinct_tickers = sorted({m.ticker for m in all_members})
    print(f"Totale record storici memorizzati: {len(all_members)}")
    print(f"Totale ticker unici storici finora: {len(distinct_tickers)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestione costituenti STOXX 600 a blocchi.")
    parser.add_argument("--start-year", type=int, required=True, help="Anno di inizio (es. 2021)")
    parser.add_argument("--end-year", type=int, required=True, help="Anno di fine (es. 2025)")
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=date.today().isoformat(),
        help="Data massima di taglio YYYY-MM-DD",
    )
    args = parser.parse_args()

    run_block_ingestion(args.start_year, args.end_year, args.cutoff_date)


if __name__ == "__main__":
    main()
