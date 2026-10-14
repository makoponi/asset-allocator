"""Script CLI per l'ingestione controllata a batch delle metriche fondamentali Point-in-Time.

Estrae serie storiche mensili BDH (Per='M') da Bloomberg per l'universo STOXX Europe 600,
con persistenza diretta nel database locale SQLite market_store.db.
Rispetta le quote Bloomberg (burst rate throttle 0.4s, zero impatto su Securities Count)
e applica un lag di 90 giorni sulle grandezze di bilancio.
"""

from __future__ import annotations

import argparse
import time
from datetime import date

from src.ingestion.fundamentals_ingestion import (
    CORE_FUNDAMENTAL_FIELDS,
    FundamentalIngestionManager,
)
from src.ingestion.storage import MarketStore


def run_fundamentals_ingestion(
    start_date_str: str,
    end_date_str: str,
    batch_size: int = 50,
    max_tickers: int | None = None,
    throttle_sec: float = 0.4,
    resume: bool = True,
) -> None:
    """Execute historical fundamental batch ingestion pipeline."""
    start_d = date.fromisoformat(start_date_str)
    end_d = date.fromisoformat(end_date_str)

    store = MarketStore()
    manager = FundamentalIngestionManager(market_store=store)

    # 1. Fetch universe tickers from SQLite
    all_members = store.get_universe_members(is_active_only=False)
    all_tickers = sorted({m.ticker for m in all_members})

    # 2. Check resume state
    already_done = manager.get_already_ingested_tickers()
    target_tickers = [t for t in all_tickers if t not in already_done] if resume else all_tickers

    if max_tickers is not None:
        target_tickers = target_tickers[:max_tickers]

    total_target = len(target_tickers)
    chunks = [target_tickers[i : i + batch_size] for i in range(0, total_target, batch_size)]

    print("==================================================")
    print("Ingestione Fondamentali Point-in-Time BDH (Per='M')")
    print(f"Finestra temporale: {start_date_str} -> {end_date_str}")
    fields_desc = ", ".join(CORE_FUNDAMENTAL_FIELDS)
    print(f"Campi interrogati ({len(CORE_FUNDAMENTAL_FIELDS)}): {fields_desc}")
    print(f"Universo totale titoli censiti: {len(all_tickers)}")
    print(f"Titoli gia presenti nel database: {len(already_done)}")
    print(f"Titoli da scaricare in questa sessione: {total_target}")
    print(f"Dimensione Batch: {batch_size} (Totale Chunk: {len(chunks)})")
    print(f"Throttle tra i batch: {throttle_sec}s")
    print("==================================================")

    if total_target == 0:
        print("Tutti i titoli sono gia presenti nel database. Nessuna operazione necessaria.")
        return

    t_start = time.time()
    total_records = 0

    for idx, chunk in enumerate(chunks, 1):
        t0 = time.time()
        written = manager.ingest_fundamentals_batch(
            tickers=chunk,
            start_date=start_d,
            end_date=end_d,
            fields=CORE_FUNDAMENTAL_FIELDS,
            lag_days=90,
            periodicity="M",
        )
        elapsed = time.time() - t0
        total_records += written
        print(
            f"[{idx}/{len(chunks)}] Scaricati {len(chunk)} titoli -> "
            f"{written} record scritti in SQLite ({elapsed:.2f}s)"
        )

        if throttle_sec > 0 and idx < len(chunks):
            time.sleep(throttle_sec)

    total_time = time.time() - t_start
    print("--------------------------------------------------")
    print(
        f"Completato! Scritti {total_records} record fondamentali in {total_time:.1f}s "
        f"({total_time / total_target:.2f}s per titolo)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingestione serie storiche fondamentali Point-in-Time da Bloomberg BDH."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2011-01-01",
        help="Data inizio YYYY-MM-DD (default: 2011-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=date.today().isoformat(),
        help="Data fine YYYY-MM-DD (default: oggi)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Dimensione batch titoli per chiamata (default: 50)",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Limite massimo titoli da scaricare (opzionale)",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.4,
        help="Sleep in secondi tra i batch (default: 0.4)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disabilita il resume automatico e forza il riscarico",
    )
    args = parser.parse_args()

    run_fundamentals_ingestion(
        start_date_str=args.start_date,
        end_date_str=args.end_date,
        batch_size=args.batch_size,
        max_tickers=args.max_tickers,
        throttle_sec=args.throttle,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
