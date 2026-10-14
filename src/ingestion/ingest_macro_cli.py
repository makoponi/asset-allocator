"""CLI for macroeconomic indicators ingestion into SQLite."""

from __future__ import annotations

import argparse
import logging

from src.ingestion.macro_ingestion import MacroIngestionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_macro_cli")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest macroeconomic signals (HY OAS, Bund 10Y-2Y, V2TX) into SQLite."
    )
    parser.add_argument(
        "--source",
        choices=["bloomberg", "synthetic"],
        default="bloomberg",
        help="Data source: 'bloomberg' via xbbg or 'synthetic' calibrated simulation.",

    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2015-01-01",
        help="Start date YYYY-MM-DD (default: 2015-01-01).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: latest).",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.4,
        help="Throttle delay in seconds between Bloomberg requests (default: 0.4s).",
    )

    args = parser.parse_args()
    pipeline = MacroIngestionPipeline()

    if args.source == "bloomberg":
        try:
            count = pipeline.ingest_from_bloomberg(
                start_date=args.start_date,
                end_date=args.end_date,
                throttle_seconds=args.throttle,
            )
            logger.info("Successfully ingested %d records from Bloomberg.", count)
        except Exception as e:
            logger.error("Bloomberg ingestion failed: %s", e)
            logger.info("Falling back to calibrated synthetic history...")
            count = pipeline.ingest_synthetic_history(
                start_date=args.start_date,
                end_date=args.end_date,
            )
            logger.info("Successfully ingested %d fallback synthetic records.", count)
    else:
        count = pipeline.ingest_synthetic_history(
            start_date=args.start_date,
            end_date=args.end_date,
        )
        logger.info("Successfully ingested %d synthetic records.", count)


if __name__ == "__main__":
    main()
