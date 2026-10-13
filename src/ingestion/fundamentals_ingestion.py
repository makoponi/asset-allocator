"""Downloader parsimonioso per le metriche fondamentali Point-in-Time da Bloomberg.

Gestisce in modo controllato e batch i campi fondamentali attivi in BDH:
- Multipli di mercato mensili: PE_RATIO, BEST_PE_RATIO, PX_TO_BOOK_RATIO
- Metriche contabili di bilancio: PROF_MARGIN, OPER_MARGIN, TOT_DEBT_TO_TOT_EQY, QUICK_RATIO
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING

from src.ingestion.storage import FundamentalPiTRecord, MarketStore

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Core fundamental fields verified on Bloomberg BDH
CORE_FUNDAMENTAL_FIELDS = [
    "PE_RATIO",
    "BEST_PE_RATIO",
    "PX_TO_BOOK_RATIO",
    "PROF_MARGIN",
    "OPER_MARGIN",
    "TOT_DEBT_TO_TOT_EQY",
    "QUICK_RATIO",
    "CF_FREE_CASH_FLOW",
]


class FundamentalIngestionManager:
    """Manages parsimonious batch downloading of Point-in-Time fundamentals into SQLite."""

    def __init__(self, market_store: MarketStore | None = None) -> None:
        self.market_store = market_store or MarketStore()

    def ingest_fundamentals_batch(
        self,
        tickers: Sequence[str],
        start_date: date,
        end_date: date,
        fields: Sequence[str] | None = None,
        lag_days: int = 90,
    ) -> int:
        """Download fundamental time series for tickers and insert into fundamentals_pit.

        Applies a prudent filing lag of lag_days (default: 90 days) for accounting metrics
        to prevent look-ahead bias. For market multiples, filing_date equals as_of_date.
        """
        from xbbg import blp

        target_fields = list(fields or CORE_FUNDAMENTAL_FIELDS)
        eq_tickers = [f"{t} Equity" if not t.endswith(" Equity") else t for t in tickers]
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        try:
            df = blp.bdh(eq_tickers, target_fields, start_str, end_str)
        except Exception as e:
            logger.error("Error fetching fundamental bdh: %s", e)
            return 0

        if df is None or len(df) == 0:
            return 0

        records: list[FundamentalPiTRecord] = []
        market_multiples = {"PE_RATIO", "BEST_PE_RATIO", "PX_TO_BOOK_RATIO"}

        if hasattr(df, "iter_rows"):
            for r in df.iter_rows():
                val_str = str(r[3]).strip()
                if val_str == "None" or not val_str:
                    continue

                try:
                    val_float = float(val_str)
                except ValueError, TypeError:
                    continue

                raw_t = str(r[0]).replace(" Equity", "").strip()
                d_raw = r[1]
                as_of_str = str(d_raw)[:10]
                field_name = str(r[2]).strip()

                # Determine filing_date: immediate for market multiples, lagged for statements
                if field_name in market_multiples:
                    filing_str = as_of_str
                else:
                    d_obj = date.fromisoformat(as_of_str)
                    filing_d = d_obj + timedelta(days=lag_days)
                    filing_str = filing_d.isoformat()

                records.append(
                    FundamentalPiTRecord(
                        as_of_date=as_of_str,
                        filing_date=filing_str,
                        ticker=raw_t,
                        metric_name=field_name,
                        metric_value=val_float,
                    )
                )

        if not records:
            return 0

        return self.market_store.insert_fundamentals_pit(records)
