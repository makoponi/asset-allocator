"""Downloader parsimonioso per le metriche fondamentali Point-in-Time da Bloomberg.

Gestisce in modo controllato e batch i campi fondamentali attivi in BDH (Per='M'):
- Multipli di mercato mensili (0-lag):
  PE_RATIO, BEST_PE_RATIO, PX_TO_BOOK_RATIO, FREE_CASH_FLOW_YIELD
- Metriche contabili di bilancio (+90d lag):
  RETURN_COM_EQY, PROF_MARGIN, OPER_MARGIN, TOT_DEBT_TO_TOT_EQY, QUICK_RATIO, CF_FREE_CASH_FLOW
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

from src.ingestion.storage import FundamentalPiTRecord, MarketStore

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# 10 core fundamental fields verified on Bloomberg BDH
CORE_FUNDAMENTAL_FIELDS: list[str] = [
    "PE_RATIO",
    "BEST_PE_RATIO",
    "PX_TO_BOOK_RATIO",
    "FREE_CASH_FLOW_YIELD",
    "RETURN_COM_EQY",
    "PROF_MARGIN",
    "OPER_MARGIN",
    "TOT_DEBT_TO_TOT_EQY",
    "QUICK_RATIO",
    "CF_FREE_CASH_FLOW",
]

# Market multiples with zero filing lag (coincident with market price date)
MARKET_MULTIPLES: set[str] = {
    "PE_RATIO",
    "BEST_PE_RATIO",
    "PX_TO_BOOK_RATIO",
    "FREE_CASH_FLOW_YIELD",
}


def parse_bdh_fundamental_rows(
    df: Any,
    market_multiples: set[str] | None = None,
    lag_days: int = 90,
) -> list[FundamentalPiTRecord]:
    """Parse raw Bloomberg BDH response into strongly-typed FundamentalPiTRecords.

    Supports both Narwhals/Polars DataFrames (with .iter_rows()) and pandas DataFrames.
    Applies zero lag for market multiples and a prudent lag_days (default: 90) for financial
    statement items to enforce zero look-ahead bias.
    """
    if df is None:
        return []

    multiples = market_multiples if market_multiples is not None else MARKET_MULTIPLES
    records: list[FundamentalPiTRecord] = []

    # Format 1: Long format with .iter_rows() (narwhals/polars or converted)
    # Expected row: [ticker, date, field, value]
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

            if field_name in multiples:
                filing_str = as_of_str
            else:
                try:
                    d_obj = date.fromisoformat(as_of_str)
                    filing_d = d_obj + timedelta(days=lag_days)
                    filing_str = filing_d.isoformat()
                except ValueError:
                    filing_str = as_of_str

            records.append(
                FundamentalPiTRecord(
                    as_of_date=as_of_str,
                    filing_date=filing_str,
                    ticker=raw_t,
                    metric_name=field_name,
                    metric_value=val_float,
                )
            )
        return records

    # Format 2: pandas DataFrame with MultiIndex columns (ticker, field)
    if hasattr(df, "columns") and hasattr(df, "index"):
        import pandas as pd

        if isinstance(df, pd.DataFrame):
            for col in df.columns:
                if isinstance(col, tuple) and len(col) >= 2:
                    ticker_raw, field_name = str(col[0]), str(col[1])
                else:
                    ticker_raw, field_name = str(col), "VALUE"

                raw_t = ticker_raw.replace(" Equity", "").strip()
                series = df[col].dropna()

                for d_val, val in series.items():
                    try:
                        val_float = float(val)
                    except ValueError, TypeError:
                        continue

                    as_of_str = str(d_val)[:10]
                    if field_name in multiples:
                        filing_str = as_of_str
                    else:
                        try:
                            d_obj = date.fromisoformat(as_of_str)
                            filing_str = (d_obj + timedelta(days=lag_days)).isoformat()
                        except ValueError:
                            filing_str = as_of_str

                    records.append(
                        FundamentalPiTRecord(
                            as_of_date=as_of_str,
                            filing_date=filing_str,
                            ticker=raw_t,
                            metric_name=field_name,
                            metric_value=val_float,
                        )
                    )

    return records


class FundamentalIngestionManager:
    """Manages parsimonious batch downloading of Point-in-Time fundamentals into SQLite."""

    def __init__(self, market_store: MarketStore | None = None) -> None:
        self.market_store = market_store or MarketStore()

    def get_already_ingested_tickers(self) -> set[str]:
        """Fetch set of tickers that already have records in fundamentals_pit."""
        with self.market_store._connection() as conn:
            cursor = conn.execute("SELECT DISTINCT ticker FROM fundamentals_pit;")
            return {row[0] for row in cursor.fetchall()}

    def get_missing_tickers(self, all_tickers: Sequence[str]) -> list[str]:
        """Filter out tickers that are already stored in fundamentals_pit."""
        done = self.get_already_ingested_tickers()
        return [t for t in all_tickers if t not in done]

    def ingest_fundamentals_batch(
        self,
        tickers: Sequence[str],
        start_date: date,
        end_date: date,
        fields: Sequence[str] | None = None,
        lag_days: int = 90,
        periodicity: str = "M",
    ) -> int:
        """Download fundamental time series for tickers and insert into fundamentals_pit.

        Uses BDH with Per='M' to minimize quote burst rate and capture monthly multiples
        as well as financial statements.
        Applies a prudent filing lag of lag_days (default: 90 days) for accounting metrics
        to prevent look-ahead bias. For market multiples, filing_date equals as_of_date.
        """
        if not tickers:
            return 0

        from xbbg import blp

        target_fields = list(fields or CORE_FUNDAMENTAL_FIELDS)
        eq_tickers = [f"{t} Equity" if not t.endswith(" Equity") else t for t in tickers]
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        try:
            df = blp.bdh(eq_tickers, target_fields, start_str, end_str, Per=periodicity)
        except Exception as e:
            logger.error("Error fetching fundamental bdh: %s", e)
            return 0

        if df is None:
            return 0

        records = parse_bdh_fundamental_rows(
            df=df,
            market_multiples=MARKET_MULTIPLES,
            lag_days=lag_days,
        )

        if not records:
            return 0

        return self.market_store.insert_fundamentals_pit(records)
