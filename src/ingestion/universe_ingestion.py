"""Ingestion manager for STOXX Europe 600 historical constituents and Point-in-Time tracking."""

from __future__ import annotations

import calendar
from datetime import date
from typing import TYPE_CHECKING

from src.ingestion.storage import MarketStore, UniverseMemberRecord

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.ingestion.bql_client import BQLClientProtocol


def get_monthly_business_end_dates(start_year: int, end_year: int) -> list[str]:
    """Calculate the last business day (Mon-Fri) for each month in the specified range.

    Dates are formatted as YYYY-MM-DD.
    """
    dates: list[str] = []
    for yr in range(start_year, end_year + 1):
        for m in range(1, 13):
            last_day = calendar.monthrange(yr, m)[1]
            cur_date = date(yr, m, last_day)
            # Roll back if Saturday (weekday 5) or Sunday (weekday 6)
            while cur_date.weekday() >= 5:
                cur_date = date(cur_date.year, cur_date.month, cur_date.day - 1)
            dates.append(cur_date.isoformat())
    return dates


class UniverseIngestionManager:
    """Orchestrates historical STOXX Europe 600 constituent ingestion.

    Guarantees idempotence: skips dates already present in MarketStore to preserve Bloomberg limits.
    """

    def __init__(
        self,
        bql_client: BQLClientProtocol,
        market_store: MarketStore | None = None,
        index_ticker: str = "SXXP Index",
    ) -> None:
        self.bql_client = bql_client
        self.market_store = market_store or MarketStore()
        self.index_ticker = index_ticker

    def get_missing_dates(self, target_dates: Sequence[str]) -> list[str]:
        """Identify which target dates do not yet have active constituents stored in database."""
        missing: list[str] = []
        for d in target_dates:
            existing = self.market_store.get_universe_members(as_of_date=d, is_active_only=True)
            if not existing:
                missing.append(d)
        return missing

    def ingest_constituents_for_date(self, as_of_date: str) -> int:
        """Fetch and persist constituents for a specific date if not already present.

        Returns the number of constituents inserted.
        """
        existing = self.market_store.get_universe_members(
            as_of_date=as_of_date, is_active_only=True
        )
        if existing:
            return 0  # Already cached, skip BQL call

        raw_constituents = self.bql_client.get_index_constituents(
            index_ticker=self.index_ticker, as_of_date=as_of_date
        )
        if not raw_constituents:
            return 0

        records = [
            UniverseMemberRecord(
                date=as_of_date,
                ticker=c.ticker,
                gics_sector=c.gics_sector,
                is_active=1 if c.is_active else 0,
            )
            for c in raw_constituents
        ]
        return self.market_store.insert_universe_members(records)

    def ingest_universe_range(
        self, start_year: int, end_year: int, cutoff_date: str | None = None
    ) -> dict[str, int]:
        """Ingest all monthly end-of-business dates in range up to cutoff_date.

        Returns a dictionary mapping date to number of inserted records.
        """
        candidate_dates = get_monthly_business_end_dates(start_year, end_year)
        if cutoff_date is not None:
            candidate_dates = [d for d in candidate_dates if d <= cutoff_date]

        missing_dates = self.get_missing_dates(candidate_dates)
        results: dict[str, int] = {}

        for d in missing_dates:
            count = self.ingest_constituents_for_date(d)
            results[d] = count

        return results

    def get_unique_historical_tickers(self) -> list[str]:
        """Return distinct list of all tickers that ever appeared in the index."""
        members = self.market_store.get_universe_members(is_active_only=False)
        return sorted({m.ticker for m in members})
