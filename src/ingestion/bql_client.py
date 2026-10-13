"""Bloomberg BQL and market data client abstraction layer.

Provides a typed protocol and mock client for deterministic offline testing and
development, alongside a production-ready real Bloomberg BQL client skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date


@dataclass(frozen=True, slots=True)
class ConstituentData:
    """Historical constituent record returned by Bloomberg query."""

    ticker: str
    gics_sector: str
    is_active: bool = True


@dataclass(frozen=True, slots=True)
class HistoricalBar:
    """Single end-of-day market bar record."""

    date: date
    ticker: str
    px_last: float
    volume: float


@runtime_checkable
class BQLClientProtocol(Protocol):
    """Protocol defining required Bloomberg BQL market data access methods."""

    def get_index_constituents(
        self, index_ticker: str, as_of_date: str
    ) -> Sequence[ConstituentData]:
        """Fetch index constituents point-in-time for a specific date (YYYY-MM-DD)."""
        ...

    def get_historical_bars(
        self, tickers: Sequence[str], start_date: date, end_date: date
    ) -> Sequence[HistoricalBar]:
        """Fetch daily historical price and volume bars between start_date and end_date."""
        ...


class MockBloombergClient:
    """Mock client providing synthetic, reproducible data for tests and offline development."""

    def __init__(
        self,
        constituents_by_date: dict[str, Sequence[ConstituentData]] | None = None,
        bars: Sequence[HistoricalBar] | None = None,
    ) -> None:
        self._constituents = dict(constituents_by_date) if constituents_by_date else {}
        self._bars = list(bars) if bars else []

    def set_constituents_for_date(
        self, as_of_date: str, constituents: Sequence[ConstituentData]
    ) -> None:
        """Register constituents for a specific date in the mock state."""
        self._constituents[as_of_date] = constituents

    def get_index_constituents(
        self, index_ticker: str, as_of_date: str
    ) -> Sequence[ConstituentData]:
        """Return configured constituents for the given date, or an empty sequence."""
        return self._constituents.get(as_of_date, [])

    def get_historical_bars(
        self, tickers: Sequence[str], start_date: date, end_date: date
    ) -> Sequence[HistoricalBar]:
        """Filter configured bars matching the requested tickers and date window."""
        tickers_set = set(tickers)
        return [
            b for b in self._bars if b.ticker in tickers_set and start_date <= b.date <= end_date
        ]


class RealBloombergClient:
    """Production Bloomberg client interacting with Bloomberg BQL/BLPAPI.

    Enforces defensive batching and explicit error checking.
    """

    def __init__(self) -> None:
        self._initialized = False

    def _ensure_connection(self) -> None:
        if not self._initialized:
            try:
                import blpapi  # noqa: F401
            except ImportError as err:
                msg = (
                    "blpapi is not installed or Bloomberg Terminal is not running. "
                    "Ensure Bloomberg C++ SDK/wheel is configured."
                )
                raise RuntimeError(msg) from err
            self._initialized = True

    def get_index_constituents(
        self, index_ticker: str, as_of_date: str
    ) -> Sequence[ConstituentData]:
        """Fetch index constituents via BQL query."""
        self._ensure_connection()
        # BQL query placeholder logic to be used when terminal is running
        msg = f"Real Bloomberg BQL query for {index_ticker} on {as_of_date} not yet wired."
        raise NotImplementedError(msg)

    def get_historical_bars(
        self, tickers: Sequence[str], start_date: date, end_date: date
    ) -> Sequence[HistoricalBar]:
        """Fetch daily historical price bars via BLPAPI."""
        self._ensure_connection()
        msg = f"Real Bloomberg historical bars for {len(tickers)} tickers not yet wired."
        raise NotImplementedError(msg)
