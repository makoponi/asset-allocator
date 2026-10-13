"""Bloomberg client abstraction layer supporting Mock and Real connections via xbbg/blpapi."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


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
    """Protocol defining required Bloomberg market data access methods."""

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
        """Register constituents for a specific date in mock state."""
        self._constituents[as_of_date] = constituents

    def get_index_constituents(
        self, index_ticker: str, as_of_date: str
    ) -> Sequence[ConstituentData]:
        """Return configured constituents for given date, or empty sequence."""
        return self._constituents.get(as_of_date, [])

    def get_historical_bars(
        self, tickers: Sequence[str], start_date: date, end_date: date
    ) -> Sequence[HistoricalBar]:
        """Filter configured bars matching requested tickers and date window."""
        tickers_set = set(tickers)
        return [
            b for b in self._bars if b.ticker in tickers_set and start_date <= b.date <= end_date
        ]


class RealBloombergClient:
    """Production Bloomberg client using xbbg and Desktop API (bbcomm)."""

    def __init__(self) -> None:
        self._check_environment()

    def _check_environment(self) -> None:
        try:
            import blpapi  # noqa: F401
            import xbbg  # noqa: F401
        except ImportError as err:
            msg = (
                "blpapi or xbbg is not installed. "
                "Ensure Bloomberg C++ SDK/wheel and dependencies are configured."
            )
            raise RuntimeError(msg) from err

    def get_index_constituents(
        self, index_ticker: str, as_of_date: str
    ) -> Sequence[ConstituentData]:
        """Fetch Point-in-Time constituents of index on as_of_date (YYYY-MM-DD).

        Uses INDX_MWEIGHT_HIST with END_DATE_OVERRIDE for historical dates,
        and queries GICS sectors in batch.
        """
        from xbbg import blp

        clean_date = as_of_date.replace("-", "")

        # Try historical field first
        try:
            raw_members = blp.bds(index_ticker, "INDX_MWEIGHT_HIST", END_DATE_OVERRIDE=clean_date)
        except Exception as e:
            logger.warning(
                "Error querying INDX_MWEIGHT_HIST for %s on %s: %s", index_ticker, as_of_date, e
            )
            raw_members = None

        member_tickers: list[str] = []

        if raw_members is not None and len(raw_members) > 0:
            cols = [str(c) for c in raw_members.columns]
            target_col = None
            for candidate in ["Index Member", "Member Ticker and Exchange Code", "ticker"]:
                if candidate in cols:
                    target_col = candidate
                    break
            if target_col is None and len(cols) >= 3:
                target_col = cols[2]

            if target_col is not None:
                if hasattr(raw_members, "iter_rows"):
                    col_idx = cols.index(target_col)
                    for r in raw_members.iter_rows():
                        val = str(r[col_idx]).strip()
                        if val and val != "None":
                            member_tickers.append(val)
                elif hasattr(raw_members, "__getitem__"):
                    series = raw_members[target_col]
                    vals = series.tolist() if hasattr(series, "tolist") else list(series)
                    member_tickers = [str(v).strip() for v in vals if str(v).strip()]

        # Fallback to current weight if historical returned empty and date is very recent
        if not member_tickers:
            raw_curr = blp.bds(index_ticker, "INDX_MWEIGHT")
            if raw_curr is not None and len(raw_curr) > 0:
                cols = [str(c) for c in raw_curr.columns]
                target_col = (
                    "Member Ticker and Exchange Code"
                    if "Member Ticker and Exchange Code" in cols
                    else cols[2]
                )
                if hasattr(raw_curr, "iter_rows"):
                    col_idx = cols.index(target_col)
                    for r in raw_curr.iter_rows():
                        val = str(r[col_idx]).strip()
                        if val:
                            member_tickers.append(val)
                elif hasattr(raw_curr, "__getitem__"):
                    series = raw_curr[target_col]
                    vals = series.tolist() if hasattr(series, "tolist") else list(series)
                    member_tickers = [str(v).strip() for v in vals if str(v).strip()]

        if not member_tickers:
            return []

        # Normalize ticker symbols to standard Equity convention (e.g. 'MC FP' -> 'MC FP Equity')
        eq_tickers = [f"{t} Equity" if not t.endswith(" Equity") else t for t in member_tickers]

        # Batch query GICS sectors (max 100 per call for reliability)
        sector_map: dict[str, str] = {}
        batch_size = 100
        for i in range(0, len(eq_tickers), batch_size):
            chunk = eq_tickers[i : i + batch_size]
            try:
                sec_df = blp.bdp(chunk, ["GICS_SECTOR_NAME"])
                if sec_df is not None and len(sec_df) > 0:
                    if hasattr(sec_df, "iter_rows"):
                        cols_sec = [str(c) for c in sec_df.columns]
                        t_idx = cols_sec.index("ticker") if "ticker" in cols_sec else 0
                        val_idx = cols_sec.index("value") if "value" in cols_sec else 2
                        for r in sec_df.iter_rows():
                            t_val = str(r[t_idx]).replace(" Equity", "").strip()
                            s_val = str(r[val_idx]).strip()
                            sector_map[t_val] = s_val
                    elif hasattr(sec_df, "index"):
                        for t_idx, val in zip(sec_df.index, sec_df.iloc[:, 0], strict=False):
                            t_clean = str(t_idx).replace(" Equity", "").strip()
                            sector_map[t_clean] = str(val).strip()
            except Exception as e:
                logger.warning("Error fetching GICS sector for chunk: %s", e)

        results: list[ConstituentData] = []
        for raw_t in member_tickers:
            clean_t = raw_t.replace(" Equity", "").strip()
            sector = sector_map.get(clean_t, "Unknown")
            results.append(
                ConstituentData(
                    ticker=clean_t,
                    gics_sector=sector,
                    is_active=True,
                )
            )

        return results

    def get_historical_bars(
        self, tickers: Sequence[str], start_date: date, end_date: date
    ) -> Sequence[HistoricalBar]:
        """Fetch daily historical price and volume bars via xbbg bdh."""
        from xbbg import blp

        eq_tickers = [f"{t} Equity" if not t.endswith(" Equity") else t for t in tickers]
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        try:
            df = blp.bdh(eq_tickers, ["PX_LAST", "VOLUME"], start_str, end_str)
        except Exception as e:
            logger.error("Error fetching historical bars: %s", e)
            return []

        results: list[HistoricalBar] = []
        if df is None or len(df) == 0:
            return results

        # Parse DataFrame structure
        # (Handling multi-index columns from xbbg: (Ticker, Field))
        if hasattr(df, "to_dict"):
            pydict: dict[Any, Any] = df.to_dict()
            for (t_sym, field_name), date_map in pydict.items():
                if field_name != "PX_LAST":
                    continue
                clean_t = str(t_sym).replace(" Equity", "").strip()
                vol_map = pydict.get((t_sym, "VOLUME"), {})
                for d_val, px in date_map.items():
                    if px is None or px != px:  # check NaN
                        continue
                    vol = vol_map.get(d_val, 0.0)
                    if hasattr(d_val, "date"):
                        d_obj = d_val.date()
                    elif isinstance(d_val, date):
                        d_obj = d_val
                    else:
                        d_obj = date.fromisoformat(str(d_val)[:10])

                    results.append(
                        HistoricalBar(
                            date=d_obj,
                            ticker=clean_t,
                            px_last=float(px),
                            volume=float(vol) if vol == vol and vol is not None else 0.0,
                        )
                    )

        return results
