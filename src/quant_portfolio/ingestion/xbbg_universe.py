"""Bloomberg xbbg historical index constituent extractor and metadata sync.

Extracts point-in-time index members directly via xbbg.blp.bds using INDX_MWEIGHT_HIST
with END_DATE_OVERRIDE, eliminates Excel dependency, and uses local SQLite caching
to minimize Bloomberg monthly 'Securities Count' quota usage.
"""

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Final

import pandas as pd

from quant_portfolio.config import FINANCIALS_GICS_SECTOR, RegionBloc
from quant_portfolio.ingestion.bql_templates import INDEX_REGION_MAP
from quant_portfolio.ingestion.xbbg_client import chunk_list
from quant_portfolio.storage.schema import (
    SQLiteStore,
    safe_int,
    safe_optional_str,
    safe_str,
)

logger = logging.getLogger(__name__)

BDSCallable = Callable[..., pd.DataFrame]
BDPCallable = Callable[..., pd.DataFrame]

METADATA_FIELDS: Final[tuple[str, ...]] = (
    "ID_ISIN",
    "GICS_SECTOR",
    "CRNCY",
    "NAME",
)
BDP_BATCH_SIZE: Final[int] = 100


def clean_bloomberg_member_ticker(raw_ticker: str) -> str:
    """Normalize raw Bloomberg ticker from BDS output.

    Strips trailing ' Equity', preserves market codes and delisted prefixes (e.g. 1CNHI IM).
    Example:
        'AAPL US Equity' -> 'AAPL US'
        '1CNHI IM Equity' -> '1CNHI IM'
        'SAP GY' -> 'SAP GY'
    """
    clean = str(raw_ticker).strip()
    if clean.upper().endswith(" EQUITY"):
        clean = clean[:-7].strip()
    elif clean.upper().endswith(" CORP"):
        clean = clean[:-5].strip()
    elif clean.upper().endswith(" INDEX"):
        clean = clean[:-6].strip()
    return clean


def _extract_ticker_list_from_bds(df: pd.DataFrame) -> list[str]:
    """Find member column in BDS response and extract clean ticker list."""
    if df.empty:
        return []

    # Common BDS member column variants
    candidate_cols = (
        "index_member",
        "Index Member",
        "member_ticker_and_exchange_code",
        "Member Ticker and Exchange Code",
        "Security",
        "security",
        "ticker",
    )
    col_found: str | None = None
    for cand in candidate_cols:
        for c in df.columns:
            if str(c).lower().strip() == cand.lower():
                col_found = str(c)
                break
        if col_found is not None:
            break

    if col_found is None:
        # Fallback to first text column
        col_found = str(df.columns[0])

    tickers: list[str] = []
    for item in df[col_found].dropna():
        cleaned = clean_bloomberg_member_ticker(str(item))
        if cleaned:
            tickers.append(cleaned)

    return sorted(set(tickers))


def _ensure_pandas(res: Any) -> pd.DataFrame:
    """Safely convert xbbg / narwhals result to standard pandas DataFrame."""
    if res is None:
        return pd.DataFrame()
    if hasattr(res, "to_pandas"):
        return res.to_pandas()  # type: ignore[no-any-return]
    if isinstance(res, pd.DataFrame):
        return res
    return pd.DataFrame(res)


class BloombergUniverseExtractor:
    """Extracts historical index members via xbbg with quota-protected metadata caching."""

    def __init__(
        self,
        store: SQLiteStore,
        bds_fn: BDSCallable | None = None,
        bdp_fn: BDPCallable | None = None,
        throttle_seconds: float = 0.35,
    ) -> None:
        self.store = store
        self._bds_fn = bds_fn
        self._bdp_fn = bdp_fn
        self.throttle_seconds = throttle_seconds

    def _get_bds(self) -> BDSCallable:
        if self._bds_fn is not None:
            return self._bds_fn
        try:
            from xbbg import blp

            return blp.bds
        except ImportError as exc:
            raise ImportError("xbbg is not installed.") from exc

    def _get_bdp(self) -> BDPCallable:
        if self._bdp_fn is not None:
            return self._bdp_fn
        try:
            from xbbg import blp

            return blp.bdp
        except ImportError as exc:
            raise ImportError("xbbg is not installed.") from exc

    def fetch_historical_index_members(
        self,
        index_ticker: str,
        as_of_date: str,
    ) -> list[str]:
        """Fetch index members for a specific historical date via INDX_MEMBERS."""
        dt = pd.to_datetime(as_of_date)
        date_str = dt.strftime("%Y%m%d")
        bds = self._get_bds()

        # Primary: INDX_MWEIGHT_HIST (respects END_DATE_OVERRIDE for true Point-in-Time)
        # Fallback: INDX_MEMBERS (only if INDX_MWEIGHT_HIST returns empty or current date)
        for field in ("INDX_MWEIGHT_HIST", "INDX_MEMBERS"):
            try:
                res = bds(index_ticker, field, END_DATE_OVERRIDE=date_str)
                df = _ensure_pandas(res)
                if not df.empty:
                    members = _extract_ticker_list_from_bds(df)
                    if members:
                        logger.info(
                            "Retrieved %d members for %s using %s",
                            len(members),
                            index_ticker,
                            field,
                        )
                        return members
            except Exception as exc:
                logger.debug("Field %s query for %s failed: %s", field, index_ticker, exc)

        return []

    def fetch_security_metadata(
        self,
        tickers: Sequence[str],
        batch_size: int = BDP_BATCH_SIZE,
    ) -> dict[str, dict[str, Any]]:
        """Fetch static metadata with zero redundant BDP calls using SQLite cache."""
        if not tickers:
            return {}

        unique_tickers = sorted(set(tickers))
        # 1. Read existing from cache
        cached = self.store.get_security_metadata_cache(unique_tickers)
        missing = [t for t in unique_tickers if t not in cached]

        if not missing:
            return cached

        logger.info(
            "Found %d tickers in local cache, fetching BDP metadata for %d new tickers",
            len(cached),
            len(missing),
        )

        bdp = self._get_bdp()
        chunks = chunk_list(missing, batch_size)
        new_records: list[dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            # Query Bloomberg BDP with ' Equity' appended for exchange resolution
            query_tickers = [
                f"{t} Equity" if (" " in t and not t.upper().endswith(" EQUITY")) else t
                for t in chunk
            ]
            try:
                raw_bdp = bdp(query_tickers, flds=list(METADATA_FIELDS))
                res_df = _ensure_pandas(raw_bdp)
                if not res_df.empty:
                    # If returned in tidy/long format (xbbg 1.5+ narwhals), pivot to wide
                    if {"ticker", "field", "value"}.issubset(res_df.columns):
                        res_df = res_df.pivot(index="ticker", columns="field", values="value")

                    # Map row for each ticker
                    for orig_t, query_t in zip(chunk, query_tickers, strict=True):
                        row_data: dict[str, Any] = {}
                        if query_t in res_df.index:
                            row_data = {str(k): v for k, v in res_df.loc[query_t].to_dict().items()}
                        elif orig_t in res_df.index:
                            row_data = {str(k): v for k, v in res_df.loc[orig_t].to_dict().items()}

                        isin_val = safe_str(
                            row_data.get("id_isin") or row_data.get("ID_ISIN"),
                            default="",
                        )
                        gics_val = row_data.get("gics_sector") or row_data.get("GICS_SECTOR")
                        crncy_val = safe_str(
                            row_data.get("crncy") or row_data.get("CRNCY"),
                            default="EUR",
                        ).upper()
                        name_val = row_data.get("name") or row_data.get("NAME")

                        record = {
                            "ticker": orig_t,
                            "isin": isin_val,
                            "gics_sector": safe_int(gics_val, default=0),
                            "local_currency": crncy_val,
                            "name": safe_optional_str(name_val),
                        }
                        new_records.append(record)
                        cached[orig_t] = record
            except Exception as exc:
                logger.error("Error fetching BDP batch %d/%d: %s", i + 1, len(chunks), exc)

            if i < len(chunks) - 1 and self.throttle_seconds > 0:
                time.sleep(self.throttle_seconds)

        if new_records:
            self.store.upsert_security_metadata_cache(new_records)

        return cached

    def sync_index_universe_date(
        self,
        index_ticker: str,
        as_of_date: str,
    ) -> int:
        """Fetch members for one date, enrich with metadata, and upsert to universe_pit."""
        if index_ticker not in INDEX_REGION_MAP:
            raise ValueError(f"Unsupported index '{index_ticker}'.")

        region: RegionBloc = INDEX_REGION_MAP[index_ticker]
        date_formatted = pd.to_datetime(as_of_date).strftime("%Y-%m-%d")

        # 1. Fetch member tickers
        members = self.fetch_historical_index_members(index_ticker, as_of_date=date_formatted)
        if not members:
            logger.warning("No members returned for %s at %s", index_ticker, date_formatted)
            return 0

        # 2. Fetch metadata with cache protection
        metadata_map = self.fetch_security_metadata(members)

        # 3. Construct universe records
        rows: list[dict[str, Any]] = []
        for ticker in members:
            meta = metadata_map.get(ticker, {})
            isin = safe_str(meta.get("isin"), default="")
            gics = safe_int(meta.get("gics_sector"), default=0)
            crncy = safe_str(meta.get("local_currency"), default="EUR")
            is_fin = bool(gics == FINANCIALS_GICS_SECTOR)

            rows.append(
                {
                    "as_of_date": date_formatted,
                    "ticker": ticker,
                    "isin": isin,
                    "region_bloc": region,
                    "source_index": index_ticker,
                    "gics_sector": gics,
                    "local_currency": crncy,
                    "is_financial": is_fin,
                }
            )

        df_universe = pd.DataFrame(rows)
        # Purge stale records for this index/date to guarantee pure PiT constituents
        self.store.delete_universe_pit(as_of_date=date_formatted, source_index=index_ticker)
        return self.store.upsert_universe_pit(df_universe)
