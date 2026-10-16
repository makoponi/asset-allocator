"""Bloomberg xbbg client for point-in-time fundamental metrics ingestion.

Fetches CORE_FUNDAMENTAL_METRICS with Currency='EUR', enforcing 50-ticker batching,
caching existing records to protect Bloomberg monthly Securities Count limits,
and upserting into SQLite fundamentals_pit.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from quant_portfolio.config import CORE_FUNDAMENTAL_METRICS
from quant_portfolio.ingestion.xbbg_client import (
    _ensure_pandas,
    chunk_list,
    ensure_bloomberg_security_suffix,
)

if TYPE_CHECKING:
    from quant_portfolio.config import PortfolioConfig
    from quant_portfolio.storage.schema import SQLiteStore

logger = logging.getLogger(__name__)

BATCH_SIZE: Final[int] = 50
THROTTLE_SECONDS: Final[float] = 0.35

PRICE_BASED_METRICS: Final[set[str]] = {
    "PE_RATIO",
    "PX_TO_BOOK_RATIO",
    "EV_TO_T12M_EBITDA",
    "FREE_CASH_FLOW_YIELD",
}

BDPCallable = Callable[..., Any]
BDHCallable = Callable[..., Any]


def _default_bdp(securities: list[str], flds: list[str] | None = None, **kwargs: Any) -> Any:
    """Default caller for xbbg.blp.bdp."""
    try:
        from xbbg import blp

        target_flds = flds or []
        return blp.bdp(securities=securities, flds=target_flds, **kwargs)
    except ImportError:
        logger.error("xbbg is not installed or Bloomberg API not available.")
        return pd.DataFrame()


def _default_bdh(
    tickers: list[str], flds: list[str], start_date: str, end_date: str, **kwargs: Any
) -> Any:
    """Default caller for xbbg.blp.bdh."""
    try:
        from xbbg import blp

        return blp.bdh(
            tickers=tickers,
            flds=flds,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
    except ImportError:
        logger.error("xbbg is not installed or Bloomberg API not available.")
        return pd.DataFrame()


class BloombergFundamentalsDownloader:
    """Fetches fundamental data from Bloomberg with quota protection and SQLite caching."""

    def __init__(
        self,
        store: SQLiteStore,
        config: PortfolioConfig | None = None,
        bdp_func: BDPCallable | None = None,
        bdh_func: BDHCallable | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self.bdp_func = bdp_func or _default_bdp
        self.bdh_func = bdh_func or _default_bdh

    def fetch_and_sync_fundamentals(
        self,
        tickers: Sequence[str],
        as_of_date: str,
        metrics: Sequence[str] = CORE_FUNDAMENTAL_METRICS,
        batch_size: int = BATCH_SIZE,
        throttle: float = THROTTLE_SECONDS,
        currency: str = "EUR",
        skip_existing: bool = True,
    ) -> int:
        """Fetch fundamental metrics for tickers, apply caching, and upsert to SQLite."""
        if not tickers:
            return 0

        needed_tickers = list(tickers)
        if skip_existing:
            existing = self.store.get_fundamentals_pit(as_of_date=as_of_date)
            if not existing.empty:
                existing_tickers = set(existing["ticker"].unique())
                needed_tickers = [t for t in needed_tickers if t not in existing_tickers]

        if not needed_tickers:
            logger.info("All %d tickers already cached for %s.", len(tickers), as_of_date)
            return 0

        logger.info(
            "Fetching fundamentals for %d tickers as of %s in batches of %d",
            len(needed_tickers),
            as_of_date,
            batch_size,
        )

        total_upserted = 0
        chunks = chunk_list(needed_tickers, batch_size)

        for i, chunk in enumerate(chunks, 1):
            sec_keys = [ensure_bloomberg_security_suffix(t) for t in chunk]
            clean_to_raw = dict(zip(sec_keys, chunk, strict=False))

            try:
                raw_res = self.bdp_func(
                    securities=sec_keys,
                    flds=list(metrics),
                    currency=currency,
                )
                df = _ensure_pandas(raw_res)

                if not df.empty:
                    records = self._format_bdp_fundamentals(
                        df=df,
                        clean_to_raw=clean_to_raw,
                        as_of_date=as_of_date,
                    )
                    if not records.empty:
                        count = self.store.upsert_fundamentals_pit(records)
                        total_upserted += count
            except Exception as exc:
                logger.error("Error fetching fundamentals for batch %d: %s", i, exc)

            if i < len(chunks) and throttle > 0:
                time.sleep(throttle)

        return total_upserted

    def _format_bdp_fundamentals(
        self,
        df: pd.DataFrame,
        clean_to_raw: dict[str, str],
        as_of_date: str,
    ) -> pd.DataFrame:
        """Convert BDP output DataFrame into standardized fundamentals_pit schema."""
        out_rows: list[dict[str, Any]] = []

        # Tickers are typically in df.index or column
        df_work = df.set_index("ticker") if "ticker" in df.columns else df.copy()

        for sec_key, row in df_work.iterrows():
            ticker_raw = clean_to_raw.get(str(sec_key), str(sec_key))
            # Remove any trailing " Equity" if not in clean_to_raw
            if ticker_raw.endswith(" Equity"):
                ticker_raw = ticker_raw[:-7].strip()

            for col in df_work.columns:
                val = row[col]
                if pd.notna(val):
                    try:
                        num_val = float(val)
                        if not np.isinf(num_val):
                            metric_name = str(col).upper().strip()
                            out_rows.append(
                                {
                                    "as_of_date": as_of_date,
                                    "filing_date": as_of_date,
                                    "ticker": ticker_raw,
                                    "metric": metric_name,
                                    "value": num_val,
                                }
                            )
                    except (ValueError, TypeError):
                        continue

        if not out_rows:
            return pd.DataFrame(columns=["as_of_date", "filing_date", "ticker", "metric", "value"])

        return pd.DataFrame(out_rows)

    def fetch_and_sync_historical_fundamentals(
        self,
        tickers: Sequence[str],
        start_date: str = "2010-01-01",
        end_date: str = "2026-12-31",
        metrics: Sequence[str] = CORE_FUNDAMENTAL_METRICS,
        batch_size: int = BATCH_SIZE,
        throttle: float = THROTTLE_SECONDS,
        currency: str = "EUR",
        periodicity: str = "M",
        skip_existing: bool = True,
        limit_tickers: int | None = None,
        apply_reporting_lag_days: int = 60,
    ) -> int:
        """Fetch historical monthly fundamentals via xbbg BDH without affecting securities count."""
        if not tickers:
            return 0

        target_tickers = list(tickers)
        if skip_existing:
            query = "SELECT DISTINCT ticker FROM fundamentals_pit"
            with self.store.get_connection() as conn:
                existing_df = pd.read_sql_query(query, conn)
            existing_set = set(existing_df["ticker"].unique()) if not existing_df.empty else set()
            target_tickers = [t for t in target_tickers if t not in existing_set]

        if limit_tickers is not None and limit_tickers > 0:
            target_tickers = target_tickers[:limit_tickers]

        if not target_tickers:
            print("All requested tickers already have fundamentals cached in SQLite.")
            return 0

        print(
            f"Fetching historical fundamentals via BDH for {len(target_tickers)} tickers "
            f"({start_date} to {end_date}, batch_size={batch_size})..."
        )

        total_upserted = 0
        chunks = chunk_list(target_tickers, batch_size)

        for i, chunk in enumerate(chunks, 1):
            sec_keys = [ensure_bloomberg_security_suffix(t) for t in chunk]
            clean_to_raw = dict(zip(sec_keys, chunk, strict=False))

            try:
                raw_res = self.bdh_func(
                    tickers=sec_keys,
                    flds=list(metrics),
                    start_date=start_date,
                    end_date=end_date,
                    Currency=currency,
                    Per=periodicity,
                )
                df = _ensure_pandas(raw_res)

                if not df.empty:
                    records = self._format_bdh_fundamentals(
                        df=df,
                        clean_to_raw=clean_to_raw,
                        apply_reporting_lag_days=apply_reporting_lag_days,
                    )
                    if not records.empty:
                        count = self.store.upsert_fundamentals_pit(records)
                        total_upserted += count
                        print(
                            f"  Batch {i}/{len(chunks)} ({len(chunk)} tickers): "
                            f"upserted {count} fundamental records"
                        )
            except Exception as exc:
                logger.error("Error fetching historical fundamentals for batch %d: %s", i, exc)
                print(f"  [!] Error on batch {i}: {exc}")

            if i < len(chunks) and throttle > 0:
                time.sleep(throttle)

        return total_upserted

    def _format_bdh_fundamentals(
        self,
        df: pd.DataFrame,
        clean_to_raw: dict[str, str],
        apply_reporting_lag_days: int = 60,
    ) -> pd.DataFrame:
        """Convert BDH output (tidy long or MultiIndex) into fundamentals_pit schema."""
        if df.empty:
            return pd.DataFrame(columns=["as_of_date", "filing_date", "ticker", "metric", "value"])

        records: list[dict[str, Any]] = []

        # Case A: Tidy long format
        if {"ticker", "date", "field", "value"}.issubset(df.columns):
            for _, row in df.iterrows():
                v = row["value"]
                if pd.isna(v) or np.isinf(v):
                    continue
                d_str = str(row["date"])[:10]
                raw_t = str(row["ticker"])
                clean_t = clean_to_raw.get(raw_t, raw_t)
                if clean_t.upper().endswith(" EQUITY"):
                    clean_t = clean_t[:-7].strip()
                field_str = str(row["field"]).upper().strip()
                if field_str in CORE_FUNDAMENTAL_METRICS:
                    if field_str in PRICE_BASED_METRICS or apply_reporting_lag_days <= 0:
                        eff_date = d_str
                    else:
                        eff_date = (
                            pd.Timestamp(d_str) + pd.Timedelta(days=apply_reporting_lag_days)
                        ).strftime("%Y-%m-%d")

                    records.append(
                        {
                            "as_of_date": eff_date,
                            "filing_date": eff_date,
                            "ticker": clean_t,
                            "metric": field_str,
                            "value": float(v),
                        }
                    )
        elif isinstance(df.columns, pd.MultiIndex):
            # Case B: MultiIndex (ticker, field) with date index
            for dt_idx, row in df.iterrows():
                d_str = str(dt_idx)[:10]
                for raw_sec, field_name in df.columns:
                    v = row[(raw_sec, field_name)]
                    if pd.isna(v) or np.isinf(v):
                        continue
                    field_str = str(field_name).upper().strip()
                    if field_str not in CORE_FUNDAMENTAL_METRICS:
                        continue
                    clean_t = clean_to_raw.get(str(raw_sec), str(raw_sec))
                    if clean_t.upper().endswith(" EQUITY"):
                        clean_t = clean_t[:-7].strip()

                    if field_str in PRICE_BASED_METRICS or apply_reporting_lag_days <= 0:
                        eff_date = d_str
                    else:
                        eff_date = (
                            pd.Timestamp(d_str) + pd.Timedelta(days=apply_reporting_lag_days)
                        ).strftime("%Y-%m-%d")

                    records.append(
                        {
                            "as_of_date": eff_date,
                            "filing_date": eff_date,
                            "ticker": clean_t,
                            "metric": field_str,
                            "value": float(v),
                        }
                    )

        if not records:
            return pd.DataFrame(columns=["as_of_date", "filing_date", "ticker", "metric", "value"])

        out = pd.DataFrame(records)
        return out.drop_duplicates(subset=["as_of_date", "ticker", "metric"], keep="last")
