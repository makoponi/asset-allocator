"""Bloomberg xbbg market data client for transatlantic historical EUR series.

Fetches TOT_RETURN_INDEX_GROSS_DVDS, PX_LAST, and PX_VOLUME with Currency='EUR',
enforcing 50-ticker batching and a 0.3s-0.5s throttle to respect Bloomberg DAPI burst rate.
"""

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Final

import pandas as pd

from quant_portfolio.config import RegionBloc

logger = logging.getLogger(__name__)

DEFAULT_SERIES_FIELDS: Final[tuple[str, ...]] = (
    "TOT_RETURN_INDEX_GROSS_DVDS",
    "PX_LAST",
    "PX_VOLUME",
)
BATCH_SIZE: Final[int] = 50
THROTTLE_SECONDS: Final[float] = 0.35

BDHCallable = Callable[..., pd.DataFrame]


def chunk_list[T](items: Sequence[T], chunk_size: int) -> list[list[T]]:
    """Partition a sequence into sub-lists of maximum size chunk_size."""
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]


def _ensure_pandas(res: Any) -> pd.DataFrame:
    """Safely convert xbbg / narwhals result to standard pandas DataFrame."""
    if res is None:
        return pd.DataFrame()
    if hasattr(res, "to_pandas"):
        return res.to_pandas()  # type: ignore[no-any-return]
    if isinstance(res, pd.DataFrame):
        return res
    return pd.DataFrame(res)


def ensure_bloomberg_security_suffix(ticker: str, default_suffix: str = "Equity") -> str:
    """Ensure ticker has an asset class suffix for Bloomberg requests."""
    t = ticker.strip()
    upper_t = t.upper()
    if upper_t.endswith(" EQUITY") or upper_t.endswith(" INDEX") or upper_t.endswith(" CURNCY"):
        return t
    return f"{t} {default_suffix}"


def format_xbbg_bdh_output(
    raw_df: pd.DataFrame,
    ticker_region_map: dict[str, RegionBloc],
) -> pd.DataFrame:
    """Format bdh output (tidy long or MultiIndex) into standard daily market schema.

    Schema: [date, ticker, region_bloc, px_last_eur, tot_return_idx_eur, volume_eur]
    """
    df = _ensure_pandas(raw_df)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "region_bloc",
                "px_last_eur",
                "tot_return_idx_eur",
                "volume_eur",
            ]
        )

    records: list[dict[str, Any]] = []

    # Case A: Tidy long format (xbbg 1.5+ narwhals with ['ticker', 'date', 'field', 'value'])
    if {"ticker", "date", "field", "value"}.issubset(df.columns):
        piv = df.pivot(index=["date", "ticker"], columns="field", values="value").reset_index()
        for _, row in piv.iterrows():
            date_str = str(row["date"])[:10]
            raw_t = str(row["ticker"])
            clean_t = raw_t[:-7].strip() if raw_t.upper().endswith(" EQUITY") else raw_t.strip()
            px = row.get("PX_LAST")
            tr = row.get("TOT_RETURN_INDEX_GROSS_DVDS")
            vol = row.get("PX_VOLUME")

            if pd.isna(px) and pd.isna(tr):
                continue

            reg = ticker_region_map.get(clean_t, ticker_region_map.get(raw_t, "NA"))
            records.append(
                {
                    "date": date_str,
                    "ticker": clean_t,
                    "region_bloc": reg,
                    "px_last_eur": float(px) if pd.notna(px) else 0.0,
                    "tot_return_idx_eur": float(tr) if pd.notna(tr) else float(px or 0.0),
                    "volume_eur": float(vol) if pd.notna(vol) else 0.0,
                }
            )
    elif isinstance(df.columns, pd.MultiIndex):
        # Case B: MultiIndex format: (ticker, field) with date index
        tickers = df.columns.levels[0]
        for dt_idx, row in df.iterrows():
            date_str = str(dt_idx)[:10]
            for raw_ticker in tickers:
                raw_t = str(raw_ticker)
                clean_t = raw_t[:-7].strip() if raw_t.upper().endswith(" EQUITY") else raw_t.strip()
                px_col = (
                    (raw_ticker, "PX_LAST")
                    if (raw_ticker, "PX_LAST") in df.columns
                    else (clean_t, "PX_LAST")
                )
                if px_col not in df.columns:
                    continue

                tr_col = (
                    (raw_ticker, "TOT_RETURN_INDEX_GROSS_DVDS")
                    if (raw_ticker, "TOT_RETURN_INDEX_GROSS_DVDS") in df.columns
                    else (clean_t, "TOT_RETURN_INDEX_GROSS_DVDS")
                )
                vol_col = (
                    (raw_ticker, "PX_VOLUME")
                    if (raw_ticker, "PX_VOLUME") in df.columns
                    else (clean_t, "PX_VOLUME")
                )

                px = row.get(px_col)
                tr = row.get(tr_col)
                vol = row.get(vol_col)

                if pd.isna(px) and pd.isna(tr):
                    continue

                reg = ticker_region_map.get(clean_t, ticker_region_map.get(raw_t, "NA"))
                records.append(
                    {
                        "date": date_str,
                        "ticker": clean_t,
                        "region_bloc": reg,
                        "px_last_eur": float(px) if pd.notna(px) else 0.0,
                        "tot_return_idx_eur": float(tr) if pd.notna(tr) else float(px or 0.0),
                        "volume_eur": float(vol) if pd.notna(vol) else 0.0,
                    }
                )
    else:
        # Single ticker case or flat DataFrame
        # Expecting index as Date and fields as columns
        for dt_idx, row in df.iterrows():
            date_str = str(dt_idx)[:10]
            for _col in df.columns:
                raw_ticker = list(ticker_region_map.keys())[0] if ticker_region_map else "UNKNOWN"
                clean_t = (
                    raw_ticker[:-7].strip()
                    if raw_ticker.upper().endswith(" EQUITY")
                    else raw_ticker.strip()
                )
                px = row.get("PX_LAST")
                tr = row.get("TOT_RETURN_INDEX_GROSS_DVDS")
                vol = row.get("PX_VOLUME")
                if pd.isna(px) and pd.isna(tr):
                    continue
                records.append(
                    {
                        "date": date_str,
                        "ticker": clean_t,
                        "region_bloc": ticker_region_map.get(clean_t, "NA"),
                        "px_last_eur": float(px) if pd.notna(px) else 0.0,
                        "tot_return_idx_eur": float(tr) if pd.notna(tr) else float(px or 0.0),
                        "volume_eur": float(vol) if pd.notna(vol) else 0.0,
                    }
                )
                break

    if not records:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "region_bloc",
                "px_last_eur",
                "tot_return_idx_eur",
                "volume_eur",
            ]
        )

    out_df = pd.DataFrame(records)
    return (
        out_df.drop_duplicates(subset=["date", "ticker"])
        .sort_values(by=["date", "ticker"])
        .reset_index(drop=True)
    )


class BloombergSeriesDownloader:
    """Batch-aware downloader for Bloomberg daily historical series in EUR via xbbg."""

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        throttle_seconds: float = THROTTLE_SECONDS,
        bdh_fn: BDHCallable | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.throttle_seconds = throttle_seconds
        self._bdh_fn = bdh_fn

    def _get_bdh(self) -> BDHCallable:
        """Resolve bdh function: user-provided or real xbbg.blp.bdh."""
        if self._bdh_fn is not None:
            return self._bdh_fn

        try:
            from xbbg import blp

            return blp.bdh
        except ImportError as exc:
            raise ImportError(
                "xbbg is not installed. Install with 'uv add xbbg' or configure Bloomberg."
            ) from exc

    def fetch_series_batch(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        fields: tuple[str, ...] = DEFAULT_SERIES_FIELDS,
        currency: str = "EUR",
    ) -> pd.DataFrame:
        """Execute a single bdh query for a chunk of tickers."""
        if not tickers:
            return pd.DataFrame()

        bdh = self._get_bdh()
        bbg_tickers = [ensure_bloomberg_security_suffix(t) for t in tickers]
        logger.info(
            "Fetching BDH batch for %d tickers from %s to %s (Currency=%s)",
            len(bbg_tickers),
            start_date,
            end_date,
            currency,
        )
        res = bdh(
            tickers=bbg_tickers,
            flds=list(fields),
            start_date=start_date,
            end_date=end_date,
            Currency=currency,
        )
        return _ensure_pandas(res)

    def stream_all_series(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        ticker_region_map: dict[str, RegionBloc],
        fields: tuple[str, ...] = DEFAULT_SERIES_FIELDS,
        currency: str = "EUR",
    ) -> list[pd.DataFrame]:
        """Fetch historical series yielding formatted batches of max batch_size."""
        if not tickers:
            return []

        unique_tickers = sorted(set(tickers))
        batches = chunk_list(unique_tickers, self.batch_size)
        logger.info(
            "Streaming %d tickers across %d batches of max %d",
            len(unique_tickers),
            len(batches),
            self.batch_size,
        )

        batch_frames: list[pd.DataFrame] = []
        for i, batch in enumerate(batches):
            try:
                batch_raw = self.fetch_series_batch(
                    tickers=batch,
                    start_date=start_date,
                    end_date=end_date,
                    fields=fields,
                    currency=currency,
                )
                if not batch_raw.empty:
                    formatted = format_xbbg_bdh_output(batch_raw, ticker_region_map)
                    if not formatted.empty:
                        batch_frames.append(formatted)
            except Exception as e:
                logger.error("Error fetching batch %d/%d: %s", i + 1, len(batches), e)

            if i < len(batches) - 1 and self.throttle_seconds > 0:
                time.sleep(self.throttle_seconds)

        return batch_frames

    def fetch_all_series(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        ticker_region_map: dict[str, RegionBloc],
        fields: tuple[str, ...] = DEFAULT_SERIES_FIELDS,
        currency: str = "EUR",
    ) -> pd.DataFrame:
        """Fetch all historical series in 50-ticker throttled batches and format output."""
        frames = self.stream_all_series(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            ticker_region_map=ticker_region_map,
            fields=fields,
            currency=currency,
        )
        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        return (
            combined.drop_duplicates(subset=["date", "ticker"])
            .sort_values(by=["date", "ticker"])
            .reset_index(drop=True)
        )
