"""Transatlantic trading calendar alignment and controlled forward-fill.

Synchronizes European (LSE/Euronext) and North American (NYSE/TSX) daily series
onto a unified master trading calendar, applying forward-fill (max 3 business days)
for asynchronous holidays without look-ahead bias and setting volume to 0.0 on closed days.
"""

from collections.abc import Iterable, Mapping
from typing import Any, Final

import numpy as np
import pandas as pd

MAX_FORWARD_FILL_DAYS: Final[int] = 3


def build_master_trading_calendar(
    dates_by_region: Mapping[str, Iterable[Any]],
) -> pd.DatetimeIndex:
    """Build a master trading calendar as the sorted union of active trading days across regions."""
    all_dates: set[pd.Timestamp] = set()
    for _region, dates in dates_by_region.items():
        dt_idx = pd.to_datetime(list(dates))
        for item in dt_idx:
            all_dates.add(pd.Timestamp(item))
    return pd.DatetimeIndex(sorted(all_dates))


def align_single_ticker_series(
    ticker_df: pd.DataFrame,
    master_calendar: pd.DatetimeIndex,
    max_ffill_days: int = MAX_FORWARD_FILL_DAYS,
) -> pd.DataFrame:
    """Align a single security's daily series to the master calendar.

    Applies forward-fill (limit=max_ffill_days) to price and total return index.
    Zeroes out volume on days where the asset did not trade.
    Preserves ticker and region_bloc metadata.
    """
    if ticker_df.empty:
        return ticker_df.copy()

    df = ticker_df.copy()
    ticker = df["ticker"].iloc[0]
    region = df["region_bloc"].iloc[0]

    df["dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("dt").drop_duplicates(subset=["dt"]).set_index("dt")

    # Filter master calendar to start from asset's first available date
    # (prevents backward filling before asset existence/listing)
    first_date = df.index.min()
    relevant_calendar = master_calendar[master_calendar >= first_date]

    reindexed = df.reindex(relevant_calendar)

    # Detect days that were reindexed (missing in original)
    was_missing = reindexed["px_last_eur"].isna()

    # Forward-fill prices and total return index with strict limit
    reindexed["px_last_eur"] = reindexed["px_last_eur"].ffill(limit=max_ffill_days)
    reindexed["tot_return_idx_eur"] = reindexed["tot_return_idx_eur"].ffill(limit=max_ffill_days)

    # Volume: for missing days (market closed), set volume to 0.0
    reindexed["volume_eur"] = np.where(was_missing, 0.0, reindexed["volume_eur"].fillna(0.0))

    # Fill metadata
    reindexed["ticker"] = ticker
    reindexed["region_bloc"] = region
    reindexed["date"] = pd.DatetimeIndex(reindexed.index).strftime("%Y-%m-%d")

    # Drop rows where price is still NaN (i.e. missing for > max_ffill_days)
    valid_mask = reindexed["px_last_eur"].notna()
    result = reindexed[valid_mask].reset_index(drop=True)

    cols = [
        "date",
        "ticker",
        "region_bloc",
        "px_last_eur",
        "tot_return_idx_eur",
        "volume_eur",
    ]
    return result[cols]


def align_market_panel(
    panel_df: pd.DataFrame,
    max_ffill_days: int = MAX_FORWARD_FILL_DAYS,
) -> pd.DataFrame:
    """Align an entire multi-asset market panel across transatlantic regions."""
    if panel_df.empty:
        return panel_df.copy()

    panel = panel_df.copy()
    panel["dt"] = pd.to_datetime(panel["date"])

    # Extract unique active dates per region
    dates_by_region: dict[str, list[Any]] = {}
    for region, group in panel.groupby("region_bloc"):
        dates_by_region[str(region)] = list(group["dt"].unique())

    master_calendar = build_master_trading_calendar(dates_by_region)

    aligned_frames: list[pd.DataFrame] = []
    for _ticker, ticker_group in panel.groupby("ticker"):
        aligned = align_single_ticker_series(
            ticker_group,
            master_calendar=master_calendar,
            max_ffill_days=max_ffill_days,
        )
        aligned_frames.append(aligned)

    if not aligned_frames:
        return pd.DataFrame(columns=panel_df.columns)

    combined = pd.concat(aligned_frames, ignore_index=True)
    return combined.sort_values(by=["date", "ticker"]).reset_index(drop=True)
