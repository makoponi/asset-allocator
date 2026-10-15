"""Multi-Day Return Synchronizer for Transatlantic Equity Portfolios.

Transfers daily total return indices in EUR into non-overlapping 3-business-day returns
(or weekly Friday-to-Friday returns) to eliminate timezone misalignment between
European markets (closing at 17:30 CET) and North American markets (closing at 22:00 CET).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quant_portfolio.config import PortfolioConfig, RiskConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_portfolio.storage.schema import ParquetMarketStore

logger = logging.getLogger(__name__)


class MultiDaySynchronizer:
    """Synchronizes multi-market daily equity prices into 3-day non-overlapping returns."""

    def __init__(
        self,
        parquet_store: ParquetMarketStore,
        config: PortfolioConfig | None = None,
    ) -> None:
        self.parquet_store = parquet_store
        self.config = config or PortfolioConfig()
        self.risk_config: RiskConfig = self.config.risk or RiskConfig()

    def get_aligned_price_matrix(
        self,
        tickers: Sequence[str],
        as_of_date: str,
        lookback_days: int | None = None,
        max_ffill_days: int = 3,
    ) -> pd.DataFrame:
        """Extract daily Total Return in EUR and pivot to an aligned date x ticker DataFrame.

        Applies controlled forward-filling up to max_ffill_days to handle regional holidays.
        """
        if not tickers:
            return pd.DataFrame()

        n_days = lookback_days or self.risk_config.history_days
        # Approximate calendar window needed for n_days business days (e.g. 1.5 * n_days)
        dt_end = pd.Timestamp(as_of_date)
        dt_start = dt_end - pd.Timedelta(days=int(n_days * 1.6) + 30)
        start_str = dt_start.strftime("%Y-%m-%d")

        raw_prices = self.parquet_store.read_daily_market(
            start_date=start_str,
            end_date=as_of_date,
            tickers=list(tickers),
        )

        if raw_prices.empty:
            logger.warning("No price history found for tickers up to %s", as_of_date)
            return pd.DataFrame()

        # Pivot to date x ticker
        pivot_df = raw_prices.pivot(
            index="date",
            columns="ticker",
            values="tot_return_idx_eur",
        ).sort_index()

        # Controlled forward-fill for asynchronous exchange holidays
        filled_df = pivot_df.ffill(limit=max_ffill_days)

        # Drop tickers with excessive missing history (>10% NaN after ffill)
        valid_cols = [c for c in filled_df.columns if filled_df[c].isna().mean() <= 0.10]
        aligned_df = filled_df[valid_cols].dropna()
        return aligned_df

    def compute_synchronized_returns(
        self,
        price_matrix: pd.DataFrame,
        resampling_days: int | None = None,
    ) -> pd.DataFrame:
        """Compute non-overlapping k-day discrete percentage returns from daily price matrix.

        Sampling every k business days eliminates the intraday closing lag between
        European and North American exchanges.
        """
        if price_matrix.empty or len(price_matrix) < 2:
            return pd.DataFrame()

        k = resampling_days or self.risk_config.resampling_days
        if k < 1:
            raise ValueError(f"resampling_days must be >= 1, got {k}")

        # Subsample rows every k days starting from the most recent date backwards
        # to ensure the most recent observation as_of_date is included
        n_rows = len(price_matrix)
        indices = list(range(n_rows - 1, -1, -k))
        indices.reverse()

        subsampled = price_matrix.iloc[indices]
        if len(subsampled) < 2:
            return pd.DataFrame()

        # Discrete percentage returns: P(t) / P(t-k) - 1
        returns = subsampled.pct_change().dropna(how="all")

        # Replace any inf or NaN with 0.0 or forward values
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return returns

    def extract_synchronized_returns(
        self,
        tickers: Sequence[str],
        as_of_date: str,
        lookback_days: int | None = None,
        resampling_days: int | None = None,
    ) -> pd.DataFrame:
        """End-to-end extraction: load prices, align grid, and compute 3-day returns."""
        prices = self.get_aligned_price_matrix(
            tickers=tickers,
            as_of_date=as_of_date,
            lookback_days=lookback_days,
        )
        if prices.empty:
            return pd.DataFrame()

        return self.compute_synchronized_returns(
            price_matrix=prices,
            resampling_days=resampling_days,
        )
