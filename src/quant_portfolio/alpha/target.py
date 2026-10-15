"""Target generation for Phase 4 Machine Learning ranking.

Calculates 21-day forward return in excess of the cross-sectional regional median:
    R_{i, t -> t+H} = (TOT_RETURN_INDEX_{i, t+H} / TOT_RETURN_INDEX_{i, t}) - 1
    y_{i, t} = R_{i, t -> t+H} - Median_{j in Region(i)}(R_{j, t -> t+H})
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quant_portfolio.config import AlphaConfig, PortfolioConfig
from quant_portfolio.storage.schema import ParquetMarketStore

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class TargetGenerator:
    """Computes forward excess returns over regional cross-sectional median."""

    def __init__(
        self,
        parquet_store: ParquetMarketStore | None = None,
        config: PortfolioConfig | None = None,
    ) -> None:
        self.config = config or PortfolioConfig()
        paths = self.config.paths.resolve_paths()
        self.parquet_store = parquet_store or ParquetMarketStore(paths.parquet_dir)
        self.alpha_config: AlphaConfig = self.config.alpha or AlphaConfig()

    def compute_forward_returns(
        self,
        as_of_date: str,
        tickers: Sequence[str],
        horizon_days: int | None = None,
    ) -> pd.DataFrame:
        """Compute the forward total return over horizon_days starting from as_of_date.

        Returns DataFrame with columns: ['ticker', 'fwd_return'].
        """
        h = horizon_days or self.alpha_config.forward_horizon_days
        if not tickers:
            return pd.DataFrame(columns=["ticker", "fwd_return"])

        # Determine year range needed: from as_of_date to roughly as_of_date + 60 days
        dt_start = pd.Timestamp(as_of_date)
        dt_end = dt_start + pd.Timedelta(days=max(h * 3, 60))

        # Read market prices for the selected tickers
        df = self.parquet_store.read_daily_market(
            start_date=as_of_date,
            end_date=dt_end.strftime("%Y-%m-%d"),
            tickers=list(tickers),
        )

        if df.empty:
            return pd.DataFrame({"ticker": list(tickers), "fwd_return": np.nan})

        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= dt_start].sort_values(["ticker", "date"]).reset_index(drop=True)

        results: list[dict[str, object]] = []
        for ticker in tickers:
            t_df = df[df["ticker"] == ticker].reset_index(drop=True)
            if len(t_df) <= h or t_df.iloc[0]["date"] != dt_start:
                # If exact as_of_date is missing or fewer than h future trading days available
                results.append({"ticker": ticker, "fwd_return": np.nan})
                continue

            p0 = float(t_df.iloc[0]["tot_return_idx_eur"])
            ph = float(t_df.iloc[h]["tot_return_idx_eur"])

            ret = (ph / p0) - 1.0 if p0 > 0 else np.nan
            results.append({"ticker": ticker, "fwd_return": ret})

        return pd.DataFrame(results)

    def compute_target_at_date(
        self,
        as_of_date: str,
        universe_df: pd.DataFrame,
        horizon_days: int | None = None,
    ) -> pd.DataFrame:
        """Compute the regional median excess target y_{i,t} for the given universe at as_of_date.

        universe_df must contain at least ['ticker', 'region_bloc'].
        Returns DataFrame with:
            ['as_of_date', 'ticker', 'region_bloc', 'fwd_return', 'regional_median', 'target'].
        """
        h = horizon_days or self.alpha_config.forward_horizon_days
        if universe_df.empty:
            return pd.DataFrame(
                columns=[
                    "as_of_date",
                    "ticker",
                    "region_bloc",
                    "fwd_return",
                    "regional_median",
                    "target",
                ]
            )

        tickers = universe_df["ticker"].tolist()
        fwd_df = self.compute_forward_returns(
            as_of_date=as_of_date, tickers=tickers, horizon_days=h
        )

        merged = universe_df[["ticker", "region_bloc"]].merge(fwd_df, on="ticker", how="left")
        merged["as_of_date"] = as_of_date

        # Calculate cross-sectional median forward return for each regional sleeve
        merged["regional_median"] = np.nan
        merged["target"] = np.nan

        for reg in merged["region_bloc"].unique():
            reg_mask = merged["region_bloc"] == reg
            valid_returns = merged.loc[reg_mask, "fwd_return"].dropna()
            if not valid_returns.empty:
                med = float(valid_returns.median())
                merged.loc[reg_mask, "regional_median"] = med
                merged.loc[reg_mask, "target"] = merged.loc[reg_mask, "fwd_return"] - med

        cols = ["as_of_date", "ticker", "region_bloc", "fwd_return", "regional_median", "target"]
        return merged[cols].copy()

    def compute_target_panel(
        self,
        dates: Sequence[str],
        universe_panel: pd.DataFrame,
        horizon_days: int | None = None,
    ) -> pd.DataFrame:
        """Compute target matrix across a panel of dates."""
        frames: list[pd.DataFrame] = []
        for d in sorted(dates):
            uni_d = universe_panel[universe_panel["as_of_date"] == d]
            if not uni_d.empty:
                t_df = self.compute_target_at_date(
                    as_of_date=d, universe_df=uni_d, horizon_days=horizon_days
                )
                frames.append(t_df)

        if not frames:
            return pd.DataFrame(
                columns=[
                    "as_of_date",
                    "ticker",
                    "region_bloc",
                    "fwd_return",
                    "regional_median",
                    "target",
                ]
            )

        return pd.concat(frames, ignore_index=True)
