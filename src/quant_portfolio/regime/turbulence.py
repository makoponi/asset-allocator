"""Mahalanobis financial turbulence metric for cross-sector equity returns."""

from typing import Any

import numpy as np
import pandas as pd


class MahalanobisTurbulence:
    """Computes Mahalanobis financial turbulence on cross-sectional or sector returns.

    Reference: Kritzman, M., & Li, Y. (2010). Skulls, Financial Turbulence, and Risk Management.
    Financial Analysts Journal, 66(5), 30-41.

    Formula at date t:
        d_t^2 = (y_t - mu_{t-1})^T * Sigma_{t-1}^{-1} * (y_t - mu_{t-1})

    To strictly prevent look-ahead bias, mu and Sigma at time t are estimated solely using
    historical observations up to t - 1.
    """

    def __init__(
        self,
        window: int = 252,
        min_periods: int = 63,
        shrinkage: float = 0.01,
    ) -> None:
        if window < 10:
            raise ValueError(f"window must be at least 10, got {window}")
        if min_periods < 5:
            raise ValueError(f"min_periods must be at least 5, got {min_periods}")
        if not (0.0 <= shrinkage < 1.0):
            raise ValueError(f"shrinkage must be in [0, 1), got {shrinkage}")

        self.window = window
        self.min_periods = min_periods
        self.shrinkage = shrinkage

    def compute_turbulence_series(self, returns_df: pd.DataFrame) -> pd.Series:
        """Calculate historical turbulence series across dates.

        Parameters
        ----------
        returns_df : pd.DataFrame
            DataFrame indexed by date (or sorted chronologically), where columns represent
            asset or sector returns (e.g. 10 GICS sectors).

        Returns
        -------
        pd.Series
            Series of Mahalanobis turbulence values indexed identically to returns_df.
            Dates before min_periods are filled with np.nan.
        """
        if returns_df.empty:
            return pd.Series(dtype=float)

        df = returns_df.dropna(how="all").copy()
        if len(df) == 0:
            return pd.Series(dtype=float, index=returns_df.index)

        # Impute isolated NaNs with column medians or zero to avoid matrix corruption
        df_clean = df.fillna(0.0)
        n_obs, n_assets = df_clean.shape

        if n_assets == 0:
            return pd.Series(dtype=float, index=returns_df.index)

        values = df_clean.to_numpy(dtype=np.float64)
        turbulence = np.full(n_obs, np.nan, dtype=np.float64)

        for t in range(self.min_periods, n_obs):
            # Strict anti-look-ahead: use data up to t - 1
            start_idx = max(0, t - self.window)
            hist = values[start_idx:t, :]

            # Mean and covariance of historical window
            mu = np.mean(hist, axis=0)
            cov = np.cov(hist, rowvar=False)

            if n_assets == 1:
                var = float(cov) + 1e-8
                inv_cov = np.array([[1.0 / var]])
                mu_vec = np.array([mu])
            else:
                # Regularize covariance to guarantee positive-definiteness
                trace_avg = np.trace(cov) / float(n_assets) if np.trace(cov) > 0 else 1e-4
                reg_cov = (1.0 - self.shrinkage) * cov + self.shrinkage * trace_avg * np.eye(
                    n_assets
                )
                try:
                    inv_cov = np.linalg.pinv(reg_cov)
                except np.linalg.LinAlgError:
                    inv_cov = np.linalg.pinv(cov + 1e-6 * np.eye(n_assets))
                mu_vec = mu

            # Current observation at date t
            y_t = values[t, :]
            diff = y_t - mu_vec
            quad = diff.T @ inv_cov @ diff
            dist = float(np.asarray(quad).item())
            turbulence[t] = max(0.0, dist)

        return pd.Series(turbulence, index=df_clean.index, name="turbulence")

    @staticmethod
    def aggregate_sector_returns(
        stock_returns: pd.DataFrame,
        ticker_to_sector: dict[str, Any],
    ) -> pd.DataFrame:
        """Aggregate individual stock returns into equal-weighted sector returns.

        Parameters
        ----------
        stock_returns : pd.DataFrame
            DataFrame where index is date and columns are stock tickers.
        ticker_to_sector : dict[str, Any]
            Mapping from ticker to sector identifier (e.g. GICS sector code 10, 15, 20...).

        Returns
        -------
        pd.DataFrame
            DataFrame where columns are unique sector identifiers and values are
            equal-weighted sector returns.
        """
        if stock_returns.empty:
            return pd.DataFrame()

        # Map available tickers to sectors
        valid_cols = [c for c in stock_returns.columns if c in ticker_to_sector]
        if not valid_cols:
            return pd.DataFrame(index=stock_returns.index)

        sector_series: dict[Any, pd.Series] = {}
        # Group tickers by sector
        sectors_map: dict[Any, list[str]] = {}
        for t in valid_cols:
            sec = ticker_to_sector[t]
            sectors_map.setdefault(sec, []).append(t)

        for sec, sec_tickers in sectors_map.items():
            sector_series[sec] = stock_returns[sec_tickers].mean(axis=1, skipna=True)

        return pd.DataFrame(sector_series, index=stock_returns.index)
