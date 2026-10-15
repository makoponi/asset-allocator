"""Technical, price momentum, and CAPM risk features calculated from daily EUR market series.

Features computed:
- MOM_1M: 21-day price reversal: (P_t / P_{t-21}) - 1
- MOM_12M_1M: 12-month momentum excluding recent month: (P_{t-21} / P_{t-252}) - 1
- BETA_REGIONAL_252D: 252-day OLS slope against regional benchmark (SPX Index / SXXP Index)
- IDIO_VOL_252D: Annualized standard deviation of residual returns from regional CAPM regression
- AMIHUD_ILLIQUIDITY_21D: 21-day average ratio of absolute return to trading volume in EUR (* 1e6)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quant_portfolio.config import FeatureConfig, RegionBloc

if TYPE_CHECKING:
    from quant_portfolio.storage.schema import ParquetMarketStore

DEFAULT_BENCHMARKS: dict[RegionBloc, str] = {
    "NA": "SPX Index",
    "EU": "SXXP Index",
}


class TechnicalFeatureCalculator:
    """Calculates price momentum, regional beta, and idiosyncratic risk as of as_of_date."""

    def __init__(
        self,
        parquet_store: ParquetMarketStore,
        config: FeatureConfig | None = None,
        benchmark_map: dict[RegionBloc, str] | None = None,
    ) -> None:
        self.parquet_store = parquet_store
        self.config = config or FeatureConfig()
        self.benchmark_map = benchmark_map or DEFAULT_BENCHMARKS

    def compute_features_at_date(
        self,
        as_of_date: str,
        tickers: list[str],
        region_map: dict[str, RegionBloc],
    ) -> pd.DataFrame:
        """Compute the 5 technical features for the specified tickers as of as_of_date.

        Enforces zero look-ahead bias by only querying market records where date <= as_of_date.
        """
        if not tickers:
            return pd.DataFrame(
                columns=[
                    "as_of_date",
                    "ticker",
                    "MOM_1M",
                    "MOM_12M_1M",
                    "IDIO_VOL_252D",
                    "BETA_REGIONAL_252D",
                    "AMIHUD_ILLIQUIDITY_21D",
                ]
            )

        # Include regional benchmarks to compute Beta and Idiosyncratic Volatility
        bmk_tickers = list(set(self.benchmark_map.values()))
        all_required_tickers = sorted(set(tickers) | set(bmk_tickers))

        # We look back up to ~500 calendar days to guarantee at least 252 trading days
        start_date = (pd.to_datetime(as_of_date) - pd.Timedelta(days=550)).strftime("%Y-%m-%d")

        market_df = self.parquet_store.read_daily_market(
            start_date=start_date,
            end_date=as_of_date,
            tickers=all_required_tickers,
        )

        if market_df.empty:
            return self._build_empty_result(as_of_date, tickers)

        # Precompute daily returns for benchmarks
        bmk_returns: dict[str, pd.Series[float]] = {}
        for bmk in bmk_tickers:
            bmk_sub = market_df[market_df["ticker"] == bmk].sort_values("date")
            if not bmk_sub.empty:
                s_prices = bmk_sub.set_index("date")["tot_return_idx_eur"].astype(float)
                s_rets = s_prices.pct_change().dropna()
                bmk_returns[bmk] = s_rets

        results: list[dict[str, float | str]] = []
        for ticker in tickers:
            reg_bloc = region_map.get(ticker, "NA")
            bmk_ticker = self.benchmark_map.get(reg_bloc, "SPX Index")
            bmk_series = bmk_returns.get(bmk_ticker, pd.Series(dtype=float))

            sub = market_df[market_df["ticker"] == ticker].sort_values("date")
            feat_row = self._compute_single_ticker_features(
                ticker=ticker,
                as_of_date=as_of_date,
                ticker_df=sub,
                bmk_returns=bmk_series,
            )
            results.append(feat_row)

        return pd.DataFrame(results)

    def _compute_single_ticker_features(
        self,
        ticker: str,
        as_of_date: str,
        ticker_df: pd.DataFrame,
        bmk_returns: pd.Series[float],
    ) -> dict[str, float | str]:
        """Compute 5 core technical features for a single ticker."""
        row: dict[str, float | str] = {
            "as_of_date": as_of_date,
            "ticker": ticker,
            "MOM_1M": 0.0,
            "MOM_12M_1M": 0.0,
            "IDIO_VOL_252D": 0.20,
            "BETA_REGIONAL_252D": 1.0,
            "AMIHUD_ILLIQUIDITY_21D": 0.0,
        }

        if len(ticker_df) < 5:
            return row

        prices = ticker_df["tot_return_idx_eur"].to_numpy(dtype=np.float64)
        dates = ticker_df["date"].to_numpy(dtype=str)
        volumes = ticker_df["volume_eur"].to_numpy(dtype=np.float64)

        n_pts = len(prices)
        rev_w = self.config.momentum_reversal_window  # 21 days
        trend_w = self.config.momentum_trend_window  # 252 days

        # 1. MOM_1M: (P_t / P_{t-21}) - 1
        if n_pts > rev_w and prices[-(rev_w + 1)] > 1e-8:
            row["MOM_1M"] = float((prices[-1] / prices[-(rev_w + 1)]) - 1.0)

        # 2. MOM_12M_1M: (P_{t-21} / P_{t-252}) - 1
        if n_pts > trend_w:
            p_past = prices[-(trend_w + 1)]
            p_recent = prices[-(rev_w + 1)]
            if p_past > 1e-8:
                row["MOM_12M_1M"] = float((p_recent / p_past) - 1.0)
        elif n_pts > self.config.min_history_days_beta:
            p_past = prices[0]
            p_recent = prices[-(rev_w + 1)]
            if p_past > 1e-8:
                row["MOM_12M_1M"] = float((p_recent / p_past) - 1.0)

        # Daily returns of the stock
        with np.errstate(divide="ignore", invalid="ignore"):
            stock_rets = np.diff(prices) / prices[:-1]
        stock_ret_dates = dates[1:]
        stock_vols = volumes[1:]

        # 3. AMIHUD_ILLIQUIDITY_21D: Mean(|r_tau| / (volume_tau + 1.0)) * 1e6
        if len(stock_rets) >= 5:
            sub_rets = stock_rets[-rev_w:]
            sub_vols = stock_vols[-rev_w:]
            amihud_vals = np.abs(sub_rets) / (sub_vols + 1.0)
            row["AMIHUD_ILLIQUIDITY_21D"] = float(np.mean(amihud_vals) * 1e6)

        # 4. & 5. BETA_REGIONAL_252D and IDIO_VOL_252D
        if len(stock_rets) >= self.config.min_history_days_beta and not bmk_returns.empty:
            s_series = pd.Series(stock_rets, index=stock_ret_dates)
            joint = pd.concat([s_series, bmk_returns], axis=1, join="inner").dropna()
            joint.columns = ["stock", "benchmark"]

            # Use up to the last 252 daily observations
            if len(joint) > trend_w:
                joint = joint.iloc[-trend_w:]

            n_obs = len(joint)
            if n_obs >= self.config.min_history_days_beta:
                r_s = joint["stock"].to_numpy(dtype=np.float64)
                r_b = joint["benchmark"].to_numpy(dtype=np.float64)

                var_b = float(np.var(r_b, ddof=1))
                if var_b > 1e-10:
                    cov_sb = float(np.cov(r_s, r_b)[0, 1])
                    beta = cov_sb / var_b
                    alpha = float(np.mean(r_s) - beta * np.mean(r_b))
                    residuals = r_s - (alpha + beta * r_b)

                    # Annualized residual volatility
                    ddof = 2 if n_obs > 2 else 1
                    sigma_resid = float(np.std(residuals, ddof=ddof))
                    row["BETA_REGIONAL_252D"] = float(beta)
                    row["IDIO_VOL_252D"] = float(sigma_resid * np.sqrt(252.0))

        return row

    def _build_empty_result(self, as_of_date: str, tickers: list[str]) -> pd.DataFrame:
        """Construct a default neutral DataFrame when market series are not found."""
        return pd.DataFrame(
            [
                {
                    "as_of_date": as_of_date,
                    "ticker": t,
                    "MOM_1M": 0.0,
                    "MOM_12M_1M": 0.0,
                    "IDIO_VOL_252D": 0.20,
                    "BETA_REGIONAL_252D": 1.0,
                    "AMIHUD_ILLIQUIDITY_21D": 0.0,
                }
                for t in tickers
            ]
        )
