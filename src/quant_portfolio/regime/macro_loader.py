"""Macro series loader and feature builder for transatlantic regime detection."""

import pandas as pd

from quant_portfolio.config import MACRO_SERIES_EU, MACRO_SERIES_NA, PortfolioConfig, RegionBloc
from quant_portfolio.regime.turbulence import MahalanobisTurbulence
from quant_portfolio.storage.schema import ParquetMarketStore, SQLiteStore


class RegimeFeatureBuilder:
    """Builds aligned daily feature panels for NA and EU regime models."""

    def __init__(
        self,
        sqlite_store: SQLiteStore,
        parquet_store: ParquetMarketStore | None = None,
        config: PortfolioConfig | None = None,
    ) -> None:
        self.store = sqlite_store
        self.parquet_store = parquet_store
        self.config = config or PortfolioConfig()
        self.turb_engine = MahalanobisTurbulence(window=252, min_periods=63, shrinkage=0.01)

    def load_macro_dataframe(
        self,
        region_bloc: RegionBloc,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Load raw macro series from SQLite and pivot into a wide daily DataFrame.

        Parameters
        ----------
        region_bloc : RegionBloc
            'NA' or 'EU'.
        start_date : str | None
            Start date string (YYYY-MM-DD).
        end_date : str | None
            End date string (YYYY-MM-DD).

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by date where columns are series_ids.
        """
        raw = self.store.get_macro_series(
            start_date=start_date,
            end_date=end_date,
            region_bloc=region_bloc,
        )
        if raw.empty:
            return pd.DataFrame()

        pivoted = raw.pivot(index="date", columns="series_id", values="value")
        pivoted.index.name = "date"
        pivoted = pivoted.sort_index()
        return pivoted

    def compute_turbulence_for_region(
        self,
        region_bloc: RegionBloc,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """Compute Mahalanobis turbulence for region using historical market returns.

        If ParquetMarketStore is available and returns are present, computes turbulence
        over equal-weighted GICS sectors. Otherwise returns empty Series.
        """
        if self.parquet_store is None:
            return pd.Series(dtype=float)

        market_df = self.parquet_store.read_daily_market(
            start_date=start_date,
            end_date=end_date,
            region_bloc=region_bloc,
        )
        if market_df.empty or "ticker" not in market_df.columns:
            return pd.Series(dtype=float)

        # Retrieve sector metadata
        tickers = market_df["ticker"].unique().tolist()
        meta = self.store.get_security_metadata_cache(tickers)
        ticker_to_sector = {
            t: info["gics_sector"]
            for t, info in meta.items()
            if info["gics_sector"] > 0 and info["gics_sector"] != self.config.financials_gics_sector
        }

        if not ticker_to_sector:
            return pd.Series(dtype=float)

        # Build daily price matrix and compute returns
        px_wide = market_df.pivot(index="date", columns="ticker", values="px_last_eur").sort_index()
        px_wide = px_wide[[c for c in px_wide.columns if c in ticker_to_sector]]
        ret_wide = px_wide.pct_change(fill_method=None).dropna(how="all")

        sec_returns = MahalanobisTurbulence.aggregate_sector_returns(ret_wide, ticker_to_sector)
        if sec_returns.empty:
            return pd.Series(dtype=float)

        turb_series = self.turb_engine.compute_turbulence_series(sec_returns)
        turb_series.name = f"TURBULENCE_{region_bloc}"
        return turb_series

    def build_regime_features(
        self,
        region_bloc: RegionBloc,
        as_of_date: str,
        history_days: int = 504,
        max_ffill_days: int = 3,
    ) -> pd.DataFrame:
        """Build clean, aligned feature panel up to as_of_date for regime modeling.

        Ensures zero look-ahead bias: strictly filters date <= as_of_date.
        """
        start_date = (pd.Timestamp(as_of_date) - pd.DateOffset(days=history_days * 2)).strftime(
            "%Y-%m-%d"
        )
        macro_df = self.load_macro_dataframe(
            region_bloc=region_bloc,
            start_date=start_date,
            end_date=as_of_date,
        )

        expected_series = MACRO_SERIES_NA if region_bloc == "NA" else MACRO_SERIES_EU
        turb_col = f"TURBULENCE_{region_bloc}"

        # If turbulence is not in macro_df, compute dynamically if possible
        if turb_col not in macro_df.columns and self.parquet_store is not None:
            turb_s = self.compute_turbulence_for_region(
                region_bloc=region_bloc,
                start_date=start_date,
                end_date=as_of_date,
            )
            if not turb_s.empty:
                if macro_df.empty:
                    macro_df = pd.DataFrame({turb_col: turb_s})
                else:
                    macro_df = macro_df.join(turb_s, how="outer")

        if macro_df.empty:
            return pd.DataFrame()

        # Strict cutoff at as_of_date
        macro_df = macro_df[macro_df.index <= as_of_date].sort_index()

        # Calendar reindexing and controlled forward fill
        if len(macro_df) > 1:
            all_bdays = pd.bdate_range(start=macro_df.index[0], end=as_of_date).strftime("%Y-%m-%d")
            macro_aligned = macro_df.reindex(all_bdays)
            macro_aligned = macro_aligned.ffill(limit=max_ffill_days)
        else:
            macro_aligned = macro_df

        # Filter down to expected series that are available
        available_cols = [c for c in expected_series if c in macro_aligned.columns]
        if not available_cols:
            return pd.DataFrame()

        clean_panel = macro_aligned[available_cols].dropna()
        if len(clean_panel) > history_days:
            clean_panel = clean_panel.iloc[-history_days:]

        return clean_panel
