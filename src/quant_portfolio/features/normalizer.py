"""Transatlantic cross-sectional feature normalizer.

Applies:
1. Winsorization at 1% and 99% quantiles within each region_bloc.
2. Z-score standardization within each (region_bloc, gics_sector) subgroup:
   z_{i,t} = (x_{i,t} - mu_{b,s,t}) / sigma_{b,s,t}
3. Residual NaN imputation to 0.0 (neutral sector/region benchmark).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quant_portfolio.config import FeatureConfig


class TransatlanticCrossSectionalNormalizer:
    """Normalizes features cross-sectionally neutralizing regional and GICS sector biases."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        if config is None:
            from quant_portfolio.config import FeatureConfig as FC

            self.config = FC()
        else:
            self.config = config

    def winsorize_by_region(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        region_col: str = "region_bloc",
    ) -> pd.DataFrame:
        """Winsorize each feature column at lower and upper quantiles per region_bloc."""
        if not isinstance(df, pd.DataFrame) or df.empty or not feature_cols:
            return df.copy()

        out = df.copy()
        q_low = self.config.winsor_lower_quantile
        q_high = self.config.winsor_upper_quantile

        for _, group_idx in out.groupby(region_col).groups.items():
            for col in feature_cols:
                vals = out.loc[group_idx, col]
                valid_vals = vals.dropna()
                if len(valid_vals) >= 5:
                    low_val = float(np.quantile(valid_vals, q_low))
                    high_val = float(np.quantile(valid_vals, q_high))
                    out.loc[group_idx, col] = np.clip(vals, low_val, high_val)

        return out

    def zscore_by_region_and_sector(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        group_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Standardize features into Z-scores within each (region_bloc, gics_sector) group.

        If a subgroup has fewer than min_observations or zero variance, z-score defaults to 0.0.
        """
        if df.empty or not feature_cols:
            return df.copy()

        cols = group_cols or ["region_bloc", "gics_sector"]
        out = df.copy()
        min_obs = self.config.min_observations_for_zscore

        # Ensure feature columns are float64
        for col in feature_cols:
            out[col] = out[col].astype(np.float64)

        for _, group_idx in out.groupby(cols).groups.items():
            for col in feature_cols:
                series = out.loc[group_idx, col]
                valid = series.dropna()

                if len(valid) < min_obs:
                    out.loc[group_idx, col] = 0.0
                    continue

                mu = float(valid.mean())
                sigma = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0

                if sigma < 1e-8 or np.isnan(sigma):
                    out.loc[group_idx, col] = 0.0
                else:
                    z = (series - mu) / sigma
                    out.loc[group_idx, col] = z

        return out

    def impute_neutral_residuals(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> pd.DataFrame:
        """Impute any residual NaN values to 0.0 (neutral sector/region relative value)."""
        out = df.copy()
        for col in feature_cols:
            out[col] = out[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        return out

    def normalize(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        group_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run complete normalization pipeline: Winsorize -> Z-score -> Impute NaNs to 0.0."""
        if df.empty:
            return df.copy()

        # Step 1: Regional Winsorization
        winsorized = self.winsorize_by_region(df, feature_cols=feature_cols)

        # Step 2: (Region, Sector) Z-Score standardization
        standardized = self.zscore_by_region_and_sector(
            winsorized, feature_cols=feature_cols, group_cols=group_cols
        )

        # Step 3: Impute residual NaNs to 0.0
        final_df = self.impute_neutral_residuals(standardized, feature_cols=feature_cols)

        return final_df
