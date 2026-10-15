"""Stratified Top-Decile Sleeve Selection for transatlantic portfolio candidates.

Extracts top-ranked stocks per regional sleeve:
- 30-35 stocks for North America (NA)
- 20-25 stocks for Europe (EU)
Total: 50-60 candidate assets for covariance estimation and portfolio optimization.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from quant_portfolio.config import AllocationConstraints, PortfolioConfig

logger = logging.getLogger(__name__)


class StratifiedSleeveSelector:
    """Selects top-ranked assets separately for NA and EU sleeves to prevent regional bias."""

    def __init__(self, config: PortfolioConfig | None = None) -> None:
        self.config = config or PortfolioConfig()
        self.allocation_config: AllocationConstraints = (
            self.config.allocation or AllocationConstraints()
        )
        self.financials_sector: int = self.config.financials_gics_sector

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate required columns and enforce exclusion of Financials (GICS 40)."""
        required_cols = {"ticker", "region_bloc", "score"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in scored universe: {missing}")

        # Enforce strict financial exclusion rule
        if "gics_sector" in df.columns:
            fin_count = (df["gics_sector"] == self.financials_sector).sum()
            if fin_count > 0:
                raise ValueError(
                    f"Found {fin_count} Financials (GICS {self.financials_sector}) stocks. "
                    f"Financials must be excluded from investible universe."
                )

    def rank_full_universe(
        self,
        scored_universe: pd.DataFrame,
        top_na: int | None = None,
        top_eu: int | None = None,
    ) -> pd.DataFrame:
        """Rank all assets within their regional sleeve and tag top selections."""
        if scored_universe.empty:
            cols = list(scored_universe.columns) + [
                "regional_rank",
                "regional_decile",
                "is_selected",
            ]
            return pd.DataFrame(columns=cols)

        self._validate_input(scored_universe)
        df = scored_universe.copy()

        k_na = top_na or self.allocation_config.na_top_stocks
        k_eu = top_eu or self.allocation_config.eu_top_stocks

        limits: dict[str, int] = {"NA": k_na, "EU": k_eu}
        ranked_frames: list[pd.DataFrame] = []

        for reg in sorted(df["region_bloc"].unique()):
            reg_df = df[df["region_bloc"] == reg].copy()
            # Sort by score descending; break ties deterministically by ticker
            reg_df = reg_df.sort_values(
                by=["score", "ticker"],
                ascending=[False, True],
            ).reset_index(drop=True)

            n_stocks = len(reg_df)
            reg_df["regional_rank"] = np.arange(1, n_stocks + 1, dtype=np.int64)

            # Compute deciles: 1 = top 10%, 10 = bottom 10%
            if n_stocks >= 10:
                # Use qcut with labels 1..10 (1 is highest score)
                ranks = np.arange(n_stocks)
                reg_df["regional_decile"] = (ranks * 10 // n_stocks + 1).astype(np.int64)
            else:
                reg_df["regional_decile"] = 1

            k_limit = limits.get(str(reg), 30)
            reg_df["is_selected"] = reg_df["regional_rank"] <= k_limit

            ranked_frames.append(reg_df)

        combined = pd.concat(ranked_frames, ignore_index=True)
        return combined

    def select_candidates(
        self,
        scored_universe: pd.DataFrame,
        top_na: int | None = None,
        top_eu: int | None = None,
    ) -> pd.DataFrame:
        """Filter universe to only the top selected candidate assets across both sleeves."""
        ranked = self.rank_full_universe(
            scored_universe=scored_universe,
            top_na=top_na,
            top_eu=top_eu,
        )
        selected = ranked[ranked["is_selected"]].reset_index(drop=True)

        n_na = (selected["region_bloc"] == "NA").sum()
        n_eu = (selected["region_bloc"] == "EU").sum()
        logger.info(
            "Selected %d candidates: %d NA stocks, %d EU stocks",
            len(selected),
            n_na,
            n_eu,
        )
        return selected
