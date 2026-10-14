"""Cross-sectional feature engineering and target generation engine.

Implements:
1. GICS sector-relative Z-score normalization with symmetric 3-sigma winsorization
   and robust sector-median imputation for missing fundamental metrics.
2. 21-day forward excess return target calculation against the active cross-sectional median.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from src.features.pit_aligner import AlignedFeatureRow
    from src.ingestion.storage import PriceEODRecord

# Default 14 quantitative features (4 technical + 10 fundamental)
CORE_FEATURE_NAMES: list[str] = [
    "mom_1m",
    "mom_12m",
    "beta_252d",
    "ivol_252d",
    "pe_ratio",
    "best_pe_ratio",
    "px_to_book",
    "fcf_yield",
    "roe",
    "prof_margin",
    "oper_margin",
    "debt_to_equity",
    "quick_ratio",
    "free_cash_flow",
]


class CrossSectionalTransformer:
    """Transforms raw point-in-time features into GICS-neutral, winsorized Z-scores."""

    def __init__(
        self,
        feature_cols: Sequence[str] | None = None,
        clip_sigma: float = 3.0,
    ) -> None:
        self.feature_cols = list(feature_cols or CORE_FEATURE_NAMES)
        self.clip_sigma = clip_sigma

    def rows_to_dataframe(self, rows: Sequence[AlignedFeatureRow]) -> pd.DataFrame:
        """Convert a sequence of AlignedFeatureRow dataclasses into a structured DataFrame."""
        if not rows:
            cols = ["date", "ticker", "gics_sector", *self.feature_cols]
            return pd.DataFrame(columns=cols)

        data: list[dict[str, Any]] = []
        for r in rows:
            item: dict[str, Any] = {
                "date": r.date,
                "ticker": r.ticker,
                "gics_sector": r.gics_sector,
                "mom_1m": r.mom_1m,
                "mom_12m": r.mom_12m,
                "beta_252d": r.beta_252d,
                "ivol_252d": r.ivol_252d,
                "pe_ratio": r.pe_ratio,
                "best_pe_ratio": r.best_pe_ratio,
                "px_to_book": r.px_to_book,
                "fcf_yield": r.fcf_yield,
                "roe": r.roe,
                "prof_margin": r.prof_margin,
                "oper_margin": r.oper_margin,
                "debt_to_equity": r.debt_to_equity,
                "quick_ratio": r.quick_ratio,
                "free_cash_flow": r.free_cash_flow,
            }
            data.append(item)

        df = pd.DataFrame(data)
        for col in self.feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sector-relative cross-sectional normalization.

        Steps:
        1. Group by (date, gics_sector).
        2. Impute missing values with sector median (fallback: date global median, then 0.0).
        3. Standardize: z = (x - mean) / std.
        4. Winsorize: clip to [-clip_sigma, +clip_sigma].
        """
        if df.empty:
            return df.copy()

        out_df = df.copy()
        active_cols = [c for c in self.feature_cols if c in out_df.columns]

        # Process each date cross-section independently to guarantee zero time leakage
        for _d_val, d_group in out_df.groupby("date"):
            date_indices = d_group.index

            for col in active_cols:
                series = d_group[col].copy()

                # 1. Sector-median imputation
                sector_medians = d_group.groupby("gics_sector")[col].transform("median")
                series = series.fillna(sector_medians)

                # Fallback to cross-sectional global median for empty sectors
                global_med = series.median()
                if pd.isna(global_med):
                    global_med = 0.0
                series = series.fillna(global_med)

                # 2. Sector-relative Z-score: z = (x - mu_sector) / sigma_sector
                sector_means = d_group.groupby("gics_sector")[col].transform("mean")
                sector_stds = d_group.groupby("gics_sector")[col].transform("std")

                # If std is zero or NaN (single stock in sector), neutral Z-score is 0.0
                denom = sector_stds.replace(0.0, np.nan)
                z_score = (series - sector_means) / denom
                z_score = z_score.fillna(0.0)

                # 3. Symmetric winsorization
                z_score = z_score.clip(lower=-self.clip_sigma, upper=self.clip_sigma)

                out_df.loc[date_indices, col] = z_score

        # Enforce float64 dtype on all active numerical features
        for col in active_cols:
            out_df[col] = pd.to_numeric(out_df[col], errors="coerce").fillna(0.0).astype(float)

        return out_df


class TargetGenerator:
    """Computes forward excess return target vectors for cross-sectional ranking."""

    def __init__(self, forward_bars: int = 21) -> None:
        self.forward_bars = forward_bars

    def compute_forward_excess_returns(
        self,
        price_records: Sequence[PriceEODRecord],
        eval_dates: Sequence[date],
    ) -> dict[tuple[str, str], float]:
        """Compute relative forward return against active median for each (date, ticker).

        Target: y_{i,t} = R_{i, t -> t+21} - median_{j in U_t}(R_{j, t -> t+21})
        """
        if not price_records or not eval_dates:
            return {}

        # Group prices by ticker sorted chronologically
        px_by_ticker: dict[str, list[tuple[date, float]]] = {}
        for r in price_records:
            px_by_ticker.setdefault(r.ticker, []).append((r.date, r.px_last))

        for records in px_by_ticker.values():
            records.sort(key=lambda x: x[0])

        targets: dict[tuple[str, str], float] = {}

        for eval_d in eval_dates:
            eval_str = eval_d.isoformat()
            raw_returns: dict[str, float] = {}

            for ticker, bars in px_by_ticker.items():
                # Find index of evaluation date
                d_list = [b[0] for b in bars]
                # Look for exact match or latest bar on or before eval_d
                past_indices = [idx for idx, d in enumerate(d_list) if d <= eval_d]
                if not past_indices:
                    continue

                t_idx = past_indices[-1]
                t_date, p_t = bars[t_idx]

                # Check that t_date is within 5 calendar days of eval_d
                if (eval_d - t_date).days > 5:
                    continue

                # Target price at forward_bars ahead
                target_idx = t_idx + self.forward_bars
                if target_idx < len(bars):
                    _fwd_date, p_fwd = bars[target_idx]
                    if p_t > 0:
                        raw_returns[ticker] = (p_fwd / p_t) - 1.0

            if not raw_returns:
                continue

            # Calculate cross-sectional active median return
            all_rets = list(raw_returns.values())
            median_ret = float(np.median(all_rets))

            # Store excess returns
            for ticker, ret in raw_returns.items():
                targets[(eval_str, ticker)] = ret - median_ret

        return targets
