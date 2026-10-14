"""Portfolio Allocator orchestrator combining RMT Denoising and Risk Allocation (HRP / CVaR).

Integrates:
1. Marcenko-Pastur spectral covariance denoising.
2. Hierarchical Risk Parity (HRP) or Mean-CVaR optimization.
3. Issuer weight constraints (default max 4% per single stock).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from src.risk.cvar_optimizer import MeanCVaROptimizer
from src.risk.hrp import HierarchicalRiskParity
from src.risk.rmt_denoiser import denoise_covariance

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


class PortfolioAllocator:
    """End-to-end portfolio construction engine for selected candidates."""

    def __init__(
        self,
        method: Literal["hrp", "cvar"] = "hrp",
        denoise: bool = True,
        max_weight: float = 0.04,
        confidence_level: float = 0.95,
    ) -> None:
        self.method = method
        self.denoise = denoise
        self.max_weight = max_weight
        self.confidence_level = confidence_level

    def allocate(
        self,
        returns_df: pd.DataFrame,
        candidate_tickers: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Compute optimal target portfolio weights for selected candidate tickers.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Daily returns with columns as tickers and index as dates.
        candidate_tickers : Sequence[str], optional
            Subset of tickers to allocate (e.g. top decile from Phase 4).

        Returns
        -------
        dict[str, float]
            Mapping from ticker symbol to optimal target weight w_i.
        """
        if candidate_tickers is not None:
            available = [t for t in candidate_tickers if t in returns_df.columns]
            sub_df = returns_df[available]
        else:
            sub_df = returns_df

        tickers = list(sub_df.columns)
        n = len(tickers)
        if n == 0:
            return {}
        if n == 1:
            return {tickers[0]: 1.0}

        r_matrix = sub_df.to_numpy(dtype=float)

        if self.method == "hrp":
            if self.denoise and r_matrix.shape[0] > r_matrix.shape[1]:
                cov = denoise_covariance(r_matrix)
            else:
                cov = np.cov(r_matrix, rowvar=False)

            hrp = HierarchicalRiskParity(max_weight=self.max_weight)
            weights = hrp.optimize(cov)
        else:
            cvar = MeanCVaROptimizer(
                confidence_level=self.confidence_level,
                max_weight=self.max_weight,
            )
            weights = cvar.optimize(r_matrix)

        result: dict[str, float] = {}
        for t, w in zip(tickers, weights, strict=True):
            if w > 1e-6:
                result[t] = float(w)

        # Normalize to exact sum of 1.0
        total_w = sum(result.values())
        if total_w > 0:
            return {k: v / total_w for k, v in result.items()}
        return {t: 1.0 / n for t in tickers}
