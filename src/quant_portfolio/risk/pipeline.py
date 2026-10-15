"""End-to-End Transatlantic Portfolio Allocation Pipeline.

Coordinates:
1. Candidate ingestion from Phase 4 (StratifiedSleeveSelector)
2. 3-day non-overlapping return synchronization (MultiDaySynchronizer)
3. Marcenko-Pastur spectral covariance denoising (MarcenkoPasturDenoiser)
4. Allocation optimization via HRP, Mean-CVaR, or 50/50 Ensemble
5. Risk metrics calculation (volatility, CVaR, MCTR) and compliance validation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant_portfolio.config import PortfolioConfig, RiskConfig
from quant_portfolio.risk.cvar import MeanCVaROptimizer
from quant_portfolio.risk.denoiser import MarcenkoPasturDenoiser
from quant_portfolio.risk.hrp import HierarchicalRiskParity
from quant_portfolio.risk.synchronizer import MultiDaySynchronizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_portfolio.storage.schema import ParquetMarketStore

logger = logging.getLogger(__name__)

AllocationMethod = Literal["hrp", "cvar", "ensemble"]


class PortfolioRiskSummary(BaseModel):
    """Aggregate risk and regional metrics for an allocated portfolio."""

    as_of_date: str
    method: str
    n_stocks: int
    n_na: int
    n_eu: int
    na_weight: float = Field(ge=0.0, le=1.0)
    eu_weight: float = Field(ge=0.0, le=1.0)
    annualized_volatility: float = Field(ge=0.0)
    cvar_95: float = Field(ge=0.0)


class PortfolioAllocationPipeline:
    """Orchestrates multi-day synchronization, RMT denoising, and constrained allocation."""

    def __init__(
        self,
        parquet_store: ParquetMarketStore,
        config: PortfolioConfig | None = None,
        synchronizer: MultiDaySynchronizer | None = None,
        denoiser: MarcenkoPasturDenoiser | None = None,
        hrp_optimizer: HierarchicalRiskParity | None = None,
        cvar_optimizer: MeanCVaROptimizer | None = None,
    ) -> None:
        self.config = config or PortfolioConfig()
        self.risk_config: RiskConfig = self.config.risk or RiskConfig()
        self.parquet_store = parquet_store

        self.synchronizer = synchronizer or MultiDaySynchronizer(
            parquet_store=self.parquet_store,
            config=self.config,
        )
        self.denoiser = denoiser or MarcenkoPasturDenoiser()
        self.hrp_optimizer = hrp_optimizer or HierarchicalRiskParity(config=self.config)
        self.cvar_optimizer = cvar_optimizer or MeanCVaROptimizer(config=self.config)

    def allocate_candidates(
        self,
        candidates_df: pd.DataFrame,
        as_of_date: str,
        method: AllocationMethod | None = None,
        lookback_days: int | None = None,
    ) -> tuple[pd.DataFrame, PortfolioRiskSummary]:
        """Execute risk modeling and optimal weight allocation on candidate assets."""
        opt_method = method or self.risk_config.default_optimizer
        if candidates_df.empty:
            empty_summary = PortfolioRiskSummary(
                as_of_date=as_of_date,
                method=opt_method,
                n_stocks=0,
                n_na=0,
                n_eu=0,
                na_weight=0.0,
                eu_weight=0.0,
                annualized_volatility=0.0,
                cvar_95=0.0,
            )
            return pd.DataFrame(), empty_summary

        required_cols = {"ticker", "region_bloc"}
        missing = required_cols - set(candidates_df.columns)
        if missing:
            raise ValueError(f"candidates_df missing required columns: {missing}")

        tickers: Sequence[str] = list(candidates_df["ticker"].unique())
        region_map: dict[str, str] = dict(
            zip(candidates_df["ticker"], candidates_df["region_bloc"], strict=False)
        )

        # 1. Multi-Day synchronization (3 days)
        sync_returns = self.synchronizer.extract_synchronized_returns(
            tickers=tickers,
            as_of_date=as_of_date,
            lookback_days=lookback_days or self.risk_config.history_days,
            resampling_days=self.risk_config.resampling_days,
        )

        if sync_returns.empty:
            logger.warning("Could not synchronize prices for date %s", as_of_date)
            empty_summary = PortfolioRiskSummary(
                as_of_date=as_of_date,
                method=opt_method,
                n_stocks=0,
                n_na=0,
                n_eu=0,
                na_weight=0.0,
                eu_weight=0.0,
                annualized_volatility=0.0,
                cvar_95=0.0,
            )
            return pd.DataFrame(), empty_summary

        # Filter candidates to only those with valid synchronized returns
        valid_tickers = list(sync_returns.columns)
        regions = [region_map[t] for t in valid_tickers]

        # 2. Spectral denoising (Marcenko-Pastur)
        denoised_cov, _ = self.denoiser.denoise_covariance(
            returns=sync_returns,
            tickers=valid_tickers,
        )

        # 3. Allocation optimization
        scores: np.ndarray | None = None
        if "score" in candidates_df.columns:
            score_map = dict(zip(candidates_df["ticker"], candidates_df["score"], strict=False))
            scores = np.array([score_map.get(t, 0.0) for t in valid_tickers], dtype=np.float64)

        if opt_method == "hrp":
            alloc_df = self.hrp_optimizer.optimize(
                cov_matrix=denoised_cov,
                region_blocs=regions,
                tickers=valid_tickers,
            )
        elif opt_method == "cvar":
            alloc_df = self.cvar_optimizer.optimize(
                returns=sync_returns,
                region_blocs=regions,
                expected_returns=scores,
                tickers=valid_tickers,
            )
        elif opt_method == "ensemble":
            w_hrp = self.hrp_optimizer.optimize(
                cov_matrix=denoised_cov,
                region_blocs=regions,
                tickers=valid_tickers,
            )["weight"].to_numpy()
            w_cvar = self.cvar_optimizer.optimize(
                returns=sync_returns,
                region_blocs=regions,
                expected_returns=scores,
                tickers=valid_tickers,
            )["weight"].to_numpy()
            w_ens = 0.5 * w_hrp + 0.5 * w_cvar
            # Ensure projected compliance
            w_proj = self.hrp_optimizer.project_transatlantic_constraints(w_ens, regions)
            alloc_df = pd.DataFrame(
                {
                    "ticker": valid_tickers,
                    "region_bloc": regions,
                    "weight": w_proj,
                }
            )
        else:
            raise ValueError(f"Unknown allocation method: {opt_method}")

        # 4. Attach metadata and risk metrics
        weights = alloc_df["weight"].to_numpy()
        cov_np = denoised_cov.to_numpy()

        # 3-day portfolio variance -> annualized
        # Annualization factor for 3-business-day returns: 252 / 3 = 84
        var_3d = float(weights.T @ cov_np @ weights)
        ann_factor = 252.0 / float(self.risk_config.resampling_days)
        ann_vol = float(np.sqrt(max(var_3d, 0.0) * ann_factor))

        # Empirical portfolio 3d returns
        port_returns = sync_returns.to_numpy() @ weights
        pct_q = (1.0 - self.risk_config.cvar_confidence) * 100.0
        cvar_cutoff = float(np.percentile(port_returns, pct_q))
        tail_losses = -port_returns[port_returns <= cvar_cutoff]
        cvar_val = float(np.mean(tail_losses)) if len(tail_losses) > 0 else 0.0

        # Marginal contribution to risk (MCTR)
        port_vol_3d = np.sqrt(max(var_3d, 1e-12))
        mctr = (cov_np @ weights) / port_vol_3d

        alloc_df["as_of_date"] = as_of_date
        alloc_df["mctr"] = mctr
        if scores is not None:
            alloc_df["score"] = scores

        n_na = int((alloc_df["region_bloc"] == "NA").sum())
        n_eu = int((alloc_df["region_bloc"] == "EU").sum())
        na_w = float(alloc_df[alloc_df["region_bloc"] == "NA"]["weight"].sum())
        eu_w = float(alloc_df[alloc_df["region_bloc"] == "EU"]["weight"].sum())

        summary = PortfolioRiskSummary(
            as_of_date=as_of_date,
            method=opt_method,
            n_stocks=len(alloc_df),
            n_na=n_na,
            n_eu=n_eu,
            na_weight=na_w,
            eu_weight=eu_w,
            annualized_volatility=ann_vol,
            cvar_95=cvar_val,
        )

        return alloc_df, summary
