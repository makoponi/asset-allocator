"""Convex Mean-CVaR (Conditional Value-at-Risk) Portfolio Optimizer.

Formulates and solves the Rockafellar & Uryasev (2000) linear/convex optimization problem:
    min_{w, gamma, u} gamma + 1 / ((1 - beta) * T) * sum(u_t) - lambda_ret * mu^T w

Subject to:
    u_t >= - r_t^T w - gamma
    u_t >= 0
    sum(w) == 1.0
    min_stock_weight <= w_i <= max_stock_weight
    na_weight_min <= sum_{i in NA} w_i <= na_weight_max
    eu_weight_min <= sum_{j in EU} w_j <= eu_weight_max
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np
import pandas as pd

from quant_portfolio.config import AllocationConstraints, PortfolioConfig, RiskConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class MeanCVaROptimizer:
    """Convex Conditional Value-at-Risk (CVaR) optimizer under transatlantic constraints."""

    def __init__(
        self,
        config: PortfolioConfig | None = None,
        beta: float | None = None,
        lambda_ret: float | None = None,
    ) -> None:
        self.config = config or PortfolioConfig()
        self.constraints: AllocationConstraints = self.config.allocation or AllocationConstraints()
        self.risk_config: RiskConfig = self.config.risk or RiskConfig()
        self.beta = beta or self.risk_config.cvar_confidence
        self.lambda_ret = (
            lambda_ret if lambda_ret is not None else self.risk_config.cvar_lambda_return
        )

    def optimize(
        self,
        returns: pd.DataFrame | np.ndarray,
        region_blocs: Sequence[str],
        expected_returns: pd.Series | np.ndarray | None = None,
        tickers: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Solve convex CVaR minimization problem.

        Args:
            returns: T x N matrix of synchronized returns.
            region_blocs: List/array of length N indicating 'NA' or 'EU'.
            expected_returns: Optional expected return or ML alpha score vector.
            tickers: Optional asset identifiers.

        Returns:
            DataFrame with columns ['ticker', 'region_bloc', 'weight'].
        """
        if isinstance(returns, pd.DataFrame):
            ticker_list = list(returns.columns)
            r_mat = returns.to_numpy(dtype=np.float64)
        else:
            r_mat = np.asarray(returns, dtype=np.float64)
            n_assets = r_mat.shape[1]
            ticker_list = (
                list(tickers) if tickers is not None else [f"A{i}" for i in range(n_assets)]
            )

        t_samples, n_assets = r_mat.shape
        if n_assets == 0 or t_samples < 2:
            return pd.DataFrame(columns=["ticker", "region_bloc", "weight"])

        if len(region_blocs) != n_assets:
            raise ValueError(
                f"Mismatch: {n_assets} tickers but {len(region_blocs)} region_bloc entries"
            )

        # Variables
        w = cp.Variable(n_assets)
        gamma = cp.Variable()  # VaR threshold
        u = cp.Variable(t_samples)  # tail loss exceedances

        min_w = self.constraints.min_stock_weight
        max_w = self.constraints.max_stock_weight

        na_mask = np.array([r == "NA" for r in region_blocs], dtype=bool)
        eu_mask = np.array([r == "EU" for r in region_blocs], dtype=bool)

        # Portfolio loss per time step: L_t = - r_t^T w
        # Constraints: u_t >= - r_t @ w - gamma, u_t >= 0
        constraints = [
            u >= -r_mat @ w - gamma,
            u >= 0.0,
            cp.sum(w) == 1.0,  # type: ignore[attr-defined]
            w >= min_w,
            w <= max_w,
        ]

        if np.any(na_mask):
            constraints.extend(
                [
                    cp.sum(w[na_mask]) >= self.constraints.na_weight_min,  # type: ignore[attr-defined]
                    cp.sum(w[na_mask]) <= self.constraints.na_weight_max,  # type: ignore[attr-defined]
                ]
            )

        if np.any(eu_mask):
            constraints.extend(
                [
                    cp.sum(w[eu_mask]) >= self.constraints.eu_weight_min,  # type: ignore[attr-defined]
                    cp.sum(w[eu_mask]) <= self.constraints.eu_weight_max,  # type: ignore[attr-defined]
                ]
            )

        # Objective: CVaR_beta = gamma + (1 / ((1 - beta) * T)) * sum(u_t)
        cvar_term = gamma + (1.0 / ((1.0 - self.beta) * float(t_samples))) * cp.sum(u)  # type: ignore[attr-defined]

        if expected_returns is not None and self.lambda_ret > 0.0:
            if isinstance(expected_returns, pd.Series):
                mu = expected_returns.to_numpy(dtype=np.float64)
            else:
                mu = np.asarray(expected_returns, dtype=np.float64)
            objective = cp.Minimize(cvar_term - self.lambda_ret * (mu @ w))
        else:
            objective = cp.Minimize(cvar_term)

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.CLARABEL)  # type: ignore[no-untyped-call]
        except Exception:
            try:
                prob.solve()  # type: ignore[no-untyped-call]
            except Exception as e:
                logger.error("CVaR solver failed: %s", e)

        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            opt_w = np.asarray(w.value, dtype=np.float64).flatten()
            opt_w = np.clip(opt_w, min_w, max_w)
            opt_w = opt_w / np.sum(opt_w)
        else:
            logger.warning(
                "CVaR optimization status %s; falling back to regional equal weighting",
                prob.status,
            )
            opt_w = self._equal_weight_fallback(n_assets, na_mask, eu_mask)

        out_df = pd.DataFrame(
            {
                "ticker": ticker_list,
                "region_bloc": list(region_blocs),
                "weight": opt_w,
            }
        )
        return out_df

    def _equal_weight_fallback(
        self,
        n_assets: int,
        na_mask: np.ndarray,
        eu_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute fallback compliant weights split between NA (55%) and EU (45%)."""
        w = np.zeros(n_assets, dtype=np.float64)
        n_na = int(np.sum(na_mask))
        n_eu = int(np.sum(eu_mask))

        if n_na > 0:
            w[na_mask] = 0.55 / float(n_na)
        if n_eu > 0:
            w[eu_mask] = 0.45 / float(n_eu)

        w = np.clip(w, self.constraints.min_stock_weight, self.constraints.max_stock_weight)
        return np.asarray(w / np.sum(w), dtype=np.float64)
