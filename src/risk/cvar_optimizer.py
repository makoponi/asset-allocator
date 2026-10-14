"""Conditional Value at Risk (CVaR) Portfolio Optimization using Convex Programming.

Implements the Rockafellar-Uryasev linear programming formulation:
min_{w, gamma, u} gamma + (1 / (T * (1 - alpha))) * sum(u_t)
s.t.
    u_t >= -w^T r_t - gamma
    u_t >= 0
    sum(w_i) == 1
    0 <= w_i <= max_weight
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


class MeanCVaROptimizer:
    """Convex solver for minimizing CVaR (Expected Shortfall) at confidence level alpha."""

    def __init__(
        self,
        confidence_level: float = 0.95,
        max_weight: float = 0.04,
    ) -> None:
        self.confidence_level = confidence_level
        self.max_weight = max_weight

    def optimize(
        self,
        returns_matrix: np.ndarray | pd.DataFrame,
    ) -> np.ndarray:
        """Solve CVaR linear program over historical return scenarios.

        Parameters
        ----------
        returns_matrix : Array or DataFrame of shape (T, N)
            Historical asset returns.

        Returns
        -------
        np.ndarray
            Optimal portfolio weight vector of length N summing to 1.0.
        """
        import cvxpy as cp

        r = np.asarray(returns_matrix, dtype=float)
        t_samples, n_assets = r.shape

        if n_assets == 0:
            return np.array([])
        if n_assets == 1:
            return np.array([1.0])

        # Adjust max_weight if unfeasible (e.g. max_weight * N < 1)
        effective_max_w = max(self.max_weight, 1.0 / n_assets)

        # Decision variables
        w = cp.Variable(n_assets)
        gamma = cp.Variable()  # Value-at-Risk threshold
        u = cp.Variable(t_samples)  # Excess loss scenarios

        alpha = self.confidence_level
        tail_prob = max(1e-4, 1.0 - alpha)

        # Objective: VaR + (1 / ((1 - alpha) * T)) * sum(losses beyond VaR)
        objective = cp.Minimize(gamma + (1.0 / (t_samples * tail_prob)) * cp.sum(u))  # type: ignore[attr-defined]

        # Constraints
        # -r @ w represents portfolio loss in scenario t
        constraints: list[Any] = [
            u >= -r @ w - gamma,
            u >= 0,
            cp.sum(w) == 1.0,  # type: ignore[attr-defined]
            w >= 0,
            w <= effective_max_w,
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()  # type: ignore[no-untyped-call]

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or w.value is None:
            # Fallback to uniform weighting
            return np.asarray(np.ones(n_assets, dtype=float) / n_assets, dtype=float)

        res_w = np.asarray(w.value, dtype=float)
        # Ensure non-negativity and exact sum to 1
        res_w = np.maximum(0.0, res_w)
        return np.asarray(res_w / np.sum(res_w), dtype=float)
