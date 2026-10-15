"""Hierarchical Risk Parity (HRP) Portfolio Optimizer with Transatlantic Constraints.

Implements the López de Prado (2016) HRP formulation:
1. Correlation-based distance matrix: d_ij = sqrt(0.5 * (1 - rho_ij))
2. Hierarchical clustering using Ward linkage
3. Quasi-diagonalization via dendrogram leaf ordering
4. Recursive bisection based on inverse cluster variance
5. Convex projection to enforce individual stock bounds [0.5%, 3.5%] and
   regional sleeve bands ([50%, 65%] NA, [35%, 50%] EU).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from quant_portfolio.config import AllocationConstraints, PortfolioConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class HierarchicalRiskParity:
    """Hierarchical Risk Parity optimizer with Ward linkage and constraint projection."""

    def __init__(
        self,
        config: PortfolioConfig | None = None,
        linkage_method: str = "ward",
    ) -> None:
        self.config = config or PortfolioConfig()
        self.constraints: AllocationConstraints = self.config.allocation or AllocationConstraints()
        self.linkage_method = linkage_method

    @staticmethod
    def compute_distance_matrix(corr: np.ndarray) -> np.ndarray:
        """Compute the López de Prado distance matrix: d_ij = sqrt(0.5 * (1 - rho_ij))."""
        clipped_corr = np.clip(corr, -1.0, 1.0)
        dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - clipped_corr)))
        np.fill_diagonal(dist, 0.0)
        return np.asarray(dist, dtype=np.float64)

    def quasi_diagonalize(self, link: np.ndarray) -> list[int]:
        """Return the sorted asset indices following the dendrogram leaves order."""
        ordered = leaves_list(link)
        return [int(x) for x in ordered]

    @staticmethod
    def _get_cluster_variance(cov: np.ndarray, cluster_indices: list[int]) -> float:
        """Compute the variance of a cluster under inverse-variance asset weighting."""
        sub_cov = cov[np.ix_(cluster_indices, cluster_indices)]
        diag_vars = np.diag(sub_cov)
        inv_vars = 1.0 / np.maximum(diag_vars, 1e-12)
        inv_weights = inv_vars / np.sum(inv_vars)
        cluster_var = float(inv_weights.T @ sub_cov @ inv_weights)
        return max(cluster_var, 1e-12)

    def recursive_bisection(self, cov: np.ndarray, ordered_indices: list[int]) -> np.ndarray:
        """Perform recursive bisection to compute unconstrained HRP weights."""
        n_assets = len(ordered_indices)
        weights = pd.Series(1.0, index=ordered_indices)
        clusters = [ordered_indices]

        while len(clusters) > 0:
            next_clusters: list[list[int]] = []
            for cluster in clusters:
                if len(cluster) > 1:
                    mid = len(cluster) // 2
                    c1 = cluster[:mid]
                    c2 = cluster[mid:]

                    var1 = self._get_cluster_variance(cov, c1)
                    var2 = self._get_cluster_variance(cov, c2)

                    alpha = 1.0 - var1 / (var1 + var2)
                    weights.loc[c1] *= alpha
                    weights.loc[c2] *= 1.0 - alpha

                    if len(c1) > 1:
                        next_clusters.append(c1)
                    if len(c2) > 1:
                        next_clusters.append(c2)

            clusters = next_clusters

        # Return weights aligned to original 0..n_assets-1 ordering
        aligned_w = np.zeros(n_assets, dtype=np.float64)
        for idx_val in ordered_indices:
            aligned_w[idx_val] = float(weights.loc[idx_val])
        return aligned_w

    def project_transatlantic_constraints(
        self,
        raw_weights: np.ndarray,
        region_blocs: Sequence[str],
    ) -> np.ndarray:
        """Project raw weights onto the transatlantic constraint polytope via quadratic programming.

        Minimizes ||w - w_raw||^2 subject to:
        - sum(w) == 1.0
        - min_stock_weight <= w_i <= max_stock_weight
        - na_weight_min <= sum_{i in NA} w_i <= na_weight_max
        - eu_weight_min <= sum_{j in EU} w_j <= eu_weight_max
        """
        n = len(raw_weights)
        w = cp.Variable(n)

        min_w = self.constraints.min_stock_weight
        max_w = self.constraints.max_stock_weight

        # Check total capacity feasibility
        if n * max_w < 1.0 or n * min_w > 1.0:
            logger.warning(
                "Capacity constraints conflict with N=%d stocks (min=%.3f, max=%.3f)",
                n,
                min_w,
                max_w,
            )

        na_mask = np.array([r == "NA" for r in region_blocs], dtype=bool)
        eu_mask = np.array([r == "EU" for r in region_blocs], dtype=bool)

        constraints = [
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

        # Objective: minimize squared Euclidean distance to unconstrained HRP weights
        objective = cp.Minimize(0.5 * cp.sum_squares(w - raw_weights))  # type: ignore[attr-defined]
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.CLARABEL)  # type: ignore[no-untyped-call]
        except Exception:
            try:
                prob.solve()  # type: ignore[no-untyped-call]
            except Exception as e:
                logger.error("Projection failed: %s. Falling back to normalized clipping.", e)

        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            proj_w = np.asarray(w.value, dtype=np.float64).flatten()
            # Clean tiny numeric precision errors
            proj_w = np.clip(proj_w, min_w, max_w)
            proj_w = proj_w / np.sum(proj_w)
            return np.asarray(proj_w, dtype=np.float64)

        # Fallback: iterative clip and rescale
        logger.warning("Solver status %s; using clip and rescale fallback", prob.status)
        clipped = np.clip(raw_weights, min_w, max_w)
        return np.asarray(clipped / np.sum(clipped), dtype=np.float64)

    def optimize(
        self,
        cov_matrix: pd.DataFrame | np.ndarray,
        region_blocs: Sequence[str],
        tickers: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Execute end-to-end HRP optimization with constraint projection."""
        if isinstance(cov_matrix, pd.DataFrame):
            ticker_list = list(cov_matrix.columns)
            cov = cov_matrix.to_numpy(dtype=np.float64)
        else:
            cov = np.asarray(cov_matrix, dtype=np.float64)
            n_assets = cov.shape[0]
            ticker_list = (
                list(tickers) if tickers is not None else [f"A{i}" for i in range(n_assets)]
            )

        n_assets = len(ticker_list)
        if n_assets == 0:
            return pd.DataFrame(columns=["ticker", "region_bloc", "weight"])

        if len(region_blocs) != n_assets:
            raise ValueError(
                f"Mismatch: {n_assets} tickers but {len(region_blocs)} region_bloc entries"
            )

        # 1. Standard deviations and correlation matrix
        stds = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        inv_stds = 1.0 / stds
        corr = np.diag(inv_stds) @ cov @ np.diag(inv_stds)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

        # 2. Distance matrix & Ward linkage
        dist = self.compute_distance_matrix(corr)
        condensed_dist = squareform(dist, checks=False)
        link = linkage(condensed_dist, method=self.linkage_method)

        # 3. Quasi-diagonalization
        ordered_idx = self.quasi_diagonalize(link)

        # 4. Recursive bisection
        raw_w = self.recursive_bisection(cov, ordered_idx)

        # 5. Constrained projection
        final_w = self.project_transatlantic_constraints(raw_w, region_blocs)

        out_df = pd.DataFrame(
            {
                "ticker": ticker_list,
                "region_bloc": list(region_blocs),
                "weight": final_w,
            }
        )
        return out_df
