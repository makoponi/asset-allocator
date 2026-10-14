"""Hierarchical Risk Parity (HRP) portfolio optimization engine.

Implements Marcos López de Prado's HRP algorithm:
1. Tree clustering from correlation distance d_{i,j} = sqrt((1 - rho_{i,j}) / 2).
2. Quasi-diagonalization of the covariance matrix.
3. Recursive bisection inverse-variance allocation with max-weight capping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

if TYPE_CHECKING:
    from collections.abc import Sequence


def correlation_to_distance(corr: np.ndarray) -> np.ndarray:
    """Compute topological correlation distance matrix: d_{i,j} = sqrt((1 - rho_{i,j}) / 2)."""
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr)))
    np.fill_diagonal(dist, 0.0)
    return dist


def get_quasi_diagonal(link: np.ndarray) -> list[int]:
    """Quasi-diagonalize the linkage matrix by sorting leaves hierarchically."""
    link_int = link.astype(int)
    num_items = link_int.shape[0] + 1
    root = 2 * num_items - 2

    order: list[int] = [root]
    while any(i >= num_items for i in order):
        new_order: list[int] = []
        for i in order:
            if i >= num_items:
                left = link_int[i - num_items, 0]
                right = link_int[i - num_items, 1]
                new_order.extend([left, right])
            else:
                new_order.append(i)
        order = new_order
    return order


def get_cluster_variance(cov: np.ndarray, cluster_indices: Sequence[int]) -> float:
    """Calculate the variance of a cluster under inverse-variance weighting."""
    idx = list(cluster_indices)
    cov_slice = cov[np.ix_(idx, idx)]
    # Inverse-variance allocation within the sub-cluster
    iv = 1.0 / np.maximum(1e-12, np.diag(cov_slice))
    weights = iv / np.sum(iv)
    val = float(weights @ cov_slice @ weights)
    return max(1e-12, val)


def recursive_bisection(cov: np.ndarray, sorted_indices: Sequence[int]) -> np.ndarray:
    """Compute HRP weights through top-down recursive bisection."""
    weights = np.ones(len(sorted_indices), dtype=float)
    clusters: list[list[int]] = [list(sorted_indices)]

    while len(clusters) > 0:
        new_clusters: list[list[int]] = []
        for c in clusters:
            if len(c) > 1:
                mid = len(c) // 2
                c1 = c[:mid]
                c2 = c[mid:]

                var1 = get_cluster_variance(cov, c1)
                var2 = get_cluster_variance(cov, c2)

                alpha = 1.0 - (var1 / (var1 + var2))

                # Scale weights
                for item in c1:
                    weights[item] *= alpha
                for item in c2:
                    weights[item] *= 1.0 - alpha

                new_clusters.append(c1)
                new_clusters.append(c2)
        clusters = new_clusters

    return weights


def cap_and_redistribute_weights(
    weights: np.ndarray,
    max_weight: float = 0.04,
    max_iter: int = 100,
) -> np.ndarray:
    """Cap weights exceeding max_weight and redistribute excess to remaining assets."""
    w = weights.copy()
    w = w / np.sum(w)
    n = len(w)

    if max_weight * n < 1.0:
        # Feasibility check: if max_weight * N < 1, can only uniform weight
        return np.ones(n, dtype=float) / n

    for _ in range(max_iter):
        excess = np.maximum(0.0, w - max_weight)
        total_excess = float(np.sum(excess))
        if total_excess <= 1e-8:
            break

        uncapped_mask = w < max_weight
        if not np.any(uncapped_mask):
            break

        w = np.minimum(w, max_weight)
        uncapped_sum = float(np.sum(w[uncapped_mask]))
        if uncapped_sum > 0:
            w[uncapped_mask] += total_excess * (w[uncapped_mask] / uncapped_sum)
        else:
            w[uncapped_mask] += total_excess / np.sum(uncapped_mask)

    return np.asarray(w / np.sum(w), dtype=float)


class HierarchicalRiskParity:
    """Hierarchical Risk Parity portfolio allocator."""

    def __init__(
        self,
        linkage_method: str = "single",
        max_weight: float | None = 0.04,
    ) -> None:
        self.linkage_method = linkage_method
        self.max_weight = max_weight

    def optimize(
        self,
        cov: np.ndarray,
    ) -> np.ndarray:
        """Compute optimal risk parity portfolio weights for a given covariance matrix.

        Parameters
        ----------
        cov : np.ndarray
            N x N covariance matrix.

        Returns
        -------
        np.ndarray
            Optimal portfolio weight vector w summing to 1.0.
        """
        n = cov.shape[0]
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])

        # 1. Derive correlation matrix
        stds = np.sqrt(np.maximum(1e-12, np.diag(cov)))
        corr = cov / np.outer(stds, stds)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)

        # 2. Distance and Linkage
        dist = correlation_to_distance(corr)
        # Symmetrize distance
        dist = 0.5 * (dist + dist.T)
        np.fill_diagonal(dist, 0.0)

        compressed_dist = squareform(dist, checks=False)
        link = sch.linkage(compressed_dist, method=self.linkage_method)

        # 3. Quasi-diagonalization
        ordered_indices = get_quasi_diagonal(link)

        # 4. Recursive Bisection
        weights = recursive_bisection(cov, ordered_indices)
        weights = weights / np.sum(weights)

        # 5. Cap and redistribute
        if self.max_weight is not None and self.max_weight > 0:
            weights = cap_and_redistribute_weights(weights, max_weight=self.max_weight)

        return np.asarray(weights, dtype=float)
