"""Unit tests for Hierarchical Risk Parity (HRP) portfolio optimization."""

from __future__ import annotations

import numpy as np
import pytest

from src.risk.hrp import (
    HierarchicalRiskParity,
    cap_and_redistribute_weights,
    correlation_to_distance,
)


def test_correlation_to_distance() -> None:
    corr = np.array(
        [
            [1.0, 0.8, -0.5],
            [0.8, 1.0, 0.0],
            [-0.5, 0.0, 1.0],
        ]
    )
    dist = correlation_to_distance(corr)

    # Diagonal must be 0
    assert np.allclose(np.diag(dist), 0.0)
    # Higher correlation -> lower distance
    # d(0, 1) with rho=0.8 is sqrt((1 - 0.8)/2) = sqrt(0.1) approx 0.316
    assert dist[0, 1] == pytest.approx(np.sqrt(0.1), abs=1e-4)
    # d(0, 2) with rho=-0.5 is sqrt((1 - (-0.5))/2) = sqrt(0.75) approx 0.866
    assert dist[0, 2] == pytest.approx(np.sqrt(0.75), abs=1e-4)


def test_hrp_inverse_variance_property() -> None:
    # 2 independent assets with different volatilities
    # Asset 0: sigma = 0.10, Asset 1: sigma = 0.20
    cov = np.diag([0.10**2, 0.20**2])

    hrp = HierarchicalRiskParity(max_weight=None)
    weights = hrp.optimize(cov)

    assert len(weights) == 2
    assert np.sum(weights) == pytest.approx(1.0)
    # Low-volatility asset must receive higher weight than high-volatility asset
    assert weights[0] > weights[1]
    # In 2-asset case with zero correlation, w0 / w1 = (1/sigma0^2) / (1/sigma1^2) = 4.0
    assert (weights[0] / weights[1]) == pytest.approx(4.0, abs=1e-2)


def test_cap_and_redistribute_weights() -> None:
    # 50 assets: initial weights with one dominating asset (0.20)
    n = 50
    weights = np.ones(n) * (0.80 / 49)
    weights[0] = 0.20

    max_w = 0.04
    capped = cap_and_redistribute_weights(weights, max_weight=max_w)

    assert len(capped) == n
    assert np.sum(capped) == pytest.approx(1.0)
    assert np.all(capped <= max_w + 1e-6)
    assert np.all(capped >= 0.0)


def test_hrp_optimization_with_max_weight_cap() -> None:
    # 40 assets
    n = 40
    rng = np.random.default_rng(42)
    fake_cov = rng.standard_normal((n, n))
    cov = fake_cov @ fake_cov.T + np.eye(n) * 0.1

    hrp = HierarchicalRiskParity(max_weight=0.04)
    w = hrp.optimize(cov)

    assert len(w) == n
    assert np.sum(w) == pytest.approx(1.0)
    # Feasibility: 40 * 0.04 = 1.6 > 1.0, so cap of 0.04 is strictly respected
    assert np.all(w <= 0.04 + 1e-6)
    assert np.all(w >= 0.0)
