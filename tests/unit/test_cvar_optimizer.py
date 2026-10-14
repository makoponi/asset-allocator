"""Unit tests for Mean-CVaR convex optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from src.risk.cvar_optimizer import MeanCVaROptimizer


def test_cvar_optimizer_budget_and_constraints() -> None:
    t_samples = 250
    n_assets = 30
    rng = np.random.default_rng(99)

    # Asset 0 has high negative mean return and high volatility
    # Asset 1-29 have normal behavior
    returns = rng.standard_normal((t_samples, n_assets)) * 0.01
    returns[:, 0] -= 0.05  # Severe losses

    optimizer = MeanCVaROptimizer(confidence_level=0.95, max_weight=0.04)
    w = optimizer.optimize(returns)

    assert len(w) == n_assets
    assert np.sum(w) == pytest.approx(1.0, abs=1e-4)
    assert np.all(w >= -1e-6)
    # Effective max weight constraint (with 30 assets, 1/30 = 0.0333 <= 0.04)
    assert np.all(w <= 0.04 + 1e-4)

    # Toxic asset 0 should have lowest weight or close to 0
    assert w[0] < np.mean(w[1:])
