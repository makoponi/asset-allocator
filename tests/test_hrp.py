"""Unit tests for Hierarchical Risk Parity (HRP) optimizer."""

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.config import AllocationConstraints, PortfolioConfig
from quant_portfolio.risk.hrp import HierarchicalRiskParity


@pytest.fixture
def synthetic_transatlantic_cov() -> tuple[pd.DataFrame, list[str]]:
    """Construct a synthetic block covariance matrix for 35 NA assets and 25 EU assets."""
    np.random.seed(42)
    n_na = 35
    n_eu = 25
    n_total = n_na + n_eu

    tickers = [f"NA_{i:02d}" for i in range(n_na)] + [f"EU_{i:02d}" for i in range(n_eu)]
    regions = ["NA"] * n_na + ["EU"] * n_eu

    # Random correlation matrix with intra-region block structure
    factors = np.random.normal(0, 0.02, size=(300, 3))
    # Loadings: Factor 0 is global, Factor 1 is NA-specific, Factor 2 is EU-specific
    loadings = np.zeros((n_total, 3))
    loadings[:, 0] = np.random.uniform(0.5, 1.0, size=n_total)
    loadings[:n_na, 1] = np.random.uniform(0.5, 1.2, size=n_na)
    loadings[n_na:, 2] = np.random.uniform(0.5, 1.2, size=n_eu)

    idio = np.random.normal(0, 0.015, size=(300, n_total))
    returns = factors @ loadings.T + idio

    cov = np.cov(returns, rowvar=False)
    cov_df = pd.DataFrame(cov, index=tickers, columns=tickers)
    return cov_df, regions


def test_hrp_distance_and_quasi_diagonalize() -> None:
    """Verify correlation distance properties and leaf ordering."""
    corr = np.array([[1.0, 0.8, 0.1], [0.8, 1.0, 0.2], [0.1, 0.2, 1.0]])
    dist = HierarchicalRiskParity.compute_distance_matrix(corr)
    assert dist.shape == (3, 3)
    assert dist[0, 0] == 0.0
    assert dist[0, 1] < dist[0, 2]  # highly correlated assets are closer


def test_hrp_transatlantic_constraints_enforcement(
    synthetic_transatlantic_cov: tuple[pd.DataFrame, list[str]],
) -> None:
    """Verify that HRP strictly satisfies individual bounds and regional sleeve bands."""
    cov_df, regions = synthetic_transatlantic_cov
    cfg = PortfolioConfig(
        allocation=AllocationConstraints(
            min_stock_weight=0.005,
            max_stock_weight=0.035,
            na_weight_min=0.50,
            na_weight_max=0.65,
            eu_weight_min=0.35,
            eu_weight_max=0.50,
        )
    )
    hrp = HierarchicalRiskParity(config=cfg)
    alloc = hrp.optimize(cov_matrix=cov_df, region_blocs=regions)

    assert not alloc.empty
    weights = alloc["weight"].to_numpy()

    # 1. Budget constraint sum(w) == 1.0
    assert abs(float(np.sum(weights)) - 1.0) < 1e-5

    # 2. Individual stock bounds [0.5%, 3.5%]
    assert np.all(weights >= 0.005 - 1e-6)
    assert np.all(weights <= 0.035 + 1e-6)

    # 3. Sleeve bounds
    na_weight = float(alloc[alloc["region_bloc"] == "NA"]["weight"].sum())
    eu_weight = float(alloc[alloc["region_bloc"] == "EU"]["weight"].sum())

    assert 0.50 - 1e-5 <= na_weight <= 0.65 + 1e-5
    assert 0.35 - 1e-5 <= eu_weight <= 0.50 + 1e-5


def test_hrp_empty_cov_handled() -> None:
    """Verify handling of empty covariance input."""
    hrp = HierarchicalRiskParity()
    res = hrp.optimize(cov_matrix=pd.DataFrame(), region_blocs=[])
    assert res.empty
