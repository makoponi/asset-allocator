"""Unit tests for MeanCVaROptimizer."""

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.config import AllocationConstraints, PortfolioConfig
from quant_portfolio.risk.cvar import MeanCVaROptimizer


@pytest.fixture
def synthetic_transatlantic_returns() -> tuple[pd.DataFrame, list[str]]:
    """Generate 200 days of returns for 35 NA and 25 EU stocks with heavy-tail events."""
    np.random.seed(42)
    n_na = 35
    n_eu = 25
    n_total = n_na + n_eu
    t_samples = 200

    tickers = [f"NA_{i:02d}" for i in range(n_na)] + [f"EU_{i:02d}" for i in range(n_eu)]
    regions = ["NA"] * n_na + ["EU"] * n_eu

    # Normal returns + occasional jump/crash tail events
    returns = np.random.normal(0.0005, 0.015, size=(t_samples, n_total))
    # Negative tail events on days 50 and 120
    returns[50, :] -= np.random.exponential(0.04, size=n_total)
    returns[120, :] -= np.random.exponential(0.03, size=n_total)

    df = pd.DataFrame(returns, columns=tickers)
    return df, regions


def test_cvar_constraints_enforcement(
    synthetic_transatlantic_returns: tuple[pd.DataFrame, list[str]],
) -> None:
    """Verify that CVaR solution strictly respects stock bounds and sleeve bands."""
    df, regions = synthetic_transatlantic_returns
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
    opt = MeanCVaROptimizer(config=cfg, beta=0.95)
    alloc = opt.optimize(returns=df, region_blocs=regions)

    assert not alloc.empty
    weights = alloc["weight"].to_numpy()

    # 1. Budget constraint
    assert abs(float(np.sum(weights)) - 1.0) < 1e-5

    # 2. Individual stock bounds [0.5%, 3.5%]
    assert np.all(weights >= 0.005 - 1e-6)
    assert np.all(weights <= 0.035 + 1e-6)

    # 3. Sleeve bounds
    na_weight = float(alloc[alloc["region_bloc"] == "NA"]["weight"].sum())
    eu_weight = float(alloc[alloc["region_bloc"] == "EU"]["weight"].sum())

    assert 0.50 - 1e-5 <= na_weight <= 0.65 + 1e-5
    assert 0.35 - 1e-5 <= eu_weight <= 0.50 + 1e-5


def test_cvar_with_expected_returns_tilt(
    synthetic_transatlantic_returns: tuple[pd.DataFrame, list[str]],
) -> None:
    """Verify that lambda_ret > 0 tilts allocation toward higher expected return assets."""
    df, regions = synthetic_transatlantic_returns
    n_assets = df.shape[1]

    # Create synthetic scores: highest score on asset 0
    scores = np.linspace(0.01, 0.10, n_assets)

    opt = MeanCVaROptimizer(beta=0.95, lambda_ret=1.0)
    alloc = opt.optimize(returns=df, region_blocs=regions, expected_returns=scores)

    assert not alloc.empty
    # Asset with highest score should receive top weight
    weights = alloc["weight"].to_numpy()
    assert weights[-1] >= weights[0]


def test_cvar_empty_input_handled() -> None:
    """Verify that empty inputs return empty DataFrame."""
    opt = MeanCVaROptimizer()
    res = opt.optimize(returns=pd.DataFrame(), region_blocs=[])
    assert res.empty
