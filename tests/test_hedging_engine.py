"""Unit tests for the Dual-Sleeve Beta-Adjusted Futures Hedging Engine."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.hedging.engine import DualSleeveHedgeEngine

if TYPE_CHECKING:
    from quant_portfolio.config import RegionBloc


@pytest.fixture
def sample_allocations() -> pd.DataFrame:
    """Sample transatlantic portfolio allocation from Phase 5."""
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "ENI IM", "AIR FP"],
            "region_bloc": ["NA", "NA", "EU", "EU"],
            "weight": [0.30, 0.30, 0.20, 0.20],  # 60% NA, 40% EU
            "beta": [1.10, 1.20, 0.80, 0.90],
        }
    )


def test_compute_sleeve_beta(sample_allocations: pd.DataFrame) -> None:
    """Verify capital-weighted beta calculations for each regional sleeve."""
    engine = DualSleeveHedgeEngine()
    beta_na = engine.compute_sleeve_beta(sample_allocations, "NA")
    beta_eu = engine.compute_sleeve_beta(sample_allocations, "EU")

    # NA weights are equal (0.30 each -> 50% relative weight each)
    expected_na = 0.5 * 1.10 + 0.5 * 1.20  # 1.15
    np.testing.assert_allclose(beta_na, expected_na, rtol=1e-6)

    # EU weights are equal (0.20 each -> 50% relative weight each)
    expected_eu = 0.5 * 0.80 + 0.5 * 0.90  # 0.85
    np.testing.assert_allclose(beta_eu, expected_eu, rtol=1e-6)


def test_calculate_hedging_plan_states(sample_allocations: pd.DataFrame) -> None:
    """Verify contract counts and notional calculations across different regime states."""
    engine = DualSleeveHedgeEngine()
    portfolio_val = 10_000_000.0  # 10M EUR

    quotes = {
        "ES": 5000.0,  # 5000 * 50 = $250,000 / 1.0 = 250,000 EUR
        "FXXP": 500.0,  # 500 * 50 = 25,000 EUR
    }

    # Case 1: Both calm (State 0) -> 0% hedge
    regimes_calm: dict[RegionBloc, int] = {"NA": 0, "EU": 0}
    plan_calm = engine.calculate_hedging_plan(
        as_of_date="2023-01-05",
        allocations_df=sample_allocations,
        filtered_regimes=regimes_calm,
        portfolio_value_eur=portfolio_val,
        futures_quotes=quotes,
        eur_usd_rate=1.0,  # 1:1 for easy math
    )

    assert plan_calm.na_sleeve.contracts_rounded == 0
    assert plan_calm.eu_sleeve.contracts_rounded == 0
    assert plan_calm.total_hedged_notional_eur == 0.0

    # Case 2: NA in Severe Stress (State 2 -> 100% * Beta), EU in Alert (State 1 -> 50% * Beta)
    regimes_stress: dict[RegionBloc, int] = {"NA": 2, "EU": 1}
    plan_stress = engine.calculate_hedging_plan(
        as_of_date="2023-01-05",
        allocations_df=sample_allocations,
        filtered_regimes=regimes_stress,
        portfolio_value_eur=portfolio_val,
        futures_quotes=quotes,
        eur_usd_rate=1.0,
    )

    # NA: 6M EUR * (1.0 * 1.15) = 6.9M EUR notional
    # ES contract val = $250k = 250k EUR
    # Contracts = 6,900,000 / 250,000 = 27.6 -> round to 28
    assert plan_stress.na_sleeve.hedge_ratio == pytest.approx(1.15)
    np.testing.assert_allclose(plan_stress.na_sleeve.hedged_notional_eur, 6_900_000.0, rtol=1e-5)
    assert plan_stress.na_sleeve.contracts_rounded == 28

    # EU: 4M EUR * (0.5 * 0.85) = 4M * 0.425 = 1.7M EUR notional
    # FXXP contract val = 500 * 50 = 25k EUR
    # Contracts = 1,700,000 / 25,000 = 68.0 -> 68
    assert plan_stress.eu_sleeve.hedge_ratio == pytest.approx(0.425)
    np.testing.assert_allclose(plan_stress.eu_sleeve.hedged_notional_eur, 1_700_000.0, rtol=1e-5)
    assert plan_stress.eu_sleeve.contracts_rounded == 68

    # Export to dataframe
    df = plan_stress.to_dataframe()
    assert len(df) == 2
    assert "contracts_short" in df.columns


def test_micro_futures_scaling(sample_allocations: pd.DataFrame) -> None:
    """Verify that MES produces exactly 10x the contract count of ES."""
    engine = DualSleeveHedgeEngine()
    portfolio_val = 1_000_000.0

    quotes = {"ES": 5000.0, "MES": 5000.0}
    regimes: dict[RegionBloc, int] = {"NA": 2, "EU": 0}

    plan_es = engine.calculate_hedging_plan(
        as_of_date="2023-01-05",
        allocations_df=sample_allocations,
        filtered_regimes=regimes,
        portfolio_value_eur=portfolio_val,
        futures_quotes=quotes,
        custom_na_future="ES",
    )

    plan_mes = engine.calculate_hedging_plan(
        as_of_date="2023-01-05",
        allocations_df=sample_allocations,
        filtered_regimes=regimes,
        portfolio_value_eur=portfolio_val,
        futures_quotes=quotes,
        custom_na_future="MES",
    )

    # Ratio of exact contracts must be exactly 10.0
    ratio = plan_mes.na_sleeve.contracts_exact / plan_es.na_sleeve.contracts_exact
    np.testing.assert_allclose(ratio, 10.0, rtol=1e-6)
