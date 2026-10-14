"""Unit tests for HedgingOverlay rules, hysteresis filtering, and future sizing."""

from __future__ import annotations

import pandas as pd
import pytest

from src.risk.hedging_overlay import HedgingOverlay


def test_hedging_overlay_threshold_rules() -> None:
    overlay = HedgingOverlay(
        hysteresis_days=3,
        threshold_prob=0.65,
        hedge_ratio_state_0=0.0,
        hedge_ratio_state_1=0.40,
        hedge_ratio_state_2=0.90,
    )

    # 1. State 0 dominates
    regime, h = overlay.compute_target_hedge_ratio(0.80, 0.15, 0.05)
    assert regime == 0
    assert h == 0.0

    # 2. State 1 dominates
    regime, h = overlay.compute_target_hedge_ratio(0.10, 0.70, 0.20)
    assert regime == 1
    assert h == 0.40

    # 3. State 2 dominates
    regime, h = overlay.compute_target_hedge_ratio(0.05, 0.15, 0.80)
    assert regime == 2
    assert h == 0.90

    # 4. Ambiguous regime (no state >= 0.65) -> Blended expectation
    regime, h = overlay.compute_target_hedge_ratio(0.40, 0.35, 0.25)
    assert regime == 0
    expected_h = 0.40 * 0.0 + 0.35 * 0.40 + 0.25 * 0.90
    assert h == pytest.approx(expected_h, rel=1e-4)


def test_hedging_overlay_hysteresis_filter() -> None:
    overlay = HedgingOverlay(hysteresis_days=3, threshold_prob=0.65)

    # Sequence of probabilities:
    # Days 1-4: Calm (State 0)
    # Days 5-6: Panic spike (State 2) -> only 2 days, should NOT trigger effective hedge
    # Day 7: Back to Calm (State 0)
    # Days 8-11: Sustained Panic (State 2) -> 4 days, should trigger effective hedge at Day 10
    probs_data = [
        # Days 1-4 (State 0)
        [0.90, 0.08, 0.02],
        [0.85, 0.10, 0.05],
        [0.88, 0.08, 0.04],
        [0.92, 0.05, 0.03],
        # Days 5-6 (Transient shock State 2)
        [0.05, 0.10, 0.85],
        [0.05, 0.10, 0.85],
        # Day 7 (Calm State 0)
        [0.85, 0.10, 0.05],
        # Days 8-11 (Sustained crisis State 2)
        [0.05, 0.05, 0.90],
        [0.05, 0.05, 0.90],
        [0.05, 0.05, 0.90],
        [0.05, 0.05, 0.90],
    ]

    dates = pd.date_range("2022-01-01", periods=len(probs_data), freq="B")
    df_probs = pd.DataFrame(
        probs_data,
        index=dates,
        columns=["prob_state_0", "prob_state_1", "prob_state_2"],
    )

    df_res = overlay.simulate_overlay_series(df_probs)

    # Days 1-4: Effective hedge is 0.0
    assert df_res.iloc[0]["effective_hedge_ratio"] == 0.0
    assert df_res.iloc[3]["effective_hedge_ratio"] == 0.0

    # Days 5-6 (transient shock): Target is 0.90, but effective stays 0.0 because of hysteresis
    assert df_res.iloc[4]["target_hedge_ratio"] == 0.90
    assert df_res.iloc[4]["effective_hedge_ratio"] == 0.0
    assert df_res.iloc[5]["effective_hedge_ratio"] == 0.0

    # Day 7: Back to Calm
    assert df_res.iloc[6]["effective_hedge_ratio"] == 0.0

    # Days 8-9: Shock restarts, pending days = 1, 2 -> effective stays 0.0
    assert df_res.iloc[7]["effective_hedge_ratio"] == 0.0
    assert df_res.iloc[8]["effective_hedge_ratio"] == 0.0

    # Day 10: 3rd consecutive day of State 2 ->
    # Hysteresis confirms and effective transitions to 0.90
    assert df_res.iloc[9]["effective_hedge_ratio"] == 0.90
    assert df_res.iloc[10]["effective_hedge_ratio"] == 0.90


def test_hedging_overlay_contract_sizing() -> None:
    overlay = HedgingOverlay(
        hysteresis_days=1,
        future_multiplier=10.0,
        hedge_ratio_state_2=1.00,
    )

    df_probs = pd.DataFrame(
        [[0.02, 0.08, 0.90]],
        index=pd.date_range("2022-03-01", periods=1),
        columns=["prob_state_0", "prob_state_1", "prob_state_2"],
    )

    nav_series = pd.Series([10_000_000.0])  # 10M EUR NAV
    future_px = pd.Series([4_000.0])        # Future at 4,000 pts -> 40,000 EUR per contract
    # 10M EUR * 1.00 hedge * 1.0 beta / 40,000 EUR = 250 contracts

    df_res = overlay.simulate_overlay_series(
        df_probs,
        nav_series=nav_series,
        future_prices=future_px,
        portfolio_beta=1.0,
    )

    assert df_res.iloc[0]["num_future_contracts"] == 250
