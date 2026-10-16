"""Unit tests for the 3-day inertia band persistence filter."""

from quant_portfolio.regime.hmm_detector import RegimeStateResult
from quant_portfolio.regime.inertia_filter import InertiaBandFilter


def make_result(
    date: str,
    p_calm: float,
    p_alert: float,
    p_stress: float,
) -> RegimeStateResult:
    """Helper to construct RegimeStateResult with automatic argmax state."""
    state = 0
    if p_alert >= p_calm and p_alert >= p_stress:
        state = 1
    elif p_stress >= p_calm and p_stress >= p_alert:
        state = 2
    return RegimeStateResult(
        as_of_date=date,
        region_bloc="NA",
        state=state,
        prob_calm=p_calm,
        prob_alert=p_alert,
        prob_stress=p_stress,
    )


def test_inertia_prevents_1day_and_2day_spikes() -> None:
    """Verify that 1-day or 2-day spikes into stress do not trigger hedge escalation."""
    filter_engine = InertiaBandFilter(inertia_days=3, stress_open_prob=0.70, stress_close_prob=0.40)

    # Initial calm series
    calm = [make_result(f"2023-01-0{i}", 0.85, 0.10, 0.05) for i in range(1, 4)]
    assert filter_engine.filter_step(calm, previous_filtered_state=0) == 0

    # Day 4: sudden single-day severe shock (prob_stress = 0.95)
    spike_1 = calm + [make_result("2023-01-04", 0.02, 0.03, 0.95)]
    assert filter_engine.filter_step(spike_1, previous_filtered_state=0) == 0  # Preserved!

    # Day 5: second shock day (prob_stress = 0.90)
    spike_2 = spike_1 + [make_result("2023-01-05", 0.05, 0.05, 0.90)]
    assert filter_engine.filter_step(spike_2, previous_filtered_state=0) == 0  # Still preserved!

    # Day 6: market calms down immediately (prob_calm = 0.80)
    revert = spike_2 + [make_result("2023-01-06", 0.80, 0.15, 0.05)]
    assert filter_engine.filter_step(revert, previous_filtered_state=0) == 0


def test_inertia_triggers_after_3_consecutive_days() -> None:
    """Verify that 3 consecutive confirmations trigger transition to stress."""
    filter_engine = InertiaBandFilter(inertia_days=3, stress_open_prob=0.70, stress_close_prob=0.40)

    history = [
        make_result("2023-01-01", 0.90, 0.08, 0.02),
        make_result("2023-01-02", 0.85, 0.10, 0.05),
        make_result("2023-01-03", 0.10, 0.15, 0.75),  # day 1 of stress
        make_result("2023-01-04", 0.05, 0.10, 0.85),  # day 2 of stress
        make_result("2023-01-05", 0.02, 0.08, 0.90),  # day 3 of stress -> Escalation!
    ]

    states = filter_engine.filter_sequence(history, initial_state=0)
    assert states == [0, 0, 0, 0, 2]


def test_inertia_de_escalation_requires_3_consecutive_days() -> None:
    """Verify that de-escalating out of stress state requires 3 consecutive calm days."""
    filter_engine = InertiaBandFilter(inertia_days=3, stress_open_prob=0.70, stress_close_prob=0.40)

    history = [
        make_result("2023-01-01", 0.05, 0.10, 0.85),  # stress
        make_result("2023-01-02", 0.02, 0.08, 0.90),  # stress
        make_result("2023-01-03", 0.02, 0.08, 0.90),  # stress (state is 2)
        make_result("2023-01-04", 0.70, 0.20, 0.10),  # calm day 1
        make_result("2023-01-05", 0.75, 0.15, 0.10),  # calm day 2
        make_result("2023-01-06", 0.80, 0.15, 0.05),  # calm day 3 -> De-escalate to 0!
    ]

    states = filter_engine.filter_sequence(history, initial_state=2)
    # Days 1-3 remain 2; days 4 and 5 remain 2 (insufficient calm days); day 6 de-escalates to 0
    assert states[0] == 2
    assert states[1] == 2
    assert states[2] == 2
    assert states[3] == 2
    assert states[4] == 2
    assert states[5] == 0
