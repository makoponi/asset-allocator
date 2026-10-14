"""Unit tests for GaussianRegimeHMM and regime detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.macro_regime import GaussianRegimeHMM


def test_gaussian_hmm_fit_and_ordering() -> None:
    # Build synthetic 3-regime series with clear separation
    rng = np.random.default_rng(42)
    n = 100

    # Low risk (State 0): V2TX ~ 13, OAS ~ 200
    r0 = np.column_stack([rng.normal(50, 5, n), rng.normal(200, 15, n), rng.normal(13, 1, n)])
    # Mid risk (State 1): V2TX ~ 18, OAS ~ 300
    r1 = np.column_stack([rng.normal(20, 5, n), rng.normal(300, 20, n), rng.normal(18, 1.5, n)])
    # High risk (State 2): V2TX ~ 28, OAS ~ 450
    r2 = np.column_stack([rng.normal(-10, 5, n), rng.normal(450, 25, n), rng.normal(28, 2, n)])

    data = np.vstack([r0, r1, r2, r0, r2])
    dates = pd.date_range("2020-01-01", periods=len(data), freq="B")
    df = pd.DataFrame(data, index=dates, columns=["BUND_10Y_2Y", "EU_HY_OAS", "V2TX"])

    hmm = GaussianRegimeHMM(n_states=3, n_iter=30)
    hmm.fit(df)

    assert hmm.is_fitted
    summaries = hmm.get_state_summary()
    assert len(summaries) == 3

    # Check canonical economic ordering: State 0 V2TX < State 1 V2TX < State 2 V2TX
    v2tx_idx = df.columns.get_loc("V2TX")
    assert summaries[0].mean[v2tx_idx] < summaries[1].mean[v2tx_idx]
    assert summaries[1].mean[v2tx_idx] < summaries[2].mean[v2tx_idx]

    # Test filtered probabilities
    probs_df = hmm.predict_filtered_probabilities(df)
    assert len(probs_df) == len(df)
    assert "prob_state_0" in probs_df.columns
    assert "prob_state_1" in probs_df.columns
    assert "prob_state_2" in probs_df.columns

    # Probabilities must sum to 1.0 for each row
    row_sums = probs_df[["prob_state_0", "prob_state_1", "prob_state_2"]].sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_gaussian_hmm_requires_minimum_observations() -> None:
    df_small = pd.DataFrame(
        np.zeros((20, 3)),
        columns=["BUND_10Y_2Y", "EU_HY_OAS", "V2TX"],
    )
    hmm = GaussianRegimeHMM()
    with pytest.raises(ValueError, match="At least 50 observations"):
        hmm.fit(df_small)
