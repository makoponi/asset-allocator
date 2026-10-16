"""Unit tests for ExpandingGaussianHMM and DualExpandingRegimeDetector."""

from pathlib import Path

import numpy as np

from quant_portfolio.config import PortfolioConfig
from quant_portfolio.regime.hmm_detector import DualExpandingRegimeDetector, ExpandingGaussianHMM
from quant_portfolio.storage.schema import SQLiteStore


def generate_synthetic_3regime_data(
    n_samples_per_regime: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic time series transitioning through 3 distinct volatility regimes."""
    np.random.seed(42)
    # Regime 0: Low vol (Calm) - mean 12.0, vol 1.0
    r0 = np.random.normal(loc=12.0, scale=1.0, size=(n_samples_per_regime, 2))
    # Regime 1: Medium vol (Alert) - mean 22.0, vol 3.0
    r1 = np.random.normal(loc=22.0, scale=3.0, size=(n_samples_per_regime, 2))
    # Regime 2: High vol (Stress) - mean 45.0, vol 8.0
    r2 = np.random.normal(loc=45.0, scale=8.0, size=(n_samples_per_regime, 2))

    X = np.vstack([r0, r1, r2])
    true_states = np.array(
        [0] * n_samples_per_regime + [1] * n_samples_per_regime + [2] * n_samples_per_regime
    )
    return X, true_states


def test_hmm_anti_label_switching() -> None:
    """Verify that states are strictly ordered by ascending stress/volatility."""
    X, _ = generate_synthetic_3regime_data(n_samples_per_regime=120)

    model = ExpandingGaussianHMM(n_states=3, covariance_type="diag", random_state=42)
    model.fit_and_sort_states(X)

    assert model.model is not None
    means = model.model.means_[:, 0]
    # State 0 < State 1 < State 2
    assert means[0] < means[1] < means[2]


def test_hmm_forward_probabilities_properties() -> None:
    """Verify that forward probabilities sum to 1.0 and avoid look-ahead bias."""
    X, _ = generate_synthetic_3regime_data(n_samples_per_regime=80)
    model = ExpandingGaussianHMM(n_states=3, covariance_type="diag", random_state=123)
    model.fit_and_sort_states(X)

    probs = model.compute_forward_probabilities(X)
    assert probs.shape == (len(X), 3)

    # Probabilities must sum to 1.0 at every time step
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)
    assert (probs >= 0.0).all()

    # Zero look-ahead check: changing X at index 200 should not affect forward probs at index 150
    X_modified = X.copy()
    X_modified[200:, :] = 100.0  # extreme shock in the future
    probs_modified = model.compute_forward_probabilities(X_modified)

    np.testing.assert_allclose(probs[:180], probs_modified[:180], rtol=1e-10)


def test_dual_expanding_regime_detector(tmp_path: Path) -> None:
    """Test end-to-end regime inference with SQLite store."""
    db_path = tmp_path / "test_regime.db"
    store = SQLiteStore(db_path)

    # Populate 260 days of macro data for NA and EU
    all_dates = [f"2023-{m:02d}-{d:02d}" for m in range(1, 10) for d in range(1, 29)][:260]

    np.random.seed(42)
    records = []
    for i, d in enumerate(all_dates):
        # Introduce a stress spike in the last 10 days
        is_stress = i >= 250
        vix_val = 45.0 + np.random.normal(0, 2.0) if is_stress else 15.0 + np.random.normal(0, 1.0)
        v2tx_val = 42.0 + np.random.normal(0, 2.0) if is_stress else 16.0 + np.random.normal(0, 1.0)

        records.append(
            {"date": d, "region_bloc": "NA", "series_id": "VIX", "value": float(vix_val)}
        )
        records.append(
            {
                "date": d,
                "region_bloc": "NA",
                "series_id": "T10Y2Y",
                "value": float(-0.5 + np.random.normal(0, 0.05)),
            }
        )
        records.append(
            {
                "date": d,
                "region_bloc": "NA",
                "series_id": "BAMLH0A0HYM2",
                "value": float(350.0 + np.random.normal(0, 5.0)),
            }
        )
        records.append(
            {"date": d, "region_bloc": "EU", "series_id": "V2TX", "value": float(v2tx_val)}
        )
        records.append(
            {
                "date": d,
                "region_bloc": "EU",
                "series_id": "BUND10Y2Y",
                "value": float(-0.2 + np.random.normal(0, 0.05)),
            }
        )
        records.append(
            {
                "date": d,
                "region_bloc": "EU",
                "series_id": "BAMLHE00EHYIOAS",
                "value": float(380.0 + np.random.normal(0, 5.0)),
            }
        )

    store.upsert_macro_series(records)

    cfg = PortfolioConfig()
    cfg.regime.min_history_days = 100
    detector = DualExpandingRegimeDetector(sqlite_store=store, config=cfg)

    # Test at calm period (day 200)
    calm_date = all_dates[200]
    res_calm = detector.detect_regimes_at_date(calm_date)
    assert "NA" in res_calm and "EU" in res_calm
    assert res_calm["NA"].state in (0, 1)

    # Test at stress spike period (last date)
    stress_date = all_dates[-1]
    res_stress = detector.detect_regimes_at_date(stress_date)
    assert res_stress["NA"].prob_stress > 0.5 or res_stress["NA"].state >= 1
