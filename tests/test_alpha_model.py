"""Tests for Phase 4 Machine Learning Ensemble (LightGBM + Random Forest)."""

import numpy as np
import pandas as pd

from quant_portfolio.alpha.model import AlphaEnsembleModel
from quant_portfolio.config import ALL_CORE_FEATURES, AlphaConfig


def test_alpha_ensemble_fit_predict_properties() -> None:
    """Verify that AlphaEnsembleModel trains and predicts without error or NaN."""
    np.random.seed(42)
    n_samples = 200
    n_features = len(ALL_CORE_FEATURES)

    # Synthetic Z-score features
    X_mat = np.random.randn(n_samples, n_features)
    X = pd.DataFrame(X_mat, columns=list(ALL_CORE_FEATURES))

    # Realized target with positive correlation to the first two features
    y = 0.3 * X.iloc[:, 0] + 0.2 * X.iloc[:, 1] + 0.1 * np.random.randn(n_samples)

    config = AlphaConfig(
        lgbm_max_depth=4,
        lgbm_min_child_samples=10,  # small for unit test
        lgbm_n_estimators=30,
        rf_max_depth=4,
        rf_min_samples_leaf=10,
        rf_n_estimators=30,
        ensemble_lgbm_weight=0.5,
    )
    model = AlphaEnsembleModel(config=config)
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == n_samples
    assert not np.isnan(preds).any()

    # Predictions should be positively correlated with y
    metrics = model.evaluate(X, y)
    assert metrics["ic_spearman"] > 0.3
    assert metrics["ic_pearson"] > 0.3
    assert metrics["rmse"] > 0.0

    # Feature importances check
    importances = model.get_feature_importances()
    assert len(importances) == n_features
    # Feature 0 and 1 should have high importance
    assert importances[ALL_CORE_FEATURES[0]] > 0.0
    assert np.isclose(sum(importances.values()), 1.0, atol=1e-5)


def test_alpha_ensemble_handles_missing_feature_values() -> None:
    """Verify that residual NaNs in test features are imputed to neutral 0.0 without error."""
    config = AlphaConfig(
        lgbm_max_depth=3,
        lgbm_min_child_samples=5,
        lgbm_n_estimators=10,
        rf_max_depth=3,
        rf_min_samples_leaf=5,
        rf_n_estimators=10,
    )
    model = AlphaEnsembleModel(config=config)

    n_samples = 50
    X = pd.DataFrame(
        np.random.randn(n_samples, len(ALL_CORE_FEATURES)), columns=list(ALL_CORE_FEATURES)
    )
    y = pd.Series(np.random.randn(n_samples))

    model.fit(X, y)

    # Insert NaNs in test feature matrix
    X_test = X.copy()
    X_test.iloc[0, 0] = np.nan
    X_test.iloc[1, 2] = np.nan

    preds = model.predict(X_test)
    assert len(preds) == n_samples
    assert not np.isnan(preds).any()
