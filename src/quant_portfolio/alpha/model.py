"""Ensemble Machine Learning model (LightGBM + Random Forest) for alpha ranking."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from quant_portfolio.config import ALL_CORE_FEATURES, AlphaConfig

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class AlphaEnsembleModel:
    """Combines LightGBM Regressor and Random Forest Regressor for cross-sectional ranking."""

    def __init__(self, config: AlphaConfig | None = None) -> None:
        self.config = config or AlphaConfig()
        self.feature_names = list(ALL_CORE_FEATURES)

        self.lgbm = LGBMRegressor(
            max_depth=self.config.lgbm_max_depth,
            min_child_samples=self.config.lgbm_min_child_samples,
            n_estimators=self.config.lgbm_n_estimators,
            learning_rate=self.config.lgbm_learning_rate,
            reg_alpha=self.config.lgbm_reg_alpha,
            reg_lambda=self.config.lgbm_reg_lambda,
            random_state=self.config.random_state,
            verbosity=-1,
        )

        self.rf = RandomForestRegressor(
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            n_estimators=self.config.rf_n_estimators,
            max_features="sqrt",
            random_state=self.config.random_state,
            n_jobs=-1,
        )

        self._is_fitted: bool = False
        self._feature_importances: dict[str, float] = {}

    @property
    def is_fitted(self) -> bool:
        """Indicate whether the ensemble has been trained."""
        return self._is_fitted

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required core features exist and residual NaNs are imputed to 0.0."""
        prepared = X.copy()
        for f in self.feature_names:
            if f not in prepared.columns:
                prepared[f] = 0.0
            else:
                prepared[f] = prepared[f].fillna(0.0)
        return prepared[self.feature_names]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray[Any, Any],
    ) -> AlphaEnsembleModel:
        """Train both LightGBM and Random Forest on feature matrix X and target y."""
        X_clean = self._prepare_features(X)
        y_arr = np.asarray(y, dtype=np.float64)

        # Drop any observations where target is NaN
        valid_mask = ~np.isnan(y_arr)
        if not np.any(valid_mask):
            raise ValueError("All target values are NaN; cannot train ensemble model.")

        X_train = X_clean.iloc[valid_mask]
        y_train = y_arr[valid_mask]

        # Fit LightGBM
        self.lgbm.fit(X_train, y_train)

        # Fit Random Forest
        self.rf.fit(X_train, y_train)

        self._is_fitted = True
        self._compute_combined_feature_importances()
        return self

    def _compute_combined_feature_importances(self) -> None:
        """Compute normalized weighted combination of feature importances."""
        lgb_imp = np.asarray(self.lgbm.feature_importances_, dtype=np.float64)
        rf_imp = np.asarray(self.rf.feature_importances_, dtype=np.float64)

        lgb_sum = np.sum(lgb_imp)
        rf_sum = np.sum(rf_imp)

        lgb_norm = lgb_imp / lgb_sum if lgb_sum > 0 else np.ones_like(lgb_imp) / len(lgb_imp)
        rf_norm = rf_imp / rf_sum if rf_sum > 0 else np.ones_like(rf_imp) / len(rf_imp)

        w = self.config.ensemble_lgbm_weight
        combined = w * lgb_norm + (1.0 - w) * rf_norm

        self._feature_importances = {
            feat: float(imp) for feat, imp in zip(self.feature_names, combined, strict=True)
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Predict expected forward excess return combining both models."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

        X_clean = self._prepare_features(X)
        pred_lgb = np.asarray(self.lgbm.predict(X_clean), dtype=np.float64)
        pred_rf = np.asarray(self.rf.predict(X_clean), dtype=np.float64)

        w = self.config.ensemble_lgbm_weight
        result: np.ndarray[Any, np.dtype[np.float64]] = w * pred_lgb + (1.0 - w) * pred_rf
        return result

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray[Any, Any],
    ) -> dict[str, float]:
        """Compute predictive metrics: IC (Spearman), Pearson, MSE, RMSE, decile spread."""
        preds = self.predict(X)
        y_arr = np.asarray(y, dtype=np.float64)

        valid_mask = ~(np.isnan(y_arr) | np.isnan(preds))
        if not np.any(valid_mask) or np.sum(valid_mask) < 2:
            return {
                "ic_spearman": np.nan,
                "ic_pearson": np.nan,
                "mse": np.nan,
                "rmse": np.nan,
                "top_bottom_decile_spread": np.nan,
            }

        p_valid = preds[valid_mask]
        y_valid = y_arr[valid_mask]

        # Spearman rank correlation (Information Coefficient)
        if np.all(p_valid == p_valid[0]) or np.all(y_valid == y_valid[0]):
            ic_spearman = 0.0
        else:
            res_sp = spearmanr(p_valid, y_valid)
            if hasattr(res_sp, "statistic"):
                ic_spearman = float(res_sp.statistic)
            else:
                ic_spearman = float(res_sp[0])

        # Pearson correlation
        p_std = np.std(p_valid)
        y_std = np.std(y_valid)
        if p_std > 0 and y_std > 0:
            cov = np.cov(p_valid, y_valid)[0, 1]
            ic_pearson = float(cov / (p_std * y_std))
        else:
            ic_pearson = 0.0

        mse = float(mean_squared_error(y_valid, p_valid))
        rmse = float(np.sqrt(mse))

        # Top vs Bottom decile spread
        n = len(p_valid)
        top_k = max(1, n // 10)
        sorted_indices = np.argsort(p_valid)
        bottom_idx = sorted_indices[:top_k]
        top_idx = sorted_indices[-top_k:]

        top_mean = float(np.mean(y_valid[top_idx]))
        bottom_mean = float(np.mean(y_valid[bottom_idx]))
        decile_spread = top_mean - bottom_mean

        return {
            "ic_spearman": float(ic_spearman),
            "ic_pearson": float(ic_pearson),
            "mse": mse,
            "rmse": rmse,
            "top_bottom_decile_spread": decile_spread,
        }

    def get_feature_importances(self) -> dict[str, float]:
        """Return the dictionary of combined feature importances."""
        return self._feature_importances.copy()
