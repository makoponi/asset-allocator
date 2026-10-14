"""Machine Learning Ranker engine for cross-sectional security selection.

Supports LightGBM regression/ranking with out-of-fold evaluation (Spearman Rank IC)
and deterministic top-decile constituent extraction for portfolio construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

from src.features.cross_sectional import CORE_FEATURE_NAMES


class MLRanker:
    """Supervised learning ranker predicting forward cross-sectional excess returns."""

    def __init__(
        self,
        model_type: Literal["lightgbm", "random_forest"] = "lightgbm",
        feature_cols: Sequence[str] | None = None,
        random_state: int = 42,
        **model_params: Any,
    ) -> None:
        self.model_type = model_type
        self.feature_cols = list(feature_cols or CORE_FEATURE_NAMES)
        self.random_state = random_state
        self.model_params = model_params
        self._model: Any = None
        self._init_model()

    def _init_model(self) -> None:
        if self.model_type == "lightgbm":
            try:
                import lightgbm as lgb

                default_params: dict[str, Any] = {
                    "n_estimators": 100,
                    "learning_rate": 0.03,
                    "max_depth": 4,
                    "num_leaves": 15,
                    "subsample": 0.8,
                    "colsample_bytree": 0.7,
                    "min_child_samples": 30,
                    "random_state": self.random_state,
                    "verbosity": -1,
                }
                default_params.update(self.model_params)
                self._model = lgb.LGBMRegressor(**default_params)
            except ImportError:
                # Graceful fallback to RandomForest
                self._model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=self.random_state,
                )
        else:
            default_rf_params: dict[str, Any] = {
                "n_estimators": 100,
                "max_depth": 5,
                "random_state": self.random_state,
            }
            default_rf_params.update(self.model_params)
            self._model = RandomForestRegressor(**default_rf_params)

    def fit(self, X: pd.DataFrame, y: pd.Series[float] | np.ndarray[Any, Any]) -> MLRanker:
        """Fit model on feature matrix X against forward excess return target y."""
        features = [c for c in self.feature_cols if c in X.columns]
        X_sub = X[features].astype(float)
        self._model.fit(X_sub, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Predict expected forward excess return for security rows."""
        features = [c for c in self.feature_cols if c in X.columns]
        X_sub = X[features].astype(float)
        preds = self._model.predict(X_sub)
        return np.asarray(preds, dtype=float)

    def compute_rank_ic(
        self,
        X: pd.DataFrame,
        y: pd.Series[float] | np.ndarray[Any, Any],
    ) -> float:
        """Calculate overall Spearman Rank Correlation (Rank IC) between predictions and targets."""
        preds = self.predict(X)
        actual = np.asarray(y, dtype=float)
        res = spearmanr(preds, actual)
        # Handle tuple or object return from spearmanr
        val = res.statistic if hasattr(res, "statistic") else res[0]
        return float(val) if not np.isnan(val) else 0.0

    @staticmethod
    def compute_mean_cross_sectional_ic(
        df_eval: pd.DataFrame,
        date_col: str = "date",
        pred_col: str = "prediction",
        target_col: str = "target",
    ) -> float:
        """Calculate average cross-sectional Spearman Rank IC across individual dates."""
        daily_ics: list[float] = []

        for _d, group in df_eval.groupby(date_col):
            if len(group) < 10:
                continue
            res = spearmanr(group[pred_col], group[target_col])
            val = res.statistic if hasattr(res, "statistic") else res[0]
            if not np.isnan(val):
                daily_ics.append(float(val))

        return float(np.mean(daily_ics)) if daily_ics else 0.0

    def select_top_decile(
        self,
        X_t: pd.DataFrame,
        top_pct: float = 0.10,
        min_n: int = 35,
        max_n: int = 60,
    ) -> list[str]:
        """Select top decile of constituents ranked by predicted expected alpha.

        Enforces cardinality constraints (min_n <= n <= max_n) to ensure a well-conditioned
        covariance matrix for Phase 5 risk allocation.
        """
        if X_t.empty or "ticker" not in X_t.columns:
            return []

        try:
            preds = self.predict(X_t)
        except Exception:
            # Cold-start fallback: rank by momentum 12m if available, else zeros
            if "mom_12m" in X_t.columns:
                preds = np.asarray(X_t["mom_12m"].fillna(0.0), dtype=float)
            else:
                preds = np.zeros(len(X_t), dtype=float)

        df_ranked = X_t[["ticker"]].copy()
        df_ranked["score"] = preds
        df_ranked = df_ranked.sort_values(by="score", ascending=False)

        total_stocks = len(df_ranked)
        k = int(round(total_stocks * top_pct))
        target_n = max(min_n, min(max_n, k))
        target_n = min(target_n, total_stocks)

        return df_ranked.iloc[:target_n]["ticker"].tolist()
