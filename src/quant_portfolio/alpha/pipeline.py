"""End-to-end Alpha Pipeline orchestrating features, target, CV, model, and selection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from quant_portfolio.alpha.cv import PurgedWalkForwardCV
from quant_portfolio.alpha.model import AlphaEnsembleModel
from quant_portfolio.alpha.selector import StratifiedSleeveSelector
from quant_portfolio.alpha.target import TargetGenerator
from quant_portfolio.config import ALL_CORE_FEATURES, AlphaConfig, PortfolioConfig
from quant_portfolio.features.pipeline import TransatlanticFeaturePipeline
from quant_portfolio.storage.schema import ParquetMarketStore, SQLiteStore

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class AlphaPipeline:
    """Orchestrates end-to-end Machine Learning training, evaluation, and sleeve selection."""

    def __init__(
        self,
        config: PortfolioConfig | None = None,
        sqlite_store: SQLiteStore | None = None,
        parquet_store: ParquetMarketStore | None = None,
        feature_pipeline: TransatlanticFeaturePipeline | None = None,
        target_generator: TargetGenerator | None = None,
        model: AlphaEnsembleModel | None = None,
        selector: StratifiedSleeveSelector | None = None,
    ) -> None:
        self.config = config or PortfolioConfig()
        paths = self.config.paths.resolve_paths()
        self.sqlite_store = sqlite_store or SQLiteStore(paths.db_path)
        self.parquet_store = parquet_store or ParquetMarketStore(paths.parquet_dir)
        self.alpha_config: AlphaConfig = self.config.alpha or AlphaConfig()

        self.feature_pipeline = feature_pipeline or TransatlanticFeaturePipeline(
            config=self.config,
            sqlite_store=self.sqlite_store,
            parquet_store=self.parquet_store,
        )
        self.target_generator = target_generator or TargetGenerator(
            parquet_store=self.parquet_store,
            config=self.config,
        )
        self.model = model or AlphaEnsembleModel(config=self.alpha_config)
        self.selector = selector or StratifiedSleeveSelector(config=self.config)

    def build_labeled_dataset(self, dates: Sequence[str]) -> pd.DataFrame:
        """Construct merged panel of normalized features X and forward excess targets y."""
        feature_frames: list[pd.DataFrame] = []
        target_frames: list[pd.DataFrame] = []

        for d in sorted(dates):
            feats_d = self.feature_pipeline.build_features_at_date(d, normalize=True)
            if not feats_d.empty:
                feature_frames.append(feats_d)
                targets_d = self.target_generator.compute_target_at_date(
                    as_of_date=d,
                    universe_df=feats_d[["ticker", "region_bloc"]],
                )
                target_frames.append(targets_d)

        if not feature_frames or not target_frames:
            return pd.DataFrame()

        feats_all = pd.concat(feature_frames, ignore_index=True)
        targets_all = pd.concat(target_frames, ignore_index=True)

        labeled = feats_all.merge(
            targets_all[["as_of_date", "ticker", "fwd_return", "regional_median", "target"]],
            on=["as_of_date", "ticker"],
            how="left",
        )
        return labeled

    def run_walk_forward_evaluation(
        self,
        dates: Sequence[str],
        n_splits: int = 5,
        purge_days: int | None = None,
        embargo_days: int | None = None,
    ) -> dict[str, Any]:
        """Execute Purged Walk-Forward CV across the panel and return performance metrics."""
        labeled_df = self.build_labeled_dataset(dates)
        if labeled_df.empty:
            logger.warning("No labeled observations found for walk-forward evaluation.")
            return {"mean_ic": np.nan, "fold_metrics": []}

        # Filter rows with valid targets for training and evaluation
        valid_df = labeled_df.dropna(subset=["target"]).reset_index(drop=True)
        if valid_df.empty:
            logger.warning("All target values are NaN (horizon exceeds data).")
            return {"mean_ic": np.nan, "fold_metrics": []}

        cv = PurgedWalkForwardCV(
            n_splits=n_splits,
            purge_days=purge_days or self.alpha_config.purge_days,
            embargo_days=embargo_days or self.alpha_config.embargo_days,
            expanding=True,
        )

        fold_metrics: list[dict[str, Any]] = []
        feature_cols = list(ALL_CORE_FEATURES)

        for fold, (train_idx, test_idx) in enumerate(cv.split(valid_df, date_col="as_of_date")):
            train_sub = valid_df.iloc[train_idx]
            test_sub = valid_df.iloc[test_idx]

            fold_model = AlphaEnsembleModel(config=self.alpha_config)
            fold_model.fit(train_sub[feature_cols], train_sub["target"])

            metrics = fold_model.evaluate(test_sub[feature_cols], test_sub["target"])
            fold_info: dict[str, Any] = {
                "fold": fold,
                "train_dates": (
                    f"{train_sub['as_of_date'].min()} to {train_sub['as_of_date'].max()}"
                ),
                "test_dates": (f"{test_sub['as_of_date'].min()} to {test_sub['as_of_date'].max()}"),
                "train_samples": len(train_sub),
                "test_samples": len(test_sub),
                **metrics,
            }
            fold_metrics.append(fold_info)

        if not fold_metrics:
            return {"mean_ic": np.nan, "fold_metrics": []}

        ic_vals = [m["ic_spearman"] for m in fold_metrics if not np.isnan(m["ic_spearman"])]
        mean_ic = float(np.mean(ic_vals)) if ic_vals else np.nan
        std_ic = float(np.std(ic_vals)) if len(ic_vals) > 1 else 0.0
        ir_ic = float(mean_ic / std_ic) if std_ic > 0 else np.nan

        rmse_vals = [m["rmse"] for m in fold_metrics if not np.isnan(m["rmse"])]
        mean_rmse = float(np.mean(rmse_vals)) if rmse_vals else np.nan

        spread_vals = [
            m["top_bottom_decile_spread"]
            for m in fold_metrics
            if not np.isnan(m["top_bottom_decile_spread"])
        ]
        mean_spread = float(np.mean(spread_vals)) if spread_vals else np.nan

        return {
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "ir_ic": ir_ic,
            "mean_rmse": mean_rmse,
            "mean_decile_spread": mean_spread,
            "n_splits": len(fold_metrics),
            "fold_metrics": fold_metrics,
        }

    def _resolve_historical_training_dates(self, as_of_date: str) -> list[str]:
        """Fetch historical training dates prior to as_of_date minus purge window."""
        purge_days = self.alpha_config.purge_days
        cutoff_dt = pd.Timestamp(as_of_date) - pd.tseries.offsets.BDay(purge_days)
        cutoff_str = cutoff_dt.strftime("%Y-%m-%d")

        query = (
            "SELECT DISTINCT as_of_date FROM universe_pit "
            "WHERE as_of_date <= ? ORDER BY as_of_date ASC"
        )
        with self.sqlite_store.get_connection() as conn:
            dates_df = pd.read_sql_query(query, conn, params=[cutoff_str])
        return [str(d) for d in dates_df["as_of_date"].tolist()]

    def score_and_rank_universe(
        self,
        as_of_date: str,
        training_dates: Sequence[str] | None = None,
        top_na: int | None = None,
        top_eu: int | None = None,
    ) -> pd.DataFrame:
        """Compute features, train model if needed, score, and return full ranked universe."""
        # 1. Train model if training_dates provided or not fitted
        if training_dates is not None or not self.model.is_fitted:
            dates_to_train = (
                list(training_dates)
                if training_dates is not None
                else self._resolve_historical_training_dates(as_of_date)
            )
            if dates_to_train:
                logger.info(
                    "Training ensemble model on %d historical dates...", len(dates_to_train)
                )
                train_df = self.build_labeled_dataset(dates_to_train)
                train_clean = train_df.dropna(subset=["target"]).reset_index(drop=True)
                if not train_clean.empty:
                    feature_cols = list(ALL_CORE_FEATURES)
                    self.model.fit(train_clean[feature_cols], train_clean["target"])

        # 2. Extract features at as_of_date
        features_t = self.feature_pipeline.build_features_at_date(as_of_date, normalize=True)
        if features_t.empty:
            logger.warning("No active universe found for date %s", as_of_date)
            return pd.DataFrame()

        # 3. Predict expected excess return score
        if self.model.is_fitted:
            scores = self.model.predict(features_t)
        else:
            # Fallback uniform deterministic scores if no training data was available
            scores = np.linspace(1.0, 0.0, len(features_t))

        scored_universe = features_t.copy()
        scored_universe["score"] = scores

        # 4. Return full ranked universe with regional ranks and deciles
        return self.selector.rank_full_universe(
            scored_universe=scored_universe,
            top_na=top_na,
            top_eu=top_eu,
        )

    def rank_and_select_at_date(
        self,
        as_of_date: str,
        training_dates: Sequence[str] | None = None,
        top_na: int | None = None,
        top_eu: int | None = None,
    ) -> pd.DataFrame:
        """Compute features, score universe, and select candidates for as_of_date."""
        ranked = self.score_and_rank_universe(
            as_of_date=as_of_date,
            training_dates=training_dates,
            top_na=top_na,
            top_eu=top_eu,
        )
        if ranked.empty:
            return pd.DataFrame()

        selected = ranked[ranked["is_selected"]].reset_index(drop=True)
        return selected
