"""Unified end-to-end regime detection and dual-sleeve hedging pipeline."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from quant_portfolio.config import PortfolioConfig, RegionBloc
from quant_portfolio.hedging.engine import DualSleeveHedgeEngine, DualSleeveHedgeSummary
from quant_portfolio.regime.hmm_detector import (
    DualExpandingRegimeDetector,
    ExpandingGaussianHMM,
    RegimeStateResult,
)
from quant_portfolio.regime.inertia_filter import InertiaBandFilter
from quant_portfolio.storage.schema import ParquetMarketStore, SQLiteStore

DEFAULT_FUTURES_QUOTES: dict[str, float] = {
    "ES": 5800.0,
    "MES": 5800.0,
    "FXXP": 520.0,
    "FESX": 4900.0,
}


class RegimeHedgingPipeline:
    """Orchestrates end-to-end regime estimation, inertia hysteresis, and futures hedging."""

    def __init__(
        self,
        sqlite_store: SQLiteStore,
        parquet_store: ParquetMarketStore | None = None,
        config: PortfolioConfig | None = None,
    ) -> None:
        self.store = sqlite_store
        self.parquet_store = parquet_store
        self.config = config or PortfolioConfig()
        self.detector = DualExpandingRegimeDetector(
            sqlite_store=self.store,
            parquet_store=self.parquet_store,
            config=self.config,
        )
        self.inertia_filter = InertiaBandFilter(config=self.config)
        self.hedge_engine = DualSleeveHedgeEngine(config=self.config)
        self._cached_hmm: dict[RegionBloc, ExpandingGaussianHMM] = {}
        self._days_since_fit: dict[RegionBloc, int] = {"NA": 0, "EU": 0}

    def detect_and_filter_regimes(
        self,
        as_of_date: str,
        lookback_days: int = 10,
    ) -> tuple[dict[RegionBloc, RegimeStateResult], dict[RegionBloc, int]]:
        """Run HMM regime inference and apply 3-day inertia band for NA and EU.

        Parameters
        ----------
        as_of_date : str
            Evaluation date (YYYY-MM-DD).
        lookback_days : int
            Number of recent business days to inspect for inertia confirmation.

        Returns
        -------
        tuple[dict[RegionBloc, RegimeStateResult], dict[RegionBloc, int]]
            Raw latest regime results and hysteresis-filtered states {0, 1, 2}.
        """
        latest_results: dict[RegionBloc, RegimeStateResult] = {}
        filtered_states: dict[RegionBloc, int] = {}

        for bloc in ("NA", "EU"):
            r_bloc: RegionBloc = "NA" if bloc == "NA" else "EU"
            panel = self.detector.feature_builder.build_regime_features(
                region_bloc=r_bloc,
                as_of_date=as_of_date,
                history_days=2520,
            )

            if len(panel) < self.config.regime.min_history_days:
                fallback = RegimeStateResult(
                    as_of_date=as_of_date,
                    region_bloc=r_bloc,
                    state=0,
                    prob_calm=1.0,
                    prob_alert=0.0,
                    prob_stress=0.0,
                )
                latest_results[r_bloc] = fallback
                filtered_states[r_bloc] = 0
                continue

            X = panel.to_numpy(dtype=np.float64)

            recal_freq = self.config.regime.recalibration_freq_days
            needs_fit = r_bloc not in self._cached_hmm or self._days_since_fit[r_bloc] >= recal_freq

            if needs_fit:
                detector_model = ExpandingGaussianHMM(
                    n_states=self.config.regime.n_states,
                    covariance_type=self.config.regime.covariance_type,
                )
                detector_model.fit_and_sort_states(X)
                self._cached_hmm[r_bloc] = detector_model
                self._days_since_fit[r_bloc] = 0
            else:
                detector_model = self._cached_hmm[r_bloc]
                self._days_since_fit[r_bloc] += 1

            probs = detector_model.compute_forward_probabilities(X)

            k = min(len(panel), max(self.config.regime.inertia_days, lookback_days))
            hist: list[RegimeStateResult] = []
            for idx in range(len(panel) - k, len(panel)):
                p_row = probs[idx]
                d_str = str(panel.index[idx])[:10]
                st = int(np.argmax(p_row))
                p_c = float(p_row[0]) if len(p_row) > 0 else 0.0
                p_a = float(p_row[1]) if len(p_row) > 1 else 0.0
                p_s = float(p_row[2]) if len(p_row) > 2 else 0.0
                hist.append(
                    RegimeStateResult(
                        as_of_date=d_str,
                        region_bloc=r_bloc,
                        state=st,
                        prob_calm=p_c,
                        prob_alert=p_a,
                        prob_stress=p_s,
                    )
                )

            latest_results[r_bloc] = hist[-1]
            filtered_seq = self.inertia_filter.filter_sequence(hist, initial_state=0)
            filtered_states[r_bloc] = filtered_seq[-1] if filtered_seq else 0

        return latest_results, filtered_states

    def execute_hedging_pipeline(
        self,
        as_of_date: str,
        portfolio_value_eur: float,
        allocations_df: pd.DataFrame,
        futures_quotes: Mapping[str, float] | None = None,
        eur_usd_rate: float = 1.08,
        ticker_betas: Mapping[str, float] | None = None,
        custom_na_future: str | None = None,
        custom_eu_future: str | None = None,
    ) -> DualSleeveHedgeSummary:
        """Run unified regime detection, inertia filtering, and futures overlay calculation."""
        quotes = dict(DEFAULT_FUTURES_QUOTES)
        if futures_quotes is not None:
            quotes.update(futures_quotes)

        _, filtered_regimes = self.detect_and_filter_regimes(as_of_date=as_of_date)

        return self.hedge_engine.calculate_hedging_plan(
            as_of_date=as_of_date,
            allocations_df=allocations_df,
            filtered_regimes=filtered_regimes,
            portfolio_value_eur=portfolio_value_eur,
            futures_quotes=quotes,
            eur_usd_rate=eur_usd_rate,
            ticker_betas=ticker_betas,
            custom_na_future=custom_na_future,
            custom_eu_future=custom_eu_future,
        )
