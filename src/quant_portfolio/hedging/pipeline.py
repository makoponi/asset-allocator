"""Unified end-to-end regime detection and dual-sleeve hedging pipeline."""

from collections.abc import Mapping

import pandas as pd

from quant_portfolio.config import PortfolioConfig, RegionBloc
from quant_portfolio.hedging.engine import DualSleeveHedgeEngine, DualSleeveHedgeSummary
from quant_portfolio.regime.hmm_detector import DualExpandingRegimeDetector, RegimeStateResult
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
        # Determine trailing business days up to as_of_date
        dt_end = pd.Timestamp(as_of_date)
        dt_start = dt_end - pd.DateOffset(days=lookback_days * 2)
        b_dates = (
            pd.bdate_range(start=dt_start, end=dt_end)
            .strftime("%Y-%m-%d")
            .tolist()[-max(self.config.regime.inertia_days, 5) :]
        )

        history_per_bloc: dict[RegionBloc, list[RegimeStateResult]] = {"NA": [], "EU": []}
        latest_results: dict[RegionBloc, RegimeStateResult] = {}

        for d in b_dates:
            reg_d = self.detector.detect_regimes_at_date(d)
            for bloc in ("NA", "EU"):
                r_bloc: RegionBloc = "NA" if bloc == "NA" else "EU"
                history_per_bloc[r_bloc].append(reg_d[r_bloc])
                if d == as_of_date:
                    latest_results[r_bloc] = reg_d[r_bloc]

        if not latest_results:
            latest_results = self.detector.detect_regimes_at_date(as_of_date)

        filtered_states: dict[RegionBloc, int] = {}
        for bloc in ("NA", "EU"):
            r_bloc = "NA" if bloc == "NA" else "EU"
            hist = history_per_bloc[r_bloc]
            # Sequential inertia filter over trailing window
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
