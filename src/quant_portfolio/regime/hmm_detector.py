"""Expanding Gaussian HMM regime detector with anti-label switching and pure forward filtering."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from hmmlearn import hmm
from scipy.special import logsumexp

from quant_portfolio.config import VALID_REGION_BLOCS, PortfolioConfig, RegionBloc
from quant_portfolio.regime.macro_loader import RegimeFeatureBuilder
from quant_portfolio.storage.schema import ParquetMarketStore, SQLiteStore


@dataclass(frozen=True)
class RegimeStateResult:
    """Snapshot of regime probabilities and detected state at date t."""

    as_of_date: str
    region_bloc: RegionBloc
    state: int  # 0: Calm, 1: Alert, 2: Severe Stress
    prob_calm: float
    prob_alert: float
    prob_stress: float

    @property
    def state_name(self) -> str:
        names = {0: "Calm", 1: "Alert", 2: "Severe Stress"}
        return names.get(self.state, "Unknown")


class ExpandingGaussianHMM:
    """Gaussian HMM trained on expanding windows with deterministic anti-label switching.

    Enforces zero look-ahead bias:
    State probabilities at date t are computed strictly via pure forward filtering P(S_t | y_{1:t}),
    never using smoothed probabilities or retrospective Viterbi passes.
    """

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "diag",
        n_iter: int = 150,
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.model: hmm.GaussianHMM | None = None
        self._fitted = False

    def fit_and_sort_states(self, X: np.ndarray) -> "ExpandingGaussianHMM":
        """Fit GaussianHMM on feature matrix X and sort states in ascending volatility order.

        Parameters
        ----------
        X : np.ndarray of shape (T, K)
            Historical feature matrix where the first feature (column 0) is a volatility/stress
            indicator (e.g. VIX or V2TX), or feature values increase with market stress.

        Returns
        -------
        ExpandingGaussianHMM
            Self with sorted state parameters: State 0 (Calm), 1 (Alert), 2 (Stress).
        """
        if len(X) < self.n_states * 10:
            raise ValueError(f"Insufficient samples to fit HMM ({len(X)} < {self.n_states * 10})")

        model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        model.fit(X)

        # Anti-label switching: compute expected stress metric per state
        # Primary criterion: mean of the first feature (volatility/stress).
        # Secondary criterion: trace of covariance matrix.
        stress_metric = np.zeros(self.n_states, dtype=np.float64)
        raw_covars: np.ndarray = model._covars_
        for s in range(self.n_states):
            first_feat_mean = float(model.means_[s, 0])
            if self.covariance_type == "diag":
                cov_var = float(np.sum(raw_covars[s]))
            else:
                cov_var = float(np.trace(raw_covars[s]))
            stress_metric[s] = first_feat_mean + 0.01 * np.sqrt(cov_var)

        perm = np.argsort(stress_metric)

        # Permute model parameters to guarantee State 0 < State 1 < State 2
        model.startprob_ = model.startprob_[perm]
        model.transmat_ = model.transmat_[perm, :][:, perm]
        model.means_ = model.means_[perm, :]
        model._covars_ = raw_covars[perm]

        self.model = model
        self._fitted = True
        return self

    def compute_forward_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Compute pure forward-filtered probabilities P(S_t = k | y_{1:t}) for each step t.

        Parameters
        ----------
        X : np.ndarray of shape (T, K)
            Observations up to time T.

        Returns
        -------
        np.ndarray of shape (T, n_states)
            Matrix where row t contains filtered state probabilities at time t.
        """
        if not self._fitted or self.model is None:
            raise RuntimeError("Model must be fitted before computing forward probabilities.")

        T, _ = X.shape
        framelogprob = self.model._compute_log_likelihood(X)

        # Log-space forward algorithm
        log_alpha = np.empty((T, self.n_states), dtype=np.float64)
        log_transmat = np.log(np.maximum(self.model.transmat_, 1e-15))
        log_startprob = np.log(np.maximum(self.model.startprob_, 1e-15))

        # t = 0
        log_alpha[0] = log_startprob + framelogprob[0]

        # t = 1 ... T - 1
        for t in range(1, T):
            for j in range(self.n_states):
                prev_trans = log_alpha[t - 1] + log_transmat[:, j]
                log_alpha[t, j] = framelogprob[t, j] + logsumexp(prev_trans)

        # Normalize to probabilities
        log_norm = logsumexp(log_alpha, axis=1, keepdims=True)
        prob: np.ndarray[Any, np.dtype[np.float64]] = np.asarray(
            np.exp(log_alpha - log_norm), dtype=np.float64
        )
        return prob


class DualExpandingRegimeDetector:
    """Manages independent expanding HMM regime detectors for NA and EU sleeves."""

    def __init__(
        self,
        sqlite_store: SQLiteStore,
        parquet_store: ParquetMarketStore | None = None,
        config: PortfolioConfig | None = None,
    ) -> None:
        self.store = sqlite_store
        self.parquet_store = parquet_store
        self.config = config or PortfolioConfig()
        self.feature_builder = RegimeFeatureBuilder(
            sqlite_store=self.store,
            parquet_store=self.parquet_store,
            config=self.config,
        )

    def detect_regimes_at_date(self, as_of_date: str) -> dict[RegionBloc, RegimeStateResult]:
        """Infer current market regime for NA and EU sleeves at as_of_date.

        Parameters
        ----------
        as_of_date : str
            Target date string (YYYY-MM-DD).

        Returns
        -------
        dict[RegionBloc, RegimeStateResult]
            Regime snapshot per region bloc.
        """
        results: dict[RegionBloc, RegimeStateResult] = {}

        for region_bloc in VALID_REGION_BLOCS:
            panel = self.feature_builder.build_regime_features(
                region_bloc=region_bloc,
                as_of_date=as_of_date,
                history_days=2520,
            )

            if len(panel) < self.config.regime.min_history_days:
                # Fallback to calm state if insufficient history is available
                results[region_bloc] = RegimeStateResult(
                    as_of_date=as_of_date,
                    region_bloc=region_bloc,
                    state=0,
                    prob_calm=1.0,
                    prob_alert=0.0,
                    prob_stress=0.0,
                )
                continue

            X = panel.to_numpy(dtype=np.float64)
            detector = ExpandingGaussianHMM(
                n_states=self.config.regime.n_states,
                covariance_type=self.config.regime.covariance_type,
            )
            detector.fit_and_sort_states(X)
            probs = detector.compute_forward_probabilities(X)

            # Extract latest filtered probabilities at as_of_date
            latest_prob = probs[-1]
            raw_state = int(np.argmax(latest_prob))

            p_calm = float(latest_prob[0]) if len(latest_prob) > 0 else 0.0
            p_alert = float(latest_prob[1]) if len(latest_prob) > 1 else 0.0
            p_stress = float(latest_prob[2]) if len(latest_prob) > 2 else 0.0

            results[region_bloc] = RegimeStateResult(
                as_of_date=as_of_date,
                region_bloc=region_bloc,
                state=raw_state,
                prob_calm=p_calm,
                prob_alert=p_alert,
                prob_stress=p_stress,
            )

        return results
