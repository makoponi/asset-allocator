"""Macroeconomic Regime Detection using a 3-State Gaussian Hidden Markov Model (HMM).

Pure Python / NumPy / SciPy implementation designed for Windows/Linux environments
without external C/Cython compiler dependencies (ABI-safe and strictly typed).
Identifies three canonical economic regimes:
- State 0: Normal / Bull Market (low OAS spread, steep yield curve, low V2TX volatility)
- State 1: Transition / Uncertainty (widening spread, flattening curve, rising volatility)
- State 2: Systemic Stress / Crisis (extreme OAS spread, inverted/stressed curve, volatility spike)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.special import logsumexp

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class HMMStateParams:
    """Fitted parameters for a specific latent regime state."""

    state_idx: int
    mean: NDArray[np.float64]
    var: NDArray[np.float64]
    name: str


class GaussianRegimeHMM:
    """3-State Gaussian Hidden Markov Model with canonical economic ordering.

    Supports Expectation-Maximization (Baum-Welch) fitting and exact real-time forward
    filtering P(S_t = k | X_{1:t}) to prevent look-ahead bias in walk-forward setups.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 50,
        tol: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        # Model parameters
        self.is_fitted = False
        self.feature_names: list[str] = []
        self.pi: NDArray[np.float64] = np.zeros(n_states, dtype=np.float64)
        self.trans_mat: NDArray[np.float64] = np.zeros((n_states, n_states), dtype=np.float64)
        self.means: NDArray[np.float64] = np.zeros((n_states, 1), dtype=np.float64)
        self.vars: NDArray[np.float64] = np.zeros((n_states, 1), dtype=np.float64)

    def _log_emission_density(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute log emission probabilities log P(x_t | S_t = k) for all t and k.

        Uses diagonal covariance with a regularizing variance floor (1e-4).
        Shape: (T, n_states)
        """
        t_steps, _ = X.shape
        log_densities = np.zeros((t_steps, self.n_states), dtype=np.float64)

        for k in range(self.n_states):
            mu_k = self.means[k]
            var_k = np.maximum(self.vars[k], 1e-4)

            # Multivariate Gaussian with diagonal covariance
            diff = X - mu_k
            quad = np.sum((diff**2) / var_k, axis=1)
            log_det = np.sum(np.log(2.0 * np.pi * var_k))
            log_densities[:, k] = -0.5 * (log_det + quad)

        return log_densities

    def _forward_pass(
        self, log_emiss: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float]:
        """Compute log alpha forward variables and total log likelihood.

        log_alpha[t, k] = log P(X_{1:t}, S_t = k)
        """
        t_steps = log_emiss.shape[0]
        log_alpha = np.zeros((t_steps, self.n_states), dtype=np.float64)
        log_pi = np.log(np.maximum(self.pi, 1e-12))
        log_trans = np.log(np.maximum(self.trans_mat, 1e-12))

        # t = 0
        log_alpha[0] = log_pi + log_emiss[0]

        # Induction
        for t in range(1, t_steps):
            log_alpha[t, :] = (
                logsumexp(log_alpha[t - 1, :, None] + log_trans, axis=0)
                + log_emiss[t, :]
            )

        total_log_likelihood = float(logsumexp(log_alpha[-1]))
        return log_alpha, total_log_likelihood

    def _backward_pass(self, log_emiss: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute log beta backward variables.

        log_beta[t, i] = log P(X_{t+1:T} | S_t = i)
        """
        t_steps = log_emiss.shape[0]
        log_beta = np.zeros((t_steps, self.n_states), dtype=np.float64)
        log_trans = np.log(np.maximum(self.trans_mat, 1e-12))

        # t = T - 1 (log(1) = 0)
        log_beta[-1] = 0.0

        # Backwards induction
        for t in range(t_steps - 2, -1, -1):
            log_beta[t, :] = logsumexp(
                log_trans + log_emiss[t + 1, None, :] + log_beta[t + 1, None, :],
                axis=1,
            )

        return log_beta


    def fit(self, X_df: pd.DataFrame) -> GaussianRegimeHMM:
        """Fit the 3-State HMM via Baum-Welch (EM) and order states canonically.

        Parameters
        ----------
        X_df : pd.DataFrame
            Macro signals matrix (e.g., EU_HY_OAS, BUND_10Y_2Y, V2TX).
        """
        if X_df.empty or len(X_df) < 50:
            msg = "At least 50 observations required to fit the 3-state HMM."
            raise ValueError(msg)

        self.feature_names = list(X_df.columns)
        X = np.asarray(X_df.values, dtype=np.float64)
        t_steps, n_features = X.shape

        # 1. Initialization: quantile-based clustering on first/leading risk factor
        # Sort initial centers by primary risk metric (V2TX or HY OAS if present, else 1st col)
        risk_col = 0
        for i, col in enumerate(self.feature_names):
            if "V2TX" in col.upper() or "OAS" in col.upper():
                risk_col = i
                break

        sorted_indices = np.argsort(X[:, risk_col])
        split_chunks = np.array_split(sorted_indices, self.n_states)

        self.pi = np.full(self.n_states, 1.0 / self.n_states, dtype=np.float64)
        self.trans_mat = np.full(
            (self.n_states, self.n_states),
            0.1 / (self.n_states - 1),
            dtype=np.float64,
        )
        np.fill_diagonal(self.trans_mat, 0.90)  # High regime persistence

        self.means = np.zeros((self.n_states, n_features), dtype=np.float64)
        self.vars = np.zeros((self.n_states, n_features), dtype=np.float64)

        for k in range(self.n_states):
            chunk = X[split_chunks[k]]
            self.means[k] = np.mean(chunk, axis=0)
            self.vars[k] = np.var(chunk, axis=0) + 1.0

        prev_ll = -np.inf

        # 2. Baum-Welch Iterations (EM)
        for _ in range(self.n_iter):
            # E-step
            log_emiss = self._log_emission_density(X)
            log_alpha, current_ll = self._forward_pass(log_emiss)
            log_beta = self._backward_pass(log_emiss)

            if abs(current_ll - prev_ll) < self.tol:
                break
            prev_ll = current_ll

            # Posterior state occupancies gamma[t, k] = P(S_t = k | X)
            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # Posterior transitions xi[t, i, j] = P(S_t = i, S_{t+1} = j | X)
            log_trans = np.log(np.maximum(self.trans_mat, 1e-12))
            log_xi = (
                log_alpha[:-1, :, None]
                + log_trans[None, :, :]
                + log_emiss[1:, None, :]
                + log_beta[1:, None, :]
            )
            log_xi -= logsumexp(log_xi, axis=(1, 2), keepdims=True)
            xi_sum = np.sum(np.exp(log_xi), axis=0)

            # M-step
            gamma_sum = np.sum(gamma, axis=0)  # shape (n_states,)
            gamma_sum = np.maximum(gamma_sum, 1e-12)

            # Update pi
            self.pi = gamma[0] / np.sum(gamma[0])

            # Update transition matrix A
            row_sums = np.sum(xi_sum, axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-12)
            self.trans_mat = xi_sum / row_sums

            # Update Gaussian means and variances
            for k in range(self.n_states):
                w = gamma[:, k, None]  # shape (T, 1)
                self.means[k] = np.sum(w * X, axis=0) / gamma_sum[k]
                diff = X - self.means[k]
                self.vars[k] = np.sum(w * (diff**2), axis=0) / gamma_sum[k]
                self.vars[k] = np.maximum(self.vars[k], 1e-4)

        # 3. Canonical Economic Ordering
        # Ensure State 0 is Lowest Risk, State 1 is Intermediate, State 2 is High Stress
        self._enforce_canonical_ordering(risk_col)
        self.is_fitted = True
        return self

    def _enforce_canonical_ordering(self, risk_col: int) -> None:
        """Permute states so State 0 = Lowest Risk, State 2 = Highest Risk."""
        order = np.argsort(self.means[:, risk_col])
        if np.array_equal(order, np.arange(self.n_states)):
            return

        self.pi = self.pi[order]
        self.trans_mat = self.trans_mat[order, :][:, order]
        self.means = self.means[order]
        self.vars = self.vars[order]

    def predict_filtered_probabilities(
        self, X_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute online filtered probabilities P(S_t = k | X_{1:t}) for each step.

        Strictly causal: probability at day t depends ONLY on observations up to day t.
        Returns a DataFrame indexed by date with columns
        ['prob_state_0', 'prob_state_1', 'prob_state_2'].
        """
        if not self.is_fitted:
            msg = "Model must be fitted before computing filtered probabilities."
            raise RuntimeError(msg)

        X = np.asarray(X_df[self.feature_names].values, dtype=np.float64)
        log_emiss = self._log_emission_density(X)
        log_alpha, _ = self._forward_pass(log_emiss)

        # Normalize alpha row-by-row
        log_prob = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
        probs = np.exp(log_prob)

        cols = [f"prob_state_{k}" for k in range(self.n_states)]
        df_out = pd.DataFrame(probs, index=X_df.index, columns=cols)
        df_out["dominant_regime"] = np.argmax(probs, axis=1)
        return df_out

    def get_state_summary(self) -> list[HMMStateParams]:
        """Return human-readable summary of the fitted regime states."""
        names = ["Bull / Normal", "Transition / Uncertainty", "Stress / Crisis"]
        summaries: list[HMMStateParams] = []
        for k in range(self.n_states):
            summaries.append(
                HMMStateParams(
                    state_idx=k,
                    mean=self.means[k],
                    var=self.vars[k],
                    name=names[k] if k < len(names) else f"State_{k}",
                )
            )
        return summaries
