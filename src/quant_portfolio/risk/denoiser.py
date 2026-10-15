"""Random Matrix Theory (Marcenko-Pastur) Spectral Covariance Denoiser.

Filters empirical correlation and covariance matrices of transatlantic equity returns
by isolating eigenvalues associated with genuine economic signals from those falling
within the Marcenko-Pastur noise band:
    lambda_+ = sigma^2 * (1 + sqrt(N/T))^2

Noise eigenvalues are shrunk to their constant mean while strictly preserving
the matrix trace Tr(C) = N and positive semi-definiteness.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class MarcenkoPasturDenoiser:
    """Spectral denoiser based on Marcenko-Pastur distribution and constant residual shrinkage."""

    def __init__(self, bandwidth: float = 0.05) -> None:
        self.bandwidth = bandwidth

    @staticmethod
    def marcenko_pastur_pdf(var: float, q: float, pts: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """Compute the theoretical Marcenko-Pastur probability density function."""
        e_min = var * (1.0 - np.sqrt(1.0 / q)) ** 2
        e_max = var * (1.0 + np.sqrt(1.0 / q)) ** 2
        e_vals = np.linspace(e_min, e_max, pts)
        denom = 2.0 * np.pi * var * (1.0 / q) * e_vals
        sqrt_term = np.sqrt(np.maximum(0.0, (e_max - e_vals) * (e_vals - e_min)))
        pdf = sqrt_term / denom
        return e_vals, pdf

    def fit_noise_variance(self, eigenvals: np.ndarray, q: float) -> float:
        """Estimate the noise variance sigma^2 by fitting the Marcenko-Pastur distribution."""
        if len(eigenvals) == 0:
            return 1.0

        n = len(eigenvals)

        def objective(var: float) -> float:
            e_max = var * (1.0 + np.sqrt(1.0 / q)) ** 2
            # Noise eigenvalues are those <= e_max
            noise_evals = eigenvals[eigenvals <= e_max]
            if len(noise_evals) == 0:
                return 1e6
            # Theoretical variance of noise component
            noise_var = float(np.var(noise_evals))
            # Expected theoretical variance for MP
            e_min = var * (1.0 - np.sqrt(1.0 / q)) ** 2
            expected_var = float(((e_max - e_min) ** 2) / 12.0)
            return float((noise_var - expected_var) ** 2)

        res = minimize_scalar(objective, bounds=(0.01, 1.0), method="bounded")
        if res.success and 0.0 < float(res.x) <= 1.0:
            return float(res.x)

        # Fallback: estimate variance from non-dominant eigenvalues
        # If market factor exists (dominant eigenval), exclude it
        if n > 1:
            var_est = float(np.mean(eigenvals[1:]))
            return min(max(var_est, 0.05), 1.0)
        return 1.0

    def denoise_correlation_matrix(
        self,
        corr_matrix: np.ndarray,
        t_samples: int,
        n_assets: int,
        sigma2: float | None = None,
    ) -> np.ndarray:
        """Denoise an empirical correlation matrix using constant residual eigenvalue replacement.

        Guarantees:
        - Trace preservation: Tr(C_denoised) == Tr(C_empirical) == N
        - Unit diagonal: C_denoised[i, i] == 1.0
        - Positive semi-definiteness
        """
        if t_samples <= n_assets:
            logger.warning(
                "Number of observations T=%d <= N=%d. Matrix may be singular.",
                t_samples,
                n_assets,
            )

        q = float(t_samples) / float(n_assets)

        # 1. Eigendecomposition of correlation matrix (symmetric)
        e_vals, e_vecs = np.linalg.eigh(corr_matrix)
        # Sort descending
        idx = np.argsort(e_vals)[::-1]
        e_vals = np.maximum(e_vals[idx], 0.0)
        e_vecs = e_vecs[:, idx]

        # 2. Determine noise threshold lambda_+
        var = sigma2 if sigma2 is not None else self.fit_noise_variance(e_vals, q)
        e_max = var * (1.0 + np.sqrt(1.0 / q)) ** 2

        # 3. Constant residual replacement for noise eigenvalues
        noise_mask = e_vals <= e_max
        n_noise = int(np.sum(noise_mask))

        e_vals_denoised = e_vals.copy()
        if n_noise > 0:
            mean_noise_eval = float(np.mean(e_vals[noise_mask]))
            e_vals_denoised[noise_mask] = mean_noise_eval

        # 4. Reconstruct correlation matrix
        c_denoised = e_vecs @ np.diag(e_vals_denoised) @ e_vecs.T

        # 5. Rescale diagonal to ensure strictly 1.0
        diag_inv_sqrt = 1.0 / np.sqrt(np.maximum(np.diag(c_denoised), 1e-12))
        c_denoised = np.diag(diag_inv_sqrt) @ c_denoised @ np.diag(diag_inv_sqrt)
        np.fill_diagonal(c_denoised, 1.0)

        # Symmetrize
        c_denoised = (c_denoised + c_denoised.T) / 2.0
        return np.asarray(c_denoised, dtype=np.float64)

    def denoise_covariance(
        self,
        returns: pd.DataFrame | np.ndarray,
        tickers: Sequence[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute and denoise empirical covariance and correlation matrices.

        Returns:
            (denoised_cov_df, denoised_corr_df)
        """
        if isinstance(returns, pd.DataFrame):
            ticker_names = list(returns.columns)
            r_mat = returns.to_numpy(dtype=np.float64)
        else:
            r_mat = np.asarray(returns, dtype=np.float64)
            n_cols = r_mat.shape[1]
            ticker_names = (
                list(tickers) if tickers is not None else [f"A{i}" for i in range(n_cols)]
            )

        t_samples, n_assets = r_mat.shape
        if n_assets == 0:
            return pd.DataFrame(), pd.DataFrame()

        # Compute empirical standard deviations and correlation
        stds = np.std(r_mat, axis=0, ddof=1)
        stds = np.maximum(stds, 1e-8)  # prevent div by zero

        corr_mat = np.corrcoef(r_mat, rowvar=False)
        corr_mat = np.nan_to_num(corr_mat, nan=0.0)
        np.fill_diagonal(corr_mat, 1.0)

        # Apply Marcenko-Pastur denoising to correlation matrix
        denoised_corr = self.denoise_correlation_matrix(
            corr_matrix=corr_mat,
            t_samples=t_samples,
            n_assets=n_assets,
        )

        # Reconstruct covariance matrix: Sigma = S @ C @ S
        std_diag = np.diag(stds)
        denoised_cov = std_diag @ denoised_corr @ std_diag
        denoised_cov = (denoised_cov + denoised_cov.T) / 2.0

        cov_df = pd.DataFrame(denoised_cov, index=ticker_names, columns=ticker_names)
        corr_df = pd.DataFrame(denoised_corr, index=ticker_names, columns=ticker_names)
        return cov_df, corr_df
