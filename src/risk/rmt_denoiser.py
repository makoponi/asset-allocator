"""Spectral Denoising of Empirical Covariance Matrices via Random Matrix Theory (RMT).

Applies the Marcenko-Pastur theoretical upper bound to identify and filter out
stochastic eigenvalue noise from financial correlation matrices while preserving the trace:
lambda_max = sigma^2 * (1 + sqrt(N / T))^2
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def marcenko_pastur_pdf(var: float, q: float, pts: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Calculate theoretical Marcenko-Pastur probability density function.

    q = T / N >= 1
    lambda_min = var * (1 - sqrt(1/q))^2
    lambda_max = var * (1 + sqrt(1/q))^2
    """
    lambda_min = var * (1.0 - np.sqrt(1.0 / q)) ** 2
    lambda_max = var * (1.0 + np.sqrt(1.0 / q)) ** 2
    e_val = np.linspace(lambda_min, lambda_max, pts)
    # Marcenko-Pastur analytical density
    diff = (lambda_max - e_val) * (e_val - lambda_min)
    diff = np.maximum(diff, 0.0)
    pdf = (q / (2.0 * np.pi * var * e_val)) * np.sqrt(diff)
    return e_val, pdf


def denoise_correlation_matrix(
    corr: np.ndarray,
    t_samples: int,
    n_assets: int,
    sigma_sq: float = 1.0,
) -> np.ndarray:
    """Denoise an empirical correlation matrix using RMT constant residual eigenvalue replacement.

    Parameters
    ----------
    corr : np.ndarray
        N x N empirical correlation matrix.
    t_samples : int
        Number of time-series observations (T).
    n_assets : int
        Number of assets (N).
    sigma_sq : float
        Variance of the stochastic component (default 1.0 for normalized series).

    Returns
    -------
    np.ndarray
        Denoised correlation matrix of shape (N, N) with unit diagonal and preserved trace.
    """
    if t_samples <= n_assets:
        msg = f"T ({t_samples}) must be greater than N ({n_assets}) for Marcenko-Pastur denoising"
        raise ValueError(msg)

    q = float(t_samples) / float(n_assets)
    lambda_max = sigma_sq * (1.0 + np.sqrt(1.0 / q)) ** 2

    # Eigendecomposition (hermitian matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Identify noise eigenvalues (<= lambda_max)
    noise_mask = eigenvalues <= lambda_max

    # If all or none are noise, handle gracefully
    if np.any(noise_mask) and not np.all(noise_mask):
        # Average noise eigenvalues to preserve total trace
        noise_mean = float(np.mean(eigenvalues[noise_mask]))
        denoised_evals = eigenvalues.copy()
        denoised_evals[noise_mask] = noise_mean
    else:
        denoised_evals = eigenvalues.copy()

    # Reconstruct correlation matrix: C_denoised = V * Lambda_denoised * V^T
    corr_denoised = eigenvectors @ np.diag(denoised_evals) @ eigenvectors.T

    # Rescale diagonal to exactly 1.0 (valid correlation matrix)
    diag_inv_sqrt = 1.0 / np.sqrt(np.diag(corr_denoised))
    corr_denoised = np.diag(diag_inv_sqrt) @ corr_denoised @ np.diag(diag_inv_sqrt)

    # Symmetrize
    corr_denoised = 0.5 * (corr_denoised + corr_denoised.T)
    return np.asarray(corr_denoised, dtype=float)


def denoise_covariance(
    returns_matrix: np.ndarray | pd.DataFrame,
    sigma_sq: float = 1.0,
) -> np.ndarray:
    """Compute denoised covariance matrix directly from asset return series.

    Parameters
    ----------
    returns_matrix : Array or DataFrame of shape (T, N)
        Daily return observations.
    sigma_sq : float
        Variance parameter for Marcenko-Pastur theoretical bound.

    Returns
    -------
    np.ndarray
        N x N denoised covariance matrix.
    """
    arr = np.asarray(returns_matrix, dtype=float)
    t_samples, n_assets = arr.shape

    # Standard deviations and empirical correlation
    stds = np.std(arr, axis=0, ddof=1)
    stds = np.where(stds <= 1e-12, 1e-6, stds)

    corr = np.corrcoef(arr, rowvar=False)

    # Clean potential NaNs
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    denoised_corr = denoise_correlation_matrix(
        corr=corr,
        t_samples=t_samples,
        n_assets=n_assets,
        sigma_sq=sigma_sq,
    )

    # Reconstruct covariance: Sigma = S * C_denoised * S
    diag_s = np.diag(stds)
    cov_denoised = diag_s @ denoised_corr @ diag_s
    return np.asarray(0.5 * (cov_denoised + cov_denoised.T), dtype=float)
