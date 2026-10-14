"""Unit tests for Random Matrix Theory (Marcenko-Pastur) spectral denoising."""

from __future__ import annotations

import numpy as np
import pytest

from src.risk.rmt_denoiser import (
    denoise_correlation_matrix,
    denoise_covariance,
    marcenko_pastur_pdf,
)


def test_marcenko_pastur_bounds() -> None:
    # T = 504, N = 50 -> q = 10.08
    q = 504.0 / 50.0
    var = 1.0
    evals, pdf = marcenko_pastur_pdf(var=var, q=q, pts=500)

    lambda_min = var * (1.0 - np.sqrt(1.0 / q)) ** 2
    lambda_max = var * (1.0 + np.sqrt(1.0 / q)) ** 2

    assert evals[0] == pytest.approx(lambda_min)
    assert evals[-1] == pytest.approx(lambda_max)
    assert np.all(pdf >= 0.0)


def test_denoise_correlation_matrix_properties() -> None:
    # Create synthetic correlation matrix with 1 strong factor (market) + random noise
    t_samples = 500
    n_assets = 30
    rng = np.random.default_rng(42)

    # 1 common market factor + idiosyncratic noise
    market_factor = rng.standard_normal((t_samples, 1))
    noise = rng.standard_normal((t_samples, n_assets))
    returns = 0.6 * market_factor + 0.4 * noise

    raw_corr = np.asarray(np.corrcoef(returns, rowvar=False), dtype=float)

    denoised_corr = denoise_correlation_matrix(
        corr=raw_corr,
        t_samples=t_samples,
        n_assets=n_assets,
        sigma_sq=1.0,
    )

    # 1. Dimension match
    assert denoised_corr.shape == (n_assets, n_assets)

    # 2. Symmetry
    assert np.allclose(denoised_corr, denoised_corr.T, atol=1e-8)

    # 3. Unit diagonal
    assert np.allclose(np.diag(denoised_corr), 1.0, atol=1e-6)

    # 4. Trace preservation: trace(C_denoised) == N
    assert np.trace(denoised_corr) == pytest.approx(n_assets, abs=1e-4)

    # 5. Positive semi-definiteness: all eigenvalues >= 0
    eigvals = np.linalg.eigvalsh(denoised_corr)
    assert np.all(eigvals >= -1e-8)


def test_denoise_covariance_matrix() -> None:
    t_samples = 300
    n_assets = 20
    rng = np.random.default_rng(123)

    returns = rng.standard_normal((t_samples, n_assets)) * 0.01

    cov_denoised = denoise_covariance(returns)

    assert cov_denoised.shape == (n_assets, n_assets)
    assert np.allclose(cov_denoised, cov_denoised.T, atol=1e-8)
    eigvals = np.linalg.eigvalsh(cov_denoised)
    assert np.all(eigvals >= -1e-10)
