"""Unit tests for Marcenko-Pastur spectral covariance denoiser."""

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.risk.denoiser import MarcenkoPasturDenoiser


@pytest.fixture
def synthetic_factor_returns() -> pd.DataFrame:
    """Generate T=200, N=30 synthetic returns with 1 dominant market factor and iid noise."""
    np.random.seed(42)
    t_samples = 200
    n_assets = 30

    # Common factor
    market = np.random.normal(0.0, 0.02, size=(t_samples, 1))
    betas = np.random.uniform(0.5, 1.5, size=(1, n_assets))
    idiosyncratic = np.random.normal(0.0, 0.015, size=(t_samples, n_assets))

    returns_matrix = market @ betas + idiosyncratic
    tickers = [f"STOCK_{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(returns_matrix, columns=tickers)


def test_trace_preservation_and_unit_diagonal(synthetic_factor_returns: pd.DataFrame) -> None:
    """Verify trace preservation, unit diagonal, symmetry, and positive semi-definiteness."""
    denoiser = MarcenkoPasturDenoiser()
    n_assets = synthetic_factor_returns.shape[1]

    cov_df, corr_df = denoiser.denoise_covariance(synthetic_factor_returns)

    # 1. Shape check
    assert cov_df.shape == (n_assets, n_assets)
    assert corr_df.shape == (n_assets, n_assets)

    # 2. Unit diagonal for correlation
    diag_vals = np.diag(corr_df.to_numpy())
    np.testing.assert_allclose(diag_vals, 1.0, atol=1e-8)

    # 3. Symmetry
    corr_np = corr_df.to_numpy()
    np.testing.assert_allclose(corr_np, corr_np.T, atol=1e-8)

    # 4. Trace preservation Tr(C) == N
    trace_corr = float(np.trace(corr_np))
    assert abs(trace_corr - float(n_assets)) < 1e-6

    # 5. Positive semi-definiteness: eigenvalues >= -1e-8
    evals = np.linalg.eigvalsh(corr_np)
    assert np.all(evals >= -1e-8)


def test_noise_eigenvalue_flattening() -> None:
    """Verify that pure white noise eigenvalues are collapsed to their mean."""
    np.random.seed(123)
    t_samples = 500
    n_assets = 20
    # Pure uncorrelated white noise
    noise_returns = np.random.normal(0.0, 0.01, size=(t_samples, n_assets))

    denoiser = MarcenkoPasturDenoiser()
    cov_df, corr_df = denoiser.denoise_covariance(noise_returns)

    evals = np.linalg.eigvalsh(corr_df.to_numpy())
    # In pure noise, all or almost all eigenvalues fall below lambda_+
    # so their variance should be substantially reduced
    raw_corr = np.corrcoef(noise_returns, rowvar=False)
    raw_evals = np.linalg.eigvalsh(raw_corr)

    assert np.var(evals) < np.var(raw_evals)


def test_empty_returns_handled_gracefully() -> None:
    """Verify handling of empty DataFrame or matrix."""
    denoiser = MarcenkoPasturDenoiser()
    cov, corr = denoiser.denoise_covariance(pd.DataFrame())
    assert cov.empty
    assert corr.empty
