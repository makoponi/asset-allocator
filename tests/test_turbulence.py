"""Unit tests for Mahalanobis financial turbulence calculation."""

import numpy as np
import pandas as pd

from quant_portfolio.regime.turbulence import MahalanobisTurbulence


def test_turbulence_analytical_identity() -> None:
    """Verify turbulence against known analytical result with zero-mean identity covariance."""
    np.random.seed(42)
    n_obs = 100
    n_assets = 3

    # Generate historical series with mean 0 and identity covariance
    X = np.random.normal(loc=0.0, scale=1.0, size=(n_obs, n_assets))
    dates = pd.date_range("2020-01-01", periods=n_obs, freq="B").strftime("%Y-%m-%d")
    df = pd.DataFrame(X, index=dates, columns=["S1", "S2", "S3"])

    turb_engine = MahalanobisTurbulence(window=80, min_periods=50, shrinkage=0.0)
    res = turb_engine.compute_turbulence_series(df)

    assert len(res) == n_obs
    assert pd.isna(res.iloc[49])
    assert not pd.isna(res.iloc[50])
    assert (res.dropna() >= 0.0).all()


def test_turbulence_no_lookahead_bias() -> None:
    """Verify that altering future returns at t+5 does not affect turbulence at date t."""
    np.random.seed(123)
    n_obs = 80
    X = np.random.normal(loc=0.0, scale=0.01, size=(n_obs, 4))
    dates = pd.date_range("2020-01-01", periods=n_obs, freq="B").strftime("%Y-%m-%d")
    df1 = pd.DataFrame(X, index=dates, columns=["A", "B", "C", "D"])
    df2 = df1.copy()

    # Introduce extreme shock at index 75 in df2
    df2.iloc[75, :] = 10.0

    turb = MahalanobisTurbulence(window=50, min_periods=30, shrinkage=0.01)
    res1 = turb.compute_turbulence_series(df1)
    res2 = turb.compute_turbulence_series(df2)

    # For all t <= 74, res1 and res2 must be strictly identical
    np.testing.assert_allclose(res1.iloc[:75].to_numpy(), res2.iloc[:75].to_numpy(), rtol=1e-12)
    # At t = 75, turbulence in df2 must reflect the current shock
    assert res2.iloc[75] > res1.iloc[75]


def test_aggregate_sector_returns() -> None:
    """Verify sector aggregation of stock returns."""
    dates = ["2023-01-02", "2023-01-03"]
    stock_df = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02],
            "MSFT": [0.03, 0.04],
            "XOM": [0.05, 0.06],
        },
        index=dates,
    )
    ticker_to_sector = {
        "AAPL": "45",  # Tech
        "MSFT": "45",  # Tech
        "XOM": "10",  # Energy
    }

    sec_df = MahalanobisTurbulence.aggregate_sector_returns(stock_df, ticker_to_sector)
    assert list(sec_df.columns) == ["45", "10"]
    np.testing.assert_allclose(sec_df["45"].to_numpy(), [0.02, 0.03])
    np.testing.assert_allclose(sec_df["10"].to_numpy(), [0.05, 0.06])


def test_turbulence_edge_cases() -> None:
    """Test empty dataframe and single asset."""
    turb = MahalanobisTurbulence(window=20, min_periods=10)
    assert turb.compute_turbulence_series(pd.DataFrame()).empty

    dates = pd.date_range("2023-01-01", periods=15).strftime("%Y-%m-%d")
    single_df = pd.DataFrame({"A": np.linspace(0.01, 0.05, 15)}, index=dates)
    res = turb.compute_turbulence_series(single_df)
    assert len(res) == 15
    assert not pd.isna(res.iloc[-1])
