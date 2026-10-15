"""Unit and mathematical tests for TransatlanticCrossSectionalNormalizer."""

import numpy as np
import pandas as pd

from quant_portfolio.config import FeatureConfig
from quant_portfolio.features.normalizer import TransatlanticCrossSectionalNormalizer


def test_winsorize_by_region() -> None:
    """Verify that extreme outliers are clipped independently within NA and EU."""
    cfg = FeatureConfig(winsor_lower_quantile=0.05, winsor_upper_quantile=0.95)
    norm = TransatlanticCrossSectionalNormalizer(config=cfg)

    # 100 stocks for NA with one extreme positive outlier
    na_vals = list(np.linspace(10.0, 30.0, 99)) + [10_000.0]
    # 100 stocks for EU with one extreme negative outlier
    eu_vals = [-5_000.0] + list(np.linspace(5.0, 25.0, 99))

    df = pd.DataFrame(
        {
            "ticker": [f"NA_{i}" for i in range(100)] + [f"EU_{i}" for i in range(100)],
            "region_bloc": ["NA"] * 100 + ["EU"] * 100,
            "PE_RATIO": na_vals + eu_vals,
        }
    )

    winsorized = norm.winsorize_by_region(df, feature_cols=["PE_RATIO"])

    na_pe = winsorized[winsorized["region_bloc"] == "NA"]["PE_RATIO"].to_numpy()
    eu_pe = winsorized[winsorized["region_bloc"] == "EU"]["PE_RATIO"].to_numpy()

    # Extreme NA outlier (10,000) must be clipped to <= 30.0
    assert na_pe.max() <= 30.0
    # Extreme EU outlier (-5,000) must be clipped to >= 5.0
    assert eu_pe.min() >= 5.0


def test_zscore_by_region_and_sector() -> None:
    """Verify standardisation to mean=0 and std=1 within (region_bloc, gics_sector)."""
    norm = TransatlanticCrossSectionalNormalizer()

    # Create 2 distinct subgroups: (NA, Tech 45) and (EU, Health 35)
    np.random.seed(123)
    n = 30
    df = pd.DataFrame(
        {
            "ticker": [f"T_{i}" for i in range(2 * n)],
            "region_bloc": ["NA"] * n + ["EU"] * n,
            "gics_sector": [45] * n + [35] * n,
            "FEAT_1": list(np.random.normal(50.0, 10.0, n)) + list(np.random.normal(10.0, 2.0, n)),
        }
    )

    std_df = norm.zscore_by_region_and_sector(df, feature_cols=["FEAT_1"])

    # Check (NA, 45)
    na_tech = std_df[(std_df["region_bloc"] == "NA") & (std_df["gics_sector"] == 45)]["FEAT_1"]
    assert abs(float(na_tech.mean())) < 1e-7
    assert abs(float(na_tech.std(ddof=1)) - 1.0) < 1e-7

    # Check (EU, 35)
    eu_health = std_df[(std_df["region_bloc"] == "EU") & (std_df["gics_sector"] == 35)]["FEAT_1"]
    assert abs(float(eu_health.mean())) < 1e-7
    assert abs(float(eu_health.std(ddof=1)) - 1.0) < 1e-7


def test_edge_cases_single_observation_and_zero_variance() -> None:
    """Verify that single observation or zero variance sectors gracefully map to z=0.0."""
    norm = TransatlanticCrossSectionalNormalizer()

    df = pd.DataFrame(
        {
            "ticker": ["T_SOLO", "T_FLAT1", "T_FLAT2"],
            "region_bloc": ["NA", "EU", "EU"],
            "gics_sector": [15, 20, 20],  # Sector 15 has 1 stock, Sector 20 has constant values
            "FEAT_A": [42.0, 10.0, 10.0],
        }
    )

    std_df = norm.zscore_by_region_and_sector(df, feature_cols=["FEAT_A"])
    assert (std_df["FEAT_A"] == 0.0).all()


def test_impute_neutral_residuals() -> None:
    """Verify that residual NaNs, infinities, and missing fields map to 0.0."""
    norm = TransatlanticCrossSectionalNormalizer()

    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "FEAT_1": [np.nan, np.inf, 1.25],
        }
    )

    imputed = norm.impute_neutral_residuals(df, feature_cols=["FEAT_1"])
    assert imputed["FEAT_1"].tolist() == [0.0, 0.0, 1.25]


def test_full_transatlantic_normalization_pipeline() -> None:
    """Verify complete pipeline execution: Winsorize -> Z-Score -> Impute."""
    norm = TransatlanticCrossSectionalNormalizer()

    df = pd.DataFrame(
        {
            "as_of_date": ["2024-01-31"] * 6,
            "ticker": ["A1", "A2", "A3", "E1", "E2", "E3"],
            "region_bloc": ["NA", "NA", "NA", "EU", "EU", "EU"],
            "gics_sector": [45, 45, 45, 45, 45, 45],
            "PE_RATIO": [25.0, 30.0, np.nan, 15.0, 18.0, 21.0],
            "MOM_1M": [0.05, -0.02, 0.01, 0.10, -0.05, 0.02],
        }
    )

    res = norm.normalize(df, feature_cols=["PE_RATIO", "MOM_1M"])

    assert len(res) == 6
    assert not res[["PE_RATIO", "MOM_1M"]].isna().any().any()
    assert (res["PE_RATIO"].dtype == np.float64) and (res["MOM_1M"].dtype == np.float64)
