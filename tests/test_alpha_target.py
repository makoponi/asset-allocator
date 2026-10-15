"""Tests for Phase 4 Target Generation (21-day forward excess returns)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.alpha.target import TargetGenerator
from quant_portfolio.storage.schema import ParquetMarketStore


@pytest.fixture
def synthetic_parquet_store(tmp_path: Path) -> ParquetMarketStore:
    """Create a temporary ParquetMarketStore with 50 trading days of synthetic data."""
    store = ParquetMarketStore(tmp_path / "parquet")
    dates = pd.bdate_range("2024-01-01", periods=60).strftime("%Y-%m-%d").tolist()

    # Create 4 tickers: 2 NA and 2 EU
    records: list[dict[str, object]] = []
    # Seed prices
    base_prices = {"AAPL US": 100.0, "MSFT US": 200.0, "SAP GY": 150.0, "ASML NA": 300.0}
    # Distinct drift rates per ticker
    daily_growth = {"AAPL US": 1.001, "MSFT US": 1.002, "SAP GY": 0.999, "ASML NA": 1.003}

    for t_idx, d in enumerate(dates):
        for ticker, base_px in base_prices.items():
            reg = "NA" if ticker.endswith("US") else "EU"
            px = base_px * (daily_growth[ticker] ** t_idx)
            records.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "region_bloc": reg,
                    "px_last_eur": px,
                    "tot_return_idx_eur": px,
                    "volume_eur": 1_000_000.0,
                }
            )

    store.write_daily_market(pd.DataFrame(records))
    return store


def test_target_generator_analytical_properties(
    synthetic_parquet_store: ParquetMarketStore,
) -> None:
    """Verify that regional median subtraction results in exactly zero median per region."""
    generator = TargetGenerator(parquet_store=synthetic_parquet_store)
    universe = pd.DataFrame(
        [
            {"ticker": "AAPL US", "region_bloc": "NA"},
            {"ticker": "MSFT US", "region_bloc": "NA"},
            {"ticker": "SAP GY", "region_bloc": "EU"},
            {"ticker": "ASML NA", "region_bloc": "EU"},
        ]
    )

    t0 = "2024-01-02"
    horizon = 21
    res = generator.compute_target_at_date(
        as_of_date=t0,
        universe_df=universe,
        horizon_days=horizon,
    )

    assert not res.empty
    assert len(res) == 4
    assert set(res["ticker"]) == {"AAPL US", "MSFT US", "SAP GY", "ASML NA"}

    # Analytical verification of forward returns:
    # AAPL forward return = 1.001^21 - 1
    # MSFT forward return = 1.002^21 - 1
    r_aapl = 1.001**21 - 1
    r_msft = 1.002**21 - 1
    med_na = 0.5 * (r_aapl + r_msft)

    aapl_row = res[res["ticker"] == "AAPL US"].iloc[0]
    msft_row = res[res["ticker"] == "MSFT US"].iloc[0]

    np.testing.assert_allclose(aapl_row["fwd_return"], r_aapl, rtol=1e-5)
    np.testing.assert_allclose(msft_row["fwd_return"], r_msft, rtol=1e-5)
    np.testing.assert_allclose(aapl_row["regional_median"], med_na, rtol=1e-5)

    # Key property: Target is excess over median
    np.testing.assert_allclose(aapl_row["target"], r_aapl - med_na, rtol=1e-5)
    np.testing.assert_allclose(msft_row["target"], r_msft - med_na, rtol=1e-5)

    # Verify zero median per region
    for reg in ["NA", "EU"]:
        reg_targets = res[res["region_bloc"] == reg]["target"]
        assert np.isclose(reg_targets.median(), 0.0, atol=1e-12)


def test_target_generator_handles_insufficient_history(
    synthetic_parquet_store: ParquetMarketStore,
) -> None:
    """Verify that dates without enough future trading days return NaN target without look-ahead."""
    generator = TargetGenerator(parquet_store=synthetic_parquet_store)
    universe = pd.DataFrame(
        [
            {"ticker": "AAPL US", "region_bloc": "NA"},
            {"ticker": "SAP GY", "region_bloc": "EU"},
        ]
    )

    # Date at the end of the available 60 days
    late_date = "2024-03-15"
    res = generator.compute_target_at_date(
        as_of_date=late_date,
        universe_df=universe,
        horizon_days=21,
    )

    assert not res.empty
    # When horizon exceeds available dates, target should be NaN
    assert res["target"].isna().all()
    assert res["fwd_return"].isna().all()
