import numpy as np
import pandas as pd

from quant_portfolio.config import FeatureConfig
from quant_portfolio.features.technical import TechnicalFeatureCalculator
from quant_portfolio.storage.schema import ParquetMarketStore


def test_technical_features_synthetic_analytical(parquet_store: ParquetMarketStore) -> None:
    """Verify analytical convergence of MOM, BETA, IDIO_VOL and AMIHUD with synthetic data."""
    np.random.seed(42)
    n_days = 300
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B").strftime("%Y-%m-%d")

    # Benchmark: SPX Index
    bmk_rets = np.random.normal(0.0005, 0.01, size=n_days)
    bmk_prices = 1000.0 * np.cumprod(1.0 + bmk_rets)

    # Stock A: Beta = 1.5, exact idio noise sigma = 0.005
    noise = np.random.normal(0.0, 0.005, size=n_days)
    stock_rets = 1.5 * bmk_rets + noise
    stock_prices = 100.0 * np.cumprod(1.0 + stock_rets)
    stock_vols = np.full(n_days, 1_000_000.0)

    # Store benchmark and stock in Parquet
    bmk_df = pd.DataFrame(
        {
            "date": dates,
            "ticker": "SPX Index",
            "region_bloc": "NA",
            "px_last_eur": bmk_prices,
            "tot_return_idx_eur": bmk_prices,
            "volume_eur": np.full(n_days, 1e9),
        }
    )
    stock_df = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL US",
            "region_bloc": "NA",
            "px_last_eur": stock_prices,
            "tot_return_idx_eur": stock_prices,
            "volume_eur": stock_vols,
        }
    )

    parquet_store.write_daily_market(pd.concat([bmk_df, stock_df], ignore_index=True))

    calc = TechnicalFeatureCalculator(parquet_store=parquet_store)
    as_of = dates[-1]
    res = calc.compute_features_at_date(
        as_of_date=as_of,
        tickers=["AAPL US"],
        region_map={"AAPL US": "NA"},
    )

    assert len(res) == 1
    row = res.iloc[0]

    # Verify Beta converges to ~1.5
    assert abs(row["BETA_REGIONAL_252D"] - 1.5) < 0.08

    # Verify IDIO_VOL_252D converges to 0.005 * sqrt(252) ~= 0.079
    expected_ivol = 0.005 * np.sqrt(252.0)
    assert abs(row["IDIO_VOL_252D"] - expected_ivol) < 0.02

    # Verify MOM_1M matches exact price ratio over 21 days
    expected_mom_1m = (stock_prices[-1] / stock_prices[-22]) - 1.0
    assert abs(row["MOM_1M"] - expected_mom_1m) < 1e-6

    # Verify MOM_12M_1M matches exact price ratio over 252 to 21 days
    expected_mom_12m_1m = (stock_prices[-22] / stock_prices[-253]) - 1.0
    assert abs(row["MOM_12M_1M"] - expected_mom_12m_1m) < 1e-6

    # Verify Amihud illiquidity is positive and non-zero
    assert row["AMIHUD_ILLIQUIDITY_21D"] > 0.0


def test_technical_features_zero_look_ahead_bias(parquet_store: ParquetMarketStore) -> None:
    """Verify that future market prices after as_of_date do not change calculated features."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B").strftime("%Y-%m-%d")

    # Initial data up to day 50
    p1 = 100.0 * np.cumprod(1.0 + np.full(50, 0.001))
    df_initial = pd.DataFrame(
        {
            "date": dates[:50],
            "ticker": "SAP GY",
            "region_bloc": "EU",
            "px_last_eur": p1,
            "tot_return_idx_eur": p1,
            "volume_eur": np.full(50, 500_000.0),
        }
    )
    parquet_store.write_daily_market(df_initial)

    calc = TechnicalFeatureCalculator(parquet_store=parquet_store)
    as_of = dates[49]
    res_before = calc.compute_features_at_date(
        as_of_date=as_of,
        tickers=["SAP GY"],
        region_map={"SAP GY": "EU"},
    )

    # Now append dramatic future spike on day 51..100
    p2 = 100.0 * np.cumprod(1.0 + np.full(50, 0.10))  # 10% daily jump!
    df_future = pd.DataFrame(
        {
            "date": dates[50:],
            "ticker": "SAP GY",
            "region_bloc": "EU",
            "px_last_eur": p2,
            "tot_return_idx_eur": p2,
            "volume_eur": np.full(50, 500_000.0),
        }
    )
    parquet_store.write_daily_market(df_future)

    # Recompute strictly as of day 50
    res_after = calc.compute_features_at_date(
        as_of_date=as_of,
        tickers=["SAP GY"],
        region_map={"SAP GY": "EU"},
    )

    assert res_before.iloc[0]["MOM_1M"] == res_after.iloc[0]["MOM_1M"]
    val_before = res_before.iloc[0]["AMIHUD_ILLIQUIDITY_21D"]
    val_after = res_after.iloc[0]["AMIHUD_ILLIQUIDITY_21D"]
    assert val_before == val_after


def test_technical_features_fallback_on_insufficient_history(
    parquet_store: ParquetMarketStore,
) -> None:
    """Verify fallback to neutral defaults when history is short or missing."""
    calc = TechnicalFeatureCalculator(
        parquet_store=parquet_store, config=FeatureConfig(min_history_days_beta=63)
    )

    res = calc.compute_features_at_date(
        as_of_date="2024-01-31",
        tickers=["NEW_IPO US"],
        region_map={"NEW_IPO US": "NA"},
    )

    assert len(res) == 1
    row = res.iloc[0]
    assert row["BETA_REGIONAL_252D"] == 1.0
    assert row["IDIO_VOL_252D"] == 0.20
    assert row["MOM_1M"] == 0.0
    assert row["MOM_12M_1M"] == 0.0
    assert row["AMIHUD_ILLIQUIDITY_21D"] == 0.0
