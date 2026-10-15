"""End-to-end integration tests for TransatlanticFeaturePipeline."""

import numpy as np
import pandas as pd

from quant_portfolio.config import ALL_CORE_FEATURES
from quant_portfolio.features.pipeline import TransatlanticFeaturePipeline
from quant_portfolio.storage.schema import ParquetMarketStore, SQLiteStore


def test_feature_pipeline_end_to_end(
    sqlite_store: SQLiteStore,
    parquet_store: ParquetMarketStore,
    sample_universe_df: pd.DataFrame,
) -> None:
    """Verify end-to-end extraction, assembly, and normalization of 15 features."""
    # 1. Populate Universe (has AAPL US, JPM US [financial], SAP GY, SAP US [cross-listed])
    sqlite_store.upsert_universe_pit(sample_universe_df)
    sqlite_store.upsert_universe_pit(sample_universe_df.assign(as_of_date="2024-02-29"))

    # 2. Populate Fundamentals
    funds_records = pd.DataFrame(
        [
            {
                "as_of_date": "2024-01-31",
                "filing_date": "2024-01-15",
                "ticker": "AAPL US",
                "metric": "PE_RATIO",
                "value": 30.0,
            },
            {
                "as_of_date": "2024-01-31",
                "filing_date": "2024-01-15",
                "ticker": "AAPL US",
                "metric": "RETURN_COM_EQY",
                "value": 0.40,
            },
            {
                "as_of_date": "2024-01-31",
                "filing_date": "2024-01-10",
                "ticker": "SAP GY",
                "metric": "PE_RATIO",
                "value": 24.0,
            },
            {
                "as_of_date": "2024-01-31",
                "filing_date": "2024-02-10",  # Future! Look-ahead! Should NOT be selected
                "ticker": "SAP GY",
                "metric": "FREE_CASH_FLOW_YIELD",
                "value": 0.08,
            },
        ]
    )
    # Upsert only records with filing_date <= as_of_date
    valid_funds = funds_records[funds_records["filing_date"] <= funds_records["as_of_date"]]
    sqlite_store.upsert_fundamentals_pit(valid_funds)

    # 3. Populate Parquet Market Data (benchmarks and stocks)
    dt_index = pd.date_range("2023-10-01", "2024-02-29", freq="B")
    n_days = len(dt_index)
    dates = dt_index.strftime("%Y-%m-%d")
    spx_p = 4000.0 * np.cumprod(1.0 + np.full(n_days, 0.0005))
    sxxp_p = 450.0 * np.cumprod(1.0 + np.full(n_days, 0.0003))
    aapl_p = 150.0 * np.cumprod(1.0 + np.full(n_days, 0.001))
    sap_p = 120.0 * np.cumprod(1.0 + np.full(n_days, 0.0008))

    market_records: list[pd.DataFrame] = []
    for t, p, reg in [
        ("SPX Index", spx_p, "NA"),
        ("SXXP Index", sxxp_p, "EU"),
        ("AAPL US", aapl_p, "NA"),
        ("SAP GY", sap_p, "EU"),
    ]:
        market_records.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": t,
                    "region_bloc": reg,
                    "px_last_eur": p,
                    "tot_return_idx_eur": p,
                    "volume_eur": np.full(n_days, 1_000_000.0),
                }
            )
        )
    parquet_store.write_daily_market(pd.concat(market_records, ignore_index=True))

    pipeline = TransatlanticFeaturePipeline(
        sqlite_store=sqlite_store,
        parquet_store=parquet_store,
    )

    as_of = dates[-1]
    features_df = pipeline.build_features_at_date(as_of_date=as_of, normalize=True)

    # Financials (JPM US) must be excluded
    assert "JPM US" not in features_df["ticker"].values

    # Cross-listing deduplication: only 1 SAP ticker (SAP GY) kept
    sap_tickers = [t for t in features_df["ticker"].values if "SAP" in t]
    assert len(sap_tickers) == 1

    # Exactly 2 active stocks: AAPL US and SAP GY
    assert len(features_df) == 2

    # Check presence of all 15 core features
    for feat in ALL_CORE_FEATURES:
        assert feat in features_df.columns
        # None should be NaN after normalization
        assert not features_df[feat].isna().any()

    # Multi-date panel test
    panel = pipeline.build_feature_panel(dates=["2024-01-31", "2024-02-29"], normalize=True)
    assert len(panel) == 4  # 2 stocks * 2 dates
    assert set(panel["as_of_date"].unique()) == {"2024-01-31", "2024-02-29"}
