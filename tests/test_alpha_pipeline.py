"""End-to-end integration tests for AlphaPipeline (Phase 4)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.alpha.pipeline import AlphaPipeline
from quant_portfolio.config import PortfolioConfig
from quant_portfolio.storage.schema import ParquetMarketStore, SQLiteStore


@pytest.fixture
def synthetic_alpha_env(tmp_path: Path) -> tuple[PortfolioConfig, SQLiteStore, ParquetMarketStore]:
    """Build synthetic SQLite and Parquet environment with 20 stocks over 80 days."""
    db_file = tmp_path / "quant_store.db"
    p_dir = tmp_path / "parquet"

    store = SQLiteStore(db_file)
    p_store = ParquetMarketStore(p_dir)

    cfg = PortfolioConfig()
    cfg.paths.db_path = db_file
    cfg.paths.parquet_dir = p_dir

    # Generate dates: 4 monthly rebalancing dates
    dates = pd.bdate_range("2024-01-01", periods=80).strftime("%Y-%m-%d").tolist()

    # 10 NA stocks and 10 EU stocks
    na_tickers = [f"NA_{i:02d} US" for i in range(10)]
    eu_tickers = [f"EU_{i:02d} GY" for i in range(10)]
    all_tickers = na_tickers + eu_tickers

    # 1. Populate universe_pit
    uni_records: list[dict[str, object]] = []
    for d in dates[::20]:  # snapshot every 20 days
        for t in all_tickers:
            reg = "NA" if "US" in t else "EU"
            uni_records.append(
                {
                    "as_of_date": d,
                    "ticker": t,
                    "isin": f"ISIN_{t[:5]}",
                    "region_bloc": reg,
                    "source_index": "SPX Index" if reg == "NA" else "SXXP Index",
                    "gics_sector": 45,  # Information Technology (non-financial)
                    "local_currency": "USD" if reg == "NA" else "EUR",
                    "is_financial": False,
                }
            )
    store.upsert_universe_pit(uni_records)

    # 2. Populate fundamentals_pit
    fund_records: list[dict[str, object]] = []
    for t in all_tickers:
        for metric in [
            "PE_RATIO",
            "PX_TO_BOOK_RATIO",
            "EV_TO_T12M_EBITDA",
            "FREE_CASH_FLOW_YIELD",
            "RETURN_COM_EQY",
            "OPER_MARGIN",
            "ASSET_TURNOVER",
            "TOT_DEBT_TO_TOT_EQY",
            "SALES_REV_TURN_GROWTH_1YR",
            "ACCRUALS_RATIO",
        ]:
            fund_records.append(
                {
                    "as_of_date": "2023-12-31",
                    "filing_date": "2023-12-31",
                    "ticker": t,
                    "metric": metric,
                    "value": float(np.random.uniform(5.0, 50.0)),
                }
            )
    store.upsert_fundamentals_pit(fund_records)

    # 3. Populate market parquet store
    market_records: list[dict[str, object]] = []
    for t_idx, d in enumerate(dates):
        for i, t in enumerate(all_tickers):
            reg = "NA" if "US" in t else "EU"
            # Give higher growth to tickers with higher index i
            growth = 1.0 + (0.0005 * (i + 1))
            px = 100.0 * (growth**t_idx)
            market_records.append(
                {
                    "date": d,
                    "ticker": t,
                    "region_bloc": reg,
                    "px_last_eur": px,
                    "tot_return_idx_eur": px,
                    "volume_eur": 5_000_000.0,
                }
            )
    p_store.write_daily_market(pd.DataFrame(market_records))

    return cfg, store, p_store


def test_alpha_pipeline_end_to_end(
    synthetic_alpha_env: tuple[PortfolioConfig, SQLiteStore, ParquetMarketStore],
) -> None:
    """Verify AlphaPipeline builds panel, trains ensemble, and selects candidates."""
    cfg, store, p_store = synthetic_alpha_env

    pipeline = AlphaPipeline(
        config=cfg,
        sqlite_store=store,
        parquet_store=p_store,
    )

    # Dates for training and ranking
    dates = pd.bdate_range("2024-01-01", periods=80).strftime("%Y-%m-%d").tolist()
    rebalance_dates = dates[::20]  # 4 dates

    # 1. Run Walk-Forward evaluation
    wf_results = pipeline.run_walk_forward_evaluation(
        dates=rebalance_dates,
        n_splits=2,
    )

    assert "mean_ic" in wf_results
    assert "fold_metrics" in wf_results
    assert len(wf_results["fold_metrics"]) > 0

    # 2. Select candidates at target date
    target_date = rebalance_dates[-1]
    train_dates = rebalance_dates[:-1]

    candidates = pipeline.rank_and_select_at_date(
        as_of_date=target_date,
        training_dates=train_dates,
        top_na=5,
        top_eu=5,
    )

    assert not candidates.empty
    assert len(candidates) == 10
    assert len(candidates[candidates["region_bloc"] == "NA"]) == 5
    assert len(candidates[candidates["region_bloc"] == "EU"]) == 5
    assert "score" in candidates.columns
    assert "regional_rank" in candidates.columns
    assert "is_selected" in candidates.columns


def test_alpha_pipeline_score_and_rank_universe(
    synthetic_alpha_env: tuple[PortfolioConfig, SQLiteStore, ParquetMarketStore],
) -> None:
    """Verify score_and_rank_universe produces full ranking and auto-resolves historical dates."""
    cfg, store, p_store = synthetic_alpha_env

    pipeline = AlphaPipeline(
        config=cfg,
        sqlite_store=store,
        parquet_store=p_store,
    )

    dates = pd.bdate_range("2024-01-01", periods=80).strftime("%Y-%m-%d").tolist()
    target_date = dates[::20][-1]

    # Model is not fitted yet, calling score_and_rank_universe with training_dates=None
    ranked = pipeline.score_and_rank_universe(as_of_date=target_date)
    assert not ranked.empty
    assert len(ranked) == 20  # All 20 stocks in active universe
    assert "regional_decile" in ranked.columns
    assert "regional_rank" in ranked.columns
    assert "score" in ranked.columns
    assert "is_selected" in ranked.columns

    # Verify rank_and_select_at_date also works seamlessly with training_dates=None
    selected = pipeline.rank_and_select_at_date(as_of_date=target_date, top_na=3, top_eu=3)
    assert len(selected) == 6
