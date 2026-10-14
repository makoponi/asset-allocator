"""End-to-end integration test for Phase 4 feature engineering, target generation, and ranking."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import numpy as np

from src.features.cross_sectional import CrossSectionalTransformer, TargetGenerator
from src.features.pit_aligner import PiTFeatureAligner
from src.features.technical_engine import TechnicalEngine
from src.ingestion.storage import (
    FundamentalPiTRecord,
    MarketStore,
    PriceEODRecord,
    UniverseMemberRecord,
)
from src.models.ml_ranker import MLRanker
from src.models.purged_cv import PurgedGroupTimeSeriesSplit

if TYPE_CHECKING:
    from pathlib import Path


def test_full_phase_4_pipeline_integration(tmp_path: Path) -> None:
    # 1. Setup isolated database
    store = MarketStore(db_path=tmp_path / "integration_market.db")

    eval_dates = [
        date(2023, 1, 31),
        date(2023, 2, 28),
        date(2023, 3, 31),
        date(2023, 4, 30),
        date(2023, 5, 31),
    ]

    tickers = [f"STK_{i:02d}" for i in range(40)]
    sectors = ["Technology", "Financials", "Healthcare", "Industrials"]

    # 2. Populate universe members and fundamentals
    member_records: list[UniverseMemberRecord] = []
    fund_records: list[FundamentalPiTRecord] = []

    for d in eval_dates:
        d_str = d.isoformat()
        for i, t in enumerate(tickers):
            sec = sectors[i % len(sectors)]
            member_records.append(
                UniverseMemberRecord(
                    date=d_str,
                    ticker=t,
                    gics_sector=sec,
                    is_active=1,
                )
            )

    for i, t in enumerate(tickers):
        fund_records.append(
            FundamentalPiTRecord(
                as_of_date="2022-12-31",
                filing_date="2022-12-31",
                ticker=t,
                metric_name="PE_RATIO",
                metric_value=15.0 + (i % 10),
            )
        )
        fund_records.append(
            FundamentalPiTRecord(
                as_of_date="2022-12-31",
                filing_date="2023-01-15",
                ticker=t,
                metric_name="RETURN_COM_EQY",
                metric_value=12.0 + (i % 5),
            )
        )

    store.insert_universe_members(member_records)
    store.insert_fundamentals_pit(fund_records)

    # 3. Generate 300 days of price history
    prices: list[PriceEODRecord] = []
    base_date = date(2022, 1, 1)
    end_date = date(2023, 7, 31)
    curr = base_date
    rng = np.random.default_rng(42)

    px_state = {t: 100.0 + i for i, t in enumerate(tickers)}
    while curr <= end_date:
        if curr.weekday() < 5:  # Business days
            for t in tickers:
                drift = 0.0005 if int(t.split("_")[1]) > 20 else -0.0002
                ret = drift + rng.normal(0, 0.01)
                new_px = px_state[t] * (1.0 + ret)
                px_state[t] = new_px
                prices.append(
                    PriceEODRecord(
                        date=curr,
                        ticker=t,
                        px_last=new_px,
                        volume=10000.0,
                        ret_1d=ret,
                    )
                )
        curr += timedelta(days=1)

    tech_engine = TechnicalEngine(prices)
    aligner = PiTFeatureAligner(market_store=store, technical_engine=tech_engine)

    # 4. Build aligned matrix across dates
    all_rows = []
    for d in eval_dates:
        rows = aligner.build_cross_sectional_matrix(d)
        all_rows.extend(rows)

    assert len(all_rows) > 0

    # 5. Transform features (Z-Score + Winsorization)
    transformer = CrossSectionalTransformer()
    df_raw = transformer.rows_to_dataframe(all_rows)
    df_z = transformer.transform(df_raw)

    assert len(df_z) == len(all_rows)

    # 6. Generate Targets
    tg = TargetGenerator(forward_bars=21)
    target_map = tg.compute_forward_excess_returns(prices, eval_dates)

    # Attach targets
    target_vals: list[float] = []
    valid_indices: list[int] = []
    for idx, r in df_z.iterrows():
        key = (str(r["date"]), str(r["ticker"]))
        if key in target_map:
            target_vals.append(target_map[key])
            valid_indices.append(int(str(idx)))

    df_labeled = df_z.iloc[valid_indices].copy()
    df_labeled["target"] = target_vals

    assert len(df_labeled) > 50

    # 7. Cross-Validation Purged Split
    cv = PurgedGroupTimeSeriesSplit(n_splits=2, embargo_pct=0.05, min_train_splits=1)
    splits = list(cv.split(df_labeled, groups=df_labeled["date"]))
    assert len(splits) > 0

    # 8. Train MLRanker
    train_idx, test_idx = splits[0]
    train_df = df_labeled.iloc[train_idx]
    test_df = df_labeled.iloc[test_idx]

    ranker = MLRanker(model_type="lightgbm", random_state=42)
    ranker.fit(train_df, train_df["target"])

    test_df_eval = test_df.copy()
    test_df_eval["prediction"] = ranker.predict(test_df)
    rank_ic = ranker.compute_rank_ic(test_df, test_df["target"])
    assert isinstance(rank_ic, float)

    # 9. Top Decile Selection on the last date
    last_date_df = df_labeled[df_labeled["date"] == eval_dates[-1].isoformat()]
    top_stocks = ranker.select_top_decile(last_date_df, top_pct=0.10, min_n=5, max_n=10)

    assert len(top_stocks) >= 5
    assert all(isinstance(s, str) for s in top_stocks)
