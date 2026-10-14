"""Unit tests for cross-sectional Z-score transformation and forward target generation."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from src.features.cross_sectional import CrossSectionalTransformer, TargetGenerator
from src.features.pit_aligner import AlignedFeatureRow
from src.ingestion.storage import PriceEODRecord


def test_cross_sectional_transformer_zscore_and_winsorization() -> None:
    # Create synthetic rows with 2 sectors
    rows: list[AlignedFeatureRow] = []

    # Sector Tech: 5 stocks with varying momentum
    for i in range(5):
        rows.append(
            AlignedFeatureRow(
                date="2024-01-31",
                ticker=f"TECH_{i}",
                gics_sector="Technology",
                mom_1m=float(i * 0.1),
                mom_12m=0.2,
                beta_252d=1.0,
                ivol_252d=0.2,
                pe_ratio=20.0 + i,
            )
        )

    # Sector Financials: 30 stocks plus an extreme outlier for PE (e.g. 500.0)
    for i in range(30):
        rows.append(
            AlignedFeatureRow(
                date="2024-01-31",
                ticker=f"FIN_{i}",
                gics_sector="Financials",
                mom_1m=float((i % 5) * 0.05),
                mom_12m=0.1,
                beta_252d=0.9,
                ivol_252d=0.15,
                pe_ratio=10.0 + (i % 3),
            )
        )
    # Outlier
    rows.append(
        AlignedFeatureRow(
            date="2024-01-31",
            ticker="FIN_OUTLIER",
            gics_sector="Financials",
            mom_1m=0.1,
            mom_12m=0.1,
            beta_252d=0.9,
            ivol_252d=0.15,
            pe_ratio=500.0,  # Extreme outlier
        )
    )

    transformer = CrossSectionalTransformer(clip_sigma=3.0)
    df_raw = transformer.rows_to_dataframe(rows)
    df_z = transformer.transform(df_raw)

    assert len(df_z) == 36
    assert "date" in df_z.columns
    assert "ticker" in df_z.columns

    # Verify winsorization bounds: all values must be within [-3.0, 3.0]
    pe_scores = df_z["pe_ratio"].to_numpy()
    assert np.all(pe_scores >= -3.0)
    assert np.all(pe_scores <= 3.0)

    # Outlier must be clipped at exactly 3.0
    outlier_val = df_z.loc[df_z["ticker"] == "FIN_OUTLIER", "pe_ratio"].iloc[0]
    assert outlier_val == pytest.approx(3.0)


def test_cross_sectional_transformer_missing_imputation() -> None:
    # Test imputation of missing fundamental values (None) with sector median
    rows = [
        AlignedFeatureRow(
            date="2024-01-31",
            ticker="STK_A",
            gics_sector="Industrials",
            mom_1m=0.05,
            mom_12m=0.1,
            beta_252d=1.0,
            ivol_252d=0.2,
            prof_margin=10.0,
        ),
        AlignedFeatureRow(
            date="2024-01-31",
            ticker="STK_B",
            gics_sector="Industrials",
            mom_1m=0.06,
            mom_12m=0.12,
            beta_252d=1.1,
            ivol_252d=0.22,
            prof_margin=20.0,
        ),
        AlignedFeatureRow(
            date="2024-01-31",
            ticker="STK_C",
            gics_sector="Industrials",
            mom_1m=0.04,
            mom_12m=0.08,
            beta_252d=0.95,
            ivol_252d=0.18,
            prof_margin=None,  # Missing -> imputed with median (15.0)
        ),
    ]

    transformer = CrossSectionalTransformer()
    df_raw = transformer.rows_to_dataframe(rows)
    df_z = transformer.transform(df_raw)

    # Imputed value 15.0 is the exact median of [10.0, 20.0, 15.0], so its mean is 15.0 and Z is 0.0
    z_c = df_z.loc[df_z["ticker"] == "STK_C", "prof_margin"].iloc[0]
    assert not np.isnan(z_c)
    assert z_c == pytest.approx(0.0, abs=1e-3)


def test_target_generator_forward_excess_returns() -> None:
    eval_d = date(2024, 1, 31)

    # Create 30 days of price records for 3 tickers
    prices: list[PriceEODRecord] = []
    # Ticker A: +10% over 21 days
    # Ticker B: +0% over 21 days
    # Ticker C: -10% over 21 days
    growth_rates = {"STK_A": 0.10, "STK_B": 0.00, "STK_C": -0.10}

    for t_sym, rate in growth_rates.items():
        base_px = 100.0
        for i in range(25):
            cur_d = eval_d + timedelta(days=i)
            # Price on eval_d = base_px; price at day 21 = base_px * (1 + rate)
            frac = i / 21.0 if i <= 21 else 1.0
            px = base_px * (1.0 + frac * rate)
            prices.append(
                PriceEODRecord(
                    date=cur_d,
                    ticker=t_sym,
                    px_last=px,
                    volume=1000.0,
                    ret_1d=0.001,
                )
            )

    tg = TargetGenerator(forward_bars=21)
    targets = tg.compute_forward_excess_returns(prices, eval_dates=[eval_d])

    assert len(targets) == 3
    # Active median return is 0.0 (STK_B)
    # Excess return for STK_A should be +0.10
    # Excess return for STK_B should be 0.0
    # Excess return for STK_C should be -0.10
    eval_str = eval_d.isoformat()
    assert targets[(eval_str, "STK_A")] == pytest.approx(0.10, abs=1e-3)
    assert targets[(eval_str, "STK_B")] == pytest.approx(0.00, abs=1e-3)
    assert targets[(eval_str, "STK_C")] == pytest.approx(-0.10, abs=1e-3)
