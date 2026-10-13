"""Tests for technical factor calculations (Momentum, CAPM Beta, and Idiosyncratic Volatility)."""

import math
from datetime import date, timedelta

from src.features.technical_engine import (
    TechnicalEngine,
    compute_ols_beta_and_ivol,
)
from src.ingestion.storage import PriceEODRecord


def test_compute_ols_beta_and_ivol_analytical() -> None:
    # Synthetic series where stock = 1.5 * benchmark + noise
    n = 252
    bench = [0.01 * math.sin(i / 10.0) for i in range(n)]
    stock = [1.5 * b + 0.002 * (1 if i % 2 == 0 else -1) for i, b in enumerate(bench)]

    beta, ivol = compute_ols_beta_and_ivol(stock, bench)

    # Beta must be approximately 1.5
    assert abs(beta - 1.5) < 0.05
    # IVOL must be strictly positive and finite
    assert ivol > 0.0


def test_compute_ols_beta_and_ivol_perfect_correlation() -> None:
    n = 252
    bench = [0.005 * (i % 5 - 2) for i in range(n)]
    stock = [2.0 * b for b in bench]

    beta, ivol = compute_ols_beta_and_ivol(stock, bench)

    assert abs(beta - 2.0) < 1e-6
    # With perfect correlation, residual variance is zero
    assert ivol < 1e-5


def test_technical_engine_momentum_and_window() -> None:
    start_d = date(2024, 1, 1)
    records: list[PriceEODRecord] = []

    # 300 business days of simulated prices growing linearly
    px = 100.0
    for i in range(300):
        cur_date = start_d + timedelta(days=i)
        # Price increases by 0.5 per day
        px_next = px + 0.5
        ret = (px_next / px) - 1.0
        records.append(
            PriceEODRecord(
                date=cur_date,
                ticker="TEST GY",
                px_last=px_next,
                volume=10000.0,
                ret_1d=ret,
            )
        )
        px = px_next

    engine = TechnicalEngine(records)
    eval_date = start_d + timedelta(days=299)
    factors = engine.compute_factors_for_ticker("TEST GY", eval_date)

    assert factors is not None
    assert factors.ticker == "TEST GY"
    assert factors.date == eval_date
    # Upward trending series has strictly positive momentums
    assert factors.mom_1m > 0.0
    assert factors.mom_12m > 0.0


def test_technical_engine_insufficient_history() -> None:
    start_d = date(2024, 1, 1)
    records = [
        PriceEODRecord(
            date=start_d + timedelta(days=i),
            ticker="NEW GY",
            px_last=50.0,
            volume=1000.0,
            ret_1d=0.0,
        )
        for i in range(100)  # Only 100 days < 252 required
    ]

    engine = TechnicalEngine(records)
    factors = engine.compute_factors_for_ticker("NEW GY", start_d + timedelta(days=99))
    assert factors is None
