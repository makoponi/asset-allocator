"""Unit tests for PortfolioAllocator orchestrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.portfolio_allocator import PortfolioAllocator


def test_portfolio_allocator_hrp_pipeline() -> None:
    t_samples = 400
    tickers = [f"EQ_{i:02d}" for i in range(35)]
    rng = np.random.default_rng(77)

    ret_arr = rng.standard_normal((t_samples, len(tickers))) * 0.015
    dates = pd.date_range("2022-01-01", periods=t_samples, freq="B")
    df_returns = pd.DataFrame(ret_arr, index=dates, columns=tickers)

    allocator = PortfolioAllocator(method="hrp", denoise=True, max_weight=0.04)
    weights = allocator.allocate(df_returns, candidate_tickers=tickers)

    assert len(weights) == len(tickers)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-5)
    # Check max weight cap
    assert max(weights.values()) <= 0.04 + 1e-4
    assert min(weights.values()) > 0.0


def test_portfolio_allocator_cvar_pipeline() -> None:
    t_samples = 200
    tickers = [f"STK_{i}" for i in range(25)]
    rng = np.random.default_rng(88)

    ret_arr = rng.standard_normal((t_samples, len(tickers))) * 0.01
    df_returns = pd.DataFrame(ret_arr, columns=tickers)

    allocator = PortfolioAllocator(method="cvar", max_weight=0.05)
    weights = allocator.allocate(df_returns)

    assert len(weights) > 0
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-4)
    assert max(weights.values()) <= 0.05 + 1e-4
