"""Technical factor computation engine: 100% local calculation on partitioned Parquet price store.

Computes Momentum 1m, Momentum 12m (ex-1m), CAPM Beta (252d), and Idiosyncratic Volatility (252d)
against the STOXX Europe 600 benchmark without any external Bloomberg API calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from src.ingestion.storage import PriceEODRecord


@dataclass(frozen=True, slots=True)
class TechnicalFactors:
    """Computed technical factor metrics for a security at an evaluation date."""

    date: date
    ticker: str
    mom_1m: float
    mom_12m: float
    beta_252d: float
    ivol_252d: float


def compute_ols_beta_and_ivol(
    stock_returns: Sequence[float],
    benchmark_returns: Sequence[float],
) -> tuple[float, float]:
    """Calculate CAPM OLS Beta and residual standard deviation (Idiosyncratic Volatility).

    Model: r_stock = alpha + beta * r_benchmark + epsilon.
    Requires at least 60 observations to return valid metrics; otherwise returns (1.0, 0.0).
    """
    n = len(stock_returns)
    if n != len(benchmark_returns) or n < 60:
        return 1.0, 0.0

    mean_x = sum(benchmark_returns) / n
    mean_y = sum(stock_returns) / n

    cov_xy = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(benchmark_returns, stock_returns, strict=True)
    )
    var_x = sum((x - mean_x) ** 2 for x in benchmark_returns)

    if var_x <= 1e-12:
        return 1.0, 0.0

    beta = cov_xy / var_x
    alpha = mean_y - beta * mean_x

    # Residuals: epsilon = y - (alpha + beta * x)
    residuals = [
        y - (alpha + beta * x) for x, y in zip(benchmark_returns, stock_returns, strict=True)
    ]
    residual_variance = sum(r**2 for r in residuals) / max(1, n - 2)
    ivol = math.sqrt(residual_variance)

    # Annualize daily idiosyncratic volatility (sqrt(252))
    ivol_annualized = ivol * math.sqrt(252.0)

    return beta, ivol_annualized


class TechnicalEngine:
    """Computes technical factors across all securities using local Parquet data."""

    def __init__(self, price_records: Sequence[PriceEODRecord]) -> None:
        # Group and index by ticker, sorted by date
        self._records_by_ticker: dict[str, list[PriceEODRecord]] = {}
        for r in price_records:
            self._records_by_ticker.setdefault(r.ticker, []).append(r)

        for _ticker, records in self._records_by_ticker.items():
            records.sort(key=lambda x: x.date)

    def compute_factors_for_ticker(
        self,
        ticker: str,
        as_of_date: date,
        benchmark_returns: Sequence[float] | None = None,
    ) -> TechnicalFactors | None:
        """Compute 4 core technical factors for a ticker as of a specific date."""
        history = self._records_by_ticker.get(ticker)
        if not history:
            return None

        # Filter records up to as_of_date
        past_records = [r for r in history if r.date <= as_of_date]
        if len(past_records) < 252:
            return None

        px_t = past_records[-1].px_last
        px_21 = past_records[-22].px_last if len(past_records) >= 22 else past_records[0].px_last
        px_252 = past_records[-252].px_last if len(past_records) >= 252 else past_records[0].px_last

        # Momentum 1m: ln(P_t / P_{t-21})
        mom_1m = math.log(px_t / px_21) if px_21 > 0 else 0.0

        # Momentum 12m ex 1m: ln(P_{t-21} / P_{t-252})
        mom_12m = math.log(px_21 / px_252) if px_252 > 0 else 0.0

        # Returns slice of last 252 days for Beta and IVOL
        window_252 = past_records[-252:]
        stock_returns = [r.ret_1d for r in window_252]

        if benchmark_returns is not None and len(benchmark_returns) == len(stock_returns):
            bench_ret = benchmark_returns
        else:
            # Fallback: cross-sectional proxy (neutral beta)
            bench_ret = stock_returns

        beta, ivol = compute_ols_beta_and_ivol(stock_returns, bench_ret)

        return TechnicalFactors(
            date=as_of_date,
            ticker=ticker,
            mom_1m=mom_1m,
            mom_12m=mom_12m,
            beta_252d=beta,
            ivol_252d=ivol,
        )
