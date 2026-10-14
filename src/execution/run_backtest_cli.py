"""CLI per l'esecuzione del backtesting storico walk-forward della metodologia di asset allocation.

Simula giorno per giorno la strategia (ML Ranker + RMT Denoising + HRP) contro il benchmark,
modellando i costi di transazione e il drift dei pesi.
"""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from src.execution.backtest_engine import BacktestEngine
from src.features.technical_engine import TechnicalEngine
from src.ingestion.storage import MarketStore, PriceEODRecord

PARQUET_DIR = Path("data/processed")


def load_price_records(start_year: int) -> list[PriceEODRecord]:
    """Load price records from Parquet for years >= start_year."""
    dataset = ds.dataset(str(PARQUET_DIR), format="parquet", partitioning=["year"])
    table = dataset.to_table(filter=ds.field("year") >= str(start_year))

    dates = table["date"].to_pylist()
    tickers = table["ticker"].to_pylist()
    pxs = table["px_last"].to_pylist()
    vols = table["volume"].to_pylist()
    rets = table["ret_1d"].to_pylist()

    records: list[PriceEODRecord] = []
    for d, t, p, v, r in zip(dates, tickers, pxs, vols, rets, strict=True):
        records.append(
            PriceEODRecord(
                date=d,
                ticker=t,
                px_last=float(p),
                volume=float(v),
                ret_1d=float(r),
            )
        )
    return records


def run_cli_backtest(
    start_date_str: str,
    end_date_str: str,
    retrain_freq_months: int = 6,
    cost_bps: float = 10.0,
    top_n: int = 50,
    max_weight: float = 0.04,
    enable_overlay: bool = False,
) -> None:
    start_d = date.fromisoformat(start_date_str)
    end_d = date.fromisoformat(end_date_str)

    overlay_desc = (
        "ATTIVO (Gaussian HMM 3-Stati + Isteresi 3d)"
        if enable_overlay
        else "DISATTIVATO (Long-Only Puro)"
    )
    print("==================================================")
    print("BACKTESTING WALK-FORWARD: ASSET ALLOCATOR ENGINE")
    print(f"Orizzonte: {start_date_str} -> {end_date_str}")
    print(f"Hedging Overlay Macro: {overlay_desc}")
    print(f"Frequenza Riaddestramento: ogni {retrain_freq_months} mesi")
    print(f"Costi di Transazione: {cost_bps:.1f} bps (fee + slippage)")
    print(f"Portafoglio: Top {top_n} titoli | Max Weight: {max_weight * 100:.1f}%")
    print("==================================================")

    t0 = time.time()
    store = MarketStore()

    hedging_overlay = None
    if enable_overlay:
        from src.risk.hedging_overlay import HedgingOverlay

        hedging_overlay = HedgingOverlay(
            hysteresis_days=3,
            threshold_prob=0.65,
            hedge_ratio_state_0=0.0,
            hedge_ratio_state_1=0.40,
            hedge_ratio_state_2=0.90,
        )

    # Lookback 2 years prior to start_date for 504d covariance matrix
    lookback_year = max(2011, start_d.year - 2)
    print(f"Caricamento prezzi da Parquet a partire dall'anno {lookback_year}...")
    prices = load_price_records(lookback_year)
    print(f"Caricate {len(prices):,} osservazioni di prezzo.")

    tech_engine = TechnicalEngine(prices)

    engine = BacktestEngine(
        market_store=store,
        tech_engine=tech_engine,
        price_records=prices,
        retrain_freq_months=retrain_freq_months,
        cost_bps=cost_bps,
        top_n=top_n,
        max_weight=max_weight,
        hedging_overlay=hedging_overlay,
    )

    print("Avvio simulazione walk-forward giornaliera...")
    res = engine.run_simulation(start_d, end_d, initial_capital=100.0)
    elapsed = time.time() - t0
    print(f"Simulazione completata in {elapsed:.2f}s.\n")


    p = res.metrics
    b = res.benchmark_metrics

    # 1. Print Summary Performance Table
    print("=" * 80)
    print(f"{'METRICA DI PERFORMANCE':<35} | {'STRATEGIA (Net)':<18} | {'BENCHMARK':<18}")
    print("-" * 80)
    print(f"{'Total Return':<32} | {p.total_return * 100:>14.2f}% | {b.total_return * 100:>14.2f}%")
    print(f"{'CAGR (Rendimento Ann.)':<32} | {p.cagr * 100:>14.2f}% | {b.cagr * 100:>14.2f}%")
    print(
        f"{'Volatilita Ann.':<32} | {p.annualized_volatility * 100:>14.2f}% | "
        f"{b.annualized_volatility * 100:>14.2f}%"
    )
    print(f"{'Sharpe Ratio':<32} | {p.sharpe_ratio:>15.2f} | {b.sharpe_ratio:>15.2f}")
    print(
        f"{'Max Drawdown':<32} | {p.max_drawdown * 100:>14.2f}% | "
        f"{b.max_drawdown * 100:>14.2f}%"
    )
    print(f"{'Calmar Ratio':<32} | {p.calmar_ratio:>15.2f} | {b.calmar_ratio:>15.2f}")
    print("-" * 80)
    alpha_str = f"{p.alpha * 100:+.2f}%" if p.alpha is not None else "N/A"
    beta_str = f"{p.beta:.2f}" if p.beta is not None else "N/A"
    te_str = f"{p.tracking_error * 100:.2f}%" if p.tracking_error is not None else "N/A"
    ir_str = f"{p.information_ratio:+.2f}" if p.information_ratio is not None else "N/A"
    print(f"{'Beta vs Benchmark':<32} | {beta_str:>15} | {'1.00':>15}")
    print(f"{'Jensen Alpha (Ann.)':<32} | {alpha_str:>15} | {'0.00%':>15}")
    print(f"{'Tracking Error':<32} | {te_str:>15} | {'0.00%':>15}")
    print(f"{'Information Ratio':<32} | {ir_str:>15} | {'0.00':>15}")
    print(f"{'Turnover Ann.':<32} | {p.annualized_turnover * 100:>14.1f}% | {'0.0%':>15}")
    print("=" * 80)

    # 2. Print Annual Breakdown Table
    df = res.history_df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year

    print("\n" + "=" * 80)
    print(
        f"{'ANNO':<6} | {'STRATEGIA':<12} | {'BENCHMARK':<12} | {'EXCESS RET':<12} | {'MAX DD':<12}"
    )
    print("-" * 80)

    for y, g in df.groupby("year"):
        p_ret_series = np.asarray(g["portfolio_ret"].astype(float).to_numpy(), dtype=float)
        b_ret_series = np.asarray(g["benchmark_ret"].astype(float).to_numpy(), dtype=float)

        p_y = float(np.prod(1.0 + p_ret_series) - 1.0)
        b_y = float(np.prod(1.0 + b_ret_series) - 1.0)
        diff_y = p_y - b_y

        nav_y = np.cumprod(1.0 + p_ret_series)
        peak_y = np.maximum.accumulate(nav_y)
        dd_y: np.ndarray[Any, Any] = (nav_y - peak_y) / peak_y
        max_dd_y = float(np.min(dd_y))

        print(
            f"{y:<8} | {p_y * 100:>+12.2f}% | {b_y * 100:>+12.2f}% | "
            f"{diff_y * 100:>+12.2f}% | {max_dd_y * 100:>12.2f}%"
        )
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Esecuzione del backtesting walk-forward della metodologia quantitativa."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2018-01-01",
        help="Data inizio YYYY-MM-DD (default: 2018-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-09-30",
        help="Data fine YYYY-MM-DD (default: 2024-09-30)",
    )
    parser.add_argument(
        "--retrain-freq-months",
        type=int,
        default=6,
        help="Frequenza riaddestramento ranker in mesi (default: 6)",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help="Attriti di transazione totali in basis point (default: 10.0 bps)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Numero titoli selezionati (default: 50)",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.04,
        help="Peso massimo per singolo titolo (default: 0.04)",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        default=False,
        help="Attiva copertura macroregime dinamica con future su indici e isteresi a 3 giorni.",
    )
    args = parser.parse_args()

    run_cli_backtest(
        start_date_str=args.start_date,
        end_date_str=args.end_date,
        retrain_freq_months=args.retrain_freq_months,
        cost_bps=args.cost_bps,
        top_n=args.top_n,
        max_weight=args.max_weight,
        enable_overlay=args.overlay,
    )



if __name__ == "__main__":
    main()
