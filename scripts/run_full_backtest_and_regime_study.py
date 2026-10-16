"""Execute full 2015-2026 Walk-Forward backtest and comprehensive HMM dynamic filters study."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from quant_portfolio.backtest.engine import WalkForwardBacktestEngine
from quant_portfolio.backtest.metrics import build_full_performance_report
from quant_portfolio.backtest.report import export_report_to_dataframes
from quant_portfolio.config import PortfolioConfig
from quant_portfolio.storage.schema import ParquetMarketStore, SQLiteStore


def extract_regime_episodes(
    daily_df: pd.DataFrame,
    regime_col: str,
    region_label: str,
) -> pd.DataFrame:
    """Extract contiguous active hedging episodes (state > 0) with duration and PnL metrics."""
    records: list[dict[str, Any]] = []
    in_episode = False
    start_date: str = ""
    start_idx = 0
    max_state = 0

    dates = daily_df["date"].tolist()
    states = daily_df[regime_col].tolist()
    unhedged_navs = daily_df["nav_unhedged"].tolist()
    hedged_navs = daily_df["nav_hedged"].tolist()
    futures_pnls = daily_df["futures_daily_pnl_eur"].tolist()

    for idx, (d_str, s_val) in enumerate(zip(dates, states, strict=False)):
        s_int = int(s_val) if not pd.isna(s_val) else 0

        if s_int > 0 and not in_episode:
            # Episode started
            in_episode = True
            start_date = str(d_str)
            start_idx = idx
            max_state = s_int
        elif s_int > 0 and in_episode:
            # Episode continuing
            if s_int > max_state:
                max_state = s_int
        elif s_int == 0 and in_episode:
            # Episode ended
            end_date = str(dates[idx - 1])
            duration = idx - start_idx
            u_ret = (unhedged_navs[idx - 1] - unhedged_navs[start_idx]) / unhedged_navs[start_idx]
            h_ret = (hedged_navs[idx - 1] - hedged_navs[start_idx]) / hedged_navs[start_idx]
            cum_fut_pnl = sum(futures_pnls[start_idx:idx])

            state_desc = "Alert (50% Beta)" if max_state == 1 else "Severe Stress (100% Beta)"
            records.append(
                {
                    "region": region_label,
                    "start_date": start_date,
                    "end_date": end_date,
                    "duration_days": duration,
                    "max_state": max_state,
                    "regime_name": state_desc,
                    "unhedged_ret_pct": u_ret * 100.0,
                    "hedged_ret_pct": h_ret * 100.0,
                    "delta_alpha_pct": (h_ret - u_ret) * 100.0,
                    "futures_overlay_pnl_eur": cum_fut_pnl,
                }
            )
            in_episode = False
            max_state = 0

    # If episode open at end of history
    if in_episode:
        end_date = str(dates[-1])
        duration = len(dates) - start_idx
        u_ret = (unhedged_navs[-1] - unhedged_navs[start_idx]) / unhedged_navs[start_idx]
        h_ret = (hedged_navs[-1] - hedged_navs[start_idx]) / hedged_navs[start_idx]
        cum_fut_pnl = sum(futures_pnls[start_idx:])
        state_desc = "Alert (50% Beta)" if max_state == 1 else "Severe Stress (100% Beta)"
        records.append(
            {
                "region": region_label,
                "start_date": start_date,
                "end_date": end_date,
                "duration_days": duration,
                "max_state": max_state,
                "regime_name": state_desc,
                "unhedged_ret_pct": u_ret * 100.0,
                "hedged_ret_pct": h_ret * 100.0,
                "delta_alpha_pct": (h_ret - u_ret) * 100.0,
                "futures_overlay_pnl_eur": cum_fut_pnl,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    """Run simulation and output detailed quantitative analysis."""
    print("=" * 78)
    print("  TRANSATLANTIC QUANT PORTFOLIO — FULL WALK-FORWARD BACKTEST & HMM STUDY")
    print("=" * 78)

    cfg = PortfolioConfig()
    paths = cfg.paths.resolve_paths()
    sqlite_store = SQLiteStore(paths.db_path)
    parquet_store = ParquetMarketStore(paths.parquet_dir)

    out_dir = paths.root_dir / "data" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = "2015-01-02"
    end_date = "2026-10-15"
    initial_capital = 10_000_000.0

    print(f"Simulation Range:    {start_date} to {end_date} (~11.8 Years)")
    print(f"Initial Capital:     EUR {initial_capital:,.2f}")
    print("Optimization:        Hierarchical Risk Parity (HRP) on RMT Covariance")
    print("Dynamic Hedging:     Dual-Sleeve HMM + 3-Day Inertia Hysteresis Filter")
    print(f"Artifacts Target:    {out_dir}")
    print("-" * 78)

    cached_daily = out_dir / "portfolio_daily_history.csv"
    cached_rebal = out_dir / "rebalance_log.csv"

    if cached_daily.exists() and cached_rebal.exists():
        print("[Notice] Using existing backtest history from data/backtest/...")
        daily_df = pd.read_csv(cached_daily)
        rebal_df = pd.read_csv(cached_rebal)
        report = build_full_performance_report(
            daily_history_df=daily_df,
            total_rebalances=len(rebal_df),
            config=cfg,
        )
    else:
        t0 = time.time()
        engine = WalkForwardBacktestEngine(
            sqlite_store=sqlite_store,
            parquet_store=parquet_store,
            config=cfg,
        )

        daily_df, report, rebal_df = engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            allocation_method="hrp",
            initial_capital_eur=initial_capital,
            enable_hedging=True,
            rebalance_freq_days=21,
        )
        elapsed = time.time() - t0
        print(
            f"\n[Completed] Simulation executed in {elapsed:.1f}s ({len(daily_df)} trading days)."
        )

    # 1. Export standard artifacts
    dfs = export_report_to_dataframes(report, daily_history_df=daily_df)
    dfs["summary"].to_csv(out_dir / "backtest_summary.csv", index=False)
    dfs["attribution"].to_csv(out_dir / "hedging_attribution.csv", index=False)
    dfs["sleeves"].to_csv(out_dir / "sleeve_metrics.csv", index=False)
    daily_df.to_csv(out_dir / "portfolio_daily_history.csv", index=False)
    rebal_df.to_csv(out_dir / "rebalance_log.csv", index=False)

    # 2. HMM Regime Episodes Analysis
    na_episodes = extract_regime_episodes(daily_df, "na_regime", "NA (S&P 500 / ES)")
    eu_episodes = extract_regime_episodes(daily_df, "eu_regime", "EU (STOXX 600 / FXXP)")
    all_episodes = pd.concat([na_episodes, eu_episodes], ignore_index=True)
    all_episodes = all_episodes.sort_values(by="start_date").reset_index(drop=True)
    all_episodes.to_csv(out_dir / "hmm_regime_episodes.csv", index=False)

    # 3. Print Tear Sheet
    u = report.unhedged_metrics
    h = report.hedged_metrics
    att = report.overlay_attribution
    u_final = float(daily_df["nav_unhedged"].iloc[-1])
    h_final = float(daily_df["nav_hedged"].iloc[-1])

    print("\n" + "=" * 78)
    print("  EXECUTIVE QUANTITATIVE PERFORMANCE SUMMARY (2015 - 2026)")
    print("=" * 78)
    print(f"{'Performance Metric':<32} | {'Unhedged Long-Only':<20} | {'Dual-Sleeve Hedged':<20}")
    print("-" * 78)
    ret_u_str = f"{u.total_return * 100:>18.2f}%"
    ret_h_str = f"{h.total_return * 100:>18.2f}%"
    print(f"{'Total Return (NAV)':<32} | {ret_u_str} | {ret_h_str}")
    print(f"{'Final Capital (EUR)':<32} | EUR {u_final:>14,.2f} | EUR {h_final:>14,.2f}")
    print(f"{'CAGR (Annualized Return)':<32} | {u.cagr * 100:>18.2f}% | {h.cagr * 100:>18.2f}%")
    vol_u_str = f"{u.annualized_volatility * 100:>18.2f}%"
    vol_h_str = f"{h.annualized_volatility * 100:>18.2f}%"
    print(f"{'Annualized Volatility':<32} | {vol_u_str} | {vol_h_str}")
    print(f"{'Sharpe Ratio (Rf=0%)':<32} | {u.sharpe_ratio:>18.2f}  | {h.sharpe_ratio:>18.2f} ")
    print(f"{'Sortino Ratio':<32} | {u.sortino_ratio:>18.2f}  | {h.sortino_ratio:>18.2f} ")
    dd_u_str = f"{u.max_drawdown * 100:>18.2f}%"
    dd_h_str = f"{h.max_drawdown * 100:>18.2f}%"
    print(f"{'Maximum Drawdown':<32} | {dd_u_str} | {dd_h_str}")
    cal_u_str = f"{u.calmar_ratio:>18.2f}"
    cal_h_str = f"{h.calmar_ratio:>18.2f}"
    print(f"{'Calmar Ratio (CAGR/MaxDD)':<32} | {cal_u_str}  | {cal_h_str} ")
    print(f"{'Win Rate (Daily)':<32} | {u.win_rate * 100:>18.2f}% | {h.win_rate * 100:>18.2f}%")
    print(f"{'Profit Factor':<32} | {u.profit_factor:>18.2f}  | {h.profit_factor:>18.2f} ")
    print("-" * 78)

    print("\n" + "=" * 78)
    print("  FUTURES OVERLAY ATTRIBUTION & TAIL-RISK PROTECTION")
    print("=" * 78)
    dd_mit_pts = att.max_dd_mitigation * 100
    u_dd_pts = u.max_drawdown * 100
    h_dd_pts = h.max_drawdown * 100
    print(
        f"Max Drawdown Mitigation:     {dd_mit_pts:+.2f}% pts "
        f"(from {u_dd_pts:.2f}% to {h_dd_pts:.2f}%)"
    )
    print(f"Volatility Reduction:        {att.volatility_reduction * 100:+.2f}% points")
    print(f"Gross Futures PnL:           EUR {att.total_futures_pnl_eur:+,.2f}")
    print(f"Total Futures Trading Costs: EUR {att.total_futures_cost_eur:,.2f} (1.5 bps)")
    print(f"Net Futures Contribution:    EUR {att.net_futures_contribution_eur:+,.2f}")
    print("-" * 78)

    # 4. Regime Distribution & Episode Statistics
    total_days = len(daily_df)
    print("\n" + "=" * 78)
    print("  HMM REGIME DISTRIBUTIONS & INERTIA FILTER PROPERTIES")
    print("=" * 78)

    for bloc, col_name in [("North America (NA)", "na_regime"), ("Europe (EU)", "eu_regime")]:
        s = daily_df[col_name]
        n_calm = (s == 0).sum()
        n_alert = (s == 1).sum()
        n_stress = (s == 2).sum()

        episodes = na_episodes if "NA" in bloc else eu_episodes
        n_ep = len(episodes)
        avg_dur = episodes["duration_days"].mean() if n_ep > 0 else 0.0
        med_dur = episodes["duration_days"].median() if n_ep > 0 else 0.0
        max_dur = episodes["duration_days"].max() if n_ep > 0 else 0

        p_calm = n_calm / total_days * 100
        p_alert = n_alert / total_days * 100
        p_stress = n_stress / total_days * 100

        print(f"[{bloc} Sleeve]")
        print(f"  State 0 - Calm (0% Hedge):            {n_calm:4d} days ({p_calm:5.1f}%)")
        print(f"  State 1 - Alert (50% * Beta):         {n_alert:4d} days ({p_alert:5.1f}%)")
        print(f"  State 2 - Severe Stress (100% * Beta):{n_stress:4d} days ({p_stress:5.1f}%)")
        print(f"  Total Active Hedging Episodes:        {n_ep:4d} distinct events")
        dur_msg = (
            f"  Episode Duration (Days): Avg={avg_dur:.1f}d | Med={med_dur:.0f}d | Max={max_dur}d"
        )
        print(dur_msg)
        print("")

    print("=" * 78)
    print("  KEY HISTORICAL HEDGING EPISODES (ENTRY, DURATION & ALPHA)")
    print("=" * 78)
    hdr_ep = (
        f"{'Region':<8} | {'Start Date':<10} | {'End Date':<10} | "
        f"{'Days':<5} | {'Regime Peak':<14} | {'Fut PnL EUR':<14} | {'Hedge Alpha'}"
    )
    print(hdr_ep)
    print("-" * 78)

    # Select top 12 episodes by duration or absolute PnL
    top_episodes = all_episodes.sort_values(by="duration_days", ascending=False).head(15)
    for _, ep in top_episodes.iterrows():
        r_str = "NA" if "NA" in str(ep["region"]) else "EU"
        st_name = "Stress (100%)" if ep["max_state"] == 2 else "Alert (50%)"
        print(
            f"{r_str:<8} | {ep['start_date']} | {ep['end_date']} | "
            f"{ep['duration_days']:4d}d | {st_name:<14} | "
            f"EUR {ep['futures_overlay_pnl_eur']:>10,.0f} | "
            f"{ep['delta_alpha_pct']:+6.2f}%"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
