"""Command-line interface (CLI) for quant-portfolio ingestion and workflow execution."""

import argparse
import socket
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from quant_portfolio.alpha.pipeline import AlphaPipeline
from quant_portfolio.backtest.costs import TransactionCostModel
from quant_portfolio.backtest.engine import WalkForwardBacktestEngine
from quant_portfolio.backtest.report import (
    export_report_to_dataframes,
    format_performance_summary_table,
)
from quant_portfolio.backtest.turnover import TurnoverBufferFilter
from quant_portfolio.config import ALL_CORE_FEATURES, VALID_REGION_BLOCS, PortfolioConfig
from quant_portfolio.features.pipeline import TransatlanticFeaturePipeline
from quant_portfolio.hedging.pipeline import RegimeHedgingPipeline
from quant_portfolio.ingestion.bql_parser import BQLFundamentalsIngestionEngine
from quant_portfolio.ingestion.bql_templates import (
    CORE_UNIVERSE_INDICES,
    BQLQueryGenerator,
)
from quant_portfolio.ingestion.fred_macro import FREDMacroDownloader
from quant_portfolio.ingestion.pipeline import TransatlanticIngestionPipeline
from quant_portfolio.ingestion.xbbg_fundamentals import BloombergFundamentalsDownloader
from quant_portfolio.risk.pipeline import AllocationMethod, PortfolioAllocationPipeline
from quant_portfolio.storage.schema import ParquetMarketStore, SQLiteStore


def check_bloomberg_dapi_status(
    host: str = "127.0.0.1", port: int = 8194, timeout: float = 2.0
) -> bool:
    """Check whether Bloomberg Desktop API (DAPI) port is reachable."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False


def cmd_status() -> None:
    """Check storage and Bloomberg connection status."""
    cfg = PortfolioConfig()
    paths = cfg.paths.resolve_paths()
    print("=== Quant Portfolio - Environment Status ===")
    print(f"Database path: {paths.db_path} (Exists: {paths.db_path.exists()})")
    print(f"Parquet dir:   {paths.parquet_dir} (Exists: {paths.parquet_dir.exists()})")
    print(f"Drop-zone dir: {paths.raw_bql_dir} (Exists: {paths.raw_bql_dir.exists()})")

    dapi_open = check_bloomberg_dapi_status()
    if dapi_open:
        print("[OK] Bloomberg DAPI port 8194 is OPEN (Terminal connected).")
    else:
        print("[!] Bloomberg DAPI port 8194 is CLOSED (Terminal offline or not logged in).")

    if paths.db_path.exists():
        store = SQLiteStore(paths.db_path)
        stocks = store.get_universe_pit()
        non_fin = store.get_universe_pit(exclude_financials=True)
        n_active = len(non_fin["ticker"].unique()) if not non_fin.empty else 0
        print(f"Total Universe records: {len(stocks)}")
        print(f"Active Non-Financial tickers: {n_active}")


def cmd_generate_bql(output_file: str | None = None) -> None:
    """Print or export BQL queries and Excel formulas for universe extraction."""
    gen = BQLQueryGenerator()
    lines: list[str] = [
        "# === BLOOMBERG BQL FORMULAS & QUERIES FOR TRANSATLANTIC UNIVERSE (15 YEARS) ===",
        "# Paste these formulas into Excel with Bloomberg Add-in, or execute via BQuant / BQL.",
        "",
    ]
    for idx in CORE_UNIVERSE_INDICES:
        cfg = gen.get_index_config(idx)
        lines.append(f"## {idx} (Region: {cfg.region_bloc}) - Universe Members")
        lines.append(f"Query: {gen.generate_universe_extraction_query(idx)}")
        lines.append(f"Excel Formula: {gen.generate_excel_universe_formula(idx)}")
        lines.append("")
        lines.append(f"## {idx} (Region: {cfg.region_bloc}) - Core Fundamentals (10 Fields in EUR)")
        lines.append(f"Query: {gen.generate_fundamentals_query(idx)}")
        lines.append(f"Excel Formula: {gen.generate_excel_fundamentals_formula(idx)}")
        lines.append("")

    content = "\n".join(lines)
    if output_file:
        p = Path(output_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        print(f"BQL templates saved to: {p.resolve()}")
    else:
        print(content)


def cmd_ingest_universe(drop_dir: str | None = None) -> None:
    """Ingest BQL export files from drop-zone into SQLite universe_pit."""
    pipeline = TransatlanticIngestionPipeline()
    upserted = pipeline.ingest_universe_from_drop_zone(raw_dir=drop_dir)
    print(f"[Done] Ingested {upserted} universe records into SQLite database.")


def cmd_ingest_fundamentals(drop_dir: str | None = None) -> None:
    """Ingest BQL fundamental export files from drop-zone into SQLite fundamentals_pit."""
    cfg = PortfolioConfig()
    paths = cfg.paths.resolve_paths()
    store = SQLiteStore(paths.db_path)
    target_dir = Path(drop_dir) if drop_dir else paths.raw_bql_dir
    engine = BQLFundamentalsIngestionEngine(store=store, raw_dir=target_dir)
    results = engine.ingest_drop_zone()
    total = sum(c for c in results.values() if c > 0)
    print(f"[Done] Ingested {total} fundamental observations from {len(results)} files.")


def cmd_fetch_fundamentals(
    start_date: str = "2010-01-01",
    end_date: str = "2026-12-31",
    single_date: str | None = None,
    batch_size: int = 50,
    skip_existing: bool = True,
    limit_tickers: int | None = None,
) -> None:
    """Fetch fundamental metrics via xbbg for active non-financial universe."""
    if not check_bloomberg_dapi_status():
        print("[ERROR] Bloomberg DAPI port 8194 is not reachable.")
        print("Please ensure Bloomberg Terminal is running and logged in.")
        sys.exit(1)

    cfg = PortfolioConfig()
    paths = cfg.paths.resolve_paths()
    store = SQLiteStore(paths.db_path)

    if single_date:
        universe_df = store.get_universe_pit(as_of_date=single_date, exclude_financials=True)
    else:
        universe_df = store.get_universe_pit(exclude_financials=True)

    if universe_df.empty:
        universe_df = store.get_universe_pit(exclude_financials=True)

    if universe_df.empty:
        print("[!] No active non-financial stocks found in universe_pit.")
        return

    # Prioritize active stocks by latest appearance
    tickers = (
        universe_df.sort_values(by="as_of_date", ascending=False)["ticker"]
        .drop_duplicates()
        .tolist()
    )
    downloader = BloombergFundamentalsDownloader(store=store, config=cfg)

    if single_date:
        print(f"Fetching fundamentals snapshot for {len(tickers)} stocks as of {single_date}...")
        count = downloader.fetch_and_sync_fundamentals(
            tickers=tickers,
            as_of_date=single_date,
            batch_size=batch_size,
            skip_existing=skip_existing,
        )
    else:
        print(
            f"Fetching historical fundamentals via BDH for {len(tickers)} stocks "
            f"({start_date} to {end_date}, batch_size={batch_size})..."
        )
        count = downloader.fetch_and_sync_historical_fundamentals(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            batch_size=batch_size,
            skip_existing=skip_existing,
            limit_tickers=limit_tickers,
        )

    print(f"[Done] Successfully upserted {count} fundamental records into SQLite.")


def cmd_compute_features(
    as_of_date: str,
    output_path: str | None = None,
    raw: bool = False,
) -> None:
    """Compute and normalize the 15 core features for active universe as of as_of_date."""
    pipeline = TransatlanticFeaturePipeline()
    df = pipeline.build_features_at_date(as_of_date=as_of_date, normalize=not raw)

    if df.empty:
        print(f"[!] No features computed for date {as_of_date}. Universe or market data missing.")
        return

    n_na = (df["region_bloc"] == "NA").sum()
    n_eu = (df["region_bloc"] == "EU").sum()
    print("=== Transatlantic Feature Matrix Summary ===")
    print(f"As of Date: {as_of_date}")
    print(f"Total Investible Stocks: {len(df)} (NA: {n_na}, EU: {n_eu})")
    print(f"Features Computed ({len(ALL_CORE_FEATURES)}): {', '.join(ALL_CORE_FEATURES)}")

    n_nans = df[list(ALL_CORE_FEATURES)].isna().sum().sum()
    print(f"Residual NaNs: {n_nans}")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix.lower() == ".parquet":
            df.to_parquet(out_p, index=False)
        else:
            df.to_csv(out_p, index=False)
        print(f"[Done] Saved feature matrix to: {out_p.resolve()}")


def cmd_train_alpha(
    start_date: str = "2015-01-01",
    end_date: str = "2026-12-31",
    n_splits: int = 5,
    output_path: str | None = None,
) -> None:
    """Run Purged Walk-Forward CV across historical panel dates and report metrics."""
    cfg = PortfolioConfig()
    paths = cfg.paths.resolve_paths()
    store = SQLiteStore(paths.db_path)
    with store.get_connection() as conn:
        cursor = conn.execute(
            "SELECT DISTINCT as_of_date FROM universe_pit "
            "WHERE as_of_date >= ? AND as_of_date <= ? ORDER BY as_of_date ASC",
            (start_date, end_date),
        )
        dates = [str(row[0]) for row in cursor.fetchall()]

    if not dates:
        print(f"[!] No universe dates found in database between {start_date} and {end_date}.")
        return

    print("=== Alpha Model Training & Purged Walk-Forward CV ===")
    print(f"Panel Dates: {len(dates)} ({dates[0]} to {dates[-1]})")
    print(f"Splits: {n_splits}, Purge: {cfg.alpha.purge_days}d, Embargo: {cfg.alpha.embargo_days}d")

    pipeline = AlphaPipeline(config=cfg)
    results = pipeline.run_walk_forward_evaluation(dates=dates, n_splits=n_splits)

    print("\n--- Out-of-Sample Performance Summary ---")
    print(f"Mean Information Coefficient (Spearman): {results['mean_ic']:.4f}")
    print(f"IC Standard Deviation:                  {results['std_ic']:.4f}")
    if not np.isnan(results["ir_ic"]):
        print(f"Information Ratio (IC / std):           {results['ir_ic']:.2f}")
    print(f"Mean RMSE:                              {results['mean_rmse']:.4f}")
    print(f"Mean Top-vs-Bottom Decile Spread:       {results['mean_decile_spread']:.4f}")

    print("\n--- Fold Breakdown ---")
    for m in results.get("fold_metrics", []):
        print(
            f"Fold {m['fold']}: Test [{m['test_dates']}] (N={m['test_samples']}) | "
            f"Train [{m['train_dates']}] (N={m['train_samples']}) | "
            f"IC: {m['ic_spearman']:.4f} | RMSE: {m['rmse']:.4f} | "
            f"Decile Spread: {m['top_bottom_decile_spread']:.4f}"
        )

    if output_path and results.get("fold_metrics"):
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results["fold_metrics"]).to_csv(out_p, index=False)
        print(f"\n[Done] Saved fold metrics to: {out_p.resolve()}")


def cmd_rank_universe(
    as_of_date: str,
    output_path: str | None = None,
    top_na: int = 35,
    top_eu: int = 25,
    train_history_years: int = 5,
) -> None:
    """Rank universe at as_of_date and select balanced candidates for Phase 5."""
    cfg = PortfolioConfig()
    paths = cfg.paths.resolve_paths()
    store = SQLiteStore(paths.db_path)

    # Find training dates prior to as_of_date
    start_train = (pd.Timestamp(as_of_date) - pd.DateOffset(years=train_history_years)).strftime(
        "%Y-%m-%d"
    )

    with store.get_connection() as conn:
        cursor = conn.execute(
            "SELECT DISTINCT as_of_date FROM universe_pit "
            "WHERE as_of_date >= ? AND as_of_date < ? ORDER BY as_of_date ASC",
            (start_train, as_of_date),
        )
        training_dates = [str(row[0]) for row in cursor.fetchall()]

    pipeline = AlphaPipeline(config=cfg)
    print(f"Scoring active universe for date: {as_of_date}...")
    candidates = pipeline.rank_and_select_at_date(
        as_of_date=as_of_date,
        training_dates=training_dates if training_dates else None,
        top_na=top_na,
        top_eu=top_eu,
    )

    if candidates.empty:
        print(f"[!] No candidate stocks selected for date {as_of_date}.")
        return

    n_na = (candidates["region_bloc"] == "NA").sum()
    n_eu = (candidates["region_bloc"] == "EU").sum()
    print("\n=== Stratified Sleeve Selection (Phase 4 -> Phase 5) ===")
    print(f"Total Selected Candidates: {len(candidates)} ({n_na} NA stocks, {n_eu} EU stocks)")

    print("\nTop 5 NA Candidates:")
    top_na_df = candidates[candidates["region_bloc"] == "NA"].head(5)
    for _, row in top_na_df.iterrows():
        print(f"  Rank {row['regional_rank']:2d}: {row['ticker']:<12} Score: {row['score']:.4f}")

    print("\nTop 5 EU Candidates:")
    top_eu_df = candidates[candidates["region_bloc"] == "EU"].head(5)
    for _, row in top_eu_df.iterrows():
        print(f"  Rank {row['regional_rank']:2d}: {row['ticker']:<12} Score: {row['score']:.4f}")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix.lower() == ".parquet":
            candidates.to_parquet(out_p, index=False)
        else:
            candidates.to_csv(out_p, index=False)
        print(f"\n[Done] Saved selected candidates to: {out_p.resolve()}")


def cmd_allocate(
    as_of_date: str,
    method: str = "hrp",
    candidates_file: str | None = None,
    output_path: str | None = None,
    top_na: int = 35,
    top_eu: int = 25,
) -> None:
    """Execute Phase 5 transatlantic risk modeling and portfolio allocation."""
    cfg = PortfolioConfig()
    paths = cfg.paths.resolve_paths()
    parquet_store = ParquetMarketStore(paths.parquet_dir)

    print(f"=== Quant Portfolio - Transatlantic Allocation ({as_of_date}) ===")
    print(f"Optimization Method: {method.upper()}")

    # 1. Load or generate candidates
    if candidates_file:
        c_path = Path(candidates_file)
        if not c_path.exists():
            print(f"[ERROR] Candidates file not found: {c_path}")
            sys.exit(1)
        if c_path.suffix.lower() == ".parquet":
            candidates = pd.read_parquet(c_path)
        else:
            candidates = pd.read_csv(c_path, keep_default_na=False)
            if "region_bloc" in candidates.columns:
                candidates["region_bloc"] = candidates["region_bloc"].replace("", "NA").fillna("NA")
        print(f"Loaded {len(candidates)} candidates from: {c_path.name}")
    else:
        sqlite_store = SQLiteStore(paths.db_path)
        alpha_pipe = AlphaPipeline(
            config=cfg,
            sqlite_store=sqlite_store,
            parquet_store=parquet_store,
        )
        print(f"Selecting candidates at {as_of_date} (top {top_na} NA, top {top_eu} EU)...")
        dt_end = pd.Timestamp(as_of_date)
        dt_start = dt_end - pd.DateOffset(years=2)
        dates_range = (
            pd.date_range(start=dt_start, end=dt_end, freq="BME").strftime("%Y-%m-%d").tolist()
        )
        candidates = alpha_pipe.rank_and_select_at_date(
            as_of_date=as_of_date,
            training_dates=dates_range,
            top_na=top_na,
            top_eu=top_eu,
        )

    if candidates.empty:
        print("[!] No candidates found or selected. Exiting.")
        return

    # 2. Run allocation pipeline
    alloc_method = cast("AllocationMethod", method)
    pipe = PortfolioAllocationPipeline(parquet_store=parquet_store, config=cfg)
    alloc_df, summary = pipe.allocate_candidates(
        candidates_df=candidates,
        as_of_date=as_of_date,
        method=alloc_method,
    )

    if alloc_df.empty:
        print("[!] Allocation failed or price history is missing.")
        return

    # 3. Print clean ASCII summary
    print("\n--- Allocation Summary ---")
    print(f"Total Portfolio Stocks: {summary.n_stocks}")
    na_pct = summary.na_weight * 100.0
    eu_pct = summary.eu_weight * 100.0
    na_tgt = f"{cfg.allocation.na_weight_min * 100:.0f}%-{cfg.allocation.na_weight_max * 100:.0f}%"
    eu_tgt = f"{cfg.allocation.eu_weight_min * 100:.0f}%-{cfg.allocation.eu_weight_max * 100:.0f}%"

    print(f"  NA Sleeve: {summary.n_na:2d} stocks | Weight: {na_pct:5.2f}% (Target: {na_tgt})")
    print(f"  EU Sleeve: {summary.n_eu:2d} stocks | Weight: {eu_pct:5.2f}% (Target: {eu_tgt})")
    print(f"Annualized 3D Volatility: {summary.annualized_volatility * 100.0:.2f}%")
    print(f"Empirical 3D CVaR (95%):  {summary.cvar_95 * 100.0:.2f}%")

    print("\nTop 5 Holdings:")
    top_holdings = alloc_df.sort_values(by="weight", ascending=False).head(5)
    for _, row in top_holdings.iterrows():
        print(
            f"  {row['ticker']:<12} [{row['region_bloc']}]  "
            f"Weight: {row['weight'] * 100.0:5.2f}%  MCTR: {row['mctr']:.4f}"
        )

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix.lower() == ".parquet":
            alloc_df.to_parquet(out_p, index=False)
        else:
            alloc_df.to_csv(out_p, index=False)
        print(f"\n[Done] Saved allocations to: {out_p.resolve()}")


def cmd_regime(as_of_date: str) -> None:
    """Detect market regimes and 3-day inertia filtered states for NA and EU."""
    cfg = PortfolioConfig()
    paths = cfg.paths.resolve_paths()
    sqlite_store = SQLiteStore(paths.db_path)
    parquet_store = ParquetMarketStore(paths.parquet_dir)

    pipe = RegimeHedgingPipeline(
        sqlite_store=sqlite_store,
        parquet_store=parquet_store,
        config=cfg,
    )

    print(f"=== Quant Portfolio - Transatlantic Regime Sensors ({as_of_date}) ===")
    raw_results, filtered_states = pipe.detect_and_filter_regimes(as_of_date=as_of_date)
    state_names = {0: "Calm", 1: "Alert", 2: "Severe Stress"}

    for bloc in VALID_REGION_BLOCS:
        res = raw_results[bloc]
        filt_s = filtered_states[bloc]
        print(f"\n[{bloc} Sleeve]")
        print(f"  Raw HMM State:       {res.state} ({res.state_name})")
        print(f"  Filtered State (3d): {filt_s} ({state_names.get(filt_s, 'Unknown')})")
        print(
            f"  Probabilities:       Calm: {res.prob_calm * 100:5.1f}% | "
            f"Alert: {res.prob_alert * 100:5.1f}% | "
            f"Stress: {res.prob_stress * 100:5.1f}%"
        )


def cmd_hedge(
    as_of_date: str,
    portfolio_value: float,
    allocations_file: str | None = None,
    output_path: str | None = None,
    na_future: str | None = None,
    eu_future: str | None = None,
    eur_usd_rate: float = 1.08,
) -> None:
    """Calculate dual-sleeve beta-adjusted futures overlay and sizing."""
    cfg = PortfolioConfig()
    paths = cfg.paths.resolve_paths()
    sqlite_store = SQLiteStore(paths.db_path)
    parquet_store = ParquetMarketStore(paths.parquet_dir)

    print(f"=== Quant Portfolio - Dual-Sleeve Futures Hedging ({as_of_date}) ===")
    print(f"Total Portfolio Capital: EUR {portfolio_value:,.2f}")

    # 1. Load or calculate allocations
    if allocations_file:
        c_path = Path(allocations_file)
        if not c_path.exists():
            print(f"[ERROR] Allocations file not found: {c_path}")
            sys.exit(1)
        if c_path.suffix.lower() == ".parquet":
            alloc_df = pd.read_parquet(c_path)
        else:
            alloc_df = pd.read_csv(c_path, keep_default_na=False)
            if "region_bloc" in alloc_df.columns:
                alloc_df["region_bloc"] = alloc_df["region_bloc"].replace("", "NA").fillna("NA")
        print(f"Loaded {len(alloc_df)} portfolio holdings from: {c_path.name}")
    else:
        print("[!] No allocations file provided. Generating HRP allocations...")
        alloc_pipe = PortfolioAllocationPipeline(parquet_store=parquet_store, config=cfg)
        alpha_pipe = AlphaPipeline(
            config=cfg,
            sqlite_store=sqlite_store,
            parquet_store=parquet_store,
        )
        candidates = alpha_pipe.rank_and_select_at_date(as_of_date=as_of_date)
        alloc_df, _ = alloc_pipe.allocate_candidates(
            candidates_df=candidates,
            as_of_date=as_of_date,
            method="hrp",
        )

    if alloc_df.empty:
        print("[!] Cannot compute hedging plan: portfolio allocation is empty.")
        return

    # 2. Execute pipeline
    hedge_pipe = RegimeHedgingPipeline(
        sqlite_store=sqlite_store,
        parquet_store=parquet_store,
        config=cfg,
    )

    summary = hedge_pipe.execute_hedging_pipeline(
        as_of_date=as_of_date,
        portfolio_value_eur=portfolio_value,
        allocations_df=alloc_df,
        eur_usd_rate=eur_usd_rate,
        custom_na_future=na_future,
        custom_eu_future=eu_future,
    )

    # 3. Print clean ASCII table
    print("\n--- Futures Hedging Execution Plan ---")
    for sleeve in (summary.na_sleeve, summary.eu_sleeve):
        print(f"[{sleeve.region_bloc} Sleeve]")
        print(
            f"  Capital: EUR {sleeve.sleeve_value_eur:,.2f} ({sleeve.sleeve_weight * 100:.1f}%) | "
            f"Weighted Beta: {sleeve.sleeve_beta:.3f}"
        )
        print(
            f"  Regime: State {sleeve.regime_state} ({sleeve.regime_name}) -> "
            f"Hedge Ratio: {sleeve.hedge_ratio * 100:.1f}%"
        )
        print(f"  Hedged Notional: EUR {sleeve.hedged_notional_eur:,.2f}")
        print(
            f"  Contract: {sleeve.contract_symbol} @ {sleeve.contract_price:,.2f} "
            f"({sleeve.contract_currency}) | Multiplier: {sleeve.contract_multiplier:.0f}"
        )
        print(
            f"  ORDER RECOMMENDATION: SELL (SHORT) {sleeve.contracts_rounded} "
            f"{sleeve.contract_symbol} contracts "
            f"(exact: {sleeve.contracts_exact:.2f})"
        )
        print("")

    eff_pct = summary.portfolio_effective_hedge_ratio * 100.0
    print(
        f"Consolidated Portfolio Hedged: EUR {summary.total_hedged_notional_eur:,.2f} "
        f"({eff_pct:.2f}% effective hedge)"
    )

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        summary_df = summary.to_dataframe()
        if out_p.suffix.lower() == ".parquet":
            summary_df.to_parquet(out_p, index=False)
        else:
            summary_df.to_csv(out_p, index=False)
        print(f"\n[Done] Saved hedging plan to: {out_p.resolve()}")


def cmd_fetch_macro(start_date: str = "2010-01-01", end_date: str | None = None) -> None:
    """Download macroeconomic series from FRED and persist into SQLite macro_series."""
    cfg = PortfolioConfig()
    paths = cfg.paths.resolve_paths()
    store = SQLiteStore(paths.db_path)
    downloader = FREDMacroDownloader(store=store, config=cfg)

    print(f"Downloading macro series from FRED (start: {start_date})...")
    count = downloader.download_and_sync_macro_series(start_date=start_date, end_date=end_date)
    print(f"[Done] Successfully upserted {count} macro series records into SQLite.")


def cmd_fetch_series(
    start_date: str = "2010-01-01",
    end_date: str = "2026-12-31",
    batch_size: int = 50,
    skip_existing: bool = False,
    limit_tickers: int | None = None,
) -> None:
    """Fetch EUR daily series from Bloomberg xbbg for active non-financial stocks."""
    if not check_bloomberg_dapi_status():
        print("[ERROR] Bloomberg DAPI port 8194 is not reachable.")
        print("Please ensure Bloomberg Terminal is running and logged in.")
        sys.exit(1)

    print(f"Starting xbbg download in EUR ({start_date} to {end_date}, batch_size={batch_size})...")
    pipeline = TransatlanticIngestionPipeline()
    written = pipeline.download_and_sync_market_series(
        start_date=start_date,
        end_date=end_date,
        batch_size=batch_size,
        skip_existing=skip_existing,
        limit_tickers=limit_tickers,
    )
    print(f"[Done] Successfully downloaded, aligned, and persisted {written} market observations.")


def cmd_fetch_universe(
    single_date: str | None = None,
    start_date: str = "2010-01-01",
    end_date: str = "2026-12-31",
    freq: str = "BME",
    indices_str: str | None = None,
    force: bool = False,
) -> None:
    """Fetch index constituents historically or for a single date via xbbg."""
    if not check_bloomberg_dapi_status():
        print("[ERROR] Bloomberg DAPI port 8194 is not reachable.")
        print("Please ensure Bloomberg Terminal is running and logged in.")
        sys.exit(1)

    target_indices = (
        tuple(idx.strip() for idx in indices_str.split(","))
        if indices_str
        else CORE_UNIVERSE_INDICES
    )

    pipeline = TransatlanticIngestionPipeline()

    if single_date:
        print(f"Fetching universe for single date {single_date} across {target_indices}...")
        total_synced = 0
        for idx in target_indices:
            count = pipeline.universe_extractor.sync_index_universe_date(
                idx, as_of_date=single_date
            )
            print(f"  [{idx}] Synced {count} constituents")
            total_synced += count
        print(f"[Done] Total {total_synced} constituent records synced for {single_date}.")
    else:
        print(f"Fetching monthly historical universe ({start_date} to {end_date}) via xbbg...")
        results = pipeline.fetch_and_sync_historical_universe(
            start_date=start_date,
            end_date=end_date,
            freq=freq,
            indices=target_indices,
            skip_existing=not force,
        )
        print("[Done] Historical universe extraction summary:")
        for idx, count in results.items():
            print(f"  {idx}: {count} records upserted")


def cmd_backtest(
    start_date: str = "2015-01-01",
    end_date: str = "2026-12-31",
    capital: float = 10_000_000.0,
    method: str = "hrp",
    rebalance_freq: int = 21,
    enable_hedging: bool = True,
    output_dir: str | None = None,
) -> None:
    """Run full Walk-Forward backtest simulation and output quantitative report."""
    cfg = PortfolioConfig()
    paths = cfg.paths.resolve_paths()
    sqlite_store = SQLiteStore(paths.db_path)
    parquet_store = ParquetMarketStore(paths.parquet_dir)

    print("=== Quant Portfolio - Walk-Forward Simulation (2015 to Present) ===")
    print(f"Date Range:              {start_date} to {end_date}")
    print(f"Initial Capital:         EUR {capital:,.2f}")
    print(f"Optimization Method:     {method.upper()}")
    print(f"Rebalance Frequency:     Every {rebalance_freq} business days (Monthly)")
    print(f"Dynamic Futures Hedging: {'ENABLED' if enable_hedging else 'DISABLED'}")

    engine = WalkForwardBacktestEngine(
        sqlite_store=sqlite_store,
        parquet_store=parquet_store,
        config=cfg,
    )

    daily_df, report, rebal_df = engine.run_backtest(
        start_date=start_date,
        end_date=end_date,
        allocation_method=cast("AllocationMethod", method),
        initial_capital_eur=capital,
        enable_hedging=enable_hedging,
        rebalance_freq_days=rebalance_freq,
    )

    summary_table = format_performance_summary_table(report)
    print("\n" + summary_table)

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        dfs = export_report_to_dataframes(report, daily_history_df=daily_df)
        dfs["summary"].to_csv(out_p / "backtest_summary.csv", index=False)
        dfs["attribution"].to_csv(out_p / "hedging_attribution.csv", index=False)
        dfs["sleeves"].to_csv(out_p / "sleeve_metrics.csv", index=False)
        daily_df.to_csv(out_p / "portfolio_daily_history.csv", index=False)
        rebal_df.to_csv(out_p / "rebalance_log.csv", index=False)
        print(f"\n[Done] Backtest artifacts exported to: {out_p.resolve()}")


def cmd_rebalance(
    as_of_date: str,
    capital: float = 10_000_000.0,
    method: str = "hrp",
    holdings_file: str | None = None,
    output_path: str | None = None,
    eur_usd_rate: float = 1.08,
) -> None:
    """Generate production rebalance orders and futures overlay for as_of_date."""
    cfg = PortfolioConfig()
    paths = cfg.paths.resolve_paths()
    sqlite_store = SQLiteStore(paths.db_path)
    parquet_store = ParquetMarketStore(paths.parquet_dir)

    print(f"=== Quant Portfolio - Monthly Rebalance & Production Orders ({as_of_date}) ===")
    print(f"Portfolio Capital: EUR {capital:,.2f}")
    print(f"Optimizer:         {method.upper()}")

    # 1. Load current holdings if provided
    current_weights_df = pd.DataFrame(columns=["ticker", "weight", "region_bloc"])
    current_held_tickers: list[str] = []
    if holdings_file:
        h_path = Path(holdings_file)
        if h_path.exists():
            if h_path.suffix.lower() == ".parquet":
                current_weights_df = pd.read_parquet(h_path)
            else:
                current_weights_df = pd.read_csv(h_path, keep_default_na=False)
                if "region_bloc" in current_weights_df.columns:
                    current_weights_df["region_bloc"] = (
                        current_weights_df["region_bloc"].replace("", "NA").fillna("NA")
                    )
            current_held_tickers = current_weights_df["ticker"].tolist()
            print(f"Current Holdings Loaded: {len(current_held_tickers)} assets")

    # 2. Score and Rank Active Universe
    alpha_pipe = AlphaPipeline(
        config=cfg,
        sqlite_store=sqlite_store,
        parquet_store=parquet_store,
    )
    ranked_univ = alpha_pipe.rank_and_select_at_date(as_of_date=as_of_date)

    # 3. Apply Turnover Buffer (Hold decile <= 3)
    tb_filter = TurnoverBufferFilter(config=cfg)
    candidates = tb_filter.filter_candidates_with_buffer(
        ranked_universe=ranked_univ,
        current_holdings=current_held_tickers,
    )

    # 4. Optimal Allocation
    alloc_pipe = PortfolioAllocationPipeline(parquet_store=parquet_store, config=cfg)
    target_weights_df, risk_summary = alloc_pipe.allocate_candidates(
        candidates_df=candidates,
        as_of_date=as_of_date,
        method=cast("AllocationMethod", method),
    )

    # 5. Delta Orders
    delta_orders_df = tb_filter.generate_delta_orders(
        current_weights=current_weights_df,
        target_weights=target_weights_df,
        portfolio_value_eur=capital,
    )

    # 6. Cost Estimation
    cost_model = TransactionCostModel(config=cfg)
    eq_cost, _ = cost_model.calculate_equity_costs(delta_orders_df, portfolio_value_eur=capital)

    # 7. Futures Hedging Plan
    hedge_pipe = RegimeHedgingPipeline(
        sqlite_store=sqlite_store,
        parquet_store=parquet_store,
        config=cfg,
    )
    hedge_summary = hedge_pipe.execute_hedging_pipeline(
        as_of_date=as_of_date,
        portfolio_value_eur=capital,
        allocations_df=target_weights_df,
        eur_usd_rate=eur_usd_rate,
    )

    print("\n--- Allocation & Risk Summary ---")
    print(
        f"Total Invested Assets:  {risk_summary.n_stocks} "
        f"(NA: {risk_summary.n_na}, EU: {risk_summary.n_eu})"
    )
    print(
        f"Sleeve Weights:         NA = {risk_summary.na_weight * 100:.1f}%, "
        f"EU = {risk_summary.eu_weight * 100:.1f}%"
    )
    print(f"Portfolio Annual Vol:   {risk_summary.annualized_volatility * 100:.2f}%")
    print(f"Estimated Trade Costs:  EUR {eq_cost:,.2f}")

    print("\n--- Futures Hedging Execution Overlay ---")
    na_sym = hedge_summary.na_sleeve.contract_symbol
    na_qty = hedge_summary.na_sleeve.contracts_rounded
    na_not = hedge_summary.na_sleeve.hedged_notional_eur
    eu_sym = hedge_summary.eu_sleeve.contract_symbol
    eu_qty = hedge_summary.eu_sleeve.contracts_rounded
    eu_not = hedge_summary.eu_sleeve.hedged_notional_eur

    print(f"NA Overlay ({na_sym}):  Sell {na_qty} contracts (Notional: EUR {na_not:,.0f})")
    print(f"EU Overlay ({eu_sym}):  Sell {eu_qty} contracts (Notional: EUR {eu_not:,.0f})")

    buys = delta_orders_df[delta_orders_df["action"] == "BUY"]
    sells = delta_orders_df[delta_orders_df["action"] == "SELL"]
    print(f"\n--- Order Execution Summary: {len(buys)} BUYs, {len(sells)} SELLs ---")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix.lower() == ".parquet":
            delta_orders_df.to_parquet(out_p, index=False)
        else:
            delta_orders_df.to_csv(out_p, index=False)
        print(f"[Done] Exported rebalance orders to: {out_p.resolve()}")


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Quant Portfolio Ingestion and Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # status
    subparsers.add_parser("status", help="Check storage and Bloomberg connection status")

    # generate-bql
    p_bql = subparsers.add_parser("generate-bql", help="Generate BQL formulas for Excel/terminal")
    p_bql.add_argument("--output", "-o", help="Optional output text file path")

    # ingest-universe
    p_ing = subparsers.add_parser("ingest-universe", help="Ingest BQL exports from drop-zone")
    p_ing.add_argument("--dir", "-d", help="Custom drop-zone directory")

    # fetch-universe
    p_uni = subparsers.add_parser("fetch-universe", help="Fetch index constituents via xbbg")
    p_uni.add_argument("--date", help="Single snapshot date (YYYY-MM-DD)")
    p_uni.add_argument("--start", "-s", default="2010-01-01", help="Start date (YYYY-MM-DD)")
    p_uni.add_argument("--end", "-e", default="2026-12-31", help="End date (YYYY-MM-DD)")
    p_uni.add_argument("--freq", "-f", default="BME", help="Resampling frequency (default: BME)")
    p_uni.add_argument("--indices", help="Comma-separated indices (default: all core)")
    p_uni.add_argument("--force", action="store_true", help="Re-sync already present dates")

    # fetch-series
    p_fetch = subparsers.add_parser("fetch-series", help="Download series in EUR via xbbg")
    p_fetch.add_argument("--start", "-s", default="2010-01-01", help="Start date (YYYY-MM-DD)")
    p_fetch.add_argument("--end", "-e", default="2026-12-31", help="End date (YYYY-MM-DD)")
    p_fetch.add_argument(
        "--batch-size", "-b", type=int, default=50, help="Batch size of tickers (default: 50)"
    )
    p_fetch.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tickers already present in Parquet store",
    )
    p_fetch.add_argument(
        "--limit-tickers",
        type=int,
        default=None,
        help="Limit number of tickers to download (for testing)",
    )

    # ingest-fundamentals
    p_if = subparsers.add_parser(
        "ingest-fundamentals", help="Ingest BQL fundamental exports from drop-zone"
    )
    p_if.add_argument("--dir", "-d", help="Custom drop-zone directory")

    # fetch-fundamentals
    p_ff = subparsers.add_parser("fetch-fundamentals", help="Fetch fundamental metrics via xbbg")
    p_ff.add_argument("--date", help="Optional single snapshot date (YYYY-MM-DD)")
    p_ff.add_argument("--start", "-s", default="2010-01-01", help="Start date (YYYY-MM-DD)")
    p_ff.add_argument("--end", "-e", default="2026-12-31", help="End date (YYYY-MM-DD)")
    p_ff.add_argument(
        "--batch-size", "-b", type=int, default=50, help="Batch size of tickers (default: 50)"
    )
    p_ff.add_argument("--force", action="store_true", help="Re-fetch already cached metrics")
    p_ff.add_argument(
        "--limit-tickers", type=int, default=None, help="Limit number of tickers (for testing)"
    )

    # compute-features
    p_cf = subparsers.add_parser(
        "compute-features", help="Compute and normalize the 15 core features for a date"
    )
    p_cf.add_argument("--date", required=True, help="As of date (YYYY-MM-DD)")
    p_cf.add_argument("--output", "-o", help="Optional output file path (.parquet or .csv)")
    p_cf.add_argument(
        "--raw", action="store_true", help="Do not apply cross-sectional normalization"
    )

    # train-alpha
    p_ta = subparsers.add_parser(
        "train-alpha", help="Run Purged Walk-Forward CV across panel dates"
    )
    p_ta.add_argument("--start", "-s", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    p_ta.add_argument("--end", "-e", default="2026-12-31", help="End date (YYYY-MM-DD)")
    p_ta.add_argument("--splits", type=int, default=5, help="Number of CV splits (default: 5)")
    p_ta.add_argument("--output", "-o", help="Optional output path for fold metrics CSV")

    # rank-universe
    p_ru = subparsers.add_parser(
        "rank-universe", help="Score universe and select balanced candidates for Phase 5"
    )
    p_ru.add_argument("--date", required=True, help="As of date (YYYY-MM-DD)")
    p_ru.add_argument("--output", "-o", help="Optional output file path (.parquet or .csv)")
    p_ru.add_argument("--top-na", type=int, default=35, help="Top NA stocks (default: 35)")
    p_ru.add_argument("--top-eu", type=int, default=25, help="Top EU stocks (default: 25)")
    p_ru.add_argument(
        "--history-years", type=int, default=5, help="Years of training history (default: 5)"
    )

    # allocate
    p_al = subparsers.add_parser(
        "allocate", help="Allocate transatlantic portfolio using HRP, Mean-CVaR, or Ensemble"
    )
    p_al.add_argument("--date", required=True, help="As of date (YYYY-MM-DD)")
    p_al.add_argument(
        "--method",
        "-m",
        choices=["hrp", "cvar", "ensemble"],
        default="hrp",
        help="Optimization method (default: hrp)",
    )
    p_al.add_argument(
        "--candidates",
        "-c",
        help="Optional path to ranked candidates file (.csv or .parquet)",
    )
    p_al.add_argument("--output", "-o", help="Optional output path for allocations (.csv/.parquet)")
    p_al.add_argument("--top-na", type=int, default=35, help="Top NA stocks (default: 35)")
    p_al.add_argument("--top-eu", type=int, default=25, help="Top EU stocks (default: 25)")

    # regime
    p_reg = subparsers.add_parser(
        "regime", help="Inspect NA and EU HMM regime probabilities and 3d inertia state"
    )
    p_reg.add_argument("--date", required=True, help="As of date (YYYY-MM-DD)")

    # hedge
    p_hd = subparsers.add_parser(
        "hedge", help="Calculate dual-sleeve beta-adjusted futures overlay and contract sizing"
    )
    p_hd.add_argument("--date", required=True, help="As of date (YYYY-MM-DD)")
    p_hd.add_argument(
        "--portfolio-value",
        "-v",
        type=float,
        default=10_000_000.0,
        help="Total portfolio capital in EUR (default: 10,000,000)",
    )
    p_hd.add_argument(
        "--allocations",
        "-a",
        help="Optional path to allocations CSV/Parquet file from allocate command",
    )
    p_hd.add_argument("--output", "-o", help="Optional output path for hedging plan CSV/Parquet")
    p_hd.add_argument("--na-future", choices=["ES", "MES"], help="NA future contract (default: ES)")
    p_hd.add_argument(
        "--eu-future", choices=["FXXP", "FESX"], help="EU future contract (default: FXXP)"
    )
    p_hd.add_argument(
        "--eur-usd", type=float, default=1.08, help="EUR/USD exchange rate (default: 1.08)"
    )

    # backtest
    p_bt = subparsers.add_parser(
        "backtest", help="Run transatlantic Walk-Forward backtest simulation (2015 to present)"
    )
    p_bt.add_argument("--start", "-s", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    p_bt.add_argument("--end", "-e", default="2026-12-31", help="End date (YYYY-MM-DD)")
    p_bt.add_argument(
        "--capital", "-c", type=float, default=10_000_000.0, help="Initial capital in EUR"
    )
    p_bt.add_argument(
        "--method",
        "-m",
        choices=["hrp", "cvar", "ensemble"],
        default="hrp",
        help="Optimizer (default: hrp)",
    )
    p_bt.add_argument(
        "--rebalance-freq",
        "-f",
        type=int,
        default=21,
        help="Rebalance frequency in business days (default: 21)",
    )
    p_bt.add_argument(
        "--no-hedge", action="store_true", help="Disable dynamic futures overlay hedging"
    )
    p_bt.add_argument(
        "--output-dir", "-o", help="Optional output directory to export CSV artifacts"
    )

    # rebalance
    p_reb = subparsers.add_parser(
        "rebalance", help="Generate production delta orders and futures hedging overlay"
    )
    p_reb.add_argument("--date", required=True, help="As of date (YYYY-MM-DD)")
    p_reb.add_argument(
        "--capital", "-c", type=float, default=10_000_000.0, help="Portfolio capital in EUR"
    )
    p_reb.add_argument(
        "--method",
        "-m",
        choices=["hrp", "cvar", "ensemble"],
        default="hrp",
        help="Optimizer (default: hrp)",
    )
    p_reb.add_argument(
        "--holdings", "-H", help="Optional path to current portfolio holdings (.csv or .parquet)"
    )
    p_reb.add_argument(
        "--output", "-o", help="Optional output path for delta orders (.csv or .parquet)"
    )
    p_reb.add_argument(
        "--eur-usd", type=float, default=1.08, help="EUR/USD exchange rate (default: 1.08)"
    )

    # fetch-macro
    p_fm = subparsers.add_parser(
        "fetch-macro", help="Download macroeconomic and credit spread series from FRED into SQLite"
    )
    p_fm.add_argument("--start", "-s", default="2010-01-01", help="Start date (YYYY-MM-DD)")
    p_fm.add_argument("--end", "-e", default=None, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()
    if args.command == "status":
        cmd_status()
    elif args.command == "generate-bql":
        cmd_generate_bql(output_file=args.output)
    elif args.command == "ingest-universe":
        cmd_ingest_universe(drop_dir=args.dir)
    elif args.command == "ingest-fundamentals":
        cmd_ingest_fundamentals(drop_dir=args.dir)
    elif args.command == "fetch-fundamentals":
        cmd_fetch_fundamentals(
            start_date=args.start,
            end_date=args.end,
            single_date=args.date,
            batch_size=args.batch_size,
            skip_existing=not args.force,
            limit_tickers=args.limit_tickers,
        )
    elif args.command == "fetch-universe":
        cmd_fetch_universe(
            single_date=args.date,
            start_date=args.start,
            end_date=args.end,
            freq=args.freq,
            indices_str=args.indices,
            force=args.force,
        )
    elif args.command == "fetch-series":
        cmd_fetch_series(
            start_date=args.start,
            end_date=args.end,
            batch_size=args.batch_size,
            skip_existing=args.skip_existing,
            limit_tickers=args.limit_tickers,
        )
    elif args.command == "compute-features":
        cmd_compute_features(
            as_of_date=args.date,
            output_path=args.output,
            raw=args.raw,
        )
    elif args.command == "train-alpha":
        cmd_train_alpha(
            start_date=args.start,
            end_date=args.end,
            n_splits=args.splits,
            output_path=args.output,
        )
    elif args.command == "rank-universe":
        cmd_rank_universe(
            as_of_date=args.date,
            output_path=args.output,
            top_na=args.top_na,
            top_eu=args.top_eu,
            train_history_years=args.history_years,
        )
    elif args.command == "allocate":
        cmd_allocate(
            as_of_date=args.date,
            method=args.method,
            candidates_file=args.candidates,
            output_path=args.output,
            top_na=args.top_na,
            top_eu=args.top_eu,
        )
    elif args.command == "regime":
        cmd_regime(as_of_date=args.date)
    elif args.command == "hedge":
        cmd_hedge(
            as_of_date=args.date,
            portfolio_value=args.portfolio_value,
            allocations_file=args.allocations,
            output_path=args.output,
            na_future=args.na_future,
            eu_future=args.eu_future,
            eur_usd_rate=args.eur_usd,
        )
    elif args.command == "backtest":
        cmd_backtest(
            start_date=args.start,
            end_date=args.end,
            capital=args.capital,
            method=args.method,
            rebalance_freq=args.rebalance_freq,
            enable_hedging=not args.no_hedge,
            output_dir=args.output_dir,
        )
    elif args.command == "rebalance":
        cmd_rebalance(
            as_of_date=args.date,
            capital=args.capital,
            method=args.method,
            holdings_file=args.holdings,
            output_path=args.output,
            eur_usd_rate=args.eur_usd,
        )
    elif args.command == "fetch-macro":
        cmd_fetch_macro(start_date=args.start, end_date=args.end)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
