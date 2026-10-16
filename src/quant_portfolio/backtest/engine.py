"""Walk-Forward Backtesting Engine for transatlantic equity and dynamic futures overlay."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from quant_portfolio.alpha.selector import StratifiedSleeveSelector
from quant_portfolio.backtest.costs import TransactionCostModel
from quant_portfolio.backtest.metrics import FullPerformanceReport, build_full_performance_report
from quant_portfolio.backtest.turnover import TurnoverBufferFilter
from quant_portfolio.config import (
    FUTURE_CURRENCIES,
    FUTURE_MULTIPLIERS,
    BacktestConfig,
    PortfolioConfig,
)
from quant_portfolio.hedging.pipeline import DEFAULT_FUTURES_QUOTES, RegimeHedgingPipeline
from quant_portfolio.risk.pipeline import AllocationMethod, PortfolioAllocationPipeline

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from quant_portfolio.alpha.pipeline import AlphaPipeline
    from quant_portfolio.storage.schema import ParquetMarketStore, SQLiteStore


logger = logging.getLogger(__name__)


class WalkForwardBacktestEngine:
    """Simulates the transatlantic portfolio Walk-Forward accounting from 2015 to present."""

    def __init__(
        self,
        sqlite_store: SQLiteStore,
        parquet_store: ParquetMarketStore,
        config: PortfolioConfig | None = None,
        alpha_pipeline: AlphaPipeline | None = None,
        allocation_pipeline: PortfolioAllocationPipeline | None = None,
        regime_pipeline: RegimeHedgingPipeline | None = None,
        turnover_filter: TurnoverBufferFilter | None = None,
        cost_model: TransactionCostModel | None = None,
    ) -> None:
        self.config = config or PortfolioConfig()
        self.bt_config: BacktestConfig = self.config.backtest or BacktestConfig()
        self.store = sqlite_store
        self.parquet_store = parquet_store
        self.alpha_pipeline = alpha_pipeline
        self._returns_matrix: pd.DataFrame | None = None
        self._last_alpha_train_date: str | None = None

        self.allocation_pipeline = allocation_pipeline or PortfolioAllocationPipeline(
            parquet_store=self.parquet_store,
            config=self.config,
        )
        self.regime_pipeline = regime_pipeline or RegimeHedgingPipeline(
            sqlite_store=self.store,
            parquet_store=self.parquet_store,
            config=self.config,
        )
        self.turnover_filter = turnover_filter or TurnoverBufferFilter(config=self.config)
        self.cost_model = cost_model or TransactionCostModel(config=self.config)
        self.sleeve_selector = StratifiedSleeveSelector(config=self.config)

    def _get_business_calendar(self, start_date: str, end_date: str) -> list[str]:
        """Fetch available business dates within the simulation range."""
        b_dates = pd.bdate_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()
        return [str(d) for d in b_dates]

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        allocation_method: AllocationMethod = "hrp",
        initial_capital_eur: float | None = None,
        enable_hedging: bool = True,
        rebalance_freq_days: int | None = None,
        eur_usd_rate: float = 1.08,
        fx_rate_loader: Callable[[str], float] | None = None,
        ranked_universe_loader: object | None = None,
        futures_prices_loader: object | None = None,
    ) -> tuple[pd.DataFrame, FullPerformanceReport, pd.DataFrame]:
        """Run complete Walk-Forward simulation.

        Parameters
        ----------
        start_date : str
            Simulation start date (YYYY-MM-DD).
        end_date : str
            Simulation end date (YYYY-MM-DD).
        allocation_method : AllocationMethod
            Allocation optimizer: 'hrp', 'cvar', or 'ensemble'.
        initial_capital_eur : float | None
            Starting portfolio capital in EUR (default: 10M EUR).
        enable_hedging : bool
            Whether to activate dynamic futures overlay.
        rebalance_freq_days : int | None
            Number of business days between equity rebalancings (default from config: 21).
        eur_usd_rate : float
            Base EUR/USD exchange rate for USD-denominated futures (default: 1.08).
        fx_rate_loader : callable | None
            Optional callable `(current_date: str) -> float` for dynamic historical FX rates.
        ranked_universe_loader : callable | None
            Optional callable `(as_of_date: str) -> pd.DataFrame` for custom candidate ranking.
        futures_prices_loader : callable | None
            Optional callable `(as_of_date: str) -> dict[str, float]` for market futures quotes.

        Returns
        -------
        tuple[pd.DataFrame, FullPerformanceReport, pd.DataFrame]
            (daily_history_df, performance_report, rebalance_log_df)
        """
        cap0 = float(initial_capital_eur or self.bt_config.initial_capital_eur)
        freq = rebalance_freq_days or self.bt_config.rebalance_freq_days

        calendar = self._get_business_calendar(start_date, end_date)
        if not calendar:
            raise ValueError(f"No business days found between {start_date} and {end_date}.")

        # Pre-load daily returns matrix from ParquetMarketStore for [start_date - 15 days, end_date]
        pre_start = (pd.Timestamp(start_date) - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
        try:
            m_df = self.parquet_store.read_daily_market(
                start_date=pre_start,
                end_date=end_date,
            )
            if not m_df.empty and "tot_return_idx_eur" in m_df.columns:
                p_df = m_df.pivot(index="date", columns="ticker", values="tot_return_idx_eur")
                self._returns_matrix = p_df.pct_change(fill_method=None)
            else:
                self._returns_matrix = None
        except Exception as exc:
            logger.debug("Could not pre-load returns matrix: %s", exc)
            self._returns_matrix = None

        logger.info(
            f"Starting Walk-Forward backtest ({calendar[0]} to {calendar[-1]}, "
            f"{len(calendar)} days, Capital: {cap0:,.2f} EUR)"
        )

        nav_unhedged = cap0
        nav_hedged = cap0

        current_weights_df = pd.DataFrame(columns=["ticker", "weight", "region_bloc"])
        current_contracts: dict[str, int] = {"ES": 0, "MES": 0, "FXXP": 0, "FESX": 0}
        prev_futures_quotes: dict[str, float] = dict(DEFAULT_FUTURES_QUOTES)

        daily_records: list[dict[str, object]] = []
        rebalance_records: list[dict[str, object]] = []

        total_rebalances = 0
        days_since_rebalance = freq  # Trigger rebalance on day 1

        for i, current_date in enumerate(calendar):
            is_rebalance_day = days_since_rebalance >= freq or i == 0
            daily_eq_cost = 0.0
            daily_fut_cost = 0.0

            # 1. Equity Rebalance Event
            if is_rebalance_day:
                total_rebalances += 1
                days_since_rebalance = 0

                # Load ranked candidates for date
                if ranked_universe_loader and callable(ranked_universe_loader):
                    ranked_univ = ranked_universe_loader(current_date)
                else:
                    ranked_univ = self._load_default_ranked_universe(current_date)

                current_held_tickers = (
                    current_weights_df["ticker"].tolist() if not current_weights_df.empty else []
                )

                # Filter using 3rd-decile turnover buffer
                selected_candidates = self.turnover_filter.filter_candidates_with_buffer(
                    ranked_universe=ranked_univ,
                    current_holdings=current_held_tickers,
                    top_na=self.config.allocation.na_top_stocks,
                    top_eu=self.config.allocation.eu_top_stocks,
                )

                if not selected_candidates.empty:
                    # Allocate optimal weights
                    target_weights_df, risk_summary = self.allocation_pipeline.allocate_candidates(
                        candidates_df=selected_candidates,
                        as_of_date=current_date,
                        method=allocation_method,
                    )

                    # Compute delta orders & execution costs
                    orders_df = self.turnover_filter.generate_delta_orders(
                        current_weights=current_weights_df,
                        target_weights=target_weights_df,
                        portfolio_value_eur=nav_hedged,
                    )
                    daily_eq_cost, _ = self.cost_model.calculate_equity_costs(
                        trades_df=orders_df,
                        portfolio_value_eur=nav_hedged,
                    )
                    current_weights_df = target_weights_df.copy()

                    rebalance_records.append(
                        {
                            "rebalance_date": current_date,
                            "n_candidates": len(selected_candidates),
                            "na_weight": risk_summary.na_weight,
                            "eu_weight": risk_summary.eu_weight,
                            "equity_cost_eur": daily_eq_cost,
                            "annualized_vol": risk_summary.annualized_volatility,
                        }
                    )
                    print(
                        f"[{current_date}] Rebalance #{total_rebalances:02d} | "
                        f"Allocated: {risk_summary.n_stocks} "
                        f"({risk_summary.n_na} NA, {risk_summary.n_eu} EU) | "
                        f"NAV: EUR {nav_hedged:,.2f}",
                        flush=True,
                    )
            else:
                days_since_rebalance += 1

            # 2. Daily Market Quotes & Returns
            # Fetch daily returns for held stocks
            stock_returns_map = self._fetch_daily_stock_returns(
                current_date=current_date,
                tickers=current_weights_df["ticker"].tolist(),
            )

            # Calculate weighted equity returns
            eq_ret_unhedged = 0.0
            na_sleeve_ret = 0.0
            eu_sleeve_ret = 0.0
            na_weight_sum = 0.0
            eu_weight_sum = 0.0

            if not current_weights_df.empty:
                for _, row in current_weights_df.iterrows():
                    t = str(row["ticker"])
                    w = float(row["weight"])
                    reg = str(row.get("region_bloc", "NA"))
                    ret_i = stock_returns_map.get(t, 0.0)
                    eq_ret_unhedged += w * ret_i

                    if reg == "NA":
                        na_sleeve_ret += w * ret_i
                        na_weight_sum += w
                    else:
                        eu_sleeve_ret += w * ret_i
                        eu_weight_sum += w

                if na_weight_sum > 0:
                    na_sleeve_ret /= na_weight_sum
                if eu_weight_sum > 0:
                    eu_sleeve_ret /= eu_weight_sum

            # 3. Dynamic Futures Overlay (if enabled)
            fut_pnl_eur = 0.0
            target_contracts = dict(current_contracts)

            daily_fx = (
                float(fx_rate_loader(current_date))
                if fx_rate_loader and callable(fx_rate_loader)
                else eur_usd_rate
            )

            # Get futures prices
            if futures_prices_loader and callable(futures_prices_loader):
                cur_fut_quotes = futures_prices_loader(current_date)
            else:
                cur_fut_quotes = dict(DEFAULT_FUTURES_QUOTES)

            # Compute variation margin from held contracts
            for sym, qty in current_contracts.items():
                if qty != 0:
                    p_prev = prev_futures_quotes.get(sym, cur_fut_quotes[sym])
                    p_now = cur_fut_quotes.get(sym, p_prev)
                    mult = FUTURE_MULTIPLIERS.get(sym, 50.0)
                    curr_code = FUTURE_CURRENCIES.get(sym, "EUR")
                    fx = daily_fx if curr_code == "USD" else 1.0

                    # Note: Hedging positions are SHORT, so profit when price falls:
                    # PnL = -qty * (p_now - p_prev) * mult / fx
                    sym_pnl = -float(qty) * (p_now - p_prev) * mult / fx
                    fut_pnl_eur += sym_pnl

            if enable_hedging and not current_weights_df.empty:
                # Run regime sensor and hedge sizing
                hedge_summary = self.regime_pipeline.execute_hedging_pipeline(
                    as_of_date=current_date,
                    portfolio_value_eur=nav_hedged,
                    allocations_df=current_weights_df,
                    futures_quotes=cur_fut_quotes,
                    eur_usd_rate=daily_fx,
                )

                na_sym = hedge_summary.na_sleeve.contract_symbol
                eu_sym = hedge_summary.eu_sleeve.contract_symbol
                target_contracts = {
                    "ES": hedge_summary.na_sleeve.contracts_rounded if na_sym == "ES" else 0,
                    "MES": hedge_summary.na_sleeve.contracts_rounded if na_sym == "MES" else 0,
                    "FXXP": hedge_summary.eu_sleeve.contracts_rounded if eu_sym == "FXXP" else 0,
                    "FESX": hedge_summary.eu_sleeve.contracts_rounded if eu_sym == "FESX" else 0,
                }

                # Contract adjustments and cost deduction
                contract_deltas: dict[str, int] = {}
                contract_notionals: dict[str, float] = {}

                for sym in ("ES", "MES", "FXXP", "FESX"):
                    d = target_contracts.get(sym, 0) - current_contracts.get(sym, 0)
                    if d != 0:
                        contract_deltas[sym] = d
                        p = cur_fut_quotes.get(sym, 5000.0)
                        m = FUTURE_MULTIPLIERS.get(sym, 50.0)
                        fx = daily_fx if FUTURE_CURRENCIES.get(sym, "EUR") == "USD" else 1.0
                        contract_notionals[sym] = p * m / fx

                if contract_deltas:
                    daily_fut_cost, _ = self.cost_model.calculate_futures_costs(
                        contracts_delta=contract_deltas,
                        contract_notionals_eur=contract_notionals,
                    )
                    current_contracts = dict(target_contracts)

            prev_futures_quotes = dict(cur_fut_quotes)

            # 4. Daily Portfolio Accounting
            total_daily_costs = daily_eq_cost + daily_fut_cost

            # Unhedged NAV step
            unhedged_pnl = nav_unhedged * eq_ret_unhedged - daily_eq_cost
            nav_unhedged = max(1.0, nav_unhedged + unhedged_pnl)
            unhedged_ret_step = unhedged_pnl / (nav_unhedged - unhedged_pnl)

            # Hedged NAV step
            hedged_pnl = (nav_hedged * eq_ret_unhedged) + fut_pnl_eur - total_daily_costs
            nav_hedged = max(1.0, nav_hedged + hedged_pnl)
            hedged_ret_step = hedged_pnl / (nav_hedged - hedged_pnl)

            daily_records.append(
                {
                    "date": current_date,
                    "unhedged_return": unhedged_ret_step,
                    "hedged_return": hedged_ret_step,
                    "nav_unhedged": nav_unhedged,
                    "nav_hedged": nav_hedged,
                    "futures_daily_pnl_eur": fut_pnl_eur,
                    "futures_cost_eur": daily_fut_cost,
                    "transaction_cost_eur": total_daily_costs,
                    "na_sleeve_return": na_sleeve_ret,
                    "eu_sleeve_return": eu_sleeve_ret,
                    "na_weight": na_weight_sum,
                    "eu_weight": eu_weight_sum,
                    "es_contracts": current_contracts.get("ES", 0),
                    "fxxp_contracts": current_contracts.get("FXXP", 0),
                }
            )

            if (i + 1) % 63 == 0 or (i + 1) == len(calendar):
                print(
                    f"  [Progress] Step {i + 1:3d}/{len(calendar)} ({current_date}) | "
                    f"Hedged NAV: EUR {nav_hedged:,.2f} | Unhedged NAV: EUR {nav_unhedged:,.2f}",
                    flush=True,
                )

        daily_history_df = pd.DataFrame(daily_records)
        rebalance_log_df = pd.DataFrame(rebalance_records)

        # 5. Build Final Quantitative Report
        report = build_full_performance_report(
            daily_history_df=daily_history_df,
            total_rebalances=total_rebalances,
            config=self.config,
        )

        return daily_history_df, report, rebalance_log_df

    def _load_default_ranked_universe(self, as_of_date: str) -> pd.DataFrame:
        """Load and rank candidate universe for as_of_date using AlphaPipeline or fallback."""
        if self.alpha_pipeline is not None:
            try:
                # Retrain expanding-window model if evaluating a new rebalance date
                if self._last_alpha_train_date != as_of_date:
                    purge_days = self.config.alpha.purge_days if self.config.alpha else 21
                    cutoff_dt = pd.Timestamp(as_of_date) - pd.tseries.offsets.BDay(purge_days)
                    cutoff_str = cutoff_dt.strftime("%Y-%m-%d")

                    query = (
                        "SELECT DISTINCT as_of_date FROM universe_pit "
                        "WHERE as_of_date <= ? ORDER BY as_of_date ASC"
                    )
                    with self.store.get_connection() as conn:
                        dates_df = pd.read_sql_query(query, conn, params=[cutoff_str])
                    if isinstance(dates_df, pd.DataFrame) and "as_of_date" in dates_df.columns:
                        hist_dates = [str(d) for d in dates_df["as_of_date"].tolist()]
                    else:
                        hist_dates = []

                    if len(hist_dates) >= 5:
                        labeled_data = self.alpha_pipeline.build_labeled_dataset(hist_dates)
                        clean_data = labeled_data.dropna(subset=["target"]).reset_index(drop=True)
                        if len(clean_data) >= 30:
                            from quant_portfolio.config import ALL_CORE_FEATURES

                            self.alpha_pipeline.model.fit(
                                clean_data[list(ALL_CORE_FEATURES)],
                                clean_data["target"],
                            )
                            self._last_alpha_train_date = as_of_date

                if self.alpha_pipeline.model.is_fitted:
                    features_t = self.alpha_pipeline.feature_pipeline.build_features_at_date(
                        as_of_date, normalize=True
                    )
                    if not features_t.empty:
                        scores = self.alpha_pipeline.model.predict(features_t)
                        scored_universe = features_t.copy()
                        scored_universe["score"] = scores
                        ranked = self.alpha_pipeline.selector.rank_full_universe(scored_universe)
                        if not ranked.empty:
                            return ranked
            except Exception as exc:
                logger.warning(
                    "AlphaPipeline ranking failed for %s, falling back to deterministic: %s",
                    as_of_date,
                    exc,
                )

        # Fallback loader extracting universe from SQLite if no custom loader is provided
        query = "SELECT MAX(as_of_date) FROM universe_pit WHERE as_of_date <= ?"
        with self.store.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, (as_of_date,))
            row = cur.fetchone()
            latest_date = str(row[0]) if row and row[0] is not None else None

        if latest_date is not None:
            df = self.store.get_universe_pit(as_of_date=latest_date, exclude_financials=True)
        else:
            df = self.store.get_universe_pit(exclude_financials=True)

        if df.empty:
            return pd.DataFrame(
                columns=["ticker", "region_bloc", "score", "regional_decile", "regional_rank"]
            )

        # Deterministic scoring fallback with full ranking and deciles
        unique_stocks = df.drop_duplicates(subset=["ticker"]).copy()
        unique_stocks["score"] = np.linspace(1.0, 0.0, len(unique_stocks))
        return self.sleeve_selector.rank_full_universe(unique_stocks)

    def _fetch_daily_stock_returns(
        self,
        current_date: str,
        tickers: Sequence[str],
    ) -> dict[str, float]:
        """Fetch daily return for held stocks on current_date from pre-loaded matrix or Parquet."""
        if not tickers:
            return {}

        ret_map: dict[str, float] = {}

        # 1. Try from pre-loaded returns matrix
        if self._returns_matrix is not None and current_date in self._returns_matrix.index:
            row = self._returns_matrix.loc[current_date]
            for t in tickers:
                if t in row.index:
                    val = row[t]
                    if pd.notna(val) and not np.isinf(cast("float", val)):
                        ret_map[t] = cast("float", val)
            return ret_map

        # 2. Fallback: compute (tot_return_t / tot_return_{t-1}) - 1 from recent parquet data
        try:
            start_trailing = (pd.Timestamp(current_date) - pd.Timedelta(days=7)).strftime(
                "%Y-%m-%d"
            )
            m_df = self.parquet_store.read_daily_market(
                start_date=start_trailing,
                end_date=current_date,
                tickers=list(tickers),
            )
            if not m_df.empty and "tot_return_idx_eur" in m_df.columns:
                p_df = m_df.pivot(index="date", columns="ticker", values="tot_return_idx_eur")
                pct_df = p_df.pct_change(fill_method=None)
                if current_date in pct_df.index:
                    row = pct_df.loc[current_date]
                    for t in tickers:
                        if t in row.index:
                            val = row[t]
                            if pd.notna(val) and not np.isinf(cast("float", val)):
                                ret_map[t] = cast("float", val)
        except Exception as exc:
            logger.debug("Fallback stock returns fetch failed for %s: %s", current_date, exc)

        return ret_map
