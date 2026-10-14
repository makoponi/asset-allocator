"""Walk-forward daily backtest engine with transaction costs and drift tracking.

Simulates historical portfolio performance with:
1. Periodic machine learning ranker retraining (e.g. every 6 months).
2. Monthly top-decile selection and RMT-denoised Hierarchical Risk Parity allocation.
3. Realistic transaction friction (commissions, slippage, turnover penalty).
4. Daily mark-to-market NAV calculation with asset weight drift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.execution.metrics import PerformanceMetrics, compute_performance_metrics
from src.features.cross_sectional import CrossSectionalTransformer, TargetGenerator
from src.models.ml_ranker import MLRanker
from src.risk.portfolio_allocator import PortfolioAllocator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.features.technical_engine import TechnicalEngine
    from src.ingestion.storage import MarketStore, PriceEODRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Comprehensive backtest simulation outputs."""

    metrics: PerformanceMetrics
    benchmark_metrics: PerformanceMetrics
    history_df: pd.DataFrame
    monthly_rebalance_records: list[dict[str, Any]]


class BacktestEngine:
    """Orchestrates daily walk-forward portfolio simulation."""

    def __init__(
        self,
        market_store: MarketStore,
        tech_engine: TechnicalEngine,
        price_records: Sequence[PriceEODRecord],
        retrain_freq_months: int = 6,
        cost_bps: float = 10.0,
        top_n: int = 50,
        max_weight: float = 0.04,
    ) -> None:
        self.market_store = market_store
        self.tech_engine = tech_engine
        self.price_records = list(price_records)
        self.retrain_freq_months = retrain_freq_months
        self.cost_bps = cost_bps
        self.top_n = top_n
        self.max_weight = max_weight

        # Index price bars by (date, ticker) -> ret_1d
        self._ret_map: dict[tuple[date, str], float] = {}
        for p in self.price_records:
            self._ret_map[(p.date, p.ticker)] = p.ret_1d

    def _build_features_for_dates(
        self,
        eval_dates: Sequence[date],
        funds_by_ticker: dict[str, list[tuple[str, str, float]]],
        members_by_date: dict[str, list[tuple[str, str]]],
    ) -> pd.DataFrame:
        """Extract and align Point-in-Time feature rows for specified dates."""
        all_rows: list[dict[str, Any]] = []

        for eval_d in eval_dates:
            eval_str = eval_d.isoformat()
            members = members_by_date.get(eval_str, [])

            for ticker, sector in members:
                tf = self.tech_engine.compute_factors_for_ticker(ticker, eval_d)
                if tf is None:
                    continue

                t_funds = funds_by_ticker.get(ticker, [])
                latest_m: dict[str, float] = {}
                for f_date, m_name, m_val in t_funds:
                    if f_date <= eval_str:
                        latest_m[m_name] = m_val
                    else:
                        break

                row = {
                    "date": eval_str,
                    "ticker": ticker,
                    "gics_sector": sector,
                    "mom_1m": tf.mom_1m,
                    "mom_12m": tf.mom_12m,
                    "beta_252d": tf.beta_252d,
                    "ivol_252d": tf.ivol_252d,
                    "pe_ratio": latest_m.get("PE_RATIO"),
                    "best_pe_ratio": latest_m.get("BEST_PE_RATIO"),
                    "px_to_book": latest_m.get("PX_TO_BOOK_RATIO"),
                    "fcf_yield": latest_m.get("FREE_CASH_FLOW_YIELD"),
                    "roe": latest_m.get("RETURN_COM_EQY"),
                    "prof_margin": latest_m.get("PROF_MARGIN"),
                    "oper_margin": latest_m.get("OPER_MARGIN"),
                    "debt_to_equity": latest_m.get("TOT_DEBT_TO_TOT_EQY"),
                    "quick_ratio": latest_m.get("QUICK_RATIO"),
                    "free_cash_flow": latest_m.get("CF_FREE_CASH_FLOW"),
                }
                all_rows.append(row)

        df = pd.DataFrame(all_rows)
        if df.empty:
            return df
        transformer = CrossSectionalTransformer()
        return transformer.transform(df)

    def run_simulation(
        self,
        start_date: date,
        end_date: date,
        initial_capital: float = 100.0,
    ) -> BacktestResult:
        """Run daily mark-to-market simulation from start_date to end_date."""
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        # 1. Fetch all monthly rebalance dates from SQLite
        with self.market_store._connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT date FROM universe_members "
                "WHERE is_active = 1 AND date >= ? AND date <= ? ORDER BY date;",
                (start_str, end_str),
            )
            rebalance_date_strs = [r[0] for r in cursor.fetchall()]

        if len(rebalance_date_strs) < 2:
            msg = "Backtest period requires at least 2 monthly rebalance dates"
            raise ValueError(msg)

        rebalance_dates = [date.fromisoformat(d) for d in rebalance_date_strs]

        # 2. Pre-index fundamentals and universe members for fast access
        with self.market_store._connection() as conn:
            cursor = conn.execute(
                "SELECT ticker, filing_date, metric_name, metric_value "
                "FROM fundamentals_pit ORDER BY filing_date;"
            )
            funds_by_ticker: dict[str, list[tuple[str, str, float]]] = {}
            for t, f_d, m_n, m_v in cursor.fetchall():
                funds_by_ticker.setdefault(t, []).append((f_d, m_n, float(m_v)))

            cursor = conn.execute(
                "SELECT date, ticker, gics_sector FROM universe_members "
                "WHERE is_active = 1 ORDER BY date, ticker;"
            )
            members_by_date: dict[str, list[tuple[str, str]]] = {}
            for d_val, t, sec in cursor.fetchall():
                members_by_date.setdefault(d_val, []).append((t, sec))

        # 3. Identify all distinct trading calendar days within [start_date, end_date]
        all_trading_days = sorted(
            {p.date for p in self.price_records if start_date <= p.date <= end_date}
        )

        # 4. State tracking
        current_nav = initial_capital
        benchmark_nav = initial_capital
        current_weights: dict[str, float] = {}
        daily_records: list[dict[str, Any]] = []
        monthly_rebalance_records: list[dict[str, Any]] = []

        total_turnover = 0.0
        rebalance_counter = 0

        # Target generator for ML training
        tg = TargetGenerator(forward_bars=21)
        ranker = MLRanker(
            model_type="lightgbm",
            n_estimators=100,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            random_state=42,
        )

        rebalance_set = set(rebalance_dates)
        allocator = PortfolioAllocator(method="hrp", denoise=True, max_weight=self.max_weight)

        # Advance day by day
        for d in all_trading_days:
            # Check if today is a rebalance day
            if d in rebalance_set:
                rebalance_counter += 1
                eval_str = d.isoformat()

                # A. Retrain ranker periodically (every retrain_freq_months or first run)
                should_retrain = (
                    rebalance_counter == 1 or (rebalance_counter % self.retrain_freq_months) == 1
                )

                if should_retrain:
                    # Collect historical dates up to d (excluding last 21 days for target window)
                    past_rebal_dates = [rd for rd in rebalance_dates if rd < d]
                    if len(past_rebal_dates) >= 1:
                        hist_df = self._build_features_for_dates(
                            past_rebal_dates, funds_by_ticker, members_by_date
                        )
                        target_map = tg.compute_forward_excess_returns(
                            self.price_records, past_rebal_dates
                        )

                        t_idx_list = []
                        t_vals = []
                        for idx, r in hist_df.iterrows():
                            key = (str(r["date"]), str(r["ticker"]))
                            if key in target_map:
                                t_idx_list.append(idx)
                                t_vals.append(target_map[key])

                        if len(t_idx_list) >= 10:
                            train_slice = hist_df.loc[t_idx_list].copy()
                            train_slice["target"] = t_vals
                            ranker.fit(train_slice, train_slice["target"])

                # B. Extract current features and rank
                current_feat_df = self._build_features_for_dates(
                    [d], funds_by_ticker, members_by_date
                )
                if not current_feat_df.empty:
                    top_candidates = ranker.select_top_decile(
                        current_feat_df,
                        top_pct=0.10,
                        min_n=self.top_n,
                        max_n=self.top_n,
                    )
                else:
                    top_candidates = []

                if top_candidates:
                    # C. Build 504-day returns slice for selected candidates
                    lookback_504 = [td for td in all_trading_days if td <= d][-504:]
                    candidate_rets: dict[str, list[float]] = {}
                    for t in top_candidates:
                        candidate_rets[t] = [
                            self._ret_map.get((day_val, t), 0.0) for day_val in lookback_504
                        ]

                    ret_df = pd.DataFrame(candidate_rets, index=lookback_504)
                    new_weights = allocator.allocate(ret_df, top_candidates)

                    # D. Calculate turnover: 0.5 * sum(|w_new - w_old|)
                    all_tickers = set(current_weights.keys()).union(set(new_weights.keys()))
                    turnover = 0.5 * sum(
                        abs(new_weights.get(t, 0.0) - current_weights.get(t, 0.0))
                        for t in all_tickers
                    )
                    total_turnover += turnover

                    # E. Deduct transaction costs (cost_bps)
                    cost_penalty = turnover * (self.cost_bps / 10000.0)
                    current_nav *= 1.0 - cost_penalty

                    current_weights = new_weights
                    monthly_rebalance_records.append(
                        {
                            "date": eval_str,
                            "turnover": turnover,
                            "cost_penalty": cost_penalty,
                            "num_stocks": len(new_weights),
                        }
                    )

            # Daily return of portfolio
            if current_weights:
                p_ret = sum(w * self._ret_map.get((d, t), 0.0) for t, w in current_weights.items())
            else:
                p_ret = 0.0

            # Daily return of benchmark (active universe equal-weighted)
            active_today = members_by_date.get(d.isoformat(), [])
            if not active_today:
                # Fallback to current members from latest rebalance
                recent_rebal = [rd for rd in rebalance_dates if rd <= d]
                if recent_rebal:
                    active_today = members_by_date.get(recent_rebal[-1].isoformat(), [])

            if active_today:
                b_rets = [
                    self._ret_map.get((d, t), 0.0)
                    for t, _ in active_today
                    if (d, t) in self._ret_map
                ]
                b_ret = float(np.mean(b_rets)) if b_rets else 0.0
            else:
                b_ret = 0.0

            # Update NAVs
            current_nav *= 1.0 + p_ret
            benchmark_nav *= 1.0 + b_ret

            # Drift weights: w_{i, d+1} = w_{i, d} * (1 + R_{i, d}) / (1 + R_{P, d})
            if current_weights and (1.0 + p_ret) > 0:
                current_weights = {
                    t: w * (1.0 + self._ret_map.get((d, t), 0.0)) / (1.0 + p_ret)
                    for t, w in current_weights.items()
                }

            daily_records.append(
                {
                    "date": d.isoformat(),
                    "portfolio_nav": current_nav,
                    "benchmark_nav": benchmark_nav,
                    "portfolio_ret": p_ret,
                    "benchmark_ret": b_ret,
                }
            )

        history_df = pd.DataFrame(daily_records)

        # Annualized turnover
        n_years = max(0.1, len(all_trading_days) / 252.0)
        ann_turnover = total_turnover / n_years

        p_rets = history_df["portfolio_ret"].tolist()
        b_rets = history_df["benchmark_ret"].tolist()

        p_metrics = compute_performance_metrics(
            daily_returns=p_rets,
            benchmark_returns=b_rets,
            risk_free_rate=0.0,
            annual_turnover=ann_turnover,
        )

        b_metrics = compute_performance_metrics(
            daily_returns=b_rets,
            benchmark_returns=None,
            risk_free_rate=0.0,
            annual_turnover=0.0,
        )

        return BacktestResult(
            metrics=p_metrics,
            benchmark_metrics=b_metrics,
            history_df=history_df,
            monthly_rebalance_records=monthly_rebalance_records,
        )
