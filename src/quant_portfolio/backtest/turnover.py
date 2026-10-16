"""Turnover buffer management and delta order generation for transatlantic rebalancing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import pandas as pd
from pydantic import BaseModel, Field

from quant_portfolio.config import BacktestConfig, PortfolioConfig, RegionBloc

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

OrderAction = Literal["BUY", "SELL", "HOLD"]


class DeltaOrder(BaseModel):
    """Specific order to transition an asset from current weight to target weight."""

    ticker: str
    region_bloc: RegionBloc
    current_weight: float = Field(ge=0.0, le=1.0)
    target_weight: float = Field(ge=0.0, le=1.0)
    weight_delta: float
    action: OrderAction
    estimated_trade_value_eur: float = Field(ge=0.0)


class TurnoverBufferFilter:
    """Applies regional decile buffer to held assets and selects replacements.

    Rule:
    - Held assets are maintained if regional_decile <= buffer_decile_max (default: 3).
    - Held assets are liquidated if regional_decile > buffer_decile_max or removed from universe.
    - Vacant quota in each sleeve is filled by the highest-scoring non-held assets.
    """

    def __init__(self, config: PortfolioConfig | None = None) -> None:
        self.config = config or PortfolioConfig()
        self.bt_config: BacktestConfig = self.config.backtest or BacktestConfig()
        self.max_decile: int = self.bt_config.turnover_buffer_decile_max
        self.target_na: int = self.config.allocation.na_top_stocks
        self.target_eu: int = self.config.allocation.eu_top_stocks

    def filter_candidates_with_buffer(
        self,
        ranked_universe: pd.DataFrame,
        current_holdings: Sequence[str] | pd.Series | None = None,
        top_na: int | None = None,
        top_eu: int | None = None,
    ) -> pd.DataFrame:
        """Filter universe applying 3rd-decile turnover buffer to held assets.

        Parameters
        ----------
        ranked_universe : pd.DataFrame
            DataFrame from StratifiedSleeveSelector.rank_full_universe, containing:
            `ticker`, `region_bloc`, `score`, `regional_decile`, `regional_rank`.
        current_holdings : Sequence[str] | pd.Series | None
            List or Series of currently held tickers. If empty or None, top_k selection is used.
        top_na : int | None
            Target count of NA stocks (default from config: 35).
        top_eu : int | None
            Target count of EU stocks (default from config: 25).

        Returns
        -------
        pd.DataFrame
            Selected candidate assets (totale ~50-60) with columns preserved.
        """
        if ranked_universe.empty:
            return pd.DataFrame(columns=ranked_universe.columns)

        k_na = top_na or self.target_na
        k_eu = top_eu or self.target_eu
        limits: dict[str, int] = {"NA": k_na, "EU": k_eu}

        held_set: set[str] = set(current_holdings) if current_holdings is not None else set()

        selected_frames: list[pd.DataFrame] = []

        for reg in ("NA", "EU"):
            reg_df = ranked_universe[ranked_universe["region_bloc"] == reg].copy()
            if reg_df.empty:
                continue

            k_limit = limits[reg]

            if not held_set:
                # Cold start: select top k_limit by score descending
                chosen = (
                    reg_df.sort_values(by=["score", "ticker"], ascending=[False, True])
                    .head(k_limit)
                    .copy()
                )
                selected_frames.append(chosen)
                continue

            # Identify held stocks in this region
            is_held = reg_df["ticker"].isin(held_set)
            held_df = reg_df[is_held].copy()
            unheld_df = reg_df[~is_held].copy()

            # Rule: Maintain held assets if regional_decile <= max_decile
            maintained_df = held_df[held_df["regional_decile"] <= self.max_decile].copy()

            if len(maintained_df) >= k_limit:
                # If more than k_limit qualify, keep top k_limit by score
                chosen = (
                    maintained_df.sort_values(by=["score", "ticker"], ascending=[False, True])
                    .head(k_limit)
                    .copy()
                )
            else:
                needed = k_limit - len(maintained_df)
                # Fill remaining spots with highest-scoring unheld assets
                new_additions = (
                    unheld_df.sort_values(by=["score", "ticker"], ascending=[False, True])
                    .head(needed)
                    .copy()
                )
                chosen = pd.concat([maintained_df, new_additions], ignore_index=True)

            selected_frames.append(chosen)

        if not selected_frames:
            return pd.DataFrame(columns=ranked_universe.columns)

        final_candidates = pd.concat(selected_frames, ignore_index=True)
        return final_candidates.sort_values(
            by=["region_bloc", "score"], ascending=[True, False]
        ).reset_index(drop=True)

    def generate_delta_orders(
        self,
        current_weights: pd.DataFrame,
        target_weights: pd.DataFrame,
        portfolio_value_eur: float,
        min_trade_threshold_eur: float = 100.0,
    ) -> pd.DataFrame:
        """Generate itemized rebalance orders comparing current and target weights.

        Parameters
        ----------
        current_weights : pd.DataFrame
            DataFrame with columns `ticker`, `weight` (and optionally `region_bloc`).
        target_weights : pd.DataFrame
            DataFrame with columns `ticker`, `weight`, `region_bloc`.
        portfolio_value_eur : float
            Current total portfolio value in EUR.
        min_trade_threshold_eur : float
            Minimum trade value to avoid micro-transactions.

        Returns
        -------
        pd.DataFrame
            DataFrame of DeltaOrder records.
        """
        curr = (
            current_weights.copy()
            if not current_weights.empty
            else pd.DataFrame(columns=["ticker", "weight", "region_bloc"])
        )
        targ = (
            target_weights.copy()
            if not target_weights.empty
            else pd.DataFrame(columns=["ticker", "weight", "region_bloc"])
        )

        curr_map = dict(zip(curr["ticker"], curr["weight"], strict=False))
        targ_map = dict(zip(targ["ticker"], targ["weight"], strict=False))

        # Build region mapping from target, fallback to current
        region_map: dict[str, RegionBloc] = {}
        if "region_bloc" in targ.columns:
            region_map.update(dict(zip(targ["ticker"], targ["region_bloc"], strict=False)))
        if "region_bloc" in curr.columns:
            for t, r in zip(curr["ticker"], curr["region_bloc"], strict=False):
                if t not in region_map:
                    region_map[t] = r

        all_tickers = sorted(set(curr_map.keys()) | set(targ_map.keys()))
        orders: list[dict[str, object]] = []

        for t in all_tickers:
            w_curr = float(curr_map.get(t, 0.0))
            w_targ = float(targ_map.get(t, 0.0))
            delta = w_targ - w_curr
            trade_val = abs(delta) * portfolio_value_eur

            if trade_val < min_trade_threshold_eur and abs(delta) < 1e-5:
                continue

            if delta > 1e-5:
                action = "BUY"
            elif delta < -1e-5:
                action = "SELL"
            else:
                action = "HOLD"

            reg = region_map.get(t, "NA")

            orders.append(
                {
                    "ticker": t,
                    "region_bloc": reg,
                    "current_weight": w_curr,
                    "target_weight": w_targ,
                    "weight_delta": delta,
                    "action": action,
                    "estimated_trade_value_eur": trade_val,
                }
            )

        if not orders:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "region_bloc",
                    "current_weight",
                    "target_weight",
                    "weight_delta",
                    "action",
                    "estimated_trade_value_eur",
                ]
            )

        return pd.DataFrame(orders)
