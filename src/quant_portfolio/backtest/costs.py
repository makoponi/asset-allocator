"""Transaction cost modeling for transatlantic equity and index futures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from quant_portfolio.config import BacktestConfig, PortfolioConfig

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd


class TransactionCostBreakdown(BaseModel):
    """Itemized breakdown of transaction costs for a rebalancing event."""

    na_equity_traded_eur: float = Field(default=0.0, ge=0.0)
    na_equity_cost_eur: float = Field(default=0.0, ge=0.0)
    eu_equity_traded_eur: float = Field(default=0.0, ge=0.0)
    eu_equity_cost_eur: float = Field(default=0.0, ge=0.0)
    futures_traded_eur: float = Field(default=0.0, ge=0.0)
    futures_cost_eur: float = Field(default=0.0, ge=0.0)
    total_cost_eur: float = Field(default=0.0, ge=0.0)


class TransactionCostModel:
    """Calculates realistic transaction costs for equities (NA 5 bps, EU 10 bps) and futures."""

    def __init__(self, config: PortfolioConfig | None = None) -> None:
        self.config = config or PortfolioConfig()
        self.bt_config: BacktestConfig = self.config.backtest or BacktestConfig()

        self.bps_na: float = self.bt_config.cost_bps_na_equity
        self.bps_eu: float = self.bt_config.cost_bps_eu_equity
        self.bps_futures: float = self.bt_config.cost_bps_futures

    def calculate_equity_costs(
        self,
        trades_df: pd.DataFrame,
        portfolio_value_eur: float | None = None,
    ) -> tuple[float, pd.DataFrame]:
        """Compute execution costs for equity trades.

        Parameters
        ----------
        trades_df : pd.DataFrame
            DataFrame with columns:
            - `region_bloc`: 'NA' or 'EU'
            And either:
            - `traded_value_eur`: absolute traded value in EUR
            Or:
            - `weight_delta`: change in weight (traded_value = abs(weight_delta) * portfolio_value)
        portfolio_value_eur : float | None
            Portfolio value in EUR, required if `weight_delta` is supplied.

        Returns
        -------
        tuple[float, pd.DataFrame]
            Total equity cost in EUR and enriched DataFrame with itemized `cost_eur`.
        """
        if trades_df.empty:
            empty_df = trades_df.copy()
            empty_df["cost_eur"] = 0.0
            return 0.0, empty_df

        df = trades_df.copy()
        if "traded_value_eur" not in df.columns:
            if "weight_delta" not in df.columns:
                raise ValueError(
                    "trades_df must contain either 'traded_value_eur' or 'weight_delta'."
                )
            if portfolio_value_eur is None or portfolio_value_eur <= 0:
                raise ValueError("portfolio_value_eur must be positive when using 'weight_delta'.")
            df["traded_value_eur"] = df["weight_delta"].abs() * float(portfolio_value_eur)
        else:
            df["traded_value_eur"] = df["traded_value_eur"].abs()

        if "region_bloc" not in df.columns:
            raise ValueError("trades_df must contain 'region_bloc' ('NA' or 'EU').")

        # Vectorized cost assignment based on regional sleeve
        rate_na = self.bps_na / 10_000.0
        rate_eu = self.bps_eu / 10_000.0

        is_na = df["region_bloc"].astype(str).str.upper() == "NA"
        rates = np.where(is_na, rate_na, rate_eu)

        df["cost_rate_bps"] = np.where(is_na, self.bps_na, self.bps_eu)
        df["cost_eur"] = df["traded_value_eur"] * rates

        total_cost = float(df["cost_eur"].sum())
        return total_cost, df

    def calculate_futures_costs(
        self,
        contracts_delta: Mapping[str, int],
        contract_notionals_eur: Mapping[str, float],
    ) -> tuple[float, dict[str, float]]:
        """Compute execution costs for futures contracts adjustments or rolls.

        Parameters
        ----------
        contracts_delta : Mapping[str, int]
            Number of contracts traded per symbol (absolute value used for cost).
        contract_notionals_eur : Mapping[str, float]
            Unit contract notional value in EUR for each symbol.

        Returns
        -------
        tuple[float, dict[str, float]]
            Total futures cost in EUR and dictionary of cost per contract symbol.
        """
        rate_futures = self.bps_futures / 10_000.0
        costs_by_symbol: dict[str, float] = {}
        total_cost = 0.0

        for symbol, delta in contracts_delta.items():
            abs_delta = abs(int(delta))
            if abs_delta == 0:
                costs_by_symbol[symbol] = 0.0
                continue

            notional = float(contract_notionals_eur.get(symbol, 0.0))
            traded_notional = abs_delta * notional
            cost = traded_notional * rate_futures
            costs_by_symbol[symbol] = cost
            total_cost += cost

        return total_cost, costs_by_symbol

    def calculate_total_rebalance_costs(
        self,
        equity_trades_df: pd.DataFrame,
        portfolio_value_eur: float,
        futures_delta: Mapping[str, int] | None = None,
        futures_notionals_eur: Mapping[str, float] | None = None,
    ) -> TransactionCostBreakdown:
        """Calculate complete rebalancing costs across equities and futures overlay."""
        eq_total, enriched_eq = self.calculate_equity_costs(
            trades_df=equity_trades_df,
            portfolio_value_eur=portfolio_value_eur,
        )

        if not enriched_eq.empty:
            na_mask = enriched_eq["region_bloc"].astype(str).str.upper() == "NA"
            na_traded = float(enriched_eq.loc[na_mask, "traded_value_eur"].sum())
            na_cost = float(enriched_eq.loc[na_mask, "cost_eur"].sum())
            eu_traded = float(enriched_eq.loc[~na_mask, "traded_value_eur"].sum())
            eu_cost = float(enriched_eq.loc[~na_mask, "cost_eur"].sum())
        else:
            na_traded, na_cost, eu_traded, eu_cost = 0.0, 0.0, 0.0, 0.0

        fut_traded = 0.0
        fut_cost = 0.0
        if futures_delta and futures_notionals_eur:
            for sym, d in futures_delta.items():
                abs_d = abs(int(d))
                notional = float(futures_notionals_eur.get(sym, 0.0))
                fut_traded += abs_d * notional
            fut_cost, _ = self.calculate_futures_costs(futures_delta, futures_notionals_eur)

        return TransactionCostBreakdown(
            na_equity_traded_eur=na_traded,
            na_equity_cost_eur=na_cost,
            eu_equity_traded_eur=eu_traded,
            eu_equity_cost_eur=eu_cost,
            futures_traded_eur=fut_traded,
            futures_cost_eur=fut_cost,
            total_cost_eur=na_cost + eu_cost + fut_cost,
        )
