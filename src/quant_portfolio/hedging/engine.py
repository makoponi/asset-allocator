"""Dual-Sleeve Beta-Adjusted Futures Hedging Engine."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_portfolio.config import (
    FUTURE_CURRENCIES,
    FUTURE_MULTIPLIERS,
    PortfolioConfig,
    RegionBloc,
)


@dataclass(frozen=True)
class SleeveHedgePlan:
    """Hedging plan and sizing details for an individual regional sleeve."""

    region_bloc: RegionBloc
    sleeve_weight: float
    sleeve_value_eur: float
    sleeve_beta: float
    regime_state: int  # 0: Calm, 1: Alert, 2: Stress
    regime_name: str
    hedge_ratio: float
    hedged_notional_eur: float
    contract_symbol: str
    contract_price: float
    contract_multiplier: float
    contract_currency: str
    contract_value_eur: float
    contracts_exact: float
    contracts_rounded: int


@dataclass(frozen=True)
class DualSleeveHedgeSummary:
    """Consolidated transatlantic hedging specification for portfolio management."""

    as_of_date: str
    total_portfolio_value_eur: float
    total_hedged_notional_eur: float
    portfolio_effective_hedge_ratio: float
    na_sleeve: SleeveHedgePlan
    eu_sleeve: SleeveHedgePlan

    def to_dataframe(self) -> pd.DataFrame:
        """Export summary as tabular DataFrame."""
        rows = []
        for sleeve in (self.na_sleeve, self.eu_sleeve):
            rows.append(
                {
                    "as_of_date": self.as_of_date,
                    "region_bloc": sleeve.region_bloc,
                    "sleeve_weight": sleeve.sleeve_weight,
                    "sleeve_value_eur": sleeve.sleeve_value_eur,
                    "sleeve_beta": sleeve.sleeve_beta,
                    "regime_state": sleeve.regime_state,
                    "regime_name": sleeve.regime_name,
                    "hedge_ratio": sleeve.hedge_ratio,
                    "hedged_notional_eur": sleeve.hedged_notional_eur,
                    "contract": sleeve.contract_symbol,
                    "contract_price": sleeve.contract_price,
                    "contract_currency": sleeve.contract_currency,
                    "contract_val_eur": sleeve.contract_value_eur,
                    "contracts_short": sleeve.contracts_rounded,
                }
            )
        return pd.DataFrame(rows)


class DualSleeveHedgeEngine:
    """Calculates sleeve-level beta-adjusted futures overlay sizes."""

    def __init__(self, config: PortfolioConfig | None = None) -> None:
        self.config = config or PortfolioConfig()

    def compute_sleeve_beta(
        self,
        allocations_df: pd.DataFrame,
        region_bloc: RegionBloc,
        ticker_betas: Mapping[str, float] | None = None,
    ) -> float:
        """Compute ex-ante capital-weighted beta for a regional sleeve.

        Parameters
        ----------
        allocations_df : pd.DataFrame
            DataFrame with columns 'ticker', 'region_bloc', and 'weight'.
        region_bloc : RegionBloc
            'NA' or 'EU'.
        ticker_betas : Mapping[str, float] | None
            Mapping from ticker to regional beta. If not provided or missing for a ticker,
            checks if 'beta' is already a column in allocations_df or defaults to 1.0.

        Returns
        -------
        float
            Weighted sleeve beta against regional index (SPX for NA, SXXP for EU).
        """
        sleeve_df = allocations_df[allocations_df["region_bloc"] == region_bloc].copy()
        if sleeve_df.empty:
            return 1.0

        total_weight = float(sleeve_df["weight"].sum())
        if total_weight <= 0.0:
            return 1.0

        sleeve_df["rel_weight"] = sleeve_df["weight"] / total_weight

        betas: list[float] = []
        for _, row in sleeve_df.iterrows():
            t = str(row["ticker"])
            if ticker_betas is not None and t in ticker_betas:
                betas.append(float(ticker_betas[t]))
            elif "beta" in row and not pd.isna(row["beta"]):
                betas.append(float(row["beta"]))
            elif "BETA_REGIONAL_252D" in row and not pd.isna(row["BETA_REGIONAL_252D"]):
                betas.append(float(row["BETA_REGIONAL_252D"]))
            else:
                betas.append(1.0)

        sleeve_df["beta_val"] = betas
        weighted_beta = float((sleeve_df["rel_weight"] * sleeve_df["beta_val"]).sum())
        return max(0.1, weighted_beta)

    def calculate_hedging_plan(
        self,
        as_of_date: str,
        allocations_df: pd.DataFrame,
        filtered_regimes: Mapping[RegionBloc, int],
        portfolio_value_eur: float,
        futures_quotes: Mapping[str, float],
        eur_usd_rate: float = 1.08,
        ticker_betas: Mapping[str, float] | None = None,
        custom_na_future: str | None = None,
        custom_eu_future: str | None = None,
    ) -> DualSleeveHedgeSummary:
        """Calculate complete hedging plan and futures contracts to sell.

        Parameters
        ----------
        as_of_date : str
            Date of allocation and hedging snapshot.
        allocations_df : pd.DataFrame
            Portfolio holdings DataFrame containing 'ticker', 'region_bloc', 'weight'.
        filtered_regimes : Mapping[RegionBloc, int]
            Hysteresis-filtered regime state (0: Calm, 1: Alert, 2: Stress) for 'NA' and 'EU'.
        portfolio_value_eur : float
            Total capital allocated in EUR.
        futures_quotes : Mapping[str, float]
            Dictionary of current futures prices (e.g. {'ES': 5800.0, 'FXXP': 520.0}).
        eur_usd_rate : float
            EUR/USD exchange rate.
        ticker_betas : Mapping[str, float] | None
            Optional ticker-to-beta mapping.
        custom_na_future : str | None
            Override default NA contract ('ES' or 'MES').
        custom_eu_future : str | None
            Override default EU contract ('FXXP' or 'FESX').

        Returns
        -------
        DualSleeveHedgeSummary
            Complete hedging plan with contract counts and notional amounts.
        """
        if portfolio_value_eur <= 0.0:
            raise ValueError(f"portfolio_value_eur must be positive, got {portfolio_value_eur}")
        if eur_usd_rate <= 0.0:
            raise ValueError(f"eur_usd_rate must be positive, got {eur_usd_rate}")

        na_symbol = custom_na_future or self.config.regime.default_na_future
        eu_symbol = custom_eu_future or self.config.regime.default_eu_future

        state_names = {0: "Calm", 1: "Alert", 2: "Severe Stress"}
        hedge_multipliers = self.config.regime.hedge_ratios  # {0: 0.0, 1: 0.5, 2: 1.0}

        plans: dict[RegionBloc, SleeveHedgePlan] = {}

        for bloc in ("NA", "EU"):
            region_bloc: RegionBloc = "NA" if bloc == "NA" else "EU"
            sub_df = allocations_df[allocations_df["region_bloc"] == region_bloc]
            sleeve_w = float(sub_df["weight"].sum()) if not sub_df.empty else 0.0
            sleeve_val = portfolio_value_eur * sleeve_w

            sleeve_beta = self.compute_sleeve_beta(
                allocations_df=allocations_df,
                region_bloc=region_bloc,
                ticker_betas=ticker_betas,
            )

            state = filtered_regimes.get(region_bloc, 0)
            base_ratio = hedge_multipliers.get(state, 0.0)
            # Beta-adjusted hedge ratio
            hedge_ratio = base_ratio * sleeve_beta
            hedged_notional = sleeve_val * hedge_ratio

            symbol = na_symbol if region_bloc == "NA" else eu_symbol
            px = float(futures_quotes.get(symbol, 1.0))
            multiplier = float(FUTURE_MULTIPLIERS.get(symbol, 50.0))
            currency = FUTURE_CURRENCIES.get(symbol, "EUR")

            # Contract value in EUR
            if currency == "USD":
                contract_val_eur = (px * multiplier) / eur_usd_rate
            else:
                contract_val_eur = px * multiplier

            if contract_val_eur > 0.0 and hedged_notional > 0.0:
                contracts_exact = hedged_notional / contract_val_eur
                contracts_rounded = int(np.round(contracts_exact))
            else:
                contracts_exact = 0.0
                contracts_rounded = 0

            plans[region_bloc] = SleeveHedgePlan(
                region_bloc=region_bloc,
                sleeve_weight=sleeve_w,
                sleeve_value_eur=sleeve_val,
                sleeve_beta=sleeve_beta,
                regime_state=state,
                regime_name=state_names.get(state, "Unknown"),
                hedge_ratio=hedge_ratio,
                hedged_notional_eur=hedged_notional,
                contract_symbol=symbol,
                contract_price=px,
                contract_multiplier=multiplier,
                contract_currency=currency,
                contract_value_eur=contract_val_eur,
                contracts_exact=contracts_exact,
                contracts_rounded=contracts_rounded,
            )

        total_hedged = plans["NA"].hedged_notional_eur + plans["EU"].hedged_notional_eur
        effective_ratio = total_hedged / portfolio_value_eur if portfolio_value_eur > 0 else 0.0

        return DualSleeveHedgeSummary(
            as_of_date=as_of_date,
            total_portfolio_value_eur=portfolio_value_eur,
            total_hedged_notional_eur=total_hedged,
            portfolio_effective_hedge_ratio=effective_ratio,
            na_sleeve=plans["NA"],
            eu_sleeve=plans["EU"],
        )
