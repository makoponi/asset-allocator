"""Dynamic Hedging Overlay module using Macro Regimes and Hysteresis.

Determines the capital fraction to hedge via index futures (FESX / FXXP) based on
3-state HMM filtered probabilities and enforces a 3-day persistence hysteresis band
to eliminate whipsaw turnover and trading friction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class HedgingDecision:
    """Daily hedging decision record."""

    date: str
    dominant_regime: int
    target_hedge_ratio: float
    effective_hedge_ratio: float
    consecutive_days_in_new_regime: int
    num_future_contracts: int


class HedgingOverlay:
    """Manages index future overlay with discrete probability thresholds and hysteresis.

    Rules from GEMINI.md:
    - P(State 0) > 0.65 -> Target Hedge Ratio = 0.0%
    - P(State 1) > 0.65 -> Target Hedge Ratio = 40.0%
    - P(State 2) > 0.65 -> Target Hedge Ratio = 80.0% - 100.0%
    - Hysteresis: Effective hedge ratio only transitions if target regime persists >= 3 days.
    """

    def __init__(
        self,
        hysteresis_days: int = 3,
        threshold_prob: float = 0.65,
        hedge_ratio_state_0: float = 0.0,
        hedge_ratio_state_1: float = 0.40,
        hedge_ratio_state_2: float = 0.90,
        future_multiplier: float = 10.0,  # Euro Stoxx 50 (FESX) = 10 EUR per index point
    ) -> None:
        self.hysteresis_days = hysteresis_days
        self.threshold_prob = threshold_prob
        self.hedge_ratios = {
            0: hedge_ratio_state_0,
            1: hedge_ratio_state_1,
            2: hedge_ratio_state_2,
        }
        self.future_multiplier = future_multiplier

    def compute_target_hedge_ratio(
        self, p0: float, p1: float, p2: float
    ) -> tuple[int, float]:
        """Compute the unconstrained target hedge ratio and candidate regime."""
        probs = [p0, p1, p2]
        max_idx = int(np.argmax(probs))
        max_p = probs[max_idx]

        if max_p >= self.threshold_prob:
            target_h = self.hedge_ratios[max_idx]
            cand_regime = max_idx
        else:
            # Linear blended expectation under regime ambiguity
            target_h = (
                p0 * self.hedge_ratios[0]
                + p1 * self.hedge_ratios[1]
                + p2 * self.hedge_ratios[2]
            )
            cand_regime = max_idx

        return cand_regime, float(target_h)

    def simulate_overlay_series(
        self,
        filtered_probs_df: pd.DataFrame,
        nav_series: pd.Series[float] | None = None,
        future_prices: pd.Series[float] | None = None,
        portfolio_beta: float = 1.0,
    ) -> pd.DataFrame:
        """Simulate the dynamic hedging overlay with 3-day hysteresis filtering.

        Parameters
        ----------
        filtered_probs_df : pd.DataFrame
            DataFrame containing ['prob_state_0', 'prob_state_1', 'prob_state_2'].
        nav_series : pd.Series, optional
            Portfolio NAV series for future contract sizing.
        future_prices : pd.Series, optional
            Index future price series.
        portfolio_beta : float
            Portfolio equity market beta.
        """
        required_cols = ["prob_state_0", "prob_state_1", "prob_state_2"]
        for col in required_cols:
            if col not in filtered_probs_df.columns:
                msg = f"Missing required probability column: {col}"
                raise ValueError(msg)

        n_rows = len(filtered_probs_df)
        dates = [str(d) for d in filtered_probs_df.index]

        # Tracking state
        current_effective_regime = 0
        current_effective_hedge = self.hedge_ratios[0]
        pending_regime = 0
        consecutive_pending_days = 0

        records: list[HedgingDecision] = []

        for i in range(n_rows):
            d_str = dates[i]
            row = filtered_probs_df.iloc[i]
            p0 = float(row["prob_state_0"])
            p1 = float(row["prob_state_1"])
            p2 = float(row["prob_state_2"])

            cand_regime, target_h = self.compute_target_hedge_ratio(p0, p1, p2)

            # Check hysteresis condition
            if cand_regime != current_effective_regime:
                if cand_regime == pending_regime:
                    consecutive_pending_days += 1
                else:
                    pending_regime = cand_regime
                    consecutive_pending_days = 1

                if consecutive_pending_days >= self.hysteresis_days:
                    # Regime change confirmed by persistence filter
                    current_effective_regime = cand_regime
                    current_effective_hedge = self.hedge_ratios[cand_regime]
                    consecutive_pending_days = 0
            else:
                pending_regime = current_effective_regime
                consecutive_pending_days = 0

            # Calculate contract count if pricing is available
            num_contracts = 0
            if nav_series is not None and future_prices is not None:
                curr_nav = float(nav_series.iloc[i]) if i < len(nav_series) else 100.0
                curr_px = float(future_prices.iloc[i]) if i < len(future_prices) else 4000.0
                notional_per_contract = curr_px * self.future_multiplier
                if notional_per_contract > 0:
                    hedge_capital = current_effective_hedge * curr_nav * portfolio_beta
                    num_contracts = int(np.round(hedge_capital / notional_per_contract))

            records.append(
                HedgingDecision(
                    date=d_str,
                    dominant_regime=cand_regime,
                    target_hedge_ratio=target_h,
                    effective_hedge_ratio=current_effective_hedge,
                    consecutive_days_in_new_regime=consecutive_pending_days,
                    num_future_contracts=num_contracts,
                )
            )

        df_out = pd.DataFrame(
            [
                {
                    "date": r.date,
                    "dominant_regime": r.dominant_regime,
                    "target_hedge_ratio": r.target_hedge_ratio,
                    "effective_hedge_ratio": r.effective_hedge_ratio,
                    "pending_days": r.consecutive_days_in_new_regime,
                    "num_future_contracts": r.num_future_contracts,
                }
                for r in records
            ]
        )
        df_out = df_out.set_index("date")
        return df_out
