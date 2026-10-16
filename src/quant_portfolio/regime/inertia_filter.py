"""Inertia band persistence filter for multi-state regime transitions."""

from collections.abc import Sequence

from quant_portfolio.config import PortfolioConfig
from quant_portfolio.regime.hmm_detector import RegimeStateResult


class InertiaBandFilter:
    """Applies a multi-day hysteresis inertia band to prevent regime whipsaw.

    Rules:
    - To escalate hedge state (e.g. 0 -> 1 or 1 -> 2):
      Requires target stress probability >= stress_open_prob (default 0.70)
      confirmed for inertia_days consecutive observations (default 3 days).
    - To de-escalate hedge state (e.g. 2 -> 1 or 1 -> 0):
      Requires stress probability < stress_close_prob (default 0.40)
      confirmed for inertia_days consecutive observations.
    - Otherwise, the previous filtered state is strictly preserved.
    """

    def __init__(
        self,
        inertia_days: int = 3,
        stress_open_prob: float = 0.70,
        stress_close_prob: float = 0.40,
        config: PortfolioConfig | None = None,
    ) -> None:
        if config is not None:
            self.inertia_days = config.regime.inertia_days
            self.stress_open_prob = config.regime.stress_open_prob
            self.stress_close_prob = config.regime.stress_close_prob
        else:
            self.inertia_days = inertia_days
            self.stress_open_prob = stress_open_prob
            self.stress_close_prob = stress_close_prob

        if self.inertia_days < 1:
            raise ValueError(f"inertia_days must be >= 1, got {self.inertia_days}")
        if not (0.0 <= self.stress_close_prob <= self.stress_open_prob <= 1.0):
            raise ValueError("Must satisfy 0.0 <= stress_close_prob <= stress_open_prob <= 1.0")

    def filter_step(
        self,
        recent_window: Sequence[RegimeStateResult],
        previous_filtered_state: int = 0,
    ) -> int:
        """Compute the filtered state at date t given the trailing window and previous state.

        Parameters
        ----------
        recent_window : Sequence[RegimeStateResult]
            Recent history up to and including date t. Needs at least inertia_days observations
            to trigger a state transition.
        previous_filtered_state : int
            The filtered state established at t - 1 (default 0).

        Returns
        -------
        int
            Filtered regime state in {0, 1, 2}.
        """
        if not recent_window:
            return previous_filtered_state

        if len(recent_window) < self.inertia_days:
            return previous_filtered_state

        tail = recent_window[-self.inertia_days :]

        # Escalation checks (require inertia_days consecutive confirmations)
        # Check severe stress (State 2)
        all_severe = all(r.prob_stress >= self.stress_open_prob for r in tail)
        if all_severe:
            return 2

        # Check alert / transition (State 1)
        all_alert_or_higher = all(
            (r.prob_alert + r.prob_stress) >= self.stress_open_prob for r in tail
        )
        if all_alert_or_higher:
            if previous_filtered_state == 0:
                return 1
            # If already at 2, we check de-escalation below; if at 1, remain 1
            if previous_filtered_state == 1:
                return 1

        # De-escalation checks (require inertia_days consecutive confirmations)
        if previous_filtered_state == 2:
            # If severe stress dropped below threshold for inertia_days
            all_stress_cleared = all(r.prob_stress < self.stress_close_prob for r in tail)
            if all_stress_cleared:
                # Check if it should drop all the way to 0 (calm)
                all_calm = all(
                    (r.prob_alert + r.prob_stress) < self.stress_close_prob for r in tail
                )
                return 0 if all_calm else 1
            return 2

        if previous_filtered_state == 1:
            all_calm = all((r.prob_alert + r.prob_stress) < self.stress_close_prob for r in tail)
            if all_calm:
                return 0
            return 1

        return previous_filtered_state

    def filter_sequence(
        self,
        history: Sequence[RegimeStateResult],
        initial_state: int = 0,
    ) -> list[int]:
        """Apply inertia filtering chronologically across a historical sequence.

        Parameters
        ----------
        history : Sequence[RegimeStateResult]
            Chronologically sorted sequence of raw regime detections.
        initial_state : int
            Starting filtered state (default 0: Calm).

        Returns
        -------
        list[int]
            Filtered state for each observation in history.
        """
        if not history:
            return []

        filtered_states: list[int] = []
        curr_state = initial_state

        for idx in range(len(history)):
            window = history[: idx + 1]
            curr_state = self.filter_step(
                recent_window=window,
                previous_filtered_state=curr_state,
            )
            filtered_states.append(curr_state)

        return filtered_states
