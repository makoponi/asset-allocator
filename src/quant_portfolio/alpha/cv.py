"""Purged Walk-Forward Cross-Validation with financial embargo.

Prevents target look-ahead leakage and autoregressive contamination by enforcing:
1. Purge period: gap between end of training set and start of test set equal to target horizon H.
2. Embargo period: additional safety window preventing correlation of adjacent residual errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Generator


SplitIndices = tuple[
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.int64]],
]


class PurgedWalkForwardCV:
    """Walk-Forward Cross-Validation splitter with strict purging and embargo."""

    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 21,
        embargo_days: int = 5,
        min_train_periods: int | None = None,
        expanding: bool = True,
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")
        if purge_days < 0:
            raise ValueError("purge_days cannot be negative")
        if embargo_days < 0:
            raise ValueError("embargo_days cannot be negative")

        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.min_train_periods = min_train_periods
        self.expanding = expanding

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        """Return the number of splitting iterations in the cross-validator."""
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame | np.ndarray[Any, Any],
        y: Any = None,
        groups: Any = None,
        date_col: str = "as_of_date",
    ) -> Generator[SplitIndices, None, None]:
        """Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : pd.DataFrame
            The input data containing a date column (or array with date_col in groups).
        y : Any, optional
            Ignored, present for scikit-learn API compatibility.
        groups : Any, optional
            Optional array of date labels if X is a numpy ndarray without column names.
        date_col : str, default="as_of_date"
            Column name representing the observation timestamp/date in X.
        """
        if isinstance(X, pd.DataFrame):
            if date_col in X.columns:
                dates_series = pd.Series(X[date_col].values, index=X.index)
            elif "date" in X.columns:
                dates_series = pd.Series(X["date"].values, index=X.index)
            else:
                raise ValueError(f"Neither '{date_col}' nor 'date' column found in DataFrame.")
        elif groups is not None:
            dates_series = pd.Series(groups)
        else:
            raise ValueError("Must provide either a DataFrame with date column or groups as dates.")

        unique_dates = np.array(sorted(dates_series.unique()))
        n_unique_dates = len(unique_dates)

        # Minimum required unique dates: at least n_splits + 1
        min_req = self.n_splits + 1
        if n_unique_dates < min_req:
            raise ValueError(
                f"Insufficient number of unique dates ({n_unique_dates}). "
                f"Need at least {min_req} dates for {self.n_splits} splits with "
                f"purge_days={self.purge_days}."
            )

        # Total test dates to distribute across folds
        # Reserve initial window for training
        initial_train_count = self.min_train_periods or max(
            1,
            int(n_unique_dates * 0.4),
        )
        test_available_count = max(self.n_splits, n_unique_dates - initial_train_count)
        test_fold_size = max(1, test_available_count // self.n_splits)

        for fold in range(self.n_splits):
            test_start_idx = n_unique_dates - (self.n_splits - fold) * test_fold_size
            test_start_idx = max(1, test_start_idx)

            if fold == self.n_splits - 1:
                test_end_idx = n_unique_dates
            else:
                test_end_idx = min(test_start_idx + test_fold_size, n_unique_dates)

            if test_start_idx >= test_end_idx:
                continue

            test_dates = unique_dates[test_start_idx:test_end_idx]
            min_test_date = pd.Timestamp(str(test_dates[0]))

            # Cutoff date for train: min_test_date minus purge_days business days
            purge_cutoff = min_test_date - pd.tseries.offsets.BDay(self.purge_days)
            train_dates = [d for d in unique_dates if pd.Timestamp(str(d)) <= purge_cutoff]

            if not self.expanding and len(train_dates) > initial_train_count:
                train_dates = train_dates[-initial_train_count:]

            if not train_dates:
                continue

            # Convert date sets into integer index arrays for X
            train_mask = np.asarray(dates_series.isin(train_dates).to_numpy(), dtype=bool)
            test_mask = np.asarray(dates_series.isin(test_dates).to_numpy(), dtype=bool)

            train_indices = np.where(train_mask)[0].astype(np.int64)
            test_indices = np.where(test_mask)[0].astype(np.int64)

            if len(train_indices) == 0 or len(test_indices) == 0:
                continue

            yield train_indices, test_indices
