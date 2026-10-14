"""Purged Group Time-Series Cross-Validator with Embargo.

Prevents information leakage and serial correlation between training and test sets
when supervised targets are overlapping forward return windows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


class PurgedGroupTimeSeriesSplit:
    """Time-series cross-validator with purging and embargo for cross-sectional panels.

    Splits dataset chronologically based on a date/group column.
    Ensures that between the end of train and the start of test, an embargo window
    is strictly respected.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.02,
        min_train_splits: int = 2,
    ) -> None:
        if n_splits < 2:
            msg = "n_splits must be at least 2"
            raise ValueError(msg)
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.min_train_splits = min_train_splits

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame | np.ndarray[Any, Any],
        y: Any = None,
        groups: Sequence[str] | pd.Series[str] | None = None,
    ) -> Generator[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
        """Generate train and test indices with temporal purging and embargo.

        Parameters
        ----------
        X : DataFrame or array of shape (n_samples, n_features)
        y : Target array (ignored)
        groups : Array or Series of string dates corresponding to each sample.
                 Required to identify temporal boundaries.
        """
        if groups is None:
            if isinstance(X, pd.DataFrame) and "date" in X.columns:
                group_series = X["date"].astype(str)
            else:
                msg = "groups parameter (date strings) is required for PurgedGroupTimeSeriesSplit"
                raise ValueError(msg)
        elif isinstance(groups, pd.Series):
            group_series = groups.astype(str)
        else:
            group_series = pd.Series(groups).astype(str)

        n_samples = len(group_series)
        unique_dates = np.array(sorted(group_series.unique()))
        n_dates = len(unique_dates)

        if n_dates < self.n_splits + 1:
            msg = f"Number of unique dates ({n_dates}) is too small for {self.n_splits} splits"
            raise ValueError(msg)

        # Calculate embargo size in terms of distinct dates
        embargo_steps = max(1, int(np.ceil(n_dates * self.embargo_pct)))

        # Split unique dates into n_splits + 1 chunks
        test_chunk_size = max(1, (n_dates - embargo_steps) // (self.n_splits + 1))

        indices = np.arange(n_samples)

        for i in range(self.min_train_splits, self.n_splits + 1):
            train_end_idx = i * test_chunk_size
            test_start_idx = train_end_idx + embargo_steps
            test_end_idx = min(n_dates, test_start_idx + test_chunk_size)

            if test_start_idx >= n_dates:
                break

            train_dates = set(unique_dates[:train_end_idx])
            test_dates = set(unique_dates[test_start_idx:test_end_idx])

            if not train_dates or not test_dates:
                continue

            train_mask = group_series.isin(train_dates).to_numpy()
            test_mask = group_series.isin(test_dates).to_numpy()

            train_idx = indices[train_mask]
            test_idx = indices[test_mask]

            yield train_idx, test_idx
