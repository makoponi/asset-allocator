"""Unit tests for Purged Group Time-Series Split with embargo enforcement."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.purged_cv import PurgedGroupTimeSeriesSplit


def test_purged_cv_embargo_and_temporal_order() -> None:
    # 24 monthly dates
    dates = [f"2022-{m:02d}-28" for m in range(1, 13)] + [f"2023-{m:02d}-28" for m in range(1, 13)]
    # 10 stocks per date -> 240 rows
    data: list[dict[str, str]] = []
    for d in dates:
        for s in range(10):
            data.append({"date": d, "ticker": f"STK_{s}"})

    df = pd.DataFrame(data)

    cv = PurgedGroupTimeSeriesSplit(n_splits=3, embargo_pct=0.05, min_train_splits=1)
    splits = list(cv.split(df, groups=df["date"]))

    assert len(splits) > 0

    unique_dates = sorted(df["date"].unique())

    for train_idx, test_idx in splits:
        train_dates = set(df.iloc[train_idx]["date"].unique())
        test_dates = set(df.iloc[test_idx]["date"].unique())

        # 1. No overlap between train and test dates
        assert len(train_dates.intersection(test_dates)) == 0

        # 2. Strict chronological order: max(train) < min(test)
        max_train = max(train_dates)
        min_test = min(test_dates)
        assert max_train < min_test

        # 3. Embargo gap enforcement: index in sorted dates must differ by at least 1 + embargo
        idx_train_end = unique_dates.index(max_train)
        idx_test_start = unique_dates.index(min_test)
        assert idx_test_start > idx_train_end + 1


def test_purged_cv_too_few_dates_raises() -> None:
    df = pd.DataFrame({"date": ["2023-01-31", "2023-02-28"], "ticker": ["A", "B"]})
    cv = PurgedGroupTimeSeriesSplit(n_splits=5)
    with pytest.raises(ValueError, match="too small"):
        list(cv.split(df, groups=df["date"]))
