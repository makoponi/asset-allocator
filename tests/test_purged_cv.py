"""Tests for Purged Walk-Forward Cross-Validation (PurgedWalkForwardCV)."""

import pandas as pd
import pytest

from quant_portfolio.alpha.cv import PurgedWalkForwardCV


def test_purged_walk_forward_cv_expanding_split_properties() -> None:
    """Verify that PurgedWalkForwardCV enforces purge and embargo gaps in expanding window."""
    # Create 100 consecutive business dates with 2 stocks per date (200 rows)
    dates = pd.bdate_range("2024-01-01", periods=100).strftime("%Y-%m-%d").tolist()
    records: list[dict[str, str]] = []
    for d in dates:
        records.append({"date": d, "ticker": "AAPL US"})
        records.append({"date": d, "ticker": "SAP GY"})

    df = pd.DataFrame(records)

    n_splits = 4
    purge_days = 5
    embargo_days = 2
    cv = PurgedWalkForwardCV(
        n_splits=n_splits,
        purge_days=purge_days,
        embargo_days=embargo_days,
        expanding=True,
    )

    splits = list(cv.split(df, date_col="date"))
    assert len(splits) == n_splits

    prev_test_end_idx = -1
    for fold, (train_idx, test_idx) in enumerate(splits):
        # Disjoint indices
        assert set(train_idx).isdisjoint(set(test_idx))

        train_dates = sorted(df.iloc[train_idx]["date"].unique())
        test_dates = sorted(df.iloc[test_idx]["date"].unique())

        max_train_date = train_dates[-1]
        min_test_date = test_dates[0]

        # Max train date must precede min test date by at least purge_days
        train_date_pos = dates.index(max_train_date)
        test_date_pos = dates.index(min_test_date)

        # Gap must be strictly >= purge_days
        gap = test_date_pos - train_date_pos
        assert gap >= purge_days, (
            f"Fold {fold}: Gap {gap} between max train {max_train_date} and "
            f"min test {min_test_date} is less than purge_days {purge_days}"
        )

        # Test windows should progress forward sequentially
        test_end_idx = dates.index(test_dates[-1])
        assert test_end_idx > prev_test_end_idx
        prev_test_end_idx = test_end_idx


def test_purged_walk_forward_cv_rolling_window() -> None:
    """Verify rolling window behavior where train size remains bounded."""
    dates = pd.bdate_range("2024-01-01", periods=60).strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({"date": dates, "ticker": ["AAPL US"] * 60})

    cv = PurgedWalkForwardCV(
        n_splits=3,
        purge_days=3,
        embargo_days=1,
        expanding=False,
    )

    splits = list(cv.split(df, date_col="date"))
    assert len(splits) == 3

    train_sizes = [len(tr) for tr, _ in splits]
    # In rolling window, train sizes should be approximately constant (or bounded)
    assert max(train_sizes) - min(train_sizes) <= 2


def test_purged_walk_forward_cv_insufficient_samples_raises() -> None:
    """Verify ValueError is raised if sample dates are too few for requested splits and purge."""
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "ticker": ["A", "B"]})
    cv = PurgedWalkForwardCV(n_splits=5, purge_days=10)

    with pytest.raises(ValueError, match="Insufficient number of unique dates"):
        list(cv.split(df, date_col="date"))
