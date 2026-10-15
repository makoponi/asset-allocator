"""Unit tests for transatlantic calendar alignment and forward-fill logic."""

import pandas as pd

from quant_portfolio.ingestion.calendar_alignment import (
    align_market_panel,
    align_single_ticker_series,
    build_master_trading_calendar,
)


def test_build_master_trading_calendar() -> None:
    """Verify master calendar union across NA and EU active dates."""
    na_dates = ["2024-01-02", "2024-01-03", "2024-01-05"]  # NA closed on 01-04
    eu_dates = ["2024-01-02", "2024-01-04", "2024-01-05"]  # EU closed on 01-03

    cal = build_master_trading_calendar({"NA": na_dates, "EU": eu_dates})
    expected = [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-05"),
    ]
    assert list(cal) == expected


def test_align_single_ticker_series_forward_fill_and_zero_volume() -> None:
    """Verify forward-fill of price and zeroing of volume on missing holiday."""
    # US stock traded on 2024-07-03 and 2024-07-05 (closed 07-04 for Independence Day)
    df_us = pd.DataFrame(
        [
            {
                "date": "2024-07-03",
                "ticker": "AAPL US",
                "region_bloc": "NA",
                "px_last_eur": 200.0,
                "tot_return_idx_eur": 600.0,
                "volume_eur": 1000000.0,
            },
            {
                "date": "2024-07-05",
                "ticker": "AAPL US",
                "region_bloc": "NA",
                "px_last_eur": 202.0,
                "tot_return_idx_eur": 606.0,
                "volume_eur": 1200000.0,
            },
        ]
    )
    master_cal = pd.DatetimeIndex(["2024-07-03", "2024-07-04", "2024-07-05"])

    aligned = align_single_ticker_series(df_us, master_calendar=master_cal)
    assert len(aligned) == 3

    # On holiday (2024-07-04):
    holiday_row = aligned[aligned["date"] == "2024-07-04"].iloc[0]
    assert holiday_row["px_last_eur"] == 200.0  # forward-filled from 07-03
    assert holiday_row["tot_return_idx_eur"] == 600.0
    assert holiday_row["volume_eur"] == 0.0  # volume is 0.0 on closed days!


def test_align_single_ticker_series_max_ffill_limit() -> None:
    """Verify that forward-fill stops after max_ffill_days (e.g. prolonged suspension)."""
    # Asset trades only on day 1
    df = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "ticker": "SUSP US",
                "region_bloc": "NA",
                "px_last_eur": 100.0,
                "tot_return_idx_eur": 300.0,
                "volume_eur": 50000.0,
            }
        ]
    )
    # Master calendar has 6 consecutive days
    master_cal = pd.date_range("2024-01-01", periods=6, freq="D")

    # With max_ffill_days=2, it can fill at most 2 subsequent days (day 2 and day 3)
    aligned = align_single_ticker_series(df, master_calendar=master_cal, max_ffill_days=2)
    assert len(aligned) == 3  # Day 1 + 2 ffilled days = 3 total days
    dates = aligned["date"].tolist()
    assert dates == ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_align_market_panel() -> None:
    """Verify alignment of multi-ticker transatlantic panel."""
    panel_raw = pd.DataFrame(
        [
            # US stock (closed on 2024-01-04)
            {
                "date": "2024-01-03",
                "ticker": "AAPL US",
                "region_bloc": "NA",
                "px_last_eur": 180.0,
                "tot_return_idx_eur": 500.0,
                "volume_eur": 500000.0,
            },
            {
                "date": "2024-01-05",
                "ticker": "AAPL US",
                "region_bloc": "NA",
                "px_last_eur": 182.0,
                "tot_return_idx_eur": 505.5,
                "volume_eur": 600000.0,
            },
            # EU stock (traded on all 3 days)
            {
                "date": "2024-01-03",
                "ticker": "SAP GY",
                "region_bloc": "EU",
                "px_last_eur": 140.0,
                "tot_return_idx_eur": 300.0,
                "volume_eur": 300000.0,
            },
            {
                "date": "2024-01-04",
                "ticker": "SAP GY",
                "region_bloc": "EU",
                "px_last_eur": 141.0,
                "tot_return_idx_eur": 302.1,
                "volume_eur": 350000.0,
            },
            {
                "date": "2024-01-05",
                "ticker": "SAP GY",
                "region_bloc": "EU",
                "px_last_eur": 141.5,
                "tot_return_idx_eur": 303.2,
                "volume_eur": 280000.0,
            },
        ]
    )

    aligned = align_market_panel(panel_raw, max_ffill_days=3)

    # AAPL now has 3 rows (was 2) and SAP GY has 3 rows -> Total 6 rows
    assert len(aligned) == 6

    aapl_04 = aligned[(aligned["ticker"] == "AAPL US") & (aligned["date"] == "2024-01-04")].iloc[0]
    assert aapl_04["px_last_eur"] == 180.0
    assert aapl_04["volume_eur"] == 0.0

    sap_04 = aligned[(aligned["ticker"] == "SAP GY") & (aligned["date"] == "2024-01-04")].iloc[0]
    assert sap_04["px_last_eur"] == 141.0
    assert sap_04["volume_eur"] == 350000.0
