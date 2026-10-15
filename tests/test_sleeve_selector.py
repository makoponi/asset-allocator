"""Tests for Phase 4 Stratified Top-Decile Sleeve Selection."""

import pandas as pd
import pytest

from quant_portfolio.alpha.selector import StratifiedSleeveSelector
from quant_portfolio.config import PortfolioConfig


def test_stratified_sleeve_selector_basic() -> None:
    """Verify that exactly 35 NA and 25 EU stocks are selected based on score ranking."""
    cfg = PortfolioConfig()
    selector = StratifiedSleeveSelector(config=cfg)

    # 100 NA stocks and 100 EU stocks
    na_tickers = [f"NA_{i:03d}" for i in range(100)]
    eu_tickers = [f"EU_{i:03d}" for i in range(100)]

    records: list[dict[str, object]] = []
    # Assign scores: NA_099 has highest score 99, NA_000 has lowest score 0
    for i, t in enumerate(na_tickers):
        records.append(
            {
                "as_of_date": "2024-01-31",
                "ticker": t,
                "isin": f"US{i:010d}",
                "region_bloc": "NA",
                "gics_sector": 45,
                "score": float(i),
            }
        )
    for i, t in enumerate(eu_tickers):
        records.append(
            {
                "as_of_date": "2024-01-31",
                "ticker": t,
                "isin": f"DE{i:010d}",
                "region_bloc": "EU",
                "gics_sector": 20,
                "score": float(i),
            }
        )

    df = pd.DataFrame(records)
    selected = selector.select_candidates(df, top_na=35, top_eu=25)

    assert len(selected) == 60
    assert len(selected[selected["region_bloc"] == "NA"]) == 35
    assert len(selected[selected["region_bloc"] == "EU"]) == 25

    # Check that highest scoring stocks were selected
    selected_na = selected[selected["region_bloc"] == "NA"]
    assert min(selected_na["score"]) == 65.0
    assert max(selected_na["score"]) == 99.0
    assert list(selected_na["regional_rank"]) == list(range(1, 36))
    assert (selected_na["regional_decile"] <= 4).all()

    selected_eu = selected[selected["region_bloc"] == "EU"]
    assert min(selected_eu["score"]) == 75.0
    assert max(selected_eu["score"]) == 99.0
    assert list(selected_eu["regional_rank"]) == list(range(1, 26))
    assert (selected_eu["regional_decile"] <= 3).all()


def test_stratified_sleeve_selector_rejects_financials() -> None:
    """Verify that Financials (GICS 40) are strictly rejected with an exception."""
    selector = StratifiedSleeveSelector()
    df = pd.DataFrame(
        [
            {
                "as_of_date": "2024-01-31",
                "ticker": "JPM US",
                "isin": "US0000000001",
                "region_bloc": "NA",
                "gics_sector": 40,  # Financials!
                "score": 10.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="Financials"):
        selector.select_candidates(df)


def test_stratified_sleeve_selector_handles_small_universe() -> None:
    """Verify that if available stocks are fewer than target, all available are returned."""
    selector = StratifiedSleeveSelector()
    df = pd.DataFrame(
        [
            {
                "as_of_date": "2024-01-31",
                "ticker": "AAPL US",
                "isin": "US0000000001",
                "region_bloc": "NA",
                "gics_sector": 45,
                "score": 5.0,
            },
            {
                "as_of_date": "2024-01-31",
                "ticker": "SAP GY",
                "isin": "DE0000000002",
                "region_bloc": "EU",
                "gics_sector": 45,
                "score": 4.0,
            },
        ]
    )

    selected = selector.select_candidates(df, top_na=35, top_eu=25)
    assert len(selected) == 2
    assert set(selected["ticker"]) == {"AAPL US", "SAP GY"}
