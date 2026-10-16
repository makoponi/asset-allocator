"""Unit tests for turnover buffer and delta order generation."""

import pandas as pd
import pytest

from quant_portfolio.backtest.turnover import TurnoverBufferFilter


def _create_mock_ranked_universe() -> pd.DataFrame:
    """Create synthetic ranked universe with 10 NA stocks and 10 EU stocks across deciles 1..10."""
    records = []
    # 10 NA stocks, scores from 10 down to 1, deciles 1 to 10
    for i in range(1, 11):
        records.append(
            {
                "ticker": f"NA_STOCK_{i}",
                "region_bloc": "NA",
                "score": float(11 - i),
                "regional_rank": i,
                "regional_decile": i,
                "is_selected": i <= 3,
            }
        )
    # 10 EU stocks, scores from 10 down to 1, deciles 1 to 10
    for i in range(1, 11):
        records.append(
            {
                "ticker": f"EU_STOCK_{i}",
                "region_bloc": "EU",
                "score": float(11 - i),
                "regional_rank": i,
                "regional_decile": i,
                "is_selected": i <= 3,
            }
        )
    return pd.DataFrame(records)


def test_cold_start_selection() -> None:
    """Without current holdings, buffer selector picks top K by score."""
    tb = TurnoverBufferFilter()
    universe = _create_mock_ranked_universe()

    # Request top 3 NA and top 2 EU
    selected = tb.filter_candidates_with_buffer(
        ranked_universe=universe,
        current_holdings=None,
        top_na=3,
        top_eu=2,
    )

    na_selected = selected[selected["region_bloc"] == "NA"]["ticker"].tolist()
    eu_selected = selected[selected["region_bloc"] == "EU"]["ticker"].tolist()

    assert na_selected == ["NA_STOCK_1", "NA_STOCK_2", "NA_STOCK_3"]
    assert eu_selected == ["EU_STOCK_1", "EU_STOCK_2"]


def test_turnover_buffer_maintains_decile_3_and_drops_decile_4() -> None:
    """Held stock in decile 2 and 3 should be kept; stock in decile 4 dropped."""
    tb = TurnoverBufferFilter()
    universe = _create_mock_ranked_universe()

    # Suppose we held NA_STOCK_2 (decile 2), NA_STOCK_3 (decile 3), and NA_STOCK_4 (decile 4)
    # Target K=3
    current_holdings = ["NA_STOCK_2", "NA_STOCK_3", "NA_STOCK_4"]

    selected = tb.filter_candidates_with_buffer(
        ranked_universe=universe,
        current_holdings=current_holdings,
        top_na=3,
        top_eu=2,
    )

    na_selected = selected[selected["region_bloc"] == "NA"]["ticker"].tolist()

    # NA_STOCK_2 and NA_STOCK_3 are in deciles <= 3, so maintained.
    # NA_STOCK_4 is decile 4 (> 3), so dropped!
    # The 3rd spot is filled by top unheld stock, which is NA_STOCK_1.
    assert "NA_STOCK_2" in na_selected
    assert "NA_STOCK_3" in na_selected
    assert "NA_STOCK_4" not in na_selected
    assert "NA_STOCK_1" in na_selected
    assert len(na_selected) == 3


def test_generate_delta_orders() -> None:
    """Test BUY, SELL, HOLD actions and trade values."""
    tb = TurnoverBufferFilter()

    current_weights = pd.DataFrame(
        [
            {"ticker": "AAPL US", "weight": 0.03, "region_bloc": "NA"},
            {"ticker": "MSFT US", "weight": 0.02, "region_bloc": "NA"},
            {"ticker": "EXIT EU", "weight": 0.02, "region_bloc": "EU"},
        ]
    )

    target_weights = pd.DataFrame(
        [
            {"ticker": "AAPL US", "weight": 0.035, "region_bloc": "NA"},  # BUY +0.005
            {"ticker": "MSFT US", "weight": 0.02, "region_bloc": "NA"},  # HOLD
            {"ticker": "NEW EU", "weight": 0.02, "region_bloc": "EU"},  # BUY +0.02
            # EXIT EU is missing from target -> SELL -0.02
        ]
    )

    portfolio_val = 10_000_000.0
    orders = tb.generate_delta_orders(current_weights, target_weights, portfolio_val)

    assert len(orders) >= 3

    order_map = {row["ticker"]: row for _, row in orders.iterrows()}

    # AAPL: BUY +0.005 of 10M = 50,000 EUR
    assert order_map["AAPL US"]["action"] == "BUY"
    assert pytest.approx(order_map["AAPL US"]["weight_delta"], rel=1e-5) == 0.005
    assert pytest.approx(order_map["AAPL US"]["estimated_trade_value_eur"], rel=1e-5) == 50_000.0

    # EXIT EU: SELL -0.02 of 10M = 200,000 EUR
    assert order_map["EXIT EU"]["action"] == "SELL"
    assert pytest.approx(order_map["EXIT EU"]["weight_delta"], rel=1e-5) == -0.02
    assert pytest.approx(order_map["EXIT EU"]["estimated_trade_value_eur"], rel=1e-5) == 200_000.0

    # NEW EU: BUY +0.02 of 10M = 200,000 EUR
    assert order_map["NEW EU"]["action"] == "BUY"
    assert pytest.approx(order_map["NEW EU"]["weight_delta"], rel=1e-5) == 0.02
    assert pytest.approx(order_map["NEW EU"]["estimated_trade_value_eur"], rel=1e-5) == 200_000.0
