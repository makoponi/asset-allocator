"""Unit and integration tests for PiT feature alignment and zero look-ahead bias verification."""

from datetime import date, timedelta
from typing import TYPE_CHECKING

import pytest

from src.features.pit_aligner import PiTFeatureAligner
from src.features.technical_engine import TechnicalEngine
from src.ingestion.storage import (
    FundamentalPiTRecord,
    MarketStore,
    PriceEODRecord,
    UniverseMemberRecord,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def temp_store(tmp_path: Path) -> MarketStore:
    return MarketStore(db_path=tmp_path / "test_aligner.db")


def test_pit_aligner_zero_look_ahead_bias(temp_store: MarketStore) -> None:
    eval_date = date(2024, 1, 31)

    # 1. Register active constituent
    temp_store.insert_universe_members(
        [
            UniverseMemberRecord(
                date="2024-01-31",
                ticker="ALV GY",
                gics_sector="Financials",
                is_active=1,
            )
        ]
    )

    # 2. Insert two fundamental filings:
    # - H1 2023 (as_of: 2023-06-30, filed: 2023-09-28) -> VALID at eval_date (2024-01-31)
    # - FY 2023 (as_of: 2023-12-31, filed: 2024-03-15) -> FUTURE / FORBIDDEN at eval_date!
    records = [
        FundamentalPiTRecord(
            as_of_date="2023-06-30",
            filing_date="2023-09-28",
            ticker="ALV GY",
            metric_name="PROF_MARGIN",
            metric_value=9.5,
        ),
        FundamentalPiTRecord(
            as_of_date="2023-12-31",
            filing_date="2024-03-15",
            ticker="ALV GY",
            metric_name="PROF_MARGIN",
            metric_value=12.0,  # Look-ahead value
        ),
    ]
    temp_store.insert_fundamentals_pit(records)

    # 3. Create synthetic price history (300 days) for technical engine
    start_d = eval_date - timedelta(days=320)
    prices: list[PriceEODRecord] = []
    px = 150.0
    for i in range(320):
        cur_d = start_d + timedelta(days=i)
        px_next = px + 0.1
        prices.append(
            PriceEODRecord(
                date=cur_d,
                ticker="ALV GY",
                px_last=px_next,
                volume=50000.0,
                ret_1d=(px_next / px) - 1.0,
            )
        )
        px = px_next

    tech_engine = TechnicalEngine(prices)
    aligner = PiTFeatureAligner(market_store=temp_store, technical_engine=tech_engine)

    # 4. Build aligned row for 2024-01-31
    matrix = aligner.build_cross_sectional_matrix(eval_date=eval_date)
    assert len(matrix) == 1
    row = matrix[0]

    assert row.ticker == "ALV GY"
    assert row.gics_sector == "Financials"
    assert row.mom_1m > 0.0

    # CRITICAL ASSERTION:
    # Must use H1 2023 value (9.5) and MUST NOT see FY 2023 future value (12.0)!
    assert row.prof_margin == 9.5
