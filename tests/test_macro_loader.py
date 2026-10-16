"""Unit tests for macro series loader and feature builder."""

from pathlib import Path

import pytest

from quant_portfolio.regime.macro_loader import RegimeFeatureBuilder
from quant_portfolio.storage.schema import ParquetMarketStore, SQLiteStore


@pytest.fixture
def temp_stores(tmp_path: Path) -> tuple[SQLiteStore, ParquetMarketStore]:
    """Create isolated SQLite and Parquet stores for testing."""
    db_path = tmp_path / "test_quant.db"
    parquet_dir = tmp_path / "parquet"
    sql_store = SQLiteStore(db_path)
    pq_store = ParquetMarketStore(parquet_dir)
    return sql_store, pq_store


def test_load_macro_dataframe(temp_stores: tuple[SQLiteStore, ParquetMarketStore]) -> None:
    """Test loading and pivoting macro series from SQLite."""
    sql_store, pq_store = temp_stores
    records = [
        {"date": "2023-01-02", "region_bloc": "NA", "series_id": "VIX", "value": 20.5},
        {"date": "2023-01-02", "region_bloc": "NA", "series_id": "T10Y2Y", "value": -0.5},
        {"date": "2023-01-03", "region_bloc": "NA", "series_id": "VIX", "value": 21.0},
        {"date": "2023-01-03", "region_bloc": "NA", "series_id": "T10Y2Y", "value": -0.48},
        {"date": "2023-01-02", "region_bloc": "EU", "series_id": "V2TX", "value": 18.0},
    ]
    sql_store.upsert_macro_series(records)

    builder = RegimeFeatureBuilder(sqlite_store=sql_store, parquet_store=pq_store)
    df_na = builder.load_macro_dataframe(region_bloc="NA")

    assert not df_na.empty
    assert list(df_na.columns) == ["T10Y2Y", "VIX"]
    assert len(df_na) == 2
    assert df_na.loc["2023-01-02", "VIX"] == 20.5


def test_build_regime_features_alignment_and_cutoff(
    temp_stores: tuple[SQLiteStore, ParquetMarketStore],
) -> None:
    """Verify calendar alignment, forward fill, and strict as_of_date cutoff."""
    sql_store, pq_store = temp_stores
    dates = ["2023-01-02", "2023-01-03", "2023-01-05", "2023-01-10"]  # gaps on 4th, 6th-9th
    records = []
    for d in dates:
        records.append({"date": d, "region_bloc": "NA", "series_id": "VIX", "value": 20.0})
        records.append({"date": d, "region_bloc": "NA", "series_id": "T10Y2Y", "value": -0.5})
        records.append(
            {"date": d, "region_bloc": "NA", "series_id": "BAMLH0A0HYM2", "value": 350.0}
        )

    sql_store.upsert_macro_series(records)

    builder = RegimeFeatureBuilder(sqlite_store=sql_store, parquet_store=pq_store)
    # Cutoff at 2023-01-04 (which has no raw record, but 01-03 exists and can ffill)
    features = builder.build_regime_features(
        region_bloc="NA",
        as_of_date="2023-01-04",
        history_days=10,
        max_ffill_days=2,
    )

    assert not features.empty
    # Must not contain any date > 2023-01-04
    assert (features.index <= "2023-01-04").all()
    # 2023-01-04 should be forward-filled from 2023-01-03
    assert "2023-01-04" in features.index
    assert features.loc["2023-01-04", "VIX"] == 20.0
