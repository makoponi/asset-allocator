"""Unit tests for the quant-portfolio CLI commands."""

from pathlib import Path

import pytest

from quant_portfolio.cli import (
    check_bloomberg_dapi_status,
    cmd_generate_bql,
    cmd_status,
)


def test_check_bloomberg_dapi_status_mocked() -> None:
    """Verify check_bloomberg_dapi_status returns bool without crashing."""
    status = check_bloomberg_dapi_status(host="127.0.0.1", port=99999, timeout=0.01)
    assert status is False


def test_cmd_status_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify cmd_status prints environment report."""
    cmd_status()
    captured = capsys.readouterr()
    assert "Quant Portfolio - Environment Status" in captured.out
    assert "Database path:" in captured.out


def test_cmd_generate_bql_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify cmd_generate_bql writes text and prints templates."""
    out_file = tmp_path / "bql_manifest.txt"
    cmd_generate_bql(output_file=str(out_file))
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "SPX Index" in content
    assert "SPTSX60 Index" in content
    assert "SXXP Index" in content
    assert "=BQL(" in content


def test_cmd_fetch_series_mocked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify cmd_fetch_series executes when DAPI is mocked as open."""
    from quant_portfolio import cli
    from quant_portfolio.ingestion.pipeline import TransatlanticIngestionPipeline

    monkeypatch.setattr(cli, "check_bloomberg_dapi_status", lambda: True)

    called = False

    def mock_download(self: object, **kwargs: object) -> int:
        nonlocal called
        called = True
        return 42

    monkeypatch.setattr(
        TransatlanticIngestionPipeline,
        "download_and_sync_market_series",
        mock_download,
    )

    cli.cmd_fetch_series(
        start_date="2024-01-01", end_date="2024-01-05", batch_size=25, limit_tickers=5
    )
    captured = capsys.readouterr()
    assert called is True
    assert "42 market observations" in captured.out


def test_cmd_ingest_fundamentals_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify cmd_ingest_fundamentals processes drop-zone without errors."""
    from quant_portfolio import cli
    from quant_portfolio.config import PortfolioConfig, StoragePaths

    test_paths = StoragePaths(
        root_dir=tmp_path,
        data_dir=tmp_path / "data",
        raw_bql_dir=tmp_path / "data" / "raw_bql_exports",
        db_path=tmp_path / "data" / "db" / "quant_store.db",
        parquet_dir=tmp_path / "data" / "parquet",
    )
    test_paths.raw_bql_dir.mkdir(parents=True)
    monkeypatch.setattr(
        cli,
        "PortfolioConfig",
        lambda: PortfolioConfig(paths=test_paths),
    )

    cli.cmd_ingest_fundamentals(drop_dir=str(test_paths.raw_bql_dir))
    captured = capsys.readouterr()
    assert "[Done] Ingested" in captured.out


def test_cmd_compute_features_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify cmd_compute_features runs pipeline and outputs summary."""
    import pandas as pd

    from quant_portfolio import cli
    from quant_portfolio.config import ALL_CORE_FEATURES
    from quant_portfolio.features.pipeline import TransatlanticFeaturePipeline

    dummy_df = pd.DataFrame(
        {
            "as_of_date": ["2024-01-31", "2024-01-31"],
            "ticker": ["AAPL US", "SAP GY"],
            "isin": ["US0378331005", "DE0007164600"],
            "region_bloc": ["NA", "EU"],
            "gics_sector": [45, 45],
            **{feat: [0.1, -0.1] for feat in ALL_CORE_FEATURES},
        }
    )

    monkeypatch.setattr(
        TransatlanticFeaturePipeline,
        "build_features_at_date",
        lambda self, as_of_date, normalize: dummy_df,
    )

    out_file = tmp_path / "test_features.parquet"
    cli.cmd_compute_features(as_of_date="2024-01-31", output_path=str(out_file))

    assert out_file.exists()
    captured = capsys.readouterr()
    assert "Transatlantic Feature Matrix Summary" in captured.out
    assert "Total Investible Stocks: 2 (NA: 1, EU: 1)" in captured.out
    assert "Residual NaNs: 0" in captured.out


def test_cmd_train_alpha_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify cmd_train_alpha runs walk-forward CV and reports metrics."""
    from quant_portfolio import cli
    from quant_portfolio.alpha.pipeline import AlphaPipeline
    from quant_portfolio.config import PortfolioConfig, StoragePaths
    from quant_portfolio.storage.schema import SQLiteStore

    db_path = tmp_path / "data" / "db" / "quant_store.db"
    test_paths = StoragePaths(
        root_dir=tmp_path,
        data_dir=tmp_path / "data",
        raw_bql_dir=tmp_path / "data" / "raw_bql_exports",
        db_path=db_path,
        parquet_dir=tmp_path / "data" / "parquet",
    )
    store = SQLiteStore(db_path)
    store.upsert_universe_pit(
        [
            {
                "as_of_date": "2024-01-31",
                "ticker": "AAPL US",
                "isin": "US0378331005",
                "region_bloc": "NA",
                "source_index": "SPX Index",
                "gics_sector": 45,
                "local_currency": "USD",
                "is_financial": False,
            },
            {
                "as_of_date": "2024-02-29",
                "ticker": "AAPL US",
                "isin": "US0378331005",
                "region_bloc": "NA",
                "source_index": "SPX Index",
                "gics_sector": 45,
                "local_currency": "USD",
                "is_financial": False,
            },
        ]
    )

    monkeypatch.setattr(
        cli,
        "PortfolioConfig",
        lambda: PortfolioConfig(paths=test_paths),
    )
    monkeypatch.setattr(
        AlphaPipeline,
        "run_walk_forward_evaluation",
        lambda self, dates, n_splits: {
            "mean_ic": 0.0825,
            "std_ic": 0.0210,
            "ir_ic": 3.93,
            "mean_rmse": 0.0450,
            "mean_decile_spread": 0.0320,
            "fold_metrics": [
                {
                    "fold": 0,
                    "test_dates": "2024-02-29",
                    "train_dates": "2024-01-31",
                    "test_samples": 50,
                    "train_samples": 50,
                    "ic_spearman": 0.0825,
                    "rmse": 0.0450,
                    "top_bottom_decile_spread": 0.0320,
                }
            ],
        },
    )

    cli.cmd_train_alpha(start_date="2024-01-01", end_date="2024-03-01", n_splits=1)
    captured = capsys.readouterr()
    assert "Mean Information Coefficient (Spearman): 0.0825" in captured.out
    assert "Information Ratio (IC / std):           3.93" in captured.out


def test_cmd_rank_universe_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify cmd_rank_universe scores universe and prints selected candidates."""
    import pandas as pd

    from quant_portfolio import cli
    from quant_portfolio.alpha.pipeline import AlphaPipeline
    from quant_portfolio.config import PortfolioConfig, StoragePaths
    from quant_portfolio.storage.schema import SQLiteStore

    db_path = tmp_path / "data" / "db" / "quant_store.db"
    test_paths = StoragePaths(
        root_dir=tmp_path,
        data_dir=tmp_path / "data",
        raw_bql_dir=tmp_path / "data" / "raw_bql_exports",
        db_path=db_path,
        parquet_dir=tmp_path / "data" / "parquet",
    )
    store = SQLiteStore(db_path)
    store.upsert_universe_pit(
        [
            {
                "as_of_date": "2023-12-31",
                "ticker": "AAPL US",
                "isin": "US0378331005",
                "region_bloc": "NA",
                "source_index": "SPX Index",
                "gics_sector": 45,
                "local_currency": "USD",
                "is_financial": False,
            }
        ]
    )

    dummy_selected = pd.DataFrame(
        [
            {
                "as_of_date": "2024-01-31",
                "ticker": "AAPL US",
                "region_bloc": "NA",
                "gics_sector": 45,
                "score": 0.15,
                "regional_rank": 1,
            },
            {
                "as_of_date": "2024-01-31",
                "ticker": "SAP GY",
                "region_bloc": "EU",
                "gics_sector": 45,
                "score": 0.12,
                "regional_rank": 1,
            },
        ]
    )

    monkeypatch.setattr(
        cli,
        "PortfolioConfig",
        lambda: PortfolioConfig(paths=test_paths),
    )
    monkeypatch.setattr(
        AlphaPipeline,
        "rank_and_select_at_date",
        lambda self, as_of_date, training_dates, top_na, top_eu: dummy_selected,
    )

    out_file = tmp_path / "candidates.csv"
    cli.cmd_rank_universe(as_of_date="2024-01-31", output_path=str(out_file), top_na=1, top_eu=1)

    assert out_file.exists()
    captured = capsys.readouterr()
    assert "Stratified Sleeve Selection" in captured.out
    assert "Total Selected Candidates: 2" in captured.out
    assert "AAPL US" in captured.out
    assert "SAP GY" in captured.out


def test_cmd_allocate_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify cmd_allocate executes portfolio allocation pipeline and outputs summary."""
    import pandas as pd

    from quant_portfolio import cli
    from quant_portfolio.config import PortfolioConfig, StoragePaths
    from quant_portfolio.risk.pipeline import PortfolioAllocationPipeline, PortfolioRiskSummary

    test_paths = StoragePaths(
        root_dir=tmp_path,
        data_dir=tmp_path / "data",
        raw_bql_dir=tmp_path / "data" / "raw_bql_exports",
        db_path=tmp_path / "data" / "db" / "quant_store.db",
        parquet_dir=tmp_path / "data" / "parquet",
    )
    monkeypatch.setattr(
        cli,
        "PortfolioConfig",
        lambda: PortfolioConfig(paths=test_paths),
    )

    # Prepare dummy candidate file
    cand_file = tmp_path / "candidates.csv"
    dummy_cands = pd.DataFrame(
        [
            {"ticker": "AAPL US", "region_bloc": "NA", "score": 0.05},
            {"ticker": "SAP GY", "region_bloc": "EU", "score": 0.03},
        ]
    )
    dummy_cands.to_csv(cand_file, index=False)

    dummy_alloc = pd.DataFrame(
        [
            {
                "ticker": "AAPL US",
                "region_bloc": "NA",
                "weight": 0.55,
                "mctr": 0.012,
                "score": 0.05,
                "as_of_date": "2024-01-31",
            },
            {
                "ticker": "SAP GY",
                "region_bloc": "EU",
                "weight": 0.45,
                "mctr": 0.010,
                "score": 0.03,
                "as_of_date": "2024-01-31",
            },
        ]
    )
    dummy_summary = PortfolioRiskSummary(
        as_of_date="2024-01-31",
        method="hrp",
        n_stocks=2,
        n_na=1,
        n_eu=1,
        na_weight=0.55,
        eu_weight=0.45,
        annualized_volatility=0.145,
        cvar_95=0.032,
    )

    monkeypatch.setattr(
        PortfolioAllocationPipeline,
        "allocate_candidates",
        lambda self, candidates_df, as_of_date, method: (dummy_alloc, dummy_summary),
    )

    out_file = tmp_path / "allocations.csv"
    cli.cmd_allocate(
        as_of_date="2024-01-31",
        method="hrp",
        candidates_file=str(cand_file),
        output_path=str(out_file),
    )

    assert out_file.exists()
    captured = capsys.readouterr()
    assert "Transatlantic Allocation" in captured.out
    assert "Optimization Method: HRP" in captured.out
    assert "NA Sleeve:  1 stocks | Weight: 55.00%" in captured.out
    assert "EU Sleeve:  1 stocks | Weight: 45.00%" in captured.out
    assert "AAPL US" in captured.out
