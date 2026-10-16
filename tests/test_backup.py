"""Unit tests for DataBackupManager and backup CLI command."""

from __future__ import annotations

import sqlite3
import zipfile
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from quant_portfolio.cli import cmd_backup
from quant_portfolio.config import PortfolioConfig
from quant_portfolio.storage.backup import DataBackupManager

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def populated_env(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create test environment with SQLite DB, Parquet files, and drop-zone files."""
    db_path = tmp_path / "data" / "db" / "quant_store.db"
    parquet_dir = tmp_path / "data" / "parquet"
    raw_bql_dir = tmp_path / "data" / "raw_bql_exports"
    backups_dir = tmp_path / "data" / "backups"

    db_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    raw_bql_dir.mkdir(parents=True, exist_ok=True)
    backups_dir.mkdir(parents=True, exist_ok=True)

    # 1. Populate SQLite database
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT, val REAL)")
    conn.execute("INSERT INTO test_table VALUES (1, 'AAPL', 150.5)")
    conn.execute("INSERT INTO test_table VALUES (2, 'MSFT', 320.0)")
    conn.commit()
    conn.close()

    # 2. Populate sample Parquet files
    sub_pq_dir = parquet_dir / "2024"
    sub_pq_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "price": [150.5, 320.0]})
    df.to_parquet(sub_pq_dir / "market_2024.parquet", index=False)

    # 3. Populate sample raw BQL drop-zone file
    (raw_bql_dir / "export_sample.csv").write_text("ticker,metric,value\nAAPL,PE,25.4\n")

    return db_path, parquet_dir, raw_bql_dir, backups_dir


def test_backup_creation_and_integrity(
    populated_env: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    """Verify archive creation, internal structure, and SQLite database integrity."""
    db_path, parquet_dir, raw_bql_dir, backups_dir = populated_env

    mgr = DataBackupManager(
        db_path=db_path,
        parquet_dir=parquet_dir,
        raw_bql_dir=raw_bql_dir,
        backups_dir=backups_dir,
    )

    archive = mgr.create_backup(keep_last_n=10, custom_timestamp="20261016_120000")
    assert archive.exists()
    assert archive.name == "quant_backup_20261016_120000.zip"

    # Inspect zip contents
    with zipfile.ZipFile(archive, "r") as zf:
        namelist = zf.namelist()
        assert "db/quant_store.db" in namelist
        assert "parquet/2024/market_2024.parquet" in namelist
        assert "raw_bql_exports/export_sample.csv" in namelist

        # Extract and verify SQLite integrity
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        zf.extract("db/quant_store.db", extract_dir)
        extracted_db = extract_dir / "db" / "quant_store.db"

        conn = sqlite3.connect(extracted_db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check")
        status = cursor.fetchone()[0]
        assert status == "ok"

        cursor.execute("SELECT id, name, val FROM test_table ORDER BY id")
        rows = cursor.fetchall()
        assert len(rows) == 2
        assert rows[0] == (1, "AAPL", 150.5)
        conn.close()


def test_backup_rotation(populated_env: tuple[Path, Path, Path, Path]) -> None:
    """Verify that rotation keeps only the last N backups."""
    db_path, parquet_dir, raw_bql_dir, backups_dir = populated_env

    mgr = DataBackupManager(
        db_path=db_path,
        parquet_dir=parquet_dir,
        raw_bql_dir=raw_bql_dir,
        backups_dir=backups_dir,
    )

    # Create 5 backups with sequential timestamps
    for i in range(1, 6):
        ts = f"20261016_10000{i}"
        mgr.create_backup(keep_last_n=3, custom_timestamp=ts)

    remaining = mgr.list_backups()
    assert len(remaining) == 3

    remaining_names = [p.name for p in remaining]
    assert "quant_backup_20261016_100001.zip" not in remaining_names
    assert "quant_backup_20261016_100002.zip" not in remaining_names
    assert "quant_backup_20261016_100003.zip" in remaining_names
    assert "quant_backup_20261016_100004.zip" in remaining_names
    assert "quant_backup_20261016_100005.zip" in remaining_names


def test_backup_duplicate_to_external_dir(
    populated_env: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    """Verify copying archive duplicate to an external destination directory."""
    db_path, parquet_dir, raw_bql_dir, backups_dir = populated_env
    external_dir = tmp_path / "nas_backup"

    mgr = DataBackupManager(
        db_path=db_path,
        parquet_dir=parquet_dir,
        raw_bql_dir=raw_bql_dir,
        backups_dir=backups_dir,
    )

    archive = mgr.create_backup(
        dest_dir=external_dir,
        custom_timestamp="20261016_150000",
    )

    assert archive.exists()
    copied_archive = external_dir / "quant_backup_20261016_150000.zip"
    assert copied_archive.exists()
    assert copied_archive.stat().st_size == archive.stat().st_size


def test_backup_cli(
    populated_env: tuple[Path, Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify execution of cmd_backup CLI handler."""
    db_path, parquet_dir, raw_bql_dir, backups_dir = populated_env
    external_dir = tmp_path / "remote_export"

    # Configure PortfolioConfig paths to point to tmp_path
    cfg = PortfolioConfig()
    cfg.paths.root_dir = tmp_path
    cfg.paths.db_path = db_path
    cfg.paths.parquet_dir = parquet_dir
    cfg.paths.raw_bql_dir = raw_bql_dir
    cfg.paths.backups_dir = backups_dir

    monkeypatch.setattr("quant_portfolio.cli.PortfolioConfig", lambda: cfg)

    cmd_backup(dest_dir=str(external_dir), keep_last_n=5)

    backups = list(backups_dir.glob("quant_backup_*.zip"))
    assert len(backups) == 1
    assert (external_dir / backups[0].name).exists()
