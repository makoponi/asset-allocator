"""Transactional backup manager for SQLite database and Parquet datalake."""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from quant_portfolio.config import PortfolioConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class DataBackupManager:
    """Manages atomic SQLite backups, Parquet archiving, and retention rotation."""

    def __init__(
        self,
        config: PortfolioConfig | None = None,
        db_path: Path | str | None = None,
        parquet_dir: Path | str | None = None,
        raw_bql_dir: Path | str | None = None,
        backups_dir: Path | str | None = None,
    ) -> None:
        cfg = config or PortfolioConfig()
        paths = cfg.paths.resolve_paths()

        self.db_path = Path(db_path) if db_path is not None else paths.db_path
        self.parquet_dir = Path(parquet_dir) if parquet_dir is not None else paths.parquet_dir
        self.raw_bql_dir = Path(raw_bql_dir) if raw_bql_dir is not None else paths.raw_bql_dir
        self.backups_dir = Path(backups_dir) if backups_dir is not None else paths.backups_dir

    def create_backup(
        self,
        keep_last_n: int = 10,
        dest_dir: Path | str | None = None,
        custom_timestamp: str | None = None,
    ) -> Path:
        """Execute atomic SQLite backup, archive Parquet files, rotate older backups, and duplicate.

        Parameters
        ----------
        keep_last_n : int
            Number of recent backup archives to retain in `backups_dir` (default: 10).
        dest_dir : Path | str | None
            Optional external path (e.g. NAS or network folder) to duplicate the archive.
        custom_timestamp : str | None
            Optional timestamp string (YYYYMMDD_HHMMSS) for testing deterministic naming.

        Returns
        -------
        Path
            Path to the created local backup archive.
        """
        self.backups_dir.mkdir(parents=True, exist_ok=True)

        ts = custom_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"quant_backup_{ts}.zip"
        archive_path = self.backups_dir / archive_name

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            temp_db_path = tmp_dir / "quant_store.db"

            has_db = self.db_path.exists()
            if has_db:
                self._backup_sqlite_atomically(source_path=self.db_path, target_path=temp_db_path)

            with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                if has_db and temp_db_path.exists():
                    zf.write(temp_db_path, arcname="db/quant_store.db")

                if self.parquet_dir.exists():
                    for p_file in sorted(self.parquet_dir.rglob("*.parquet")):
                        rel_path = p_file.relative_to(self.parquet_dir)
                        zf.write(p_file, arcname=f"parquet/{rel_path.as_posix()}")

                if self.raw_bql_dir.exists():
                    for b_file in sorted(self.raw_bql_dir.rglob("*")):
                        if b_file.is_file():
                            rel_path = b_file.relative_to(self.raw_bql_dir)
                            zf.write(b_file, arcname=f"raw_bql_exports/{rel_path.as_posix()}")

        logger.info("Successfully created backup archive: %s", archive_path)

        # Retention rotation
        if keep_last_n > 0:
            self._rotate_backups(keep_last_n=keep_last_n)

        # Duplicate to external destination if requested
        if dest_dir is not None:
            destination = Path(dest_dir)
            destination.mkdir(parents=True, exist_ok=True)
            target_copy = destination / archive_name
            shutil.copy2(archive_path, target_copy)
            logger.info("Copied backup duplicate to external destination: %s", target_copy)

        return archive_path

    @staticmethod
    def _backup_sqlite_atomically(source_path: Path, target_path: Path) -> None:
        """Perform hot atomic copy using SQLite native backup API."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_conn = sqlite3.connect(source_path)
        target_conn = sqlite3.connect(target_path)
        try:
            source_conn.backup(target_conn)
        finally:
            target_conn.close()
            source_conn.close()

    def _rotate_backups(self, keep_last_n: int) -> list[Path]:
        """Delete oldest archives if total count in `backups_dir` exceeds `keep_last_n`."""
        archives: list[Path] = sorted(
            self.backups_dir.glob("quant_backup_*.zip"),
            key=lambda p: (p.name, p.stat().st_mtime),
        )

        deleted: list[Path] = []
        if len(archives) > keep_last_n:
            to_remove = archives[: len(archives) - keep_last_n]
            for old_archive in to_remove:
                try:
                    os.unlink(old_archive)
                    deleted.append(old_archive)
                    logger.info("Rotated and removed old backup: %s", old_archive.name)
                except OSError as e:
                    logger.warning("Could not delete old backup %s: %s", old_archive, e)

        return deleted

    def list_backups(self) -> Sequence[Path]:
        """List all available backup archives sorted from oldest to newest."""
        if not self.backups_dir.exists():
            return []
        return sorted(
            self.backups_dir.glob("quant_backup_*.zip"),
            key=lambda p: (p.name, p.stat().st_mtime),
        )
