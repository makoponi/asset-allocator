"""Storage layer for local data persistence: SQLite PiT metadata and Parquet EOD prices."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

DEFAULT_DB_PATH = Path("data/market_store.db")
DEFAULT_PARQUET_DIR = Path("data/processed")


@dataclass(frozen=True, slots=True)
class UniverseMemberRecord:
    """Record representing an index constituent at a given point in time."""

    date: str
    ticker: str
    gics_sector: str
    is_active: int


@dataclass(frozen=True, slots=True)
class FundamentalPiTRecord:
    """Point-in-Time fundamental financial metric record."""

    as_of_date: str
    filing_date: str
    ticker: str
    metric_name: str
    metric_value: float


@dataclass(frozen=True, slots=True)
class MacroSeriesRecord:
    """Macroeconomic or market indicator time series record."""

    date: str
    series_id: str
    value: float


@dataclass(frozen=True, slots=True)
class PriceEODRecord:
    """End-of-day daily market price, volume, and return record."""

    date: date
    ticker: str
    px_last: float
    volume: float
    ret_1d: float


PRICE_SCHEMA = pa.schema(
    [
        pa.field("date", pa.date32()),
        pa.field("ticker", pa.string()),
        pa.field("px_last", pa.float64()),
        pa.field("volume", pa.float64()),
        pa.field("ret_1d", pa.float64()),
    ]
)


class MarketStore:
    """Manages SQLite database for index constituents, fundamentals, and macro series.

    Enforces strict idempotence via unique composite primary keys and UPSERT semantics.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS universe_members (
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    gics_sector TEXT NOT NULL,
                    is_active INTEGER NOT NULL,
                    PRIMARY KEY (date, ticker)
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fundamentals_pit (
                    as_of_date TEXT NOT NULL,
                    filing_date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    PRIMARY KEY (as_of_date, ticker, metric_name)
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS macro_series (
                    date TEXT NOT NULL,
                    series_id TEXT NOT NULL,
                    value REAL NOT NULL,
                    PRIMARY KEY (date, series_id)
                );
                """
            )

    def insert_universe_members(self, records: Sequence[UniverseMemberRecord]) -> int:
        """Insert or replace universe member records idempotently."""
        if not records:
            return 0
        sql = """
            INSERT INTO universe_members (date, ticker, gics_sector, is_active)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(date, ticker) DO UPDATE SET
                gics_sector = excluded.gics_sector,
                is_active = excluded.is_active;
        """
        data = [
            (record.date, record.ticker, record.gics_sector, record.is_active) for record in records
        ]
        with self._connection() as conn:
            cursor = conn.executemany(sql, data)
            return cursor.rowcount

    def get_universe_members(
        self, as_of_date: str | None = None, is_active_only: bool = True
    ) -> list[UniverseMemberRecord]:
        """Fetch universe members with optional date filter and active flag."""
        query = "SELECT date, ticker, gics_sector, is_active FROM universe_members WHERE 1=1"
        params: list[Any] = []
        if as_of_date is not None:
            query += " AND date = ?"
            params.append(as_of_date)
        if is_active_only:
            query += " AND is_active = 1"
        query += " ORDER BY date, ticker;"

        with self._connection() as conn:
            cursor = conn.execute(query, params)
            return [
                UniverseMemberRecord(
                    date=row[0], ticker=row[1], gics_sector=row[2], is_active=row[3]
                )
                for row in cursor.fetchall()
            ]

    def insert_fundamentals_pit(self, records: Sequence[FundamentalPiTRecord]) -> int:
        """Insert or replace Point-in-Time fundamentals idempotently."""
        if not records:
            return 0
        sql = """
            INSERT INTO fundamentals_pit (
                as_of_date, filing_date, ticker, metric_name, metric_value
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(as_of_date, ticker, metric_name) DO UPDATE SET
                filing_date = excluded.filing_date,
                metric_value = excluded.metric_value;
        """
        data = [
            (
                record.as_of_date,
                record.filing_date,
                record.ticker,
                record.metric_name,
                record.metric_value,
            )
            for record in records
        ]
        with self._connection() as conn:
            cursor = conn.executemany(sql, data)
            return cursor.rowcount

    def get_fundamentals_pit(
        self,
        ticker: str | None = None,
        as_of_date: str | None = None,
        metric_name: str | None = None,
    ) -> list[FundamentalPiTRecord]:
        """Query fundamentals with optional filters."""
        query = (
            "SELECT as_of_date, filing_date, ticker, metric_name, metric_value "
            "FROM fundamentals_pit WHERE 1=1"
        )
        params: list[Any] = []
        if ticker is not None:
            query += " AND ticker = ?"
            params.append(ticker)
        if as_of_date is not None:
            query += " AND as_of_date = ?"
            params.append(as_of_date)
        if metric_name is not None:
            query += " AND metric_name = ?"
            params.append(metric_name)
        query += " ORDER BY as_of_date, ticker, metric_name;"

        with self._connection() as conn:
            cursor = conn.execute(query, params)
            return [
                FundamentalPiTRecord(
                    as_of_date=row[0],
                    filing_date=row[1],
                    ticker=row[2],
                    metric_name=row[3],
                    metric_value=float(row[4]),
                )
                for row in cursor.fetchall()
            ]

    def insert_macro_series(self, records: Sequence[MacroSeriesRecord]) -> int:
        """Insert or update macro series observations idempotently."""
        if not records:
            return 0
        sql = """
            INSERT INTO macro_series (date, series_id, value)
            VALUES (?, ?, ?)
            ON CONFLICT(date, series_id) DO UPDATE SET
                value = excluded.value;
        """
        data = [(record.date, record.series_id, record.value) for record in records]
        with self._connection() as conn:
            cursor = conn.executemany(sql, data)
            return cursor.rowcount

    def get_macro_series(
        self,
        series_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[MacroSeriesRecord]:
        """Fetch macro series records with optional range filtering."""
        query = "SELECT date, series_id, value FROM macro_series WHERE 1=1"
        params: list[Any] = []
        if series_id is not None:
            query += " AND series_id = ?"
            params.append(series_id)
        if start_date is not None:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date is not None:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date, series_id;"

        with self._connection() as conn:
            cursor = conn.execute(query, params)
            return [
                MacroSeriesRecord(date=row[0], series_id=row[1], value=float(row[2]))
                for row in cursor.fetchall()
            ]


class PriceStore:
    """Manages Parquet-backed EOD market prices partitioned by year."""

    def __init__(self, base_dir: Path | str = DEFAULT_PARQUET_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_prices(self, records: Sequence[PriceEODRecord]) -> int:
        """Write prices partitioned by year, ensuring deduplication on (ticker, date).

        Returns the total number of records written across partitions.
        """
        if not records:
            return 0

        # Group records by partition year
        records_by_year: dict[int, list[PriceEODRecord]] = {}
        for record in records:
            yr = record.date.year
            records_by_year.setdefault(yr, []).append(record)

        total_written = 0
        for yr, yr_records in records_by_year.items():
            year_partition_dir = self.base_dir / f"year={yr}"
            year_partition_dir.mkdir(parents=True, exist_ok=True)
            partition_file = year_partition_dir / "prices.parquet"

            existing_records: dict[tuple[date, str], PriceEODRecord] = {}
            if partition_file.exists():
                table = pq.read_table(partition_file)
                pydict = table.to_pydict()
                for d_val, t_val, px, vol, ret in zip(
                    pydict["date"],
                    pydict["ticker"],
                    pydict["px_last"],
                    pydict["volume"],
                    pydict["ret_1d"],
                    strict=True,
                ):
                    d_obj = d_val if isinstance(d_val, date) else d_val.date()
                    existing_records[(d_obj, t_val)] = PriceEODRecord(
                        date=d_obj,
                        ticker=t_val,
                        px_last=float(px),
                        volume=float(vol),
                        ret_1d=float(ret),
                    )

            # Upsert new incoming records into map
            for r in yr_records:
                existing_records[(r.date, r.ticker)] = r

            sorted_records = sorted(existing_records.values(), key=lambda x: (x.date, x.ticker))

            table_out = pa.Table.from_pydict(
                {
                    "date": [r.date for r in sorted_records],
                    "ticker": [r.ticker for r in sorted_records],
                    "px_last": [r.px_last for r in sorted_records],
                    "volume": [r.volume for r in sorted_records],
                    "ret_1d": [r.ret_1d for r in sorted_records],
                },
                schema=PRICE_SCHEMA,
            )

            pq.write_table(table_out, partition_file, compression="SNAPPY")
            total_written += len(yr_records)

        return total_written

    def read_prices(
        self,
        tickers: Sequence[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[PriceEODRecord]:
        """Read prices across partitions with optional ticker and date filters."""
        partition_files = sorted(self.base_dir.glob("year=*/*.parquet"))
        if not partition_files:
            return []

        tickers_set = set(tickers) if tickers is not None else None
        results: list[PriceEODRecord] = []

        for p_file in partition_files:
            table = pq.read_table(p_file)
            pydict = table.to_pydict()
            for d_val, t_val, px, vol, ret in zip(
                pydict["date"],
                pydict["ticker"],
                pydict["px_last"],
                pydict["volume"],
                pydict["ret_1d"],
                strict=True,
            ):
                d_obj = d_val if isinstance(d_val, date) else d_val.date()
                if tickers_set is not None and t_val not in tickers_set:
                    continue
                if start_date is not None and d_obj < start_date:
                    continue
                if end_date is not None and d_obj > end_date:
                    continue

                results.append(
                    PriceEODRecord(
                        date=d_obj,
                        ticker=t_val,
                        px_last=float(px),
                        volume=float(vol),
                        ret_1d=float(ret),
                    )
                )

        results.sort(key=lambda x: (x.date, x.ticker))
        return results
