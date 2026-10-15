"""SQLite schemas, Parquet market store, and data persistence management."""

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DAILY_MARKET_COLUMNS: list[str] = [
    "date",
    "ticker",
    "region_bloc",
    "px_last_eur",
    "tot_return_idx_eur",
    "volume_eur",
]

DAILY_MARKET_ARROW_SCHEMA = pa.schema(
    [
        ("date", pa.string()),
        ("ticker", pa.string()),
        ("region_bloc", pa.string()),
        ("px_last_eur", pa.float64()),
        ("tot_return_idx_eur", pa.float64()),
        ("volume_eur", pa.float64()),
    ]
)


def safe_int(val: Any, default: int = 0) -> int:
    """Safely convert a value to int handling NaN, None, and strings."""
    if val is None or pd.isna(val):
        return default
    try:
        num = float(val)
        return int(num) if not pd.isna(num) else default
    except (ValueError, TypeError):
        return default


def safe_str(val: Any, default: str = "") -> str:
    """Safely convert a value to string, handling NaN and None."""
    if val is None or pd.isna(val):
        return default
    s = str(val).strip()
    return default if s.lower() == "nan" else s


def safe_optional_str(val: Any) -> str | None:
    """Safely convert an optional string, returning None for NaN/None/empty."""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    return None if (not s or s.lower() == "nan") else s


def deduplicate_universe_by_isin(
    df: pd.DataFrame,
    liquidity_col: str = "volume_eur",
    tie_breaker_col: str = "ticker",
) -> pd.DataFrame:
    """Deduplicate cross-listed securities sharing the same ISIN on the same as_of_date.

    If multiple tickers share the same ISIN for a given as_of_date, keep the line with
    the highest liquidity (e.g. trading volume in EUR). If liquidity is tied or missing,
    sort deterministically by tie_breaker_col.
    """
    if df.empty:
        return df.copy()

    required_cols = {"as_of_date", "isin", "ticker"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for ISIN deduplication: {missing}")

    has_liquidity = liquidity_col in df.columns
    sort_cols: list[str] = ["as_of_date", "isin"]
    ascending: list[bool] = [True, True]

    if has_liquidity:
        sort_cols.append(liquidity_col)
        ascending.append(False)  # Highest volume first

    if tie_breaker_col in df.columns:
        sort_cols.append(tie_breaker_col)
        ascending.append(True)

    sorted_df = df.sort_values(by=sort_cols, ascending=ascending)
    return sorted_df.drop_duplicates(subset=["as_of_date", "isin"], keep="first").reset_index(
        drop=True
    )


class SQLiteStore:
    """Manager for point-in-time universe, fundamentals, and macro SQLite storage."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create and configure a SQLite connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Create database tables and indexes if they do not exist."""
        with self.get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS universe_pit (
                    as_of_date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    isin TEXT NOT NULL,
                    region_bloc TEXT NOT NULL CHECK(region_bloc IN ('NA', 'EU')),
                    source_index TEXT NOT NULL,
                    gics_sector INTEGER NOT NULL,
                    local_currency TEXT NOT NULL,
                    is_financial INTEGER NOT NULL CHECK(is_financial IN (0, 1)),
                    PRIMARY KEY (as_of_date, ticker)
                );

                CREATE INDEX IF NOT EXISTS idx_universe_pit_isin
                    ON universe_pit(isin);
                CREATE INDEX IF NOT EXISTS idx_universe_pit_date
                    ON universe_pit(as_of_date);

                CREATE TABLE IF NOT EXISTS fundamentals_pit (
                    as_of_date TEXT NOT NULL,
                    filing_date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    value REAL NOT NULL,
                    PRIMARY KEY (as_of_date, ticker, metric),
                    CHECK(filing_date <= as_of_date)
                );

                CREATE INDEX IF NOT EXISTS idx_fundamentals_pit_ticker
                    ON fundamentals_pit(ticker);
                CREATE INDEX IF NOT EXISTS idx_fundamentals_pit_date
                    ON fundamentals_pit(as_of_date);

                CREATE TABLE IF NOT EXISTS macro_series (
                    date TEXT NOT NULL,
                    region_bloc TEXT NOT NULL CHECK(region_bloc IN ('NA', 'EU')),
                    series_id TEXT NOT NULL,
                    value REAL NOT NULL,
                    PRIMARY KEY (date, region_bloc, series_id)
                );

                CREATE INDEX IF NOT EXISTS idx_macro_series_date
                    ON macro_series(date);

                CREATE TABLE IF NOT EXISTS security_metadata_cache (
                    ticker TEXT PRIMARY KEY,
                    isin TEXT NOT NULL,
                    gics_sector INTEGER NOT NULL,
                    local_currency TEXT NOT NULL,
                    name TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_security_metadata_isin
                    ON security_metadata_cache(isin);
                """
            )

    def upsert_universe_pit(self, data: pd.DataFrame | Sequence[dict[str, Any]]) -> int:
        """Upsert universe records idempotently into universe_pit."""
        records: list[tuple[str, str, str, str, str, int, str, int]] = []

        raw_items: Sequence[dict[Any, Any]]
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return 0
            raw_items = data.to_dict(orient="records")
        else:
            raw_items = data

        for item in raw_items:
            records.append(
                (
                    str(item["as_of_date"]),
                    str(item["ticker"]),
                    safe_str(item.get("isin"), default=""),
                    str(item["region_bloc"]),
                    str(item["source_index"]),
                    safe_int(item.get("gics_sector"), default=0),
                    safe_str(item.get("local_currency"), default="EUR"),
                    1 if bool(item.get("is_financial")) else 0,
                )
            )

        if not records:
            return 0

        sql = """
            INSERT INTO universe_pit (
                as_of_date, ticker, isin, region_bloc, source_index,
                gics_sector, local_currency, is_financial
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(as_of_date, ticker) DO UPDATE SET
                isin = excluded.isin,
                region_bloc = excluded.region_bloc,
                source_index = excluded.source_index,
                gics_sector = excluded.gics_sector,
                local_currency = excluded.local_currency,
                is_financial = excluded.is_financial;
        """
        with self.get_connection() as conn:
            conn.executemany(sql, records)
        return len(records)

    def delete_universe_pit(
        self, as_of_date: str | None = None, source_index: str | None = None
    ) -> int:
        """Delete universe_pit records, optionally filtered by as_of_date and source_index."""
        query = "DELETE FROM universe_pit"
        conditions: list[str] = []
        params: list[Any] = []
        if as_of_date is not None:
            conditions.append("as_of_date = ?")
            params.append(as_of_date)
        if source_index is not None:
            conditions.append("source_index = ?")
            params.append(source_index)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with self.get_connection() as conn:
            cur = conn.execute(query, params)
            return cur.rowcount

    def upsert_fundamentals_pit(self, data: pd.DataFrame | Sequence[dict[str, Any]]) -> int:
        """Upsert fundamentals point-in-time records.

        Enforces filing_date <= as_of_date constraint to prevent look-ahead bias.
        """
        records: list[tuple[str, str, str, str, float]] = []

        raw_items: Sequence[dict[Any, Any]]
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return 0
            raw_items = data.to_dict(orient="records")
        else:
            raw_items = data

        for item in raw_items:
            as_of_date = str(item["as_of_date"])
            filing_date = str(item["filing_date"])
            if filing_date > as_of_date:
                raise ValueError(
                    f"Point-in-Time violation for {item.get('ticker')}: "
                    f"filing_date ({filing_date}) > as_of_date ({as_of_date})"
                )
            records.append(
                (
                    as_of_date,
                    filing_date,
                    str(item["ticker"]),
                    str(item["metric"]),
                    float(item["value"]),
                )
            )

        if not records:
            return 0

        sql = """
            INSERT INTO fundamentals_pit (
                as_of_date, filing_date, ticker, metric, value
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(as_of_date, ticker, metric) DO UPDATE SET
                filing_date = excluded.filing_date,
                value = excluded.value;
        """
        with self.get_connection() as conn:
            conn.executemany(sql, records)
        return len(records)

    def upsert_macro_series(self, data: pd.DataFrame | Sequence[dict[str, Any]]) -> int:
        """Upsert macro series observations idempotently."""
        records: list[tuple[str, str, str, float]] = []

        raw_items: Sequence[dict[Any, Any]]
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return 0
            raw_items = data.to_dict(orient="records")
        else:
            raw_items = data

        for item in raw_items:
            records.append(
                (
                    str(item["date"]),
                    str(item["region_bloc"]),
                    str(item["series_id"]),
                    float(item["value"]),
                )
            )

        if not records:
            return 0

        sql = """
            INSERT INTO macro_series (date, region_bloc, series_id, value)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(date, region_bloc, series_id) DO UPDATE SET
                value = excluded.value;
        """
        with self.get_connection() as conn:
            conn.executemany(sql, records)
        return len(records)

    def get_universe_pit(
        self,
        as_of_date: str | None = None,
        region_bloc: str | None = None,
        exclude_financials: bool = False,
    ) -> pd.DataFrame:
        """Retrieve point-in-time universe records with optional filters."""
        query = "SELECT * FROM universe_pit WHERE 1=1"
        params: list[Any] = []

        if as_of_date is not None:
            query += " AND as_of_date = ?"
            params.append(as_of_date)
        if region_bloc is not None:
            query += " AND region_bloc = ?"
            params.append(region_bloc)
        if exclude_financials:
            query += " AND is_financial = 0"

        query += " ORDER BY as_of_date ASC, ticker ASC"
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        df["is_financial"] = df["is_financial"].astype(bool)
        return df

    def get_fundamentals_pit(
        self,
        as_of_date: str | None = None,
        ticker: str | None = None,
        metric: str | None = None,
    ) -> pd.DataFrame:
        """Retrieve point-in-time fundamentals records with optional filters."""
        query = "SELECT * FROM fundamentals_pit WHERE 1=1"
        params: list[Any] = []

        if as_of_date is not None:
            query += " AND as_of_date = ?"
            params.append(as_of_date)
        if ticker is not None:
            query += " AND ticker = ?"
            params.append(ticker)
        if metric is not None:
            query += " AND metric = ?"
            params.append(metric)

        query += " ORDER BY as_of_date ASC, ticker ASC, metric ASC"
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_macro_series(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        region_bloc: str | None = None,
        series_id: str | None = None,
    ) -> pd.DataFrame:
        """Retrieve macro series observations with optional filters."""
        query = "SELECT * FROM macro_series WHERE 1=1"
        params: list[Any] = []

        if start_date is not None:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date is not None:
            query += " AND date <= ?"
            params.append(end_date)
        if region_bloc is not None:
            query += " AND region_bloc = ?"
            params.append(region_bloc)
        if series_id is not None:
            query += " AND series_id = ?"
            params.append(series_id)

        query += " ORDER BY date ASC, region_bloc ASC, series_id ASC"
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def upsert_security_metadata_cache(self, records: Sequence[dict[str, Any]]) -> int:
        """Upsert security metadata records into security_metadata_cache."""
        if not records:
            return 0

        rows: list[tuple[str, str, int, str, str | None, str]] = []
        for r in records:
            rows.append(
                (
                    str(r["ticker"]).strip(),
                    safe_str(r.get("isin"), default="").upper(),
                    safe_int(r.get("gics_sector"), default=0),
                    safe_str(r.get("local_currency"), default="EUR").upper(),
                    safe_optional_str(r.get("name")),
                    str(r.get("updated_at", pd.Timestamp.now().strftime("%Y-%m-%d"))),
                )
            )

        sql = """
            INSERT INTO security_metadata_cache (
                ticker, isin, gics_sector, local_currency, name, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                isin = excluded.isin,
                gics_sector = excluded.gics_sector,
                local_currency = excluded.local_currency,
                name = coalesce(excluded.name, security_metadata_cache.name),
                updated_at = excluded.updated_at;
        """
        with self.get_connection() as conn:
            conn.executemany(sql, rows)
        return len(rows)

    def get_security_metadata_cache(
        self, tickers: Sequence[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Retrieve cached metadata for tickers, checking cache table and universe_pit."""
        result: dict[str, dict[str, Any]] = {}
        with self.get_connection() as conn:
            # 1. Check dedicated cache table
            if tickers is not None:
                if not tickers:
                    return {}
                placeholders = ",".join("?" for _ in tickers)
                c_query = (
                    f"SELECT ticker, isin, gics_sector, local_currency, name "
                    f"FROM security_metadata_cache WHERE ticker IN ({placeholders})"
                )
                c_df = pd.read_sql_query(c_query, conn, params=list(tickers))
            else:
                c_df = pd.read_sql_query("SELECT * FROM security_metadata_cache", conn)

            for _, row in c_df.iterrows():
                t = str(row["ticker"])
                result[t] = {
                    "ticker": t,
                    "isin": safe_str(row.get("isin"), default=""),
                    "gics_sector": safe_int(row.get("gics_sector"), default=0),
                    "local_currency": safe_str(row.get("local_currency"), default="EUR"),
                    "name": safe_optional_str(row.get("name")),
                }

            # 2. Check universe_pit as fallback for any still missing tickers
            missing_tickers = [t for t in tickers if t not in result] if tickers is not None else []
            if missing_tickers:
                u_placeholders = ",".join("?" for _ in missing_tickers)
                u_query = (
                    f"SELECT DISTINCT ticker, isin, gics_sector, local_currency "
                    f"FROM universe_pit WHERE ticker IN ({u_placeholders}) AND isin != ''"
                )
                u_df = pd.read_sql_query(u_query, conn, params=missing_tickers)
                for _, row in u_df.iterrows():
                    t = str(row["ticker"])
                    if t not in result:
                        result[t] = {
                            "ticker": t,
                            "isin": safe_str(row.get("isin"), default=""),
                            "gics_sector": safe_int(row.get("gics_sector"), default=0),
                            "local_currency": safe_str(row.get("local_currency"), default="EUR"),
                            "name": None,
                        }

        return result


class ParquetMarketStore:
    """Idempotent yearly-partitioned Parquet storage for daily EUR market data."""

    def __init__(self, parquet_dir: Path | str) -> None:
        self.parquet_dir = Path(parquet_dir)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

    def _year_file_path(self, year: int) -> Path:
        return self.parquet_dir / f"daily_market_{year}.parquet"

    def write_daily_market(self, df: pd.DataFrame) -> int:
        """Write daily market data into yearly partitioned Parquet files idempotently.

        Enforces required schema and deduplicates by (date, ticker) preserving the latest row.
        """
        if df.empty:
            return 0

        missing = set(DAILY_MARKET_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required daily market columns: {missing}")

        formatted = df[DAILY_MARKET_COLUMNS].copy()
        formatted["date"] = formatted["date"].astype(str)
        formatted["ticker"] = formatted["ticker"].astype(str)
        formatted["region_bloc"] = formatted["region_bloc"].astype(str)
        formatted["px_last_eur"] = formatted["px_last_eur"].astype(float)
        formatted["tot_return_idx_eur"] = formatted["tot_return_idx_eur"].astype(float)
        formatted["volume_eur"] = formatted["volume_eur"].astype(float)

        formatted["_year"] = formatted["date"].str.slice(0, 4).astype(int)

        total_upserted = 0
        for year, group in formatted.groupby("_year"):
            year_int = int(str(year))
            group_clean = group.drop(columns=["_year"])
            file_path = self._year_file_path(year_int)

            if file_path.exists():
                existing = pd.read_parquet(file_path)
                combined = pd.concat([existing, group_clean], ignore_index=True)
                deduped = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
            else:
                deduped = group_clean.drop_duplicates(subset=["date", "ticker"], keep="last")

            table = pa.Table.from_pandas(
                deduped,
                schema=DAILY_MARKET_ARROW_SCHEMA,
                preserve_index=False,
            )
            pq.write_table(table, file_path, compression="snappy")
            total_upserted += len(group_clean)

        return total_upserted

    def read_daily_market(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        tickers: list[str] | None = None,
        region_bloc: str | None = None,
    ) -> pd.DataFrame:
        """Read daily market data across partitions with optional filtering."""
        files = sorted(self.parquet_dir.glob("daily_market_*.parquet"))
        if not files:
            return pd.DataFrame(columns=DAILY_MARKET_COLUMNS)

        frames: list[pd.DataFrame] = []
        for file in files:
            # Check year from file name to skip irrelevant partitions if possible
            year_str = file.stem.replace("daily_market_", "")
            if year_str.isdigit():
                file_year = int(year_str)
                if start_date is not None and file_year < int(start_date[:4]):
                    continue
                if end_date is not None and file_year > int(end_date[:4]):
                    continue

            df = pd.read_parquet(file)
            if start_date is not None:
                df = df[df["date"] >= start_date]
            if end_date is not None:
                df = df[df["date"] <= end_date]
            if tickers is not None:
                df = df[df["ticker"].isin(tickers)]
            if region_bloc is not None:
                df = df[df["region_bloc"] == region_bloc]

            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=DAILY_MARKET_COLUMNS)

        result = pd.concat(frames, ignore_index=True)
        return result.sort_values(by=["date", "ticker"]).reset_index(drop=True)

    def get_existing_tickers(self) -> set[str]:
        """Retrieve set of unique tickers already stored across all Parquet partitions."""
        files = sorted(self.parquet_dir.glob("daily_market_*.parquet"))
        if not files:
            return set()
        existing: set[str] = set()
        for file in files:
            try:
                table = pq.read_table(file, columns=["ticker"])
                t_arr = table.column("ticker").to_pylist()
                existing.update(t_arr)
            except Exception:
                continue
        return existing
