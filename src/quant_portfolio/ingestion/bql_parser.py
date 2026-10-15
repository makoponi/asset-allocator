"""Parser and normalizer for raw BQL index constituent exports.

Processes raw exported files (Parquet, CSV, Excel) from Bloomberg BQL,
maps fields to SQLite universe schema, marks Financials (GICS 40),
preserves delisted tickers with numerical prefixes, and upserts to SQLite.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from quant_portfolio.config import CORE_FUNDAMENTAL_METRICS, FINANCIALS_GICS_SECTOR, RegionBloc
from quant_portfolio.ingestion.bql_templates import INDEX_REGION_MAP
from quant_portfolio.storage.schema import SQLiteStore, deduplicate_universe_by_isin

# Flexible column aliases from BQL exports
COLUMN_ALIASES: dict[str, Sequence[str]] = {
    "as_of_date": ("as_of_date", "date", "asofdate", "dates", "period_end_date"),
    "ticker": ("ticker", "id", "security", "symbol", "bbg_ticker"),
    "isin": ("isin", "id_isin", "isin_code"),
    "source_index": ("source_index", "index", "index_ticker", "benchmark"),
    "gics_sector": ("gics_sector", "sector_code", "gics_sector_code", "gics"),
    "local_currency": ("local_currency", "crncy", "currency", "local_crncy"),
}

FUNDAMENTAL_METRIC_ALIASES: dict[str, Sequence[str]] = {
    "PE_RATIO": ("pe_ratio", "pe", "trailing_pe", "p_e"),
    "PX_TO_BOOK_RATIO": ("px_to_book_ratio", "pb", "pb_ratio", "px_to_book", "p_b"),
    "EV_TO_T12M_EBITDA": ("ev_to_t12m_ebitda", "ev_ebitda", "ev_to_ebitda"),
    "FREE_CASH_FLOW_YIELD": ("free_cash_flow_yield", "fcf_yield", "fcfy"),
    "RETURN_COM_EQY": ("return_com_eqy", "roe", "return_on_equity"),
    "OPER_MARGIN": ("oper_margin", "operating_margin", "op_margin"),
    "ASSET_TURNOVER": ("asset_turnover", "asset_turn", "turnover_assets"),
    "TOT_DEBT_TO_TOT_EQY": ("tot_debt_to_tot_eqy", "debt_to_equity", "debt_equity"),
    "SALES_REV_TURN_GROWTH_1YR": (
        "sales_rev_turn_growth_1yr",
        "sales_growth",
        "rev_growth_1yr",
        "revenue_growth_1yr",
    ),
    "ACCRUALS_RATIO": ("accruals_ratio", "accruals", "accruals_ratio_cf"),
}


def _match_column(df: pd.DataFrame, target: str, aliases: Sequence[str]) -> str:
    """Find matching column in DataFrame by alias (case-insensitive)."""
    col_map = {str(c).lower().strip(): str(c) for c in df.columns}
    for alias in aliases:
        if alias.lower() in col_map:
            return col_map[alias.lower()]
    cols = list(df.columns)
    raise ValueError(f"Required column '{target}' not found in DataFrame. Available: {cols}")


def parse_bql_universe_file(file_path: Path | str) -> pd.DataFrame:
    """Read a raw export file (Parquet, CSV, XLSX) into a pandas DataFrame."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Export file not found: {p}")

    ext = p.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(p)
    elif ext in (".csv", ".txt"):
        return pd.read_csv(p)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(p)
    else:
        raise ValueError(f"Unsupported file format '{ext}' for file: {p}")


def normalize_universe_records(
    df: pd.DataFrame,
    default_source_index: str | None = None,
    default_region: RegionBloc | None = None,
) -> pd.DataFrame:
    """Normalize raw BQL DataFrame into standardized universe_pit records.

    Rules applied:
    - Normalizes dates to YYYY-MM-DD string.
    - Strips whitespace from tickers and ISINs.
    - Preserves numerical delisted prefixes (e.g. 1CNHI IM).
    - Infers region_bloc from source_index (or fallback).
    - Sets is_financial = True for GICS Sector 40.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "as_of_date",
                "ticker",
                "isin",
                "region_bloc",
                "source_index",
                "gics_sector",
                "local_currency",
                "is_financial",
            ]
        )

    # Resolve column names
    date_col = _match_column(df, "as_of_date", COLUMN_ALIASES["as_of_date"])
    ticker_col = _match_column(df, "ticker", COLUMN_ALIASES["ticker"])
    isin_col = _match_column(df, "isin", COLUMN_ALIASES["isin"])

    try:
        source_index_col = _match_column(df, "source_index", COLUMN_ALIASES["source_index"])
        source_index_series = df[source_index_col].astype(str).str.strip()
    except ValueError:
        if default_source_index is None:
            raise ValueError(
                "Column 'source_index' missing and no default_source_index specified."
            ) from None
        source_index_series = pd.Series(default_source_index, index=df.index)

    try:
        gics_col = _match_column(df, "gics_sector", COLUMN_ALIASES["gics_sector"])
        # Cast to integer safely, filling NA with 0
        gics_series = pd.to_numeric(df[gics_col], errors="coerce").fillna(0).astype(int)
    except ValueError:
        gics_series = pd.Series(0, index=df.index, dtype=int)

    try:
        crncy_col = _match_column(df, "local_currency", COLUMN_ALIASES["local_currency"])
        crncy_series = df[crncy_col].astype(str).str.strip().str.upper()
    except ValueError:
        crncy_series = pd.Series("EUR", index=df.index)

    # Format date as YYYY-MM-DD
    date_series = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")

    # Clean ticker and isin (preserve exact delisted ticker string)
    ticker_series = df[ticker_col].astype(str).str.strip()
    isin_series = df[isin_col].astype(str).str.strip().str.upper()

    # Determine region_bloc
    def _resolve_region(idx: str) -> RegionBloc:
        if idx in INDEX_REGION_MAP:
            return INDEX_REGION_MAP[idx]
        if default_region is not None:
            return default_region
        supported = list(INDEX_REGION_MAP.keys())
        raise ValueError(
            f"Unable to resolve region for index '{idx}'. Recognized indices: {supported}"
        )

    region_series = source_index_series.map(_resolve_region)

    # Financials flag (Sector 40)
    is_financial_series = gics_series == FINANCIALS_GICS_SECTOR

    normalized = pd.DataFrame(
        {
            "as_of_date": date_series,
            "ticker": ticker_series,
            "isin": isin_series,
            "region_bloc": region_series,
            "source_index": source_index_series,
            "gics_sector": gics_series,
            "local_currency": crncy_series,
            "is_financial": is_financial_series,
        }
    )

    # Drop invalid rows (missing ticker or ISIN)
    valid = normalized["ticker"].ne("") & normalized["isin"].ne("")
    return normalized[valid].reset_index(drop=True)


class BQLUniverseIngestionEngine:
    """Orchestrates parsing of drop-zone BQL files and database upsert."""

    def __init__(self, store: SQLiteStore, raw_dir: Path | str) -> None:
        self.store = store
        self.raw_dir = Path(raw_dir)

    def ingest_file(
        self,
        file_path: Path | str,
        default_source_index: str | None = None,
        default_region: RegionBloc | None = None,
        dedup_by_isin: bool = True,
    ) -> int:
        """Parse a single export file and upsert to SQLite."""
        raw_df = parse_bql_universe_file(file_path)
        normalized = normalize_universe_records(
            raw_df,
            default_source_index=default_source_index,
            default_region=default_region,
        )
        if normalized.empty:
            return 0

        final_df = normalized
        if dedup_by_isin:
            final_df = deduplicate_universe_by_isin(normalized)

        return self.store.upsert_universe_pit(final_df)

    def ingest_drop_zone(
        self,
        glob_pattern: str = "*.*",
        dedup_by_isin: bool = True,
    ) -> dict[str, int]:
        """Ingest all supported export files from drop-zone directory."""
        if not self.raw_dir.exists():
            return {}

        results: dict[str, int] = {}
        for file in sorted(self.raw_dir.glob(glob_pattern)):
            if file.suffix.lower() in (".parquet", ".csv", ".xlsx", ".xls"):
                try:
                    count = self.ingest_file(file, dedup_by_isin=dedup_by_isin)
                    results[file.name] = count
                except Exception:
                    results[file.name] = -1  # Flag error
        return results


parse_bql_file = parse_bql_universe_file


def _canonicalize_metric_name(col_name: str) -> str | None:
    """Map a column or metric name to canonical Bloomberg metric name if recognized."""
    clean = col_name.lower().strip()
    # Check direct canonical match
    for canonical in CORE_FUNDAMENTAL_METRICS:
        if clean == canonical.lower():
            return canonical
    # Check aliases
    for canonical, aliases in FUNDAMENTAL_METRIC_ALIASES.items():
        for alias in aliases:
            if clean == alias or clean.startswith(f"{alias}(") or clean.startswith(f"{alias}_"):
                return canonical
    return None


def normalize_fundamentals_records(
    df: pd.DataFrame,
    default_as_of_date: str | None = None,
    default_lag_days: int = 90,
) -> pd.DataFrame:
    """Normalize raw BQL DataFrame into standardized fundamentals_pit records.

    Supports both long-format (as_of_date, ticker, metric, value, filing_date) and
    wide-format (as_of_date, ticker, PE_RATIO, PX_TO_BOOK_RATIO, ...).
    Enforces filing_date <= as_of_date constraint to prevent look-ahead bias.
    """
    if df.empty:
        return pd.DataFrame(columns=["as_of_date", "filing_date", "ticker", "metric", "value"])

    # Resolve date column or use default
    try:
        date_col = _match_column(df, "as_of_date", COLUMN_ALIASES["as_of_date"])
        date_series = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    except ValueError:
        if default_as_of_date is None:
            raise ValueError(
                "No date column found and default_as_of_date was not provided."
            ) from None
        date_series = pd.Series(default_as_of_date, index=df.index)

    # Resolve ticker
    ticker_col = _match_column(df, "ticker", COLUMN_ALIASES["ticker"])
    ticker_series = df[ticker_col].astype(str).str.strip()

    # Optional filing_date
    filing_col: str | None = None
    for alias in ("filing_date", "filingdate", "announce_date", "report_date", "public_date"):
        try:
            filing_col = _match_column(df, "filing_date", (alias,))
            break
        except ValueError:
            continue

    filing_series: pd.Series[str]
    if filing_col is not None:
        filing_series = pd.to_datetime(df[filing_col]).dt.strftime("%Y-%m-%d")
    else:
        # Default to as_of_date if explicit filing_date is not present
        filing_series = date_series.copy()

    # Check if df is in long format (contains metric and value columns)
    metric_col: str | None = None
    value_col: str | None = None
    for m_alias in ("metric", "field", "indicator", "item"):
        try:
            metric_col = _match_column(df, "metric", (m_alias,))
            break
        except ValueError:
            continue

    for v_alias in ("value", "val", "data"):
        try:
            value_col = _match_column(df, "value", (v_alias,))
            break
        except ValueError:
            continue

    if metric_col is not None and value_col is not None:
        # Long format processing
        metrics_canonical = df[metric_col].astype(str).map(_canonicalize_metric_name)
        values_numeric = pd.to_numeric(df[value_col], errors="coerce")

        long_df = pd.DataFrame(
            {
                "as_of_date": date_series,
                "filing_date": filing_series,
                "ticker": ticker_series,
                "metric": metrics_canonical,
                "value": values_numeric,
            }
        )
    else:
        # Wide format: find all recognized metric columns
        matched_metric_cols: dict[str, str] = {}
        for c in df.columns:
            canonical = _canonicalize_metric_name(str(c))
            if canonical is not None:
                matched_metric_cols[str(c)] = canonical

        if not matched_metric_cols:
            return pd.DataFrame(columns=["as_of_date", "filing_date", "ticker", "metric", "value"])

        records: list[dict[str, Any]] = []
        for orig_col, canonical in matched_metric_cols.items():
            s_val = pd.to_numeric(df[orig_col], errors="coerce")
            for d, f, t, v in zip(date_series, filing_series, ticker_series, s_val, strict=False):
                if pd.notna(v):
                    records.append(
                        {
                            "as_of_date": d,
                            "filing_date": f,
                            "ticker": t,
                            "metric": canonical,
                            "value": float(v),
                        }
                    )
        long_df = pd.DataFrame(records)

    if long_df.empty:
        return pd.DataFrame(columns=["as_of_date", "filing_date", "ticker", "metric", "value"])

    # Filter out unmapped metrics, NaNs, empty tickers
    valid = (
        long_df["metric"].notna()
        & long_df["metric"].isin(CORE_FUNDAMENTAL_METRICS)
        & long_df["value"].notna()
        & long_df["ticker"].ne("")
        & (long_df["filing_date"] <= long_df["as_of_date"])  # Anti Look-Ahead Bias
    )
    clean_df = long_df[valid].copy()
    clean_df["value"] = clean_df["value"].astype(float)

    # Deduplicate keeping last record per (as_of_date, ticker, metric)
    return clean_df.drop_duplicates(
        subset=["as_of_date", "ticker", "metric"], keep="last"
    ).reset_index(drop=True)


class BQLFundamentalsIngestionEngine:
    """Orchestrates parsing of drop-zone BQL fundamentals exports and SQLite upsert."""

    def __init__(self, store: SQLiteStore, raw_dir: Path | str) -> None:
        self.store = store
        self.raw_dir = Path(raw_dir)

    def ingest_file(
        self,
        file_path: Path | str,
        default_as_of_date: str | None = None,
    ) -> int:
        """Parse a single fundamental export file and upsert to SQLite."""
        raw_df = parse_bql_file(file_path)
        normalized = normalize_fundamentals_records(
            raw_df,
            default_as_of_date=default_as_of_date,
        )
        if normalized.empty:
            return 0
        return self.store.upsert_fundamentals_pit(normalized)

    def ingest_drop_zone(
        self,
        glob_pattern: str = "*fundamental*.*",
    ) -> dict[str, int]:
        """Ingest all fundamental export files from drop-zone directory."""
        if not self.raw_dir.exists():
            return {}

        results: dict[str, int] = {}
        for file in sorted(self.raw_dir.glob(glob_pattern)):
            if file.suffix.lower() in (".parquet", ".csv", ".xlsx", ".xls"):
                try:
                    count = self.ingest_file(file)
                    results[file.name] = count
                except Exception:
                    results[file.name] = -1  # Flag error
        return results
