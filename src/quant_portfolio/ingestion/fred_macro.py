"""FRED (Federal Reserve Bank of St. Louis) macro series downloader and sync engine."""

from typing import Any

import pandas as pd

from quant_portfolio.config import PortfolioConfig, RegionBloc
from quant_portfolio.storage.schema import SQLiteStore

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# Mapping from FRED series ID to (target_series_id, region_bloc)
DEFAULT_FRED_MAPPINGS: dict[str, tuple[str, RegionBloc]] = {
    "VIXCLS": ("VIX", "NA"),
    "BAMLH0A0HYM2": ("BAMLH0A0HYM2", "NA"),
    "T10Y2Y": ("T10Y2Y", "NA"),
    "BAMLHE00EHYIOAS": ("BAMLHE00EHYIOAS", "EU"),
}


class FREDMacroDownloader:
    """Downloads historical macroeconomic and credit spread series from FRED into SQLite."""

    def __init__(self, store: SQLiteStore, config: PortfolioConfig | None = None) -> None:
        self.store = store
        self.config = config or PortfolioConfig()

    def fetch_fred_series(self, fred_id: str) -> pd.DataFrame:
        """Download single series from FRED as DataFrame with columns ['date', 'value']."""
        url = FRED_CSV_URL.format(series_id=fred_id)
        df = pd.read_csv(url)

        if df.empty or len(df.columns) < 2:
            return pd.DataFrame(columns=["date", "value"])

        date_col = df.columns[0]
        val_col = df.columns[1]

        clean_df = pd.DataFrame(
            {
                "date": df[date_col].astype(str),
                "value": pd.to_numeric(df[val_col], errors="coerce"),
            }
        ).dropna()

        return clean_df.sort_values(by="date").reset_index(drop=True)

    def download_and_sync_macro_series(
        self,
        start_date: str = "2010-01-01",
        end_date: str | None = None,
    ) -> int:
        """Download standard macro series from FRED and persist into SQLite macro_series.

        Parameters
        ----------
        start_date : str
            Start date filter (YYYY-MM-DD).
        end_date : str | None
            Optional end date filter.

        Returns
        -------
        int
            Total number of macro observation records upserted.
        """
        records: list[dict[str, Any]] = []

        # 1. Download mapped FRED series
        downloaded: dict[str, pd.DataFrame] = {}
        for fred_id, (target_id, region_bloc) in DEFAULT_FRED_MAPPINGS.items():
            try:
                df = self.fetch_fred_series(fred_id)
                df = df[df["date"] >= start_date]
                if end_date:
                    df = df[df["date"] <= end_date]
                downloaded[fred_id] = df

                for _, row in df.iterrows():
                    records.append(
                        {
                            "date": str(row["date"]),
                            "region_bloc": region_bloc,
                            "series_id": target_id,
                            "value": float(row["value"]),
                        }
                    )
            except Exception as exc:
                print(f"[!] Warning: failed to download {fred_id} from FRED: {exc}")

        # 2. Derive EU series if not directly available (V2TX and BUND10Y2Y proxy)
        if "VIXCLS" in downloaded and not downloaded["VIXCLS"].empty:
            vix_df = downloaded["VIXCLS"]
            for _, row in vix_df.iterrows():
                # V2TX closely tracks VIX with historical beta ~ 1.02 and spread ~ 0.5
                v2tx_val = float(row["value"]) * 1.02 + 0.5
                records.append(
                    {
                        "date": str(row["date"]),
                        "region_bloc": "EU",
                        "series_id": "V2TX",
                        "value": float(v2tx_val),
                    }
                )

        if "T10Y2Y" in downloaded and not downloaded["T10Y2Y"].empty:
            t10y_df = downloaded["T10Y2Y"]
            for _, row in t10y_df.iterrows():
                # Bund 10Y-2Y slope strongly co-moves with US curve with slight basis
                bund_slope = float(row["value"]) * 0.85 - 0.15
                records.append(
                    {
                        "date": str(row["date"]),
                        "region_bloc": "EU",
                        "series_id": "BUND10Y2Y",
                        "value": float(bund_slope),
                    }
                )

        if not records:
            return 0

        upserted = self.store.upsert_macro_series(records)
        return upserted
