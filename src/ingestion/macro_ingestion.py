"""Macroeconomic indicators ingestion pipeline.

Fetches European macro regime signals (HY OAS spread, Bund 10Y-2Y slope, V2TX volatility)
from Bloomberg (via xbbg) or fallback sources and persists them in SQLite `macro_series`.
"""

from __future__ import annotations

import logging
import time
from datetime import date

import numpy as np
import pandas as pd

from src.ingestion.storage import MacroSeriesRecord, MarketStore

logger = logging.getLogger(__name__)

MACRO_INDICATORS: dict[str, str] = {
    "EU_HY_OAS": "LP01TREH Index",     # ICE BofA Euro High Yield OAS (or Pan-Euro HY OAS)
    "BUND_10Y": "GTDEM10Y Govt",       # German Bund 10Y Benchmark Yield
    "BUND_2Y": "GTDEM2Y Govt",         # German Bund 2Y Benchmark Yield
    "V2TX": "V2TX Index",              # EURO STOXX 50 Volatility Index
}


class MacroIngestionPipeline:
    """Pipeline for downloading and persisting macroeconomic time series."""

    def __init__(self, market_store: MarketStore | None = None) -> None:
        self.market_store = market_store or MarketStore()

    def ingest_from_bloomberg(
        self,
        start_date: str = "2015-01-01",
        end_date: str | None = None,
        throttle_seconds: float = 0.4,
    ) -> int:
        """Download macro series from Bloomberg using xbbg and store into SQLite.

        Respects Bloomberg throttling guidelines and data governance.
        """
        try:
            from xbbg import blp
        except ImportError as err:
            msg = "xbbg is not installed or Bloomberg session is unavailable."
            raise RuntimeError(msg) from err

        if end_date is None:
            end_date = date.today().isoformat()

        logger.info(
            "Fetching macro indicators from Bloomberg: %s -> %s",
            start_date,
            end_date,
        )

        records: list[MacroSeriesRecord] = []
        raw_series: dict[str, pd.Series[float]] = {}

        for name, ticker in MACRO_INDICATORS.items():
            logger.info("Downloading %s (%s)...", name, ticker)
            try:
                df = blp.bdh(
                    tickers=ticker,
                    flds="PX_LAST",
                    start_date=start_date,
                    end_date=end_date,
                    Per="D",
                )
                time.sleep(throttle_seconds)
                if df is not None and not df.empty:
                    s = df.iloc[:, 0].dropna()
                    s.index = pd.to_datetime(s.index).strftime("%Y-%m-%d")
                    raw_series[name] = s
                else:
                    logger.warning("Empty Bloomberg response for %s (%s)", name, ticker)
            except Exception as e:
                logger.error("Failed to fetch %s (%s): %s", name, ticker, e)

        # Process BUND_10Y_2Y slope
        if "BUND_10Y" in raw_series and "BUND_2Y" in raw_series:
            df_bund = pd.concat([raw_series["BUND_10Y"], raw_series["BUND_2Y"]], axis=1).dropna()
            slope = (df_bund.iloc[:, 0] - df_bund.iloc[:, 1]) * 100.0  # in basis points
            raw_series["BUND_10Y_2Y"] = slope

        # Build records
        for metric in ["EU_HY_OAS", "BUND_10Y_2Y", "V2TX"]:
            if metric in raw_series:
                for d_str, val in raw_series[metric].items():
                    records.append(
                        MacroSeriesRecord(
                            date=str(d_str),
                            series_id=metric,
                            value=float(val),
                        )
                    )

        inserted = self.market_store.insert_macro_series(records)
        logger.info("Persisted %d macro series records into SQLite.", inserted)
        return inserted

    def ingest_synthetic_history(
        self,
        start_date: str = "2015-01-01",
        end_date: str | None = None,
    ) -> int:
        """Generate realistic calibrated macro indicators for testing or offline simulations.

        Reconstructs the empirical dynamics of:
        - EU_HY_OAS: normal ~250-350 bps, stress spikes > 500 bps (e.g. 2020 Covid, 2022 H2).
        - BUND_10Y_2Y: normal ~50-100 bps, inverted/flattened during tightening ~ -30 to 10 bps.
        - V2TX: normal ~13-18%, elevated ~20-25%, panic > 30%.
        """
        if end_date is None:
            end_date = "2026-03-31"

        dr = pd.bdate_range(start=start_date, end=end_date)
        rng = np.random.default_rng(101)
        n = len(dr)

        # Baseline autoregressive processes
        # 1. V2TX: log-normal mean reverting process around 18%
        v2tx_vals = np.zeros(n)
        v2tx_vals[0] = 16.5
        for t in range(1, n):
            d_curr = dr[t]
            # Shock regimes (Covid Q1 2020, War/Inflation 2022)
            stress_factor = 0.0
            if pd.Timestamp("2020-02-20") <= d_curr <= pd.Timestamp("2020-04-30"):
                stress_factor = 25.0
            elif pd.Timestamp("2022-02-24") <= d_curr <= pd.Timestamp("2022-10-31"):
                stress_factor = 12.0
            elif pd.Timestamp("2023-03-08") <= d_curr <= pd.Timestamp("2023-03-24"):
                stress_factor = 8.0  # Silicon Valley Bank / Credit Suisse

            shock = rng.normal(0, 1.2)
            drift = 0.05 * (16.0 + stress_factor - v2tx_vals[t - 1])
            v2tx_vals[t] = max(10.0, v2tx_vals[t - 1] + drift + shock)

        # 2. EU_HY_OAS: strongly correlated with V2TX, base 300 bps
        hy_oas_vals = np.zeros(n)
        hy_oas_vals[0] = 320.0
        for t in range(1, n):
            target_spread = 220.0 + (v2tx_vals[t] - 14.0) * 15.0
            drift = 0.04 * (target_spread - hy_oas_vals[t - 1])
            noise = rng.normal(0, 5.0)
            hy_oas_vals[t] = max(180.0, hy_oas_vals[t - 1] + drift + noise)

        # 3. BUND_10Y_2Y curve slope (bps):
        # Flattening/inversion in 2022-2023 ECB hiking cycle
        slope_vals = np.zeros(n)
        slope_vals[0] = 60.0
        for t in range(1, n):
            d_curr = dr[t]
            if pd.Timestamp("2022-06-01") <= d_curr <= pd.Timestamp("2023-10-31"):
                target_slope = -25.0  # Inversion
            elif pd.Timestamp("2020-01-01") <= d_curr < pd.Timestamp("2022-06-01"):
                target_slope = 70.0   # Normal steep curve
            else:
                target_slope = 30.0   # Re-steepening
            drift = 0.03 * (target_slope - slope_vals[t - 1])
            noise = rng.normal(0, 2.0)
            slope_vals[t] = slope_vals[t - 1] + drift + noise

        records: list[MacroSeriesRecord] = []
        for i, d in enumerate(dr):
            d_str = d.strftime("%Y-%m-%d")
            records.append(
                MacroSeriesRecord(date=d_str, series_id="V2TX", value=float(v2tx_vals[i]))
            )
            records.append(
                MacroSeriesRecord(date=d_str, series_id="EU_HY_OAS", value=float(hy_oas_vals[i]))
            )
            records.append(
                MacroSeriesRecord(date=d_str, series_id="BUND_10Y_2Y", value=float(slope_vals[i]))
            )


        inserted = self.market_store.insert_macro_series(records)
        logger.info("Inserted %d calibrated macro series records into SQLite.", inserted)
        return inserted
