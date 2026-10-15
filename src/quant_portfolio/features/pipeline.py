"""Cross-sectional Feature Pipeline assembling 15 core predictive signals.

Combines:
- 10 Point-in-Time fundamental features (with strict filing_date <= as_of_date)
- 5 Technical and CAPM risk features in EUR (zero look-ahead bias from Parquet)
- Transatlantic cross-sectional normalization (regional Winsorization + Region/Sector Z-scores)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quant_portfolio.config import (
    ALL_CORE_FEATURES,
    CORE_FUNDAMENTAL_METRICS,
    FeatureConfig,
    PortfolioConfig,
    RegionBloc,
)
from quant_portfolio.features.normalizer import TransatlanticCrossSectionalNormalizer
from quant_portfolio.features.technical import TechnicalFeatureCalculator
from quant_portfolio.storage.schema import (
    ParquetMarketStore,
    SQLiteStore,
    deduplicate_universe_by_isin,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class TransatlanticFeaturePipeline:
    """Orchestrates end-to-end extraction, assembly, and normalization of the 15 core features."""

    def __init__(
        self,
        config: PortfolioConfig | None = None,
        sqlite_store: SQLiteStore | None = None,
        parquet_store: ParquetMarketStore | None = None,
        normalizer: TransatlanticCrossSectionalNormalizer | None = None,
        technical_calculator: TechnicalFeatureCalculator | None = None,
    ) -> None:
        self.config = config or PortfolioConfig()
        paths = self.config.paths.resolve_paths()
        self.sqlite_store = sqlite_store or SQLiteStore(paths.db_path)
        self.parquet_store = parquet_store or ParquetMarketStore(paths.parquet_dir)
        self.feature_config = self.config.features or FeatureConfig()
        self.normalizer = normalizer or TransatlanticCrossSectionalNormalizer(
            config=self.feature_config
        )
        self.technical_calculator = technical_calculator or TechnicalFeatureCalculator(
            parquet_store=self.parquet_store,
            config=self.feature_config,
        )

    def get_effective_universe(self, as_of_date: str) -> pd.DataFrame:
        """Retrieve the non-financial universe as of as_of_date (or latest prior date in DB)."""
        uni = self.sqlite_store.get_universe_pit(as_of_date=as_of_date, exclude_financials=True)
        if uni.empty:
            # Query the latest available universe date prior to or equal to as_of_date
            query = (
                "SELECT DISTINCT as_of_date FROM universe_pit WHERE as_of_date <= ? "
                "ORDER BY as_of_date DESC LIMIT 1"
            )
            with self.sqlite_store.get_connection() as conn:
                cursor = conn.execute(query, (as_of_date,))
                row = cursor.fetchone()
                if row:
                    latest_date = str(row[0])
                    uni = self.sqlite_store.get_universe_pit(
                        as_of_date=latest_date, exclude_financials=True
                    )

        if uni.empty:
            return pd.DataFrame(
                columns=["as_of_date", "ticker", "isin", "region_bloc", "gics_sector"]
            )

        # Deduplicate cross-listings sharing the same ISIN
        deduped = deduplicate_universe_by_isin(uni)
        deduped["as_of_date"] = as_of_date
        return deduped[["as_of_date", "ticker", "isin", "region_bloc", "gics_sector"]].copy()

    def get_point_in_time_fundamentals(
        self,
        as_of_date: str,
        tickers: Sequence[str],
    ) -> pd.DataFrame:
        """Retrieve most recent fundamental records filed strictly on or before as_of_date.

        Pivots results so each row is a ticker and columns are the 10 core fundamental metrics.
        """
        if not tickers:
            return pd.DataFrame(columns=["ticker"] + list(CORE_FUNDAMENTAL_METRICS))

        placeholders = ",".join("?" for _ in tickers)
        query = f"""
            WITH ranked AS (
                SELECT
                    ticker,
                    metric,
                    value,
                    ROW_NUMBER() OVER (
                        PARTITION BY ticker, metric
                        ORDER BY filing_date DESC, as_of_date DESC
                    ) as rn
                FROM fundamentals_pit
                WHERE filing_date <= ?
                  AND ticker IN ({placeholders})
                  AND metric IN ({",".join("?" for _ in CORE_FUNDAMENTAL_METRICS)})
            )
            SELECT ticker, metric, value
            FROM ranked
            WHERE rn = 1;
        """
        params: list[str] = [as_of_date] + list(tickers) + list(CORE_FUNDAMENTAL_METRICS)

        with self.sqlite_store.get_connection() as conn:
            raw_funds = pd.read_sql_query(query, conn, params=params)

        if raw_funds.empty:
            pivoted = pd.DataFrame(columns=["ticker"] + list(CORE_FUNDAMENTAL_METRICS))
        else:
            pivoted = raw_funds.pivot(
                index="ticker", columns="metric", values="value"
            ).reset_index()

        # Ensure all 10 fundamental columns exist
        for m in CORE_FUNDAMENTAL_METRICS:
            if m not in pivoted.columns:
                pivoted[m] = np.nan

        return pivoted

    def build_features_at_date(
        self,
        as_of_date: str,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Construct the complete cross-sectional feature matrix X_t for as_of_date."""
        universe = self.get_effective_universe(as_of_date)
        if universe.empty:
            logger.warning("No investible stocks found for date %s", as_of_date)
            cols = ["as_of_date", "ticker", "isin", "region_bloc", "gics_sector"] + list(
                ALL_CORE_FEATURES
            )
            return pd.DataFrame(columns=cols)

        tickers = universe["ticker"].tolist()
        region_map: dict[str, RegionBloc] = {
            str(row["ticker"]): "NA" if str(row["region_bloc"]) == "NA" else "EU"
            for _, row in universe.iterrows()
        }

        # 1. Fundamentals (10 features)
        funds_df = self.get_point_in_time_fundamentals(as_of_date, tickers)

        # 2. Technical and Risk features (5 features)
        tech_df = self.technical_calculator.compute_features_at_date(
            as_of_date=as_of_date,
            tickers=tickers,
            region_map=region_map,
        )

        # 3. Assemble raw feature table
        merged = universe.merge(funds_df, on="ticker", how="left")
        merged = merged.merge(tech_df.drop(columns=["as_of_date"]), on="ticker", how="left")

        # Ensure all columns exist and are cast to float64
        feature_cols = list(ALL_CORE_FEATURES)
        for col in feature_cols:
            if col not in merged.columns:
                merged[col] = np.nan
            merged[col] = merged[col].astype(np.float64)

        # 4. Cross-Sectional Double Normalization
        if normalize:
            normalized = self.normalizer.normalize(
                merged,
                feature_cols=feature_cols,
                group_cols=["region_bloc", "gics_sector"],
            )
            return normalized

        return merged

    def build_feature_panel(
        self,
        dates: Sequence[str],
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Construct cross-sectional feature matrices across a sequence of dates."""
        frames: list[pd.DataFrame] = []
        for d in sorted(dates):
            logger.info("Computing features for as_of_date: %s", d)
            df_d = self.build_features_at_date(as_of_date=d, normalize=normalize)
            if not df_d.empty:
                frames.append(df_d)

        if not frames:
            cols = ["as_of_date", "ticker", "isin", "region_bloc", "gics_sector"] + list(
                ALL_CORE_FEATURES
            )
            return pd.DataFrame(columns=cols)

        return pd.concat(frames, ignore_index=True)
