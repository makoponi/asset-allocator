"""Point-in-Time feature aligner: combines local technical factors and lagged PiT fundamentals.

Enforces zero look-ahead bias: at evaluation date T, strictly filters fundamentals
such that filing_date <= T.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    from src.features.technical_engine import TechnicalEngine
    from src.ingestion.storage import MarketStore


@dataclass(frozen=True, slots=True)
class AlignedFeatureRow:
    """Consolidated feature vector for a security at an evaluation date."""

    date: str
    ticker: str
    gics_sector: str
    # Technical factors (4)
    mom_1m: float
    mom_12m: float
    beta_252d: float
    ivol_252d: float
    # Fundamental factors (10 core fields)
    pe_ratio: float | None = None
    best_pe_ratio: float | None = None
    px_to_book: float | None = None
    fcf_yield: float | None = None
    roe: float | None = None
    prof_margin: float | None = None
    oper_margin: float | None = None
    debt_to_equity: float | None = None
    quick_ratio: float | None = None
    free_cash_flow: float | None = None


class PiTFeatureAligner:
    """Assembles cross-sectional feature matrices with point-in-time guarantees."""

    def __init__(
        self,
        market_store: MarketStore,
        technical_engine: TechnicalEngine,
    ) -> None:
        self.market_store = market_store
        self.technical_engine = technical_engine

    def build_cross_sectional_matrix(
        self,
        eval_date: date,
    ) -> list[AlignedFeatureRow]:
        """Build feature matrix for all active constituents on eval_date.

        Ensures that for all fundamental metrics, filing_date <= eval_date.
        """
        eval_date_str = eval_date.isoformat()

        # 1. Fetch active constituents for the evaluation date
        constituents = self.market_store.get_universe_members(
            as_of_date=eval_date_str, is_active_only=True
        )
        if not constituents:
            return []

        # 2. Build feature rows
        rows: list[AlignedFeatureRow] = []

        for c in constituents:
            # Local technical factors
            tf = self.technical_engine.compute_factors_for_ticker(c.ticker, eval_date)
            if tf is None:
                continue

            # Query PiT fundamentals for ticker
            all_funds = self.market_store.get_fundamentals_pit(ticker=c.ticker)

            # Strict Point-in-Time filter: filing_date <= eval_date_str
            valid_funds = [f for f in all_funds if f.filing_date <= eval_date_str]

            # Get latest available value for each metric
            latest_metrics: dict[str, float] = {}
            for f in sorted(valid_funds, key=lambda x: x.filing_date):
                latest_metrics[f.metric_name] = f.metric_value

            row = AlignedFeatureRow(
                date=eval_date_str,
                ticker=c.ticker,
                gics_sector=c.gics_sector,
                mom_1m=tf.mom_1m,
                mom_12m=tf.mom_12m,
                beta_252d=tf.beta_252d,
                ivol_252d=tf.ivol_252d,
                pe_ratio=latest_metrics.get("PE_RATIO"),
                best_pe_ratio=latest_metrics.get("BEST_PE_RATIO"),
                px_to_book=latest_metrics.get("PX_TO_BOOK_RATIO"),
                fcf_yield=latest_metrics.get("FREE_CASH_FLOW_YIELD"),
                roe=latest_metrics.get("RETURN_COM_EQY"),
                prof_margin=latest_metrics.get("PROF_MARGIN"),
                oper_margin=latest_metrics.get("OPER_MARGIN"),
                debt_to_equity=latest_metrics.get("TOT_DEBT_TO_TOT_EQY"),
                quick_ratio=latest_metrics.get("QUICK_RATIO"),
                free_cash_flow=latest_metrics.get("CF_FREE_CASH_FLOW"),
            )
            rows.append(row)

        return rows
