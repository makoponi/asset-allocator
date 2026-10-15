"""Bloomberg Query Language (BQL) string templates and Excel formula generators.

Provides standardized BQL queries for extracting transatlantic historical index members,
cross-sectional attributes (ISIN, GICS Sector, Currency), fundamentals, and EUR-converted series.
"""

from dataclasses import dataclass
from typing import Final

from quant_portfolio.config import RegionBloc

INDEX_REGION_MAP: Final[dict[str, RegionBloc]] = {
    "SPX Index": "NA",
    "SPTSX60 Index": "NA",
    "SXXP Index": "EU",
}

CORE_UNIVERSE_INDICES: Final[tuple[str, ...]] = (
    "SPX Index",
    "SPTSX60 Index",
    "SXXP Index",
)

UNIVERSE_STATIC_FIELDS: Final[tuple[str, ...]] = (
    "ID_ISIN",
    "GICS_SECTOR",
    "GICS_SECTOR_NAME",
    "NAME",
    "CRNCY",
    "EQY_PRIM_SECURITY_COMP_EXCH",
)

CORE_FUNDAMENTAL_FIELDS: Final[tuple[str, ...]] = (
    "PE_RATIO",
    "PX_TO_BOOK_RATIO",
    "EV_TO_T12M_EBITDA",
    "FREE_CASH_FLOW_YIELD",
    "RETURN_COM_EQY",
    "OPER_MARGIN",
    "ASSET_TURNOVER",
    "TOT_DEBT_TO_TOT_EQY",
    "SALES_REV_TURN_GROWTH_1YR",
    "ACCRUALS_RATIO",
)


@dataclass(frozen=True)
class IndexUniverseConfig:
    """Configuration for an index members extraction."""

    index_ticker: str
    region_bloc: RegionBloc
    frequency: str = "m"
    date_range: str = "range(-15y, 0d)"
    base_currency: str = "EUR"


class BQLQueryGenerator:
    """Generates syntactic BQL queries for terminal and Excel usage."""

    def __init__(
        self,
        default_range: str = "range(-15y, 0d)",
        default_frequency: str = "m",
        base_currency: str = "EUR",
    ) -> None:
        self.default_range = default_range
        self.default_frequency = default_frequency
        self.base_currency = base_currency

    def get_index_config(self, index_ticker: str) -> IndexUniverseConfig:
        """Resolve region and config for a recognized transatlantic index."""
        if index_ticker not in INDEX_REGION_MAP:
            supported = list(INDEX_REGION_MAP.keys())
            raise ValueError(
                f"Index '{index_ticker}' is not supported. Supported indices: {supported}"
            )
        return IndexUniverseConfig(
            index_ticker=index_ticker,
            region_bloc=INDEX_REGION_MAP[index_ticker],
            frequency=self.default_frequency,
            date_range=self.default_range,
            base_currency=self.base_currency,
        )

    def generate_members_expression(
        self,
        index_ticker: str,
        date_range: str | None = None,
        frequency: str | None = None,
    ) -> str:
        """Generate BQL universe expression: members('INDEX', dates=range(...), frq='m')."""
        rng = date_range or self.default_range
        frq = frequency or self.default_frequency
        return f"members('{index_ticker}', dates={rng}, frq='{frq}')"

    def generate_universe_extraction_query(
        self,
        index_ticker: str,
        date_range: str | None = None,
        frequency: str | None = None,
        include_fields: tuple[str, ...] | None = None,
    ) -> str:
        """Generate a complete BQL get query for historical members and static attributes."""
        universe_expr = self.generate_members_expression(
            index_ticker=index_ticker,
            date_range=date_range,
            frequency=frequency,
        )
        fields = include_fields or UNIVERSE_STATIC_FIELDS
        fields_str = ", ".join(fields)
        return f"get({fields_str}) for({universe_expr})"

    def generate_excel_universe_formula(
        self,
        index_ticker: str,
        date_range: str | None = None,
        frequency: str | None = None,
        include_fields: tuple[str, ...] | None = None,
    ) -> str:
        """Generate an Excel BQL formula `=BQL(universe, fields)`."""
        universe_expr = self.generate_members_expression(
            index_ticker=index_ticker,
            date_range=date_range,
            frequency=frequency,
        )
        fields = include_fields or UNIVERSE_STATIC_FIELDS
        fields_str = ", ".join(f"'{f}'" for f in fields)
        return f'=BQL("{universe_expr}", {fields_str})'

    def generate_fundamentals_query(
        self,
        index_ticker: str,
        metrics: tuple[str, ...] | None = None,
        date_range: str | None = None,
        frequency: str | None = None,
        currency: str | None = None,
    ) -> str:
        """Generate BQL query for fundamental metrics converted to base currency.

        Applies base currency conversion (default EUR) where applicable.
        """
        curr = currency or self.base_currency
        universe_expr = self.generate_members_expression(
            index_ticker=index_ticker,
            date_range=date_range,
            frequency=frequency,
        )
        target_metrics = metrics or CORE_FUNDAMENTAL_FIELDS
        metric_exprs: list[str] = []
        for m in target_metrics:
            metric_exprs.append(f"{m}(currency='{curr}')")
        return f"get({', '.join(metric_exprs)}) for({universe_expr})"

    def generate_excel_fundamentals_formula(
        self,
        index_ticker: str,
        metrics: tuple[str, ...] | None = None,
        date_range: str | None = None,
        frequency: str | None = None,
        currency: str | None = None,
    ) -> str:
        """Generate Excel formula for fundamental extraction in EUR."""
        curr = currency or self.base_currency
        universe_expr = self.generate_members_expression(
            index_ticker=index_ticker,
            date_range=date_range,
            frequency=frequency,
        )
        target_metrics = metrics or CORE_FUNDAMENTAL_FIELDS
        fields_args = ", ".join(f"\"{m}(currency='{curr}')\"" for m in target_metrics)
        return f'=BQL("{universe_expr}", {fields_args})'

    def generate_market_series_query(
        self,
        universe_expr: str,
        start_date: str = "-15y",
        end_date: str = "0d",
        currency: str | None = None,
    ) -> str:
        """Generate BQL query for daily EUR market series."""
        curr = currency or self.base_currency
        date_param = f"dates=range('{start_date}', '{end_date}', frq='d')"
        fields = [
            f"TOT_RETURN_INDEX_GROSS_DVDS({date_param}, currency='{curr}')",
            f"PX_LAST({date_param}, currency='{curr}')",
            f"PX_VOLUME({date_param})",
        ]
        return f"get({', '.join(fields)}) for({universe_expr})"


def generate_members_query(
    index_ticker: str,
    date_range: str = "range(-15y, 0d)",
    frequency: str = "m",
) -> str:
    """Convenience helper to generate member query string."""
    generator = BQLQueryGenerator()
    return generator.generate_universe_extraction_query(
        index_ticker=index_ticker,
        date_range=date_range,
        frequency=frequency,
    )


def generate_excel_bql_formula(
    index_ticker: str,
    date_range: str = "range(-15y, 0d)",
    frequency: str = "m",
) -> str:
    """Convenience helper to generate Excel formula string."""
    generator = BQLQueryGenerator()
    return generator.generate_excel_universe_formula(
        index_ticker=index_ticker,
        date_range=date_range,
        frequency=frequency,
    )
