"""Unit tests for Bloomberg BQL query and Excel formula templates."""

import pytest

from quant_portfolio.ingestion.bql_templates import (
    CORE_FUNDAMENTAL_FIELDS,
    CORE_UNIVERSE_INDICES,
    INDEX_REGION_MAP,
    UNIVERSE_STATIC_FIELDS,
    BQLQueryGenerator,
    generate_excel_bql_formula,
    generate_members_query,
)


def test_index_region_mapping() -> None:
    """Verify that all core transatlantic indices map to NA or EU."""
    assert INDEX_REGION_MAP["SPX Index"] == "NA"
    assert INDEX_REGION_MAP["SPTSX60 Index"] == "NA"
    assert INDEX_REGION_MAP["SXXP Index"] == "EU"
    for idx in CORE_UNIVERSE_INDICES:
        assert idx in INDEX_REGION_MAP


def test_bql_generator_index_config() -> None:
    """Verify BQLQueryGenerator index configuration resolver."""
    gen = BQLQueryGenerator()
    cfg_spx = gen.get_index_config("SPX Index")
    assert cfg_spx.index_ticker == "SPX Index"
    assert cfg_spx.region_bloc == "NA"
    assert cfg_spx.frequency == "m"
    assert cfg_spx.base_currency == "EUR"

    cfg_sxxp = gen.get_index_config("SXXP Index")
    assert cfg_sxxp.region_bloc == "EU"

    with pytest.raises(ValueError, match="Index 'UNKNOWN Index' is not supported"):
        gen.get_index_config("UNKNOWN Index")


def test_generate_members_expression() -> None:
    """Verify member expression formatting."""
    gen = BQLQueryGenerator()
    expr = gen.generate_members_expression("SPX Index")
    assert expr == "members('SPX Index', dates=range(-15y, 0d), frq='m')"

    expr_custom = gen.generate_members_expression(
        "SXXP Index", date_range="range(-10y, 0d)", frequency="q"
    )
    assert expr_custom == "members('SXXP Index', dates=range(-10y, 0d), frq='q')"


def test_generate_universe_extraction_query() -> None:
    """Verify BQL get query generation for universe constituents."""
    query = generate_members_query("SPX Index")
    assert query.startswith("get(")
    for field in UNIVERSE_STATIC_FIELDS:
        assert field in query
    assert "for(members('SPX Index', dates=range(-15y, 0d), frq='m'))" in query


def test_generate_excel_universe_formula() -> None:
    """Verify Excel =BQL formula generation for terminal exports."""
    formula = generate_excel_bql_formula("SXXP Index")
    assert formula.startswith("=BQL(\"members('SXXP Index', dates=range(-15y, 0d), frq='m')\",")
    assert "'ID_ISIN'" in formula
    assert "'GICS_SECTOR'" in formula


def test_generate_fundamentals_query() -> None:
    """Verify BQL fundamentals query with EUR conversion."""
    gen = BQLQueryGenerator()
    q = gen.generate_fundamentals_query("SPTSX60 Index")
    assert "PE_RATIO(currency='EUR')" in q
    assert "EV_TO_T12M_EBITDA(currency='EUR')" in q
    for metric in CORE_FUNDAMENTAL_FIELDS:
        assert f"{metric}(currency='EUR')" in q
    assert "for(members('SPTSX60 Index', dates=range(-15y, 0d), frq='m'))" in q


def test_generate_excel_fundamentals_formula() -> None:
    """Verify Excel formula for fundamentals."""
    gen = BQLQueryGenerator()
    formula = gen.generate_excel_fundamentals_formula("SPX Index")
    assert formula.startswith("=BQL(\"members('SPX Index', dates=range(-15y, 0d), frq='m')\",")
    assert "\"PE_RATIO(currency='EUR')\"" in formula


def test_generate_market_series_query() -> None:
    """Verify market series daily query in EUR."""
    gen = BQLQueryGenerator()
    q = gen.generate_market_series_query("members('SPX Index', dates=0d)")
    assert "TOT_RETURN_INDEX_GROSS_DVDS(dates=range('-15y', '0d', frq='d'), currency='EUR')" in q
    assert "PX_LAST(dates=range('-15y', '0d', frq='d'), currency='EUR')" in q
    assert "PX_VOLUME(dates=range('-15y', '0d', frq='d'))" in q
    assert "for(members('SPX Index', dates=0d))" in q
