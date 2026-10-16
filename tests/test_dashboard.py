"""Unit tests for dashboard components and CLI invocation."""

from __future__ import annotations

from unittest.mock import patch

from quant_portfolio.cli import cmd_dashboard
from quant_portfolio.dashboard.app import get_system_components


def test_dashboard_components_initialization() -> None:
    """Verify that dashboard can initialize system components cleanly."""
    config, sqlite_store, parquet_store = get_system_components()
    assert config is not None
    assert sqlite_store is not None
    assert parquet_store is not None


def test_cmd_dashboard_invocation(monkeypatch: object) -> None:
    """Verify that cmd_dashboard builds proper subprocess arguments."""
    with patch("subprocess.run") as mock_run:
        cmd_dashboard(host="127.0.0.1", port=8502)
        assert mock_run.called
        args = mock_run.call_args[0][0]
        assert "streamlit" in args
        assert "run" in args
        assert "--server.address" in args
        assert "127.0.0.1" in args
        assert "--server.port" in args
        assert "8502" in args
