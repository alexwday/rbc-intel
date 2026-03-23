"""Tests for ingestion.utils.ssl_certificates."""

from unittest.mock import MagicMock, patch

from ingestion.utils.ssl_certificates import setup_ssl


def test_setup_ssl_with_rbc_security():
    """Calls enable_certs when rbc_security is available."""
    mock_module = MagicMock()
    with patch(
        "ingestion.utils.ssl_certificates.importlib.import_module",
        return_value=mock_module,
    ):
        setup_ssl()
    mock_module.enable_certs.assert_called_once()


def test_setup_ssl_without_rbc_security():
    """Falls back silently when rbc_security is missing."""
    with patch(
        "ingestion.utils.ssl_certificates.importlib.import_module",
        side_effect=ImportError,
    ):
        setup_ssl()
