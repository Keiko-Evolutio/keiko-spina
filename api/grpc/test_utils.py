"""Unit Tests für KEI-RPC Utilities.

Testet konsolidierte Utility-Funktionen für Metadata-Extraktion,
Validierung und Error-Handling.
"""

import time
from unittest.mock import MagicMock

import grpc
import pytest

from .constants import ErrorCodes
from .utils import (
    create_grpc_error,
    create_timing_info,
    extract_bearer_token,
    extract_correlation_id,
    extract_metadata_value,
    extract_peer_info,
    extract_peer_ip,
    extract_tenant_id,
    handle_common_errors,
    log_operation_end,
    log_operation_start,
    validate_idempotency_key,
    validate_w3c_traceparent,
)


class TestMetadataUtilities:
    """Tests für Metadata-Utility-Funktionen."""

    def test_extract_metadata_value_success(self):
        """Testet erfolgreiche Metadata-Extraktion."""
        # Mock gRPC Context
        mock_context = MagicMock()
        mock_context.invocation_metadata.return_value = [
            ("content-type", "application/grpc"),
            ("authorization", "Bearer test-token"),
            ("x-tenant-id", "tenant-123"),
        ]

        # Test case-insensitive Extraktion
        auth_value = extract_metadata_value(mock_context, "Authorization")
        assert auth_value == "Bearer test-token"

        tenant_value = extract_metadata_value(mock_context, "x-tenant-id")
        assert tenant_value == "tenant-123"

    def test_extract_metadata_value_not_found(self):
        """Testet Metadata-Extraktion wenn Key nicht gefunden."""
        mock_context = MagicMock()
        mock_context.invocation_metadata.return_value = [
            ("content-type", "application/grpc"),
        ]

        # Test mit Default-Wert
        value = extract_metadata_value(mock_context, "missing-key", "default")
        assert value == "default"

        # Test ohne Default-Wert
        value = extract_metadata_value(mock_context, "missing-key")
        assert value is None

    def test_extract_bearer_token_success(self):
        """Testet erfolgreiche Bearer Token Extraktion."""
        mock_context = MagicMock()
        mock_context.invocation_metadata.return_value = [
            ("authorization", "Bearer abc123def456"),
        ]

        token = extract_bearer_token(mock_context)
        assert token == "abc123def456"

    def test_extract_bearer_token_missing(self):
        """Testet fehlenden Bearer Token."""
        mock_context = MagicMock()
        mock_context.invocation_metadata.return_value = []

        token = extract_bearer_token(mock_context)
        assert token is None

    def test_extract_bearer_token_invalid_format(self):
        """Testet ungültiges Bearer Token Format."""
        mock_context = MagicMock()
        mock_context.invocation_metadata.return_value = [
            ("authorization", "Basic dXNlcjpwYXNz"),  # Nicht Bearer
        ]

        token = extract_bearer_token(mock_context)
        assert token is None

    def test_extract_tenant_id_success(self):
        """Testet erfolgreiche Tenant-ID Extraktion."""
        mock_context = MagicMock()
        mock_context.invocation_metadata.return_value = [
            ("x-tenant-id", "tenant-456"),
        ]

        tenant_id = extract_tenant_id(mock_context, required=False)
        assert tenant_id == "tenant-456"

    def test_extract_tenant_id_default(self):
        """Testet Default Tenant-ID."""
        mock_context = MagicMock()
        mock_context.invocation_metadata.return_value = []

        tenant_id = extract_tenant_id(mock_context, required=False)
        assert tenant_id == "default"

    def test_extract_tenant_id_required_missing(self):
        """Testet erforderliche aber fehlende Tenant-ID."""
        mock_context = MagicMock()
        mock_context.invocation_metadata.return_value = []
        mock_context.abort = MagicMock()

        extract_tenant_id(mock_context, required=True)
        mock_context.abort.assert_called_once()

    def test_extract_correlation_id_existing(self):
        """Testet Extraktion existierender Correlation-ID."""
        mock_context = MagicMock()
        mock_context.invocation_metadata.return_value = [
            ("x-correlation-id", "corr-123-456"),
        ]

        correlation_id = extract_correlation_id(mock_context)
        assert correlation_id == "corr-123-456"

    def test_extract_correlation_id_generated(self):
        """Testet Generierung neuer Correlation-ID."""
        mock_context = MagicMock()
        mock_context.invocation_metadata.return_value = []

        correlation_id = extract_correlation_id(mock_context)
        assert correlation_id is not None
        assert len(correlation_id) > 0


class TestPeerUtilities:
    """Tests für Peer-Utility-Funktionen."""

    def test_extract_peer_ip_ipv4(self):
        """Testet IPv4 Peer-IP Extraktion."""
        mock_context = MagicMock()
        mock_context.peer.return_value = "ipv4:192.168.1.100:12345"

        ip = extract_peer_ip(mock_context)
        assert ip == "192.168.1.100"

    def test_extract_peer_ip_ipv6(self):
        """Testet IPv6 Peer-IP Extraktion."""
        mock_context = MagicMock()
        mock_context.peer.return_value = "ipv6:[2001:db8::1]:12345"

        ip = extract_peer_ip(mock_context)
        assert ip == "2001:db8::1"

    def test_extract_peer_ip_unknown(self):
        """Testet unbekanntes Peer-Format."""
        mock_context = MagicMock()
        mock_context.peer.return_value = None

        ip = extract_peer_ip(mock_context)
        assert ip == "unknown"

    def test_extract_peer_info(self):
        """Testet vollständige Peer-Info Extraktion."""
        mock_context = MagicMock()
        mock_context.peer.return_value = "ipv4:127.0.0.1:12345"
        mock_context.invocation_metadata.return_value = [
            ("user-agent", "grpc-python/1.0"),
            ("x-correlation-id", "test-corr-id"),
        ]

        peer_info = extract_peer_info(mock_context)

        assert peer_info["peer"] == "ipv4:127.0.0.1:12345"
        assert peer_info["ip"] == "127.0.0.1"
        assert peer_info["user_agent"] == "grpc-python/1.0"
        assert peer_info["correlation_id"] == "test-corr-id"


class TestValidationUtilities:
    """Tests für Validierungs-Utility-Funktionen."""

    def test_validate_w3c_traceparent_valid(self):
        """Testet gültiges W3C Traceparent."""
        valid_traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"

        assert validate_w3c_traceparent(valid_traceparent)

    def test_validate_w3c_traceparent_invalid_format(self):
        """Testet ungültiges W3C Traceparent Format."""
        # Zu wenige Teile
        assert (
            not validate_w3c_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7")
        )

        # Ungültige Version
        assert (
            not validate_w3c_traceparent("zz-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
        )

        # Trace-ID alle Nullen
        assert (
            not validate_w3c_traceparent("00-00000000000000000000000000000000-00f067aa0ba902b7-01")
        )

        # Parent-ID alle Nullen
        assert (
            not validate_w3c_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01")
        )

    def test_validate_w3c_traceparent_empty(self):
        """Testet leeres Traceparent."""
        assert not validate_w3c_traceparent("")
        assert not validate_w3c_traceparent("")  # Verwende leeren String statt None für Type-Safety

    def test_validate_idempotency_key_valid(self):
        """Testet gültigen Idempotency-Key."""
        assert validate_idempotency_key("abc123")
        assert validate_idempotency_key("test-key-123")
        assert validate_idempotency_key("test_key_456")

    def test_validate_idempotency_key_invalid(self):
        """Testet ungültigen Idempotency-Key."""
        # Leer
        assert not validate_idempotency_key("")

        # Zu lang
        assert not validate_idempotency_key("a" * 256)

        # Ungültige Zeichen
        assert not validate_idempotency_key("test@key")
        assert not validate_idempotency_key("test key")


class TestErrorHandlingUtilities:
    """Tests für Error-Handling-Utility-Funktionen."""

    def test_create_grpc_error(self):
        """Testet gRPC Error-Erstellung."""
        mock_context = MagicMock()
        mock_context.set_trailing_metadata = MagicMock()
        mock_context.abort = MagicMock()

        create_grpc_error(
            mock_context,
            grpc.StatusCode.INVALID_ARGUMENT,
            "Test Error",
            ErrorCodes.VALIDATION_ERROR,
            "corr-123",
        )

        # Prüfe Metadata-Aufruf
        mock_context.set_trailing_metadata.assert_called_once()

        # Prüfe Abort-Aufruf
        mock_context.abort.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT, "Test Error")

    def test_handle_common_errors_validation_error(self):
        """Testet Behandlung von Validation-Errors."""
        mock_context = MagicMock()
        mock_context.set_trailing_metadata = MagicMock()
        mock_context.abort = MagicMock()

        error = ValueError("Invalid input")

        handle_common_errors(mock_context, error, "corr-123")

        # Sollte als INVALID_ARGUMENT behandelt werden
        mock_context.abort.assert_called_once()
        args = mock_context.abort.call_args[0]
        assert args[0] == grpc.StatusCode.INVALID_ARGUMENT

    def test_handle_common_errors_timeout_error(self):
        """Testet Behandlung von Timeout-Errors."""
        mock_context = MagicMock()
        mock_context.set_trailing_metadata = MagicMock()
        mock_context.abort = MagicMock()

        error = TimeoutError("Operation timed out")

        handle_common_errors(mock_context, error, "corr-123")

        # Sollte als DEADLINE_EXCEEDED behandelt werden
        mock_context.abort.assert_called_once()
        args = mock_context.abort.call_args[0]
        assert args[0] == grpc.StatusCode.DEADLINE_EXCEEDED


class TestTimingUtilities:
    """Tests für Timing-Utility-Funktionen."""

    def test_create_timing_info(self):
        """Testet Timing-Info Erstellung."""
        start_time = time.time()
        end_time = start_time + 1.5  # 1.5 Sekunden später

        timing_info = create_timing_info(start_time, end_time)

        assert timing_info["start_time"] == start_time
        assert timing_info["end_time"] == end_time
        assert timing_info["duration_seconds"] == 1.5
        assert timing_info["duration_ms"] == 1500.0

    def test_create_timing_info_auto_end_time(self):
        """Testet automatische End-Zeit."""
        start_time = time.time() - 1.0  # 1 Sekunde in der Vergangenheit

        timing_info = create_timing_info(start_time)

        assert timing_info["start_time"] == start_time
        assert timing_info["end_time"] > start_time
        assert timing_info["duration_seconds"] > 0
        assert timing_info["duration_ms"] > 0


class TestLoggingUtilities:
    """Tests für Logging-Utility-Funktionen."""

    def test_log_operation_start(self):
        """Testet Operation-Start Logging."""
        peer_info = {"ip": "127.0.0.1", "user_agent": "test-agent"}

        # Test sollte ohne Fehler durchlaufen
        log_operation_start("plan", "corr-123", peer_info, additional_data="test")

    def test_log_operation_end_success(self):
        """Testet Operation-End Logging (Erfolg)."""
        timing_info = {"duration_ms": 150.5}

        # Test sollte ohne Fehler durchlaufen
        log_operation_end("plan", "corr-123", timing_info, success=True)

    def test_log_operation_end_failure(self):
        """Testet Operation-End Logging (Fehler)."""
        timing_info = {"duration_ms": 75.2}

        # Test sollte ohne Fehler durchlaufen
        log_operation_end(
            "plan", "corr-123", timing_info, success=False, error_code="AGENT_NOT_FOUND"
        )


# Pytest-Konfiguration
pytest_plugins = []

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
