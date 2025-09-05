# backend/services/clients/tests/test_common_error_handling.py
"""Tests für Common Error Handling.

Deutsche Docstrings, englische Identifiers, isolierte Tests mit Mocking.
"""

from unittest.mock import Mock, patch

import pytest

from services.clients.common.error_handling import (
    ClientServiceException,
    ContentPolicyViolationException,
    ErrorHandler,
    ServiceError,
    ServiceNotConfiguredException,
    ServiceUnavailableException,
    create_fallback_result,
    handle_http_error,
    is_content_policy_violation,
    log_service_error,
)


class TestServiceError:
    """Tests für ServiceError Dataclass."""

    def test_service_error_creation(self) -> None:
        """Prüft, dass ServiceError korrekt erstellt wird."""
        error = ServiceError(
            error_type="test_error",
            message="Test message",
            details={"key": "value"},
            recoverable=True,
            retry_after=30.0
        )

        assert error.error_type == "test_error"
        assert error.message == "Test message"
        assert error.details == {"key": "value"}
        assert error.recoverable is True
        assert error.retry_after == 30.0

    def test_service_error_defaults(self) -> None:
        """Prüft, dass ServiceError korrekte Standardwerte hat."""
        error = ServiceError(
            error_type="test_error",
            message="Test message"
        )

        assert error.details is None
        assert error.recoverable is False
        assert error.retry_after is None

    def test_service_error_slots(self) -> None:
        """Prüft, dass ServiceError __slots__ verwendet."""
        error = ServiceError("test", "message")

        # __slots__ verhindert dynamische Attribute
        with pytest.raises(AttributeError):
            error.dynamic_attribute = "test"  # type: ignore


class TestClientServiceException:
    """Tests für ClientServiceException."""

    def test_client_service_exception_creation(self) -> None:
        """Prüft, dass ClientServiceException korrekt erstellt wird."""
        exception = ClientServiceException(
            message="Test error",
            error_type="custom_error",
            details={"context": "test"},
            recoverable=True
        )

        assert str(exception) == "Test error"
        assert exception.error_type == "custom_error"
        assert exception.details == {"context": "test"}
        assert exception.recoverable is True

    def test_client_service_exception_defaults(self) -> None:
        """Prüft, dass ClientServiceException korrekte Standardwerte hat."""
        exception = ClientServiceException("Test error")

        assert exception.error_type == "client_error"
        assert exception.details == {}
        assert exception.recoverable is False

    def test_client_service_exception_inheritance(self) -> None:
        """Prüft, dass ClientServiceException von Exception erbt."""
        exception = ClientServiceException("Test")
        assert isinstance(exception, Exception)


class TestServiceNotConfiguredException:
    """Tests für ServiceNotConfiguredException."""

    def test_service_not_configured_exception_basic(self) -> None:
        """Prüft, dass ServiceNotConfiguredException korrekt erstellt wird."""
        exception = ServiceNotConfiguredException("TestService")

        assert "TestService" in str(exception)
        assert "nicht konfiguriert" in str(exception)
        assert exception.error_type == "configuration_error"
        assert exception.recoverable is False
        assert exception.details["service"] == "TestService"

    def test_service_not_configured_exception_with_missing_config(self) -> None:
        """Prüft ServiceNotConfiguredException mit fehlender Konfiguration."""
        exception = ServiceNotConfiguredException("TestService", "api_key")

        assert "TestService" in str(exception)
        assert "api_key" in str(exception)
        assert exception.details["missing_config"] == "api_key"


class TestServiceUnavailableException:
    """Tests für ServiceUnavailableException."""

    def test_service_unavailable_exception_basic(self) -> None:
        """Prüft, dass ServiceUnavailableException korrekt erstellt wird."""
        exception = ServiceUnavailableException("TestService")

        assert "TestService" in str(exception)
        assert "nicht verfügbar" in str(exception)
        assert exception.error_type == "availability_error"
        assert exception.recoverable is True
        assert exception.details["service"] == "TestService"

    def test_service_unavailable_exception_with_reason_and_retry(self) -> None:
        """Prüft ServiceUnavailableException mit Grund und Retry-Zeit."""
        exception = ServiceUnavailableException(
            "TestService",
            reason="Network timeout",
            retry_after=60.0
        )

        assert "Network timeout" in str(exception)
        assert exception.details["reason"] == "Network timeout"
        assert exception.retry_after == 60.0


class TestContentPolicyViolationException:
    """Tests für ContentPolicyViolationException."""

    def test_content_policy_violation_exception_basic(self) -> None:
        """Prüft, dass ContentPolicyViolationException korrekt erstellt wird."""
        exception = ContentPolicyViolationException("Unsafe prompt")

        assert "Content Policy Violation" in str(exception)
        assert exception.error_type == "content_policy_violation"
        assert exception.recoverable is True
        assert exception.details["original_prompt"] == "Unsafe prompt"

    def test_content_policy_violation_exception_with_sanitized_prompt(self) -> None:
        """Prüft ContentPolicyViolationException mit sanitisiertem Prompt."""
        exception = ContentPolicyViolationException(
            original_prompt="Unsafe prompt",
            sanitized_prompt="Safe prompt",
            suggestions=["Alternative 1", "Alternative 2"]
        )

        assert exception.details["sanitized_prompt"] == "Safe prompt"
        assert exception.details["suggestions"] == ["Alternative 1", "Alternative 2"]


class TestHandleHttpError:
    """Tests für handle_http_error Funktion."""

    def test_handle_http_error_400_bad_request(self) -> None:
        """Prüft Behandlung von HTTP 400 Bad Request."""
        error = handle_http_error(400, service_name="TestService")

        assert error.error_type == "bad_request"
        assert "Ungültige Anfrage" in error.message
        assert error.recoverable is False
        assert error.details["status_code"] == 400
        assert error.details["service"] == "TestService"

    def test_handle_http_error_401_unauthorized(self) -> None:
        """Prüft Behandlung von HTTP 401 Unauthorized."""
        error = handle_http_error(401)

        assert error.error_type == "authentication_error"
        assert "Authentifizierung fehlgeschlagen" in error.message
        assert error.recoverable is False

    def test_handle_http_error_403_forbidden(self) -> None:
        """Prüft Behandlung von HTTP 403 Forbidden."""
        error = handle_http_error(403)

        assert error.error_type == "authorization_error"
        assert "Zugriff verweigert" in error.message
        assert error.recoverable is False

    def test_handle_http_error_404_not_found(self) -> None:
        """Prüft Behandlung von HTTP 404 Not Found."""
        error = handle_http_error(404)

        assert error.error_type == "not_found"
        assert "Ressource nicht gefunden" in error.message
        assert error.recoverable is False

    def test_handle_http_error_429_rate_limit(self) -> None:
        """Prüft Behandlung von HTTP 429 Rate Limit."""
        response_data = {"retry_after": "120"}
        error = handle_http_error(429, response_data)

        assert error.error_type == "rate_limit_exceeded"
        assert "Rate Limit überschritten" in error.message
        assert error.recoverable is True
        assert error.retry_after == 120.0

    def test_handle_http_error_429_default_retry_after(self) -> None:
        """Prüft Behandlung von HTTP 429 ohne Retry-After Header."""
        error = handle_http_error(429)

        assert error.retry_after == 60.0  # Default-Wert

    def test_handle_http_error_500_server_error(self) -> None:
        """Prüft Behandlung von HTTP 500 Server Error."""
        error = handle_http_error(500)

        assert error.error_type == "server_error"
        assert "Server-Fehler" in error.message
        assert error.recoverable is True
        assert error.retry_after == 30.0

    def test_handle_http_error_with_response_data(self) -> None:
        """Prüft Behandlung mit Response-Daten."""
        response_data = {"error": "Custom error message"}
        error = handle_http_error(400, response_data)

        assert "Custom error message" in error.message
        assert error.details["response_data"] == response_data

    @patch("services.clients.common.error_handling.logger")
    def test_handle_http_error_logging(self, mock_logger: Mock) -> None:
        """Prüft, dass HTTP-Fehler geloggt werden."""
        handle_http_error(500, service_name="TestService")

        mock_logger.warning.assert_called_once()
        log_data = mock_logger.warning.call_args[0][0]
        assert log_data["event"] == "http_error_handled"
        assert log_data["service"] == "TestService"
        assert log_data["status"] == 500


class TestIsContentPolicyViolation:
    """Tests für is_content_policy_violation Funktion."""

    def test_detects_content_policy_violation(self) -> None:
        """Prüft, dass Content Policy Violations erkannt werden."""
        test_cases = [
            "content_policy_violation detected",
            "Safety system blocked the request",
            "Content filter triggered",
            "Policy violation occurred",
            "CONTENT_POLICY_VIOLATION",  # Case insensitive
        ]

        for error_message in test_cases:
            assert is_content_policy_violation(error_message) is True

    def test_does_not_detect_normal_errors(self) -> None:
        """Prüft, dass normale Fehler nicht als Content Policy Violations erkannt werden."""
        test_cases = [
            "Network timeout",
            "Invalid API key",
            "Rate limit exceeded",
            "Server error",
            "Connection refused",
        ]

        for error_message in test_cases:
            assert is_content_policy_violation(error_message) is False

    def test_case_insensitive_detection(self) -> None:
        """Prüft, dass Erkennung case-insensitive ist."""
        assert is_content_policy_violation("CONTENT_POLICY_VIOLATION") is True
        assert is_content_policy_violation("Content_Policy_Violation") is True
        assert is_content_policy_violation("content_policy_violation") is True


class TestLogServiceError:
    """Tests für log_service_error Funktion."""

    @patch("services.clients.common.error_handling.logger")
    def test_log_service_error_basic(self, mock_logger: Mock) -> None:
        """Prüft, dass Service-Fehler geloggt werden."""
        error = RuntimeError("Test error")

        log_service_error(error, "TestService", "test_operation")

        mock_logger.error.assert_called_once()
        log_data = mock_logger.error.call_args[0][0]

        assert log_data["event"] == "service_error"
        assert log_data["service"] == "TestService"
        assert log_data["operation"] == "test_operation"
        assert log_data["error_type"] == "RuntimeError"
        assert log_data["error_message"] == "Test error"

    @patch("services.clients.common.error_handling.logger")
    def test_log_service_error_with_client_service_exception(self, mock_logger: Mock) -> None:
        """Prüft Logging von ClientServiceException."""
        error = ClientServiceException(
            "Test error",
            error_type="custom_error",
            details={"key": "value"},
            recoverable=True
        )

        log_service_error(error, "TestService", "test_operation")

        log_data = mock_logger.error.call_args[0][0]
        assert log_data["client_error_type"] == "custom_error"
        assert log_data["recoverable"] is True
        assert log_data["details"] == {"key": "value"}

    @patch("services.clients.common.error_handling.logger")
    def test_log_service_error_with_context(self, mock_logger: Mock) -> None:
        """Prüft Logging mit zusätzlichem Kontext."""
        error = ValueError("Test error")
        context = {"request_id": "123", "user_id": "user456"}

        log_service_error(error, "TestService", "test_operation", context)

        log_data = mock_logger.error.call_args[0][0]
        assert log_data["context"] == context


class TestCreateFallbackResult:
    """Tests für create_fallback_result Funktion."""

    @patch("services.clients.common.error_handling.logger")
    def test_create_fallback_result_basic(self, mock_logger: Mock) -> None:
        """Prüft, dass Fallback-Ergebnisse korrekt erstellt werden."""
        result = create_fallback_result(
            "TestService",
            "test_operation",
            "Service unavailable"
        )

        assert result["success"] is False
        assert result["fallback"] is True
        assert result["service"] == "TestService"
        assert result["operation"] == "test_operation"
        assert result["reason"] == "Service unavailable"
        assert "timestamp" in result

        # Prüfe Logging
        mock_logger.info.assert_called_once()
        log_data = mock_logger.info.call_args[0][0]
        assert log_data["event"] == "fallback_result_created"

    @patch("services.clients.common.error_handling.logger")
    def test_create_fallback_result_with_default_value(self, _mock_logger: Mock) -> None:
        """Prüft Fallback-Ergebnis mit Default-Wert."""
        default_value = {"default": "data"}

        result = create_fallback_result(
            "TestService",
            "test_operation",
            "Service unavailable",
            default_value
        )

        assert result["default_value"] == default_value

    def test_create_fallback_result_timestamp_format(self) -> None:
        """Prüft, dass Timestamp im korrekten ISO-Format ist."""
        result = create_fallback_result("TestService", "test_op", "reason")

        timestamp = result["timestamp"]
        assert isinstance(timestamp, str)
        # ISO-Format sollte 'T' enthalten
        assert "T" in timestamp


class TestErrorHandler:
    """Tests für ErrorHandler Klasse."""

    def test_error_handler_initialization(self) -> None:
        """Prüft, dass ErrorHandler korrekt initialisiert wird."""
        handler = ErrorHandler("TestService")
        assert handler.service_name == "TestService"

    @patch("services.clients.common.error_handling.log_service_error")
    def test_error_handler_handle_exception_basic(self, mock_log: Mock) -> None:
        """Prüft, dass ErrorHandler Exceptions korrekt behandelt."""
        handler = ErrorHandler("TestService")
        error = RuntimeError("Test error")

        service_error = handler.handle_exception(error, "test_operation")

        assert isinstance(service_error, ServiceError)
        assert service_error.error_type == "unexpected_error"
        assert service_error.message == "Test error"
        assert service_error.recoverable is False

        # Prüfe, dass Logging aufgerufen wurde
        mock_log.assert_called_once_with(error, "TestService", "test_operation", None)

    @patch("services.clients.common.error_handling.log_service_error")
    def test_error_handler_handle_client_service_exception(self, _mock_log: Mock) -> None:
        """Prüft Behandlung von ClientServiceException."""
        handler = ErrorHandler("TestService")
        error = ClientServiceException(
            "Test error",
            error_type="custom_error",
            details={"key": "value"},
            recoverable=True
        )

        service_error = handler.handle_exception(error, "test_operation")

        assert service_error.error_type == "custom_error"
        assert service_error.message == "Test error"
        assert service_error.details == {"key": "value"}
        assert service_error.recoverable is True

    @patch("services.clients.common.error_handling.log_service_error")
    def test_error_handler_handle_exception_with_context(self, mock_log: Mock) -> None:
        """Prüft Exception-Behandlung mit Kontext."""
        handler = ErrorHandler("TestService")
        error = ValueError("Test error")
        context = {"request_id": "123"}

        handler.handle_exception(error, "test_operation", context)

        mock_log.assert_called_once_with(error, "TestService", "test_operation", context)
