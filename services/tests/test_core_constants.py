"""Unit-Tests für services.core.constants.

Testet alle konsolidierten Konstanten und deren Verwendung.
"""

from services.core.constants import (
    A2A_INVALID_REPLY_ERROR,
    A2A_MESSAGE_VALIDATION_ERROR,
    A2A_REQUEST_TYPE,
    CONTENT_TYPE_FORM_URLENCODED,
    # Content Types
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_MULTIPART,
    DEFAULT_A2A_TIMEOUT_SECONDS,
    # A2A Service Konstanten
    DEFAULT_A2A_VERSION,
    DEFAULT_CONNECTION_TIMEOUT,
    # Circuit Breaker Konstanten
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_HALF_OPEN_MAX_CONCURRENT,
    # Heartbeat Service Konstanten
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_HEARTBEAT_TIMEOUT,
    DEFAULT_HTTP2_MAX_CONNECTIONS,
    DEFAULT_HTTP2_MAX_KEEPALIVE,
    DEFAULT_MAX_HEARTBEAT_FAILURES,
    DEFAULT_MAX_RETRIES,
    DEFAULT_OPEN_TIMEOUT_SECONDS,
    DEFAULT_READ_TIMEOUT,
    DEFAULT_RECOVERY_BACKOFF_BASE,
    DEFAULT_RECOVERY_BACKOFF_MAX_SECONDS,
    # HTTP Client Konstanten
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_RETRY_BACKOFF_FACTOR,
    DEFAULT_RETRY_MAX_DELAY,
    # Domain Revalidation Konstanten
    DEFAULT_REVALIDATION_INTERVAL_HOURS,
    # Environment Konstanten
    ENVIRONMENT_DEVELOPMENT,
    ENVIRONMENT_PRODUCTION,
    ENVIRONMENT_TESTING,
    FORCE_REVALIDATION_INTERVAL,
    HEALTH_CHECK_INTERVAL_SECONDS,
    HEALTH_CHECK_TIMEOUT_SECONDS,
    HTTP_STATUS_ACCEPTED,
    HTTP_STATUS_BAD_GATEWAY,
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_CONFLICT,
    HTTP_STATUS_CREATED,
    HTTP_STATUS_FORBIDDEN,
    HTTP_STATUS_GATEWAY_TIMEOUT,
    HTTP_STATUS_INTERNAL_SERVER_ERROR,
    HTTP_STATUS_NO_CONTENT,
    HTTP_STATUS_NOT_FOUND,
    # HTTP Status Code Konstanten
    HTTP_STATUS_OK,
    HTTP_STATUS_SERVICE_UNAVAILABLE,
    HTTP_STATUS_TOO_MANY_REQUESTS,
    HTTP_STATUS_UNAUTHORIZED,
    HTTP_STATUS_UNPROCESSABLE_ENTITY,
    KEI_GRPC_CIRCUIT_BREAKER_CONFIG,
    KEI_MCP_CLIENT_USER_AGENT,
    KEI_RPC_CIRCUIT_BREAKER_CONFIG,
    KEI_RPC_CLIENT_USER_AGENT,
    # User Agent Strings
    KEIKO_WEBHOOK_USER_AGENT,
    MAX_HEALTH_CHECK_FAILURES,
    MCP_CLIENT_CIRCUIT_BREAKER_CONFIG,
    # Service Status Konstanten
    SERVICE_STATUS_AVAILABLE,
    SERVICE_STATUS_ERROR,
    SERVICE_STATUS_INITIALIZING,
    SERVICE_STATUS_SHUTTING_DOWN,
    SERVICE_STATUS_UNAVAILABLE,
    SERVICE_STATUS_UNREACHABLE,
)


class TestCircuitBreakerConstants:
    """Tests für Circuit Breaker Konstanten."""

    def test_default_circuit_breaker_values(self):
        """Testet Standard Circuit Breaker Werte."""
        assert DEFAULT_FAILURE_THRESHOLD == 5
        assert DEFAULT_OPEN_TIMEOUT_SECONDS == 10.0
        assert DEFAULT_HALF_OPEN_MAX_CONCURRENT == 1
        assert DEFAULT_RECOVERY_BACKOFF_BASE == 1.5
        assert DEFAULT_RECOVERY_BACKOFF_MAX_SECONDS == 30.0

    def test_kei_rpc_circuit_breaker_config(self):
        """Testet KEI RPC Circuit Breaker Konfiguration."""
        assert KEI_RPC_CIRCUIT_BREAKER_CONFIG["failure_threshold"] == DEFAULT_FAILURE_THRESHOLD
        assert KEI_RPC_CIRCUIT_BREAKER_CONFIG["open_timeout_seconds"] == DEFAULT_OPEN_TIMEOUT_SECONDS

    def test_kei_grpc_circuit_breaker_config(self):
        """Testet KEI gRPC Circuit Breaker Konfiguration."""
        assert KEI_GRPC_CIRCUIT_BREAKER_CONFIG["failure_threshold"] == DEFAULT_FAILURE_THRESHOLD
        assert KEI_GRPC_CIRCUIT_BREAKER_CONFIG["open_timeout_seconds"] == DEFAULT_OPEN_TIMEOUT_SECONDS

    def test_mcp_client_circuit_breaker_config(self):
        """Testet MCP Client Circuit Breaker Konfiguration."""
        assert MCP_CLIENT_CIRCUIT_BREAKER_CONFIG["failure_threshold"] == DEFAULT_FAILURE_THRESHOLD
        assert MCP_CLIENT_CIRCUIT_BREAKER_CONFIG["open_timeout_seconds"] == 60.0
        assert MCP_CLIENT_CIRCUIT_BREAKER_CONFIG["half_open_max_concurrent"] == 3


class TestHTTPClientConstants:
    """Tests für HTTP Client Konstanten."""

    def test_timeout_values(self):
        """Testet Timeout-Werte."""
        assert DEFAULT_REQUEST_TIMEOUT == 30.0
        assert DEFAULT_CONNECTION_TIMEOUT == 10.0
        assert DEFAULT_READ_TIMEOUT == 30.0

    def test_http2_configuration(self):
        """Testet HTTP/2 Konfiguration."""
        assert DEFAULT_HTTP2_MAX_CONNECTIONS == 100
        assert DEFAULT_HTTP2_MAX_KEEPALIVE == 20

    def test_retry_configuration(self):
        """Testet Retry-Konfiguration."""
        assert DEFAULT_MAX_RETRIES == 3
        assert DEFAULT_RETRY_BACKOFF_FACTOR == 2.0
        assert DEFAULT_RETRY_MAX_DELAY == 60.0


class TestServiceStatusConstants:
    """Tests für Service Status Konstanten."""

    def test_service_status_strings(self):
        """Testet Service Status Strings."""
        assert SERVICE_STATUS_AVAILABLE == "available"
        assert SERVICE_STATUS_UNAVAILABLE == "unavailable"
        assert SERVICE_STATUS_UNREACHABLE == "unreachable"
        assert SERVICE_STATUS_ERROR == "error"
        assert SERVICE_STATUS_INITIALIZING == "initializing"
        assert SERVICE_STATUS_SHUTTING_DOWN == "shutting_down"

    def test_health_check_constants(self):
        """Testet Health Check Konstanten."""
        assert HEALTH_CHECK_INTERVAL_SECONDS == 30.0
        assert HEALTH_CHECK_TIMEOUT_SECONDS == 5.0
        assert MAX_HEALTH_CHECK_FAILURES == 3


class TestHTTPStatusCodeConstants:
    """Tests für HTTP Status Code Konstanten."""

    def test_success_status_codes(self):
        """Testet erfolgreiche Status Codes."""
        assert HTTP_STATUS_OK == 200
        assert HTTP_STATUS_CREATED == 201
        assert HTTP_STATUS_ACCEPTED == 202
        assert HTTP_STATUS_NO_CONTENT == 204

    def test_client_error_status_codes(self):
        """Testet Client Error Status Codes."""
        assert HTTP_STATUS_BAD_REQUEST == 400
        assert HTTP_STATUS_UNAUTHORIZED == 401
        assert HTTP_STATUS_FORBIDDEN == 403
        assert HTTP_STATUS_NOT_FOUND == 404
        assert HTTP_STATUS_CONFLICT == 409
        assert HTTP_STATUS_UNPROCESSABLE_ENTITY == 422
        assert HTTP_STATUS_TOO_MANY_REQUESTS == 429

    def test_server_error_status_codes(self):
        """Testet Server Error Status Codes."""
        assert HTTP_STATUS_INTERNAL_SERVER_ERROR == 500
        assert HTTP_STATUS_BAD_GATEWAY == 502
        assert HTTP_STATUS_SERVICE_UNAVAILABLE == 503
        assert HTTP_STATUS_GATEWAY_TIMEOUT == 504


class TestHeartbeatServiceConstants:
    """Tests für Heartbeat Service Konstanten."""

    def test_heartbeat_values(self):
        """Testet Heartbeat-Werte."""
        assert DEFAULT_HEARTBEAT_INTERVAL == 30.0
        assert DEFAULT_HEARTBEAT_TIMEOUT == 5.0
        assert DEFAULT_MAX_HEARTBEAT_FAILURES == 3


class TestDomainRevalidationConstants:
    """Tests für Domain Revalidation Konstanten."""

    def test_revalidation_values(self):
        """Testet Revalidation-Werte."""
        assert DEFAULT_REVALIDATION_INTERVAL_HOURS == 24
        assert FORCE_REVALIDATION_INTERVAL == 0


class TestA2AServiceConstants:
    """Tests für A2A Service Konstanten."""

    def test_a2a_values(self):
        """Testet A2A-Werte."""
        assert DEFAULT_A2A_VERSION == 1
        assert DEFAULT_A2A_TIMEOUT_SECONDS == 10.0
        assert A2A_REQUEST_TYPE == "a2a_request"
        assert A2A_INVALID_REPLY_ERROR == "invalid_reply"
        assert A2A_MESSAGE_VALIDATION_ERROR == "A2A Nachricht ungültig"


class TestUserAgentConstants:
    """Tests für User Agent Konstanten."""

    def test_user_agent_strings(self):
        """Testet User Agent Strings."""
        assert KEIKO_WEBHOOK_USER_AGENT == "Keiko-Webhook/2"
        assert KEI_MCP_CLIENT_USER_AGENT == "KEI-MCP-Client/1.0"
        assert KEI_RPC_CLIENT_USER_AGENT == "KEI-RPC-Client/1.0"


class TestContentTypeConstants:
    """Tests für Content Type Konstanten."""

    def test_content_types(self):
        """Testet Content Types."""
        assert CONTENT_TYPE_JSON == "application/json"
        assert CONTENT_TYPE_FORM_URLENCODED == "application/x-www-form-urlencoded"
        assert CONTENT_TYPE_MULTIPART == "multipart/form-data"


class TestEnvironmentConstants:
    """Tests für Environment Konstanten."""

    def test_environment_values(self):
        """Testet Environment-Werte."""
        assert ENVIRONMENT_DEVELOPMENT == "development"
        assert ENVIRONMENT_PRODUCTION == "production"
        assert ENVIRONMENT_TESTING == "testing"


class TestConstantsIntegrity:
    """Tests für Konstanten-Integrität."""

    def test_no_duplicate_values(self):
        """Testet, dass keine duplizierten Werte existieren."""
        # Teste HTTP Status Codes auf Eindeutigkeit
        status_codes = [
            HTTP_STATUS_OK, HTTP_STATUS_CREATED, HTTP_STATUS_ACCEPTED,
            HTTP_STATUS_BAD_REQUEST, HTTP_STATUS_UNAUTHORIZED, HTTP_STATUS_FORBIDDEN,
            HTTP_STATUS_INTERNAL_SERVER_ERROR, HTTP_STATUS_BAD_GATEWAY
        ]
        assert len(status_codes) == len(set(status_codes))

    def test_string_constants_not_empty(self):
        """Testet, dass String-Konstanten nicht leer sind."""
        string_constants = [
            SERVICE_STATUS_AVAILABLE, A2A_REQUEST_TYPE, KEIKO_WEBHOOK_USER_AGENT,
            CONTENT_TYPE_JSON, ENVIRONMENT_DEVELOPMENT
        ]
        for constant in string_constants:
            assert constant
            assert len(constant.strip()) > 0

    def test_numeric_constants_positive(self):
        """Testet, dass numerische Konstanten positive Werte haben."""
        numeric_constants = [
            DEFAULT_FAILURE_THRESHOLD, DEFAULT_REQUEST_TIMEOUT,
            DEFAULT_MAX_RETRIES, HEALTH_CHECK_INTERVAL_SECONDS,
            DEFAULT_HEARTBEAT_INTERVAL, DEFAULT_A2A_VERSION
        ]
        for constant in numeric_constants:
            assert constant > 0
