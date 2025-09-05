"""Utility-Funktionen für KEI-RPC Modul.

Konsolidiert wiederkehrende Patterns und eliminiert Code-Duplikation
durch gemeinsame Helper-Funktionen.
"""

from __future__ import annotations

import re
import time
import uuid

import grpc

from kei_logging import get_logger

from .constants import ErrorCodes, MetadataKeys

logger = get_logger(__name__)


# ============================================================================
# METADATA UTILITIES
# ============================================================================


def extract_metadata_value(
    context: grpc.ServicerContext, key: str, default: str | None = None
) -> str | None:
    """Extrahiert Wert aus gRPC Metadata.

    Args:
        context: gRPC Service Context
        key: Metadata-Key (case-insensitive)
        default: Default-Wert falls Key nicht gefunden

    Returns:
        Metadata-Wert oder Default
    """
    try:
        metadata = context.invocation_metadata() or []
        key_lower = key.lower()

        for metadata_key, value in metadata:
            if metadata_key.lower() == key_lower:
                return value

        return default

    except Exception as e:
        logger.warning(f"Fehler beim Extrahieren von Metadata '{key}': {e}")
        return default


def extract_bearer_token(context: grpc.ServicerContext) -> str | None:
    """Extrahiert Bearer Token aus Authorization Header.

    Args:
        context: gRPC Service Context

    Returns:
        Token-String oder None
    """
    auth_header = extract_metadata_value(context, MetadataKeys.AUTHORIZATION)

    if auth_header and auth_header.startswith(MetadataKeys.BEARER_PREFIX):
        return auth_header[len(MetadataKeys.BEARER_PREFIX) :].strip()

    return None


def extract_tenant_id(context: grpc.ServicerContext, required: bool = True) -> str | None:
    """Extrahiert Tenant-ID aus Metadata.

    Args:
        context: gRPC Service Context
        required: Ob Tenant-ID erforderlich ist

    Returns:
        Tenant-ID oder None

    Raises:
        grpc.RpcError: Wenn Tenant-ID erforderlich aber nicht vorhanden
    """
    tenant_id = extract_metadata_value(context, MetadataKeys.TENANT_ID, "default")

    if required and not tenant_id:
        context.abort(
            grpc.StatusCode.INVALID_ARGUMENT, "Tenant-ID erforderlich aber nicht vorhanden"
        )

    return tenant_id


def extract_correlation_id(context: grpc.ServicerContext) -> str:
    """Extrahiert oder generiert Correlation-ID.

    Args:
        context: gRPC Service Context

    Returns:
        Correlation-ID (existierend oder neu generiert)
    """
    correlation_id = extract_metadata_value(context, MetadataKeys.CORRELATION_ID)

    if not correlation_id:
        correlation_id = str(uuid.uuid4())

    return correlation_id


def set_error_metadata(
    context: grpc.ServicerContext, error_code: str, correlation_id: str | None = None
) -> None:
    """Setzt Error-Metadata in gRPC Response.

    Args:
        context: gRPC Service Context
        error_code: Error-Code
        correlation_id: Optional Correlation-ID
    """
    try:
        metadata = [
            (MetadataKeys.ERROR_CODE, error_code),
            (MetadataKeys.ERROR_SEVERITY, "ERROR"),
        ]

        if correlation_id:
            metadata.append((MetadataKeys.ERROR_CORRELATION_ID, correlation_id))

        context.set_trailing_metadata(metadata)

    except Exception as e:
        logger.warning(f"Fehler beim Setzen der Error-Metadata: {e}")


# ============================================================================
# PEER UTILITIES
# ============================================================================


def extract_peer_ip(context: grpc.ServicerContext) -> str:
    """Extrahiert Peer-IP aus gRPC Context.

    Args:
        context: gRPC Service Context

    Returns:
        Peer-IP-Adresse oder "unknown"
    """
    try:
        peer = context.peer()
        if not peer:
            return "unknown"

        # Format: "ipv4:127.0.0.1:12345" oder "ipv6:[::1]:12345"
        if peer.startswith("ipv4:"):
            return peer.split(":")[1]
        if peer.startswith("ipv6:"):
            # IPv6: Extrahiere zwischen [ und ]
            match = re.search(r"\[([^\]]+)\]", peer)
            return match.group(1) if match else "unknown"
        # Fallback: alles vor dem letzten Doppelpunkt
        return peer.rsplit(":", 1)[0]

    except Exception as e:
        logger.warning(f"Fehler bei Peer-IP-Extraktion: {e}")
        return "unknown"


def extract_peer_info(context: grpc.ServicerContext) -> dict[str, str]:
    """Extrahiert vollständige Peer-Informationen.

    Args:
        context: gRPC Service Context

    Returns:
        Dictionary mit Peer-Informationen
    """
    peer = context.peer() or "unknown"

    return {
        "peer": peer,
        "ip": extract_peer_ip(context),
        "user_agent": extract_metadata_value(context, "user-agent", "unknown"),
        "correlation_id": extract_correlation_id(context),
    }


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================


def validate_w3c_traceparent(traceparent: str) -> bool:
    """Validiert W3C Traceparent Format.

    Args:
        traceparent: Traceparent-String

    Returns:
        True wenn gültiges Format
    """
    if not traceparent:
        return False

    # Format: version-trace_id-parent_id-trace_flags
    parts = traceparent.split("-")

    if len(parts) != 4:
        return False

    version, trace_id, parent_id, trace_flags = parts

    # Version: 2 hex digits
    if not re.match(r"^[0-9a-f]{2}$", version):
        return False

    # Trace ID: 32 hex digits, not all zeros
    if not re.match(r"^[0-9a-f]{32}$", trace_id) or trace_id == "0" * 32:
        return False

    # Parent ID: 16 hex digits, not all zeros
    if not re.match(r"^[0-9a-f]{16}$", parent_id) or parent_id == "0" * 16:
        return False

    # Trace flags: 2 hex digits
    return re.match(r"^[0-9a-f]{2}$", trace_flags)


def validate_idempotency_key(key: str) -> bool:
    """Validiert Idempotency-Key Format.

    Args:
        key: Idempotency-Key

    Returns:
        True wenn gültiges Format
    """
    if not key:
        return False

    # Länge zwischen 1 und 255 Zeichen
    if len(key) < 1 or len(key) > 255:
        return False

    # Nur alphanumerische Zeichen, Bindestriche und Unterstriche
    return re.match(r"^[a-zA-Z0-9_-]+$", key)


# ============================================================================
# ERROR HANDLING UTILITIES
# ============================================================================


def create_grpc_error(
    context: grpc.ServicerContext,
    status_code: grpc.StatusCode,
    message: str,
    error_code: str,
    correlation_id: str | None = None,
) -> None:
    """Erstellt standardisierten gRPC-Fehler.

    Args:
        context: gRPC Service Context
        status_code: gRPC Status-Code
        message: Fehlermeldung
        error_code: KEI Error-Code
        correlation_id: Optional Correlation-ID
    """
    # Error-Metadata setzen
    set_error_metadata(context, error_code, correlation_id)

    # Request abbrechen
    context.abort(status_code, message)


def handle_common_errors(
    context: grpc.ServicerContext, error: Exception, correlation_id: str | None = None
) -> None:
    """Behandelt häufige Fehler-Typen.

    Args:
        context: gRPC Service Context
        error: Aufgetretener Fehler
        correlation_id: Optional Correlation-ID
    """
    error_type = type(error).__name__
    error_message = str(error)

    # KEI-RPC spezifische Fehler
    if hasattr(error, "error_code"):
        error_code = error.error_code

        if error_code in (ErrorCodes.AUTH_TOKEN_MISSING, ErrorCodes.AUTH_TOKEN_INVALID):
            create_grpc_error(
                context, grpc.StatusCode.UNAUTHENTICATED, error_message, error_code, correlation_id
            )
        elif error_code == ErrorCodes.AUTH_SCOPE_INSUFFICIENT:
            create_grpc_error(
                context,
                grpc.StatusCode.PERMISSION_DENIED,
                error_message,
                error_code,
                correlation_id,
            )
        elif error_code == ErrorCodes.RATE_LIMIT_EXCEEDED:
            create_grpc_error(
                context,
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                error_message,
                error_code,
                correlation_id,
            )
        elif error_code == ErrorCodes.OPERATION_TIMEOUT:
            create_grpc_error(
                context,
                grpc.StatusCode.DEADLINE_EXCEEDED,
                error_message,
                error_code,
                correlation_id,
            )
        else:
            create_grpc_error(
                context, grpc.StatusCode.INTERNAL, error_message, error_code, correlation_id
            )

    # Standard Python Exceptions
    elif isinstance(error, ValueError):
        create_grpc_error(
            context,
            grpc.StatusCode.INVALID_ARGUMENT,
            error_message,
            ErrorCodes.VALIDATION_ERROR,
            correlation_id,
        )
    elif isinstance(error, TimeoutError):
        create_grpc_error(
            context,
            grpc.StatusCode.DEADLINE_EXCEEDED,
            error_message,
            ErrorCodes.OPERATION_TIMEOUT,
            correlation_id,
        )
    elif isinstance(error, PermissionError):
        create_grpc_error(
            context,
            grpc.StatusCode.PERMISSION_DENIED,
            error_message,
            ErrorCodes.AUTH_SCOPE_INSUFFICIENT,
            correlation_id,
        )
    else:
        # Unbekannter Fehler
        create_grpc_error(
            context,
            grpc.StatusCode.INTERNAL,
            f"Interner Fehler: {error_type}",
            ErrorCodes.INTERNAL_ERROR,
            correlation_id,
        )


# ============================================================================
# TIMING UTILITIES
# ============================================================================


def create_timing_info(start_time: float, end_time: float | None = None) -> dict[str, float]:
    """Erstellt Timing-Informationen.

    Args:
        start_time: Start-Zeitpunkt
        end_time: End-Zeitpunkt (optional, default: jetzt)

    Returns:
        Dictionary mit Timing-Informationen
    """
    if end_time is None:
        end_time = time.time()

    duration_seconds = end_time - start_time
    duration_ms = duration_seconds * 1000.0

    return {
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration_seconds,
        "duration_ms": duration_ms,
    }


# ============================================================================
# LOGGING UTILITIES
# ============================================================================


def log_operation_start(
    operation_type: str, correlation_id: str, peer_info: dict[str, str], **kwargs
) -> None:
    """Loggt Operation-Start.

    Args:
        operation_type: Typ der Operation
        correlation_id: Correlation-ID
        peer_info: Peer-Informationen
        **kwargs: Zusätzliche Log-Daten
    """
    logger.info(
        f"Operation {operation_type} gestartet",
        extra={
            "operation_type": operation_type,
            "correlation_id": correlation_id,
            "peer_ip": peer_info.get("ip"),
            "user_agent": peer_info.get("user_agent"),
            **kwargs,
        },
    )


def log_operation_end(
    operation_type: str,
    correlation_id: str,
    timing_info: dict[str, float],
    success: bool = True,
    **kwargs,
) -> None:
    """Loggt Operation-Ende.

    Args:
        operation_type: Typ der Operation
        correlation_id: Correlation-ID
        timing_info: Timing-Informationen
        success: Ob Operation erfolgreich war
        **kwargs: Zusätzliche Log-Daten
    """
    level = "info" if success else "error"
    status = "erfolgreich" if success else "fehlgeschlagen"

    getattr(logger, level)(
        f"Operation {operation_type} {status} nach {timing_info['duration_ms']:.2f}ms",
        extra={
            "operation_type": operation_type,
            "correlation_id": correlation_id,
            "duration_ms": timing_info["duration_ms"],
            "success": success,
            **kwargs,
        },
    )


__all__ = [
    # Error Handling Utilities
    "create_grpc_error",
    # Timing Utilities
    "create_timing_info",
    "extract_bearer_token",
    "extract_correlation_id",
    # Metadata Utilities
    "extract_metadata_value",
    "extract_peer_info",
    # Peer Utilities
    "extract_peer_ip",
    "extract_tenant_id",
    "handle_common_errors",
    "log_operation_end",
    # Logging Utilities
    "log_operation_start",
    "set_error_metadata",
    "validate_idempotency_key",
    # Validation Utilities
    "validate_w3c_traceparent",
]
