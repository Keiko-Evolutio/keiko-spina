"""Error Response Builder für Middleware.

Vereinfachte Error-Response-Generierung für Middleware-Komponenten
mit standardisierten Formaten und Status-Codes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from starlette.responses import JSONResponse

from kei_logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request

logger = get_logger(__name__)


class MiddlewareErrorBuilder:
    """Vereinfachte Error-Response-Builder für Middleware.

    Bietet standardisierte Error-Response-Generierung für häufige
    Middleware-Fehlerszenarien mit einheitlichen Formaten.
    """

    # Standard Error-Codes für Middleware
    ERROR_CODES = {
        "AUTHENTICATION_FAILED": "AUTH_001",
        "TOKEN_EXPIRED": "AUTH_002",
        "TOKEN_INVALID": "AUTH_003",
        "AUTHORIZATION_FAILED": "AUTH_004",
        "INSUFFICIENT_SCOPES": "AUTH_005",
        "RATE_LIMITED": "RATE_001",
        "FORBIDDEN": "ACCESS_001",
        "TENANT_REQUIRED": "TENANT_001",
        "VALIDATION_ERROR": "VALID_001",
        "INTERNAL_ERROR": "SYS_001",
        "SERVICE_UNAVAILABLE": "SYS_002",
    }

    # Standard HTTP Status-Codes
    STATUS_CODES = {
        "AUTHENTICATION_FAILED": 401,
        "TOKEN_EXPIRED": 401,
        "TOKEN_INVALID": 401,
        "AUTHORIZATION_FAILED": 403,
        "INSUFFICIENT_SCOPES": 403,
        "RATE_LIMITED": 429,
        "FORBIDDEN": 403,
        "TENANT_REQUIRED": 400,
        "VALIDATION_ERROR": 400,
        "INTERNAL_ERROR": 500,
        "SERVICE_UNAVAILABLE": 503,
    }

    def __init__(self, component_name: str = "middleware", include_details: bool = False):
        """Initialisiert Error Response Builder.

        Args:
            component_name: Name der Middleware-Komponente
            include_details: Ob detaillierte Error-Informationen enthalten sein sollen
        """
        self.component_name = component_name
        self.include_details = include_details

    def _get_timestamp(self) -> str:
        """Erstellt ISO-Zeitstempel."""
        return datetime.now(UTC).isoformat()

    def _extract_request_context(self, request: Request | None) -> dict[str, Any]:
        """Extrahiert Kontext-Informationen aus Request."""
        if not request:
            return {}

        context = {
            "method": request.method,
            "path": str(request.url.path),
        }

        # Optionale Kontext-Informationen
        if hasattr(request, "state"):
            if hasattr(request.state, "user"):
                user_info = getattr(request.state, "user", {})
                if isinstance(user_info, dict) and user_info:
                    context["user_id"] = user_info.get("sub") or user_info.get("user_id")

            if hasattr(request.state, "forced_tenant"):
                tenant = getattr(request.state, "forced_tenant", None)
                if tenant:
                    context["tenant_id"] = tenant

        # Tenant aus Headers
        if not context.get("tenant_id"):
            tenant_headers = ["X-Tenant-Id", "x-tenant-id", "x-tenant"]
            for header in tenant_headers:
                tenant = request.headers.get(header)
                if tenant:
                    context["tenant_id"] = tenant
                    break

        return context

    def create_error_response(
        self,
        error_type: str,
        message: str,
        request: Request | None = None,
        *,
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
        headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Erstellt standardisierte Error-Response-Struktur.

        Args:
            error_type: Typ des Fehlers (aus ERROR_CODES)
            message: Benutzerfreundliche Fehlermeldung
            request: Optional Request-Objekt für Kontext
            details: Zusätzliche Error-Details
            status_code: Optionaler HTTP-Status-Code
            headers: Zusätzliche Response-Headers

        Returns:
            Error-Response-Dictionary
        """
        error_code = self.ERROR_CODES.get(error_type, "UNKNOWN")

        error_response = {
            "error": {
                "code": error_code,
                "type": error_type,
                "message": message,
                "component": self.component_name,
                "timestamp": self._get_timestamp(),
            }
        }

        # Zusätzliche Details hinzufügen
        if details:
            error_response["error"]["details"] = details

        # Request-Kontext hinzufügen
        context = self._extract_request_context(request)
        if context:
            error_response["context"] = context

        # Recovery-Hinweise für bestimmte Fehlertypen
        recovery_hints = self._get_recovery_hints(error_type)
        if recovery_hints:
            error_response["recovery"] = recovery_hints

        return error_response

    def _get_recovery_hints(self, error_type: str) -> dict[str, Any] | None:
        """Erstellt Recovery-Hinweise für verschiedene Fehlertypen."""
        recovery_map = {
            "AUTHENTICATION_FAILED": {
                "action": "Provide valid authentication credentials",
                "retryable": True,
                "retry_after": None
            },
            "TOKEN_EXPIRED": {
                "action": "Refresh authentication token",
                "retryable": True,
                "retry_after": None
            },
            "TOKEN_INVALID": {
                "action": "Obtain new authentication token",
                "retryable": True,
                "retry_after": None
            },
            "RATE_LIMITED": {
                "action": "Reduce request frequency",
                "retryable": True,
                "retry_after": 60  # Sekunden
            },
            "INSUFFICIENT_SCOPES": {
                "action": "Request additional permissions",
                "retryable": False,
                "retry_after": None
            },
            "TENANT_REQUIRED": {
                "action": "Provide X-Tenant-Id header",
                "retryable": True,
                "retry_after": None
            },
            "SERVICE_UNAVAILABLE": {
                "action": "Retry request later",
                "retryable": True,
                "retry_after": 30
            }
        }

        return recovery_map.get(error_type)

    def create_json_response(
        self,
        error_type: str,
        message: str,
        request: Request | None = None,
        *,
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
        headers: dict[str, str] | None = None
    ) -> JSONResponse:
        """Erstellt JSONResponse für FastAPI/Starlette.

        Args:
            error_type: Typ des Fehlers (aus ERROR_CODES)
            message: Benutzerfreundliche Fehlermeldung
            request: Optional Request-Objekt für Kontext
            details: Zusätzliche Error-Details
            status_code: Optionaler HTTP-Status-Code
            headers: Zusätzliche Response-Headers

        Returns:
            JSONResponse mit Error-Details
        """
        error_response = self.create_error_response(
            error_type, message, request,
            details=details, status_code=status_code, headers=headers
        )

        # Status-Code bestimmen
        final_status_code = status_code or self.STATUS_CODES.get(error_type, 500)

        # Standard-Headers
        response_headers = {
            "Content-Type": "application/json",
        }

        # Zusätzliche Headers für bestimmte Fehlertypen
        if error_type in ["AUTHENTICATION_FAILED", "TOKEN_EXPIRED", "TOKEN_INVALID"]:
            response_headers["WWW-Authenticate"] = "Bearer"

        if error_type == "RATE_LIMITED":
            # Retry-After aus Details oder Recovery-Hints
            if details and "retry_after" in details:
                retry_after = details["retry_after"]
            else:
                recovery = error_response.get("recovery", {})
                retry_after = recovery.get("retry_after", 60)
            response_headers["Retry-After"] = str(retry_after)

        # Benutzerdefinierte Headers hinzufügen
        if headers:
            response_headers.update(headers)

        return JSONResponse(
            content=error_response,
            status_code=final_status_code,
            headers=response_headers
        )

    def create_simple_response(
        self,
        error_type: str,
        message: str,
        status_code: int | None = None
    ) -> JSONResponse:
        """Erstellt einfache Error-Response ohne Request-Kontext.

        Args:
            error_type: Typ des Fehlers
            message: Fehlermeldung
            status_code: HTTP-Status-Code

        Returns:
            JSONResponse mit minimalen Error-Details
        """
        final_status_code = status_code or self.STATUS_CODES.get(error_type, 500)

        simple_response = {
            "error": message,
            "code": self.ERROR_CODES.get(error_type, "UNKNOWN"),
            "timestamp": self._get_timestamp()
        }

        return JSONResponse(
            content=simple_response,
            status_code=final_status_code
        )

    # Convenience-Methoden für häufige Middleware-Fehler

    def authentication_failed(
        self,
        message: str = "Authentication failed",
        request: Request | None = None,
        **details
    ) -> JSONResponse:
        """Erstellt Authentication-Failed-Response."""
        return self.create_json_response(
            "AUTHENTICATION_FAILED", message, request, details=details
        )

    def token_expired(
        self,
        message: str = "Token expired",
        request: Request | None = None,
        **details
    ) -> JSONResponse:
        """Erstellt Token-Expired-Response."""
        return self.create_json_response(
            "TOKEN_EXPIRED", message, request, details=details
        )

    def insufficient_scopes(
        self,
        message: str = "Insufficient permissions",
        request: Request | None = None,
        required_scopes: list | None = None,
        **details
    ) -> JSONResponse:
        """Erstellt Insufficient-Scopes-Response."""
        if required_scopes:
            details["required_scopes"] = required_scopes
        return self.create_json_response(
            "INSUFFICIENT_SCOPES", message, request, details=details
        )

    def rate_limited(
        self,
        message: str = "Rate limit exceeded",
        request: Request | None = None,
        retry_after: int = 60,
        **details
    ) -> JSONResponse:
        """Erstellt Rate-Limited-Response."""
        details["retry_after"] = retry_after
        return self.create_json_response(
            "RATE_LIMITED", message, request, details=details
        )

    def forbidden(
        self,
        message: str = "Access forbidden",
        request: Request | None = None,
        **details
    ) -> JSONResponse:
        """Erstellt Forbidden-Response."""
        return self.create_json_response(
            "FORBIDDEN", message, request, details=details
        )

    def tenant_required(
        self,
        message: str = "Tenant ID required",
        request: Request | None = None,
        **details
    ) -> JSONResponse:
        """Erstellt Tenant-Required-Response."""
        return self.create_json_response(
            "TENANT_REQUIRED", message, request, details=details
        )


__all__ = ["MiddlewareErrorBuilder"]
