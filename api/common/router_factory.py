"""Konsolidierte Router-Factory für die Keiko-API.

Eliminiert Code-Duplikation durch einheitliche Router-Erstellung
und -Konfiguration für alle API-Module.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from kei_logging import get_logger

from .api_constants import APIPaths, HTTPStatus, ResponseConstants

logger = get_logger(__name__)


# ============================================================================
# STANDARD RESPONSES
# ============================================================================

def get_standard_responses() -> dict[int, dict[str, Any]]:
    """Gibt Standard-Response-Definitionen zurück.

    Returns:
        Dictionary mit Standard-Response-Definitionen
    """
    return {
        HTTPStatus.BAD_REQUEST: {
            "description": "Ungültige Anfrage",
            "content": {
                ResponseConstants.JSON_CONTENT_TYPE: {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {"type": "string"},
                            "error_code": {"type": "string"}
                        }
                    }
                }
            }
        },
        HTTPStatus.UNAUTHORIZED: {
            "description": "Nicht autorisiert",
            "content": {
                ResponseConstants.JSON_CONTENT_TYPE: {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {"type": "string"},
                            "error_code": {"type": "string"}
                        }
                    }
                }
            }
        },
        HTTPStatus.FORBIDDEN: {
            "description": "Zugriff verweigert",
            "content": {
                ResponseConstants.JSON_CONTENT_TYPE: {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {"type": "string"},
                            "error_code": {"type": "string"}
                        }
                    }
                }
            }
        },
        HTTPStatus.NOT_FOUND: {
            "description": "Ressource nicht gefunden",
            "content": {
                ResponseConstants.JSON_CONTENT_TYPE: {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {"type": "string"},
                            "error_code": {"type": "string"}
                        }
                    }
                }
            }
        },
        HTTPStatus.UNPROCESSABLE_ENTITY: {
            "description": "Validierungsfehler",
            "content": {
                ResponseConstants.JSON_CONTENT_TYPE: {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {"type": "array"},
                            "error_code": {"type": "string"}
                        }
                    }
                }
            }
        },
        HTTPStatus.TOO_MANY_REQUESTS: {
            "description": "Rate Limit überschritten",
            "content": {
                ResponseConstants.JSON_CONTENT_TYPE: {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {"type": "string"},
                            "error_code": {"type": "string"},
                            "retry_after": {"type": "integer"}
                        }
                    }
                }
            }
        },
        HTTPStatus.INTERNAL_SERVER_ERROR: {
            "description": "Interner Serverfehler",
            "content": {
                ResponseConstants.JSON_CONTENT_TYPE: {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {"type": "string"},
                            "error_code": {"type": "string"}
                        }
                    }
                }
            }
        },
        HTTPStatus.SERVICE_UNAVAILABLE: {
            "description": "Service nicht verfügbar",
            "content": {
                ResponseConstants.JSON_CONTENT_TYPE: {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {"type": "string"},
                            "error_code": {"type": "string"}
                        }
                    }
                }
            }
        }
    }


def get_health_responses() -> dict[int, dict[str, Any]]:
    """Gibt Health-spezifische Response-Definitionen zurück.

    Returns:
        Dictionary mit Health-Response-Definitionen
    """
    return {
        HTTPStatus.OK: {
            "description": "Service ist gesund",
            "content": {
                ResponseConstants.JSON_CONTENT_TYPE: {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["healthy"]},
                            "service": {"type": "string"},
                            "version": {"type": "string"},
                            "timestamp": {"type": "string", "format": "date-time"}
                        }
                    }
                }
            }
        },
        HTTPStatus.SERVICE_UNAVAILABLE: {
            "description": "Service ist nicht verfügbar",
            "content": {
                ResponseConstants.JSON_CONTENT_TYPE: {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["unhealthy"]},
                            "service": {"type": "string"},
                            "error": {"type": "string"},
                            "timestamp": {"type": "string", "format": "date-time"}
                        }
                    }
                }
            }
        }
    }


def get_chat_responses() -> dict[int, dict[str, Any]]:
    """Gibt Chat-spezifische Response-Definitionen zurück.

    Returns:
        Dictionary mit Chat-Response-Definitionen
    """
    standard = get_standard_responses()
    chat_specific = {
        HTTPStatus.CREATED: {
            "description": "Chat-Session erfolgreich erstellt",
            "content": {
                ResponseConstants.JSON_CONTENT_TYPE: {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string"},
                            "created_at": {"type": "string", "format": "date-time"}
                        }
                    }
                }
            }
        }
    }
    return {**standard, **chat_specific}


def get_function_responses() -> dict[int, dict[str, Any]]:
    """Gibt Function-spezifische Response-Definitionen zurück.

    Returns:
        Dictionary mit Function-Response-Definitionen
    """
    standard = get_standard_responses()
    function_specific = {
        HTTPStatus.ACCEPTED: {
            "description": "Funktion wird asynchron ausgeführt",
            "content": {
                ResponseConstants.JSON_CONTENT_TYPE: {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "execution_id": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending"]}
                        }
                    }
                }
            }
        }
    }
    return {**standard, **function_specific}


# ============================================================================
# ROUTER FACTORY
# ============================================================================

class RouterFactory:
    """Factory-Klasse für einheitliche Router-Erstellung.

    Konsolidiert Router-Konfiguration und eliminiert Code-Duplikation.
    """

    @staticmethod
    def create_router(
        prefix: str,
        tags: list[str],
        include_standard_responses: bool = True,
        additional_responses: dict[int, dict[str, Any]] | None = None,
        dependencies: list[Any] | None = None,
        deprecated: bool = False
    ) -> APIRouter:
        """Erstellt konfigurierten APIRouter.

        Args:
            prefix: URL-Präfix für Router
            tags: OpenAPI-Tags
            include_standard_responses: Ob Standard-Responses eingeschlossen werden sollen
            additional_responses: Zusätzliche Response-Definitionen
            dependencies: Router-Dependencies
            deprecated: Ob Router als deprecated markiert werden soll

        Returns:
            Konfigurierter APIRouter
        """
        router = APIRouter(
            prefix=prefix,
            tags=tags,
            dependencies=dependencies or [],
            deprecated=deprecated
        )

        # Standard-Responses hinzufügen
        if include_standard_responses:
            router.responses.update(get_standard_responses())

        # Zusätzliche Responses hinzufügen
        if additional_responses:
            router.responses.update(additional_responses)

        logger.debug(f"Router erstellt: prefix={prefix}, tags={tags}")
        return router

    @staticmethod
    def create_health_router(
        prefix: str = APIPaths.HEALTH,
        tags: list[str] | None = None
    ) -> APIRouter:
        """Erstellt Health-spezifischen Router.

        Args:
            prefix: URL-Präfix
            tags: OpenAPI-Tags

        Returns:
            Health-Router
        """
        return RouterFactory.create_router(
            prefix=prefix,
            tags=tags or ["health"],
            additional_responses=get_health_responses()
        )

    @staticmethod
    def create_chat_router(
        prefix: str = APIPaths.CHAT,
        tags: list[str] | None = None
    ) -> APIRouter:
        """Erstellt Chat-spezifischen Router.

        Args:
            prefix: URL-Präfix
            tags: OpenAPI-Tags

        Returns:
            Chat-Router
        """
        return RouterFactory.create_router(
            prefix=prefix,
            tags=tags or ["chat"],
            additional_responses=get_chat_responses()
        )

    @staticmethod
    def create_function_router(
        prefix: str = APIPaths.FUNCTIONS,
        tags: list[str] | None = None
    ) -> APIRouter:
        """Erstellt Function-spezifischen Router.

        Args:
            prefix: URL-Präfix
            tags: OpenAPI-Tags

        Returns:
            Function-Router
        """
        return RouterFactory.create_router(
            prefix=prefix,
            tags=tags or ["functions"],
            additional_responses=get_function_responses()
        )

    @staticmethod
    def create_agents_router(
        prefix: str = APIPaths.AGENTS,
        tags: list[str] | None = None
    ) -> APIRouter:
        """Erstellt Agents-spezifischen Router.

        Args:
            prefix: URL-Präfix
            tags: OpenAPI-Tags

        Returns:
            Agents-Router
        """
        return RouterFactory.create_router(
            prefix=prefix,
            tags=tags or ["agents"]
        )

    @staticmethod
    def create_capabilities_router(
        prefix: str = APIPaths.CAPABILITIES,
        tags: list[str] | None = None
    ) -> APIRouter:
        """Erstellt Capabilities-spezifischen Router.

        Args:
            prefix: URL-Präfix
            tags: OpenAPI-Tags

        Returns:
            Capabilities-Router
        """
        return RouterFactory.create_router(
            prefix=prefix,
            tags=tags or ["capabilities"]
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_router(
    prefix: str,
    tags: list[str],
    **kwargs
) -> APIRouter:
    """Convenience-Funktion für Router-Erstellung.

    Args:
        prefix: URL-Präfix
        tags: OpenAPI-Tags
        **kwargs: Zusätzliche Router-Parameter

    Returns:
        Konfigurierter APIRouter
    """
    return RouterFactory.create_router(prefix, tags, **kwargs)


def create_health_router(prefix: str = APIPaths.HEALTH) -> APIRouter:
    """Convenience-Funktion für Health-Router-Erstellung.

    Args:
        prefix: URL-Präfix

    Returns:
        Health-Router
    """
    return RouterFactory.create_health_router(prefix)


def create_versioned_router(
    prefix: str,
    tags: list[str],
    version: str = "v1"
) -> APIRouter:
    """Erstellt versionierten Router.

    Args:
        prefix: URL-Präfix (ohne Version)
        tags: OpenAPI-Tags
        version: API-Version

    Returns:
        Versionierter Router
    """
    versioned_prefix = f"/api/{version}{prefix}"
    return RouterFactory.create_router(versioned_prefix, tags)
