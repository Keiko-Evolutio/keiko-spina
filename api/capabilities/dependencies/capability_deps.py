"""FastAPI Dependencies für Capabilities und Feature-Flags.

Eliminiert globale Variablen und stellt testbare Dependency Injection bereit.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, Header, Query

from ..services.capability_service import CapabilityService
from ..services.feature_flag_service import FeatureFlagService

# ============================================================================
# SERVICE FACTORIES (mit Caching für Performance)
# ============================================================================

@lru_cache(maxsize=1)
def get_feature_flag_service() -> FeatureFlagService:
    """Erstellt oder gibt gecachte FeatureFlagService-Instanz zurück.

    Verwendet LRU-Cache für Singleton-Pattern ohne globale Variablen.

    Returns:
        FeatureFlagService-Instanz
    """
    return FeatureFlagService()


@lru_cache(maxsize=1)
def get_capability_service(
    feature_flag_service: FeatureFlagService = Depends(get_feature_flag_service)
) -> CapabilityService:
    """Erstellt oder gibt gecachte CapabilityService-Instanz zurück.

    Args:
        feature_flag_service: Injected FeatureFlagService

    Returns:
        CapabilityService-Instanz
    """
    return CapabilityService(feature_flag_service)


# ============================================================================
# AUTHENTICATION & CONTEXT DEPENDENCIES
# ============================================================================

def get_client_id(authorization: str | None = Header(None)) -> str | None:
    """Extrahiert Client-ID aus Authorization-Header.

    Args:
        authorization: Authorization-Header

    Returns:
        Client-ID oder None
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None

    # Vereinfachte Client-ID-Extraktion
    # In Production: JWT-Token-Parsing mit korrekter Validierung
    token = authorization.replace("Bearer ", "")
    return f"client_{hash(token) % 10000}"


def get_server_name_from_query(
    server_name: str | None = Query(
        None,
        description="MCP-Server-Name für Feature-Context"
    )
) -> str | None:
    """Extrahiert Server-Name aus Query-Parameter.

    Args:
        server_name: Server-Name aus Query

    Returns:
        Server-Name oder None
    """
    return server_name


def get_feature_context(
    client_id: str | None = Depends(get_client_id),
    server_name: str | None = Depends(get_server_name_from_query)
) -> dict[str, str | None]:
    """Erstellt Feature-Context für Request.

    Args:
        client_id: Injected Client-ID
        server_name: Injected Server-Name

    Returns:
        Dictionary mit Feature-Context
    """
    return {
        "client_id": client_id,
        "server_name": server_name
    }


# ============================================================================
# SPECIALIZED DEPENDENCIES
# ============================================================================

def get_api_version(
    api_version: str | None = Header(None, alias="X-API-Version")
) -> str:
    """Extrahiert API-Version aus Header mit Fallback.

    Args:
        api_version: API-Version aus Header

    Returns:
        API-Version (Standard: "2.0.0")
    """
    return api_version or "2.0.0"


def get_correlation_id(
    correlation_id: str | None = Header(None, alias="X-Correlation-ID")
) -> str | None:
    """Extrahiert Korrelations-ID aus Header.

    Args:
        correlation_id: Korrelations-ID aus Header

    Returns:
        Korrelations-ID oder None
    """
    return correlation_id


# ============================================================================
# COMPOSITE DEPENDENCIES
# ============================================================================

class CapabilityContext:
    """Kontext-Objekt für Capability-Requests."""

    def __init__(
        self,
        client_id: str | None,
        server_name: str | None,
        api_version: str,
        correlation_id: str | None
    ) -> None:
        """Initialisiert Capability-Context.

        Args:
            client_id: Client-ID
            server_name: Server-Name
            api_version: API-Version
            correlation_id: Korrelations-ID
        """
        self.client_id = client_id
        self.server_name = server_name
        self.api_version = api_version
        self.correlation_id = correlation_id


def get_capability_context(
    client_id: str | None = Depends(get_client_id),
    server_name: str | None = Depends(get_server_name_from_query),
    api_version: str = Depends(get_api_version),
    correlation_id: str | None = Depends(get_correlation_id)
) -> CapabilityContext:
    """Erstellt vollständigen Capability-Context.

    Args:
        client_id: Injected Client-ID
        server_name: Injected Server-Name
        api_version: Injected API-Version
        correlation_id: Injected Korrelations-ID

    Returns:
        CapabilityContext-Objekt
    """
    return CapabilityContext(
        client_id=client_id,
        server_name=server_name,
        api_version=api_version,
        correlation_id=correlation_id
    )


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def clear_service_cache() -> None:
    """Löscht Service-Cache für Tests.

    Ermöglicht frische Service-Instanzen in Unit-Tests.
    """
    get_feature_flag_service.cache_clear()
    get_capability_service.cache_clear()


def create_test_feature_context(
    client_id: str | None = None,
    server_name: str | None = None
) -> dict[str, str | None]:
    """Erstellt Test-Feature-Context.

    Args:
        client_id: Test-Client-ID
        server_name: Test-Server-Name

    Returns:
        Test-Feature-Context
    """
    return {
        "client_id": client_id,
        "server_name": server_name
    }


def create_test_capability_context(
    client_id: str | None = None,
    server_name: str | None = None,
    api_version: str = "2.0.0",
    correlation_id: str | None = None
) -> CapabilityContext:
    """Erstellt Test-Capability-Context.

    Args:
        client_id: Test-Client-ID
        server_name: Test-Server-Name
        api_version: Test-API-Version
        correlation_id: Test-Korrelations-ID

    Returns:
        Test-CapabilityContext
    """
    return CapabilityContext(
        client_id=client_id,
        server_name=server_name,
        api_version=api_version,
        correlation_id=correlation_id
    )
