"""API-Routes für Capabilities und Feature-Flags.

Extrahiert aus der monolithischen capabilities.py für bessere Modularität
und verwendet die neuen gemeinsamen Basis-Abstraktionen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from kei_logging import get_logger
from observability import trace_function

from ...common.error_handlers import StandardErrorHandler
from ...common.response_models import create_success_response
from ..dependencies.capability_deps import (
    CapabilityContext,
    get_capability_context,
    get_capability_service,
)
from ..models.capability_models import CapabilitiesResponse

if TYPE_CHECKING:
    from ..services.capability_service import CapabilityService

# ============================================================================
# ROUTER SETUP
# ============================================================================

router = APIRouter(prefix="/api/v1", tags=["capabilities"])
logger = get_logger(__name__)
error_handler = StandardErrorHandler(include_details=False)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/capabilities", response_model=CapabilitiesResponse)
@trace_function("api.capabilities.get_capabilities")
async def get_capabilities(
    context: CapabilityContext = Depends(get_capability_context),
    capability_service: CapabilityService = Depends(get_capability_service)
) -> CapabilitiesResponse:
    """Gibt verfügbare API-Capabilities und Feature-Flags zurück.

    Dieser Endpoint ermöglicht API-Consumern die Discovery verfügbarer
    Features und Capabilities basierend auf ihrem Client-Context.

    Args:
        context: Injected Capability-Context
        capability_service: Injected Capability-Service

    Returns:
        Vollständige Capabilities-Response

    Raises:
        HTTPException: Bei Fehlern während der Capability-Abfrage
    """
    try:
        logger.info(
            f"Capabilities-Abfrage für Client: {context.client_id}, "
            f"Server: {context.server_name}, API-Version: {context.api_version}",
            extra={"correlation_id": context.correlation_id}
        )

        # Capabilities-Response erstellen
        response = capability_service.get_capabilities_response(
            client_id=context.client_id,
            server_name=context.server_name,
            api_version=context.api_version
        )

        logger.info(
            f"Capabilities erfolgreich abgerufen: "
            f"{len(response.capabilities)} Capabilities, "
            f"{len(response.feature_flags)} Feature-Flags",
            extra={"correlation_id": context.correlation_id}
        )

        return response

    except Exception as exc:
        error_response = error_handler.handle_error(
            exc,
            "get_capabilities",
            context.correlation_id
        )
        # Konvertiere JSONResponse zu CapabilitiesResponse für Type-Safety
        if hasattr(error_response, "body"):
            # Fallback CapabilitiesResponse bei Fehlern
            return CapabilitiesResponse(
                capabilities=[],
                feature_flags={},
                api_version="1.0.0",
                error=True
            )
        return error_response


@router.get("/capabilities/stats")
@trace_function("api.capabilities.get_stats")
async def get_capability_stats(
    context: CapabilityContext = Depends(get_capability_context),
    capability_service: CapabilityService = Depends(get_capability_service)
):
    """Gibt Statistiken über Capabilities und Feature-Flags zurück.

    Args:
        context: Injected Capability-Context
        capability_service: Injected Capability-Service

    Returns:
        Statistiken über Capabilities und Feature-Flags
    """
    try:
        logger.info(
            "Capability-Statistiken angefordert",
            extra={"correlation_id": context.correlation_id}
        )

        # Statistiken sammeln
        capability_stats = capability_service.get_stats()
        feature_stats = capability_service.feature_flag_service.get_stats()

        stats = {
            "capabilities": capability_stats,
            "feature_flags": feature_stats,
            "summary": {
                "total_capabilities": capability_stats["total_capabilities"],
                "total_features": feature_stats["total_features"],
                "enabled_features": feature_stats["enabled_by_default"]
            }
        }

        return create_success_response(
            data=stats,
            message="Capability-Statistiken erfolgreich abgerufen",
            correlation_id=context.correlation_id
        )

    except Exception as exc:
        return error_handler.handle_error(
            exc,
            "get_capability_stats",
            context.correlation_id
        )


@router.get("/capabilities/{capability_name}")
@trace_function("api.capabilities.get_capability")
async def get_capability_details(
    capability_name: str,
    context: CapabilityContext = Depends(get_capability_context),
    capability_service: CapabilityService = Depends(get_capability_service)
):
    """Gibt Details zu einer spezifischen Capability zurück.

    Args:
        capability_name: Name der Capability
        context: Injected Capability-Context
        capability_service: Injected Capability-Service

    Returns:
        Capability-Details
    """
    try:
        logger.info(
            f"Capability-Details angefordert: {capability_name}",
            extra={"correlation_id": context.correlation_id}
        )

        # Capability abrufen
        capability = capability_service.get_capability(capability_name)
        if not capability:
            from ...common.error_handlers import NotFoundError
            raise NotFoundError(
                "Capability",
                capability_name,
                context.correlation_id
            )

        # Verfügbarkeit prüfen
        available = capability_service.is_capability_available(
            capability_name,
            context.client_id,
            context.server_name
        )

        # Response erstellen
        from ..models.capability_models import CapabilityResponse
        response_data = CapabilityResponse.from_api_capability(capability, available)

        return create_success_response(
            data=response_data.dict(),
            message=f"Capability '{capability_name}' erfolgreich abgerufen",
            correlation_id=context.correlation_id
        )

    except Exception as exc:
        return error_handler.handle_error(
            exc,
            "get_capability_details",
            context.correlation_id
        )


@router.get("/features")
@trace_function("api.capabilities.get_features")
async def get_feature_flags(
    context: CapabilityContext = Depends(get_capability_context),
    capability_service: CapabilityService = Depends(get_capability_service)
):
    """Gibt alle Feature-Flags zurück (aktiviert und deaktiviert).

    Args:
        context: Injected Capability-Context
        capability_service: Injected Capability-Service

    Returns:
        Liste aller Feature-Flags
    """
    try:
        logger.info(
            "Feature-Flags angefordert",
            extra={"correlation_id": context.correlation_id}
        )

        # Alle Features abrufen
        all_features = capability_service.feature_flag_service.get_all_features()

        return create_success_response(
            data={"features": [f.dict() for f in all_features]},
            message=f"{len(all_features)} Feature-Flags erfolgreich abgerufen",
            correlation_id=context.correlation_id
        )

    except Exception as exc:
        return error_handler.handle_error(
            exc,
            "get_feature_flags",
            context.correlation_id
        )


@router.get("/features/enabled")
@trace_function("api.capabilities.get_enabled_features")
async def get_enabled_features(
    context: CapabilityContext = Depends(get_capability_context),
    capability_service: CapabilityService = Depends(get_capability_service)
):
    """Gibt nur aktivierte Feature-Flags zurück.

    Args:
        context: Injected Capability-Context
        capability_service: Injected Capability-Service

    Returns:
        Liste aktivierter Feature-Flags
    """
    try:
        logger.info(
            f"Aktivierte Feature-Flags angefordert für Client: {context.client_id}, "
            f"Server: {context.server_name}",
            extra={"correlation_id": context.correlation_id}
        )

        # Aktivierte Features abrufen
        enabled_features = capability_service.feature_flag_service.get_enabled_features(
            context.client_id,
            context.server_name
        )

        return create_success_response(
            data={"enabled_features": [f.dict() for f in enabled_features]},
            message=f"{len(enabled_features)} aktivierte Feature-Flags abgerufen",
            correlation_id=context.correlation_id
        )

    except Exception as exc:
        return error_handler.handle_error(
            exc,
            "get_enabled_features",
            context.correlation_id
        )
