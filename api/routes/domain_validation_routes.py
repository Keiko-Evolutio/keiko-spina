"""API-Endpunkte für Domain-Validierung-Management.

Diese Endpunkte ermöglichen die Verwaltung und Überwachung der
Domain-Validierung für registrierte MCP-Server.
"""

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from agents.tools.mcp import mcp_registry
from config.domain_validation_config import (
    get_domain_validation_config,
    reload_domain_validation_config,
)
from kei_logging import get_logger
from observability import trace_function
from security.kei_mcp_auth import require_auth
from services.unified_domain_revalidation_service import UnifiedDomainRevalidationService

# Service-Instanz für API-Endpunkte
_domain_service = UnifiedDomainRevalidationService()

logger = get_logger(__name__)

router = APIRouter()


# ============================================================================
# PYDANTIC-MODELLE
# ============================================================================

class DomainValidationStatus(BaseModel):
    """Status der Domain-Validierung für einen Server."""
    server_name: str = Field(..., description="Name des MCP-Servers")
    exists: bool = Field(..., description="Ob der Server in der Registry existiert")
    domain_validated: bool = Field(..., description="Ob die Domain validiert ist")
    server_url: str | None = Field(None, description="URL des Servers")
    validation_time: float | None = Field(None, description="Zeitstempel der initialen Validierung")
    last_revalidation: float | None = Field(None, description="Zeitstempel der letzten Revalidierung")


class DomainValidationConfig(BaseModel):
    """Konfiguration der Domain-Validierung."""
    enabled: bool = Field(..., description="Ob Domain-Validierung aktiviert ist")
    allowed_domains: list[str] = Field(..., description="Liste erlaubter Domains")
    periodic_revalidation: bool = Field(..., description="Ob periodische Revalidierung aktiviert ist")
    revalidation_interval_hours: int = Field(..., description="Revalidierung-Intervall in Stunden")
    config_reload: bool = Field(..., description="Ob Config-Reload aktiviert ist")
    config_reload_interval_minutes: int = Field(..., description="Config-Reload-Intervall in Minuten")


class DomainRevalidationServiceStatus(BaseModel):
    """Status des Domain-Revalidierung-Service."""
    running: bool = Field(..., description="Ob der Service läuft")
    config: DomainValidationConfig = Field(..., description="Aktuelle Konfiguration")
    last_revalidation: float | None = Field(None, description="Zeitstempel der letzten Revalidierung")
    last_config_reload: float | None = Field(None, description="Zeitstempel des letzten Config-Reload")
    next_revalidation: float | None = Field(None, description="Zeitstempel der nächsten Revalidierung")
    next_config_reload: float | None = Field(None, description="Zeitstempel des nächsten Config-Reload")


class DomainRevalidationResult(BaseModel):
    """Ergebnis einer Domain-Revalidierung."""
    total_servers: int = Field(..., description="Gesamtanzahl der Server")
    successful_validations: int = Field(..., description="Anzahl erfolgreicher Validierungen")
    failed_validations: int = Field(..., description="Anzahl fehlgeschlagener Validierungen")
    results: dict[str, bool] = Field(..., description="Detaillierte Ergebnisse pro Server")
    duration_seconds: float = Field(..., description="Dauer der Revalidierung in Sekunden")


class DomainValidationOverview(BaseModel):
    """Übersicht über Domain-Validierung-Status."""
    service_status: DomainRevalidationServiceStatus = Field(..., description="Service-Status")
    server_statuses: list[DomainValidationStatus] = Field(..., description="Status aller Server")
    summary: dict[str, int] = Field(..., description="Zusammenfassung der Validierung-Status")


# ============================================================================
# API-ENDPUNKTE
# ============================================================================

@router.get("/domain-validation/status", response_model=DomainValidationOverview)
@trace_function("api.domain_validation.get_status")
async def get_domain_validation_status(
    _: str = Depends(require_auth)
) -> DomainValidationOverview:
    """Gibt Übersicht über Domain-Validierung-Status zurück.

    Dieser Endpunkt zeigt den Status der Domain-Validierung für alle
    registrierten MCP-Server sowie den Status des Revalidierung-Service.

    **Authentifizierung:** Bearer Token erforderlich
    """
    try:
        # Service-Status abrufen
        service_status_data = _domain_service.get_status()
        config_data = service_status_data["config"]

        service_status = DomainRevalidationServiceStatus(
            running=service_status_data["running"],
            config=DomainValidationConfig(
                enabled=config_data["enabled"],
                allowed_domains=[],  # Aus Sicherheitsgründen nicht exposieren
                periodic_revalidation=config_data["periodic_revalidation"],
                revalidation_interval_hours=config_data["revalidation_interval_hours"],
                config_reload=config_data["config_reload"],
                config_reload_interval_minutes=config_data["config_reload_interval_minutes"]
            ),
            last_revalidation=service_status_data["last_revalidation"],
            last_config_reload=service_status_data["last_config_reload"],
            next_revalidation=service_status_data["next_revalidation"],
            next_config_reload=service_status_data["next_config_reload"]
        )

        # Server-Status für alle registrierten Server abrufen
        server_names = mcp_registry.list_servers()
        server_statuses = []

        for server_name in server_names:
            status_data = mcp_registry.get_domain_validation_status(server_name)
            server_statuses.append(DomainValidationStatus(
                server_name=server_name,
                exists=status_data["exists"],
                domain_validated=status_data["domain_validated"],
                server_url=status_data.get("server_url"),
                validation_time=status_data["validation_time"],
                last_revalidation=status_data["last_revalidation"]
            ))

        # Zusammenfassung erstellen
        total_servers = len(server_statuses)
        validated_servers = sum(1 for s in server_statuses if s.domain_validated)
        unvalidated_servers = total_servers - validated_servers

        summary = {
            "total_servers": total_servers,
            "validated_servers": validated_servers,
            "unvalidated_servers": unvalidated_servers,
            "allowed_domains_count": config_data["allowed_domains_count"]
        }

        return DomainValidationOverview(
            service_status=service_status,
            server_statuses=server_statuses,
            summary=summary
        )

    except Exception as e:
        logger.exception(f"Fehler beim Abrufen des Domain-Validierung-Status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler beim Abrufen des Domain-Validierung-Status"
        )


@router.get("/domain-validation/servers/{server_name}", response_model=DomainValidationStatus)
@trace_function("api.domain_validation.get_server_status")
async def get_server_domain_validation_status(
    server_name: str,
    _: str = Depends(require_auth)
) -> DomainValidationStatus:
    """Gibt Domain-Validierung-Status für einen spezifischen Server zurück.

    **Authentifizierung:** Bearer Token erforderlich
    """
    try:
        status_data = mcp_registry.get_domain_validation_status(server_name)

        if not status_data["exists"]:
            raise HTTPException(
                status_code=404,
                detail=f"Server {server_name} nicht gefunden"
            )

        return DomainValidationStatus(
            server_name=server_name,
            exists=status_data["exists"],
            domain_validated=status_data["domain_validated"],
            server_url=status_data.get("server_url"),
            validation_time=status_data["validation_time"],
            last_revalidation=status_data["last_revalidation"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Fehler beim Abrufen des Server-Domain-Status für {server_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler beim Abrufen des Server-Domain-Status"
        )


@router.post("/domain-validation/revalidate", response_model=DomainRevalidationResult)
@trace_function("api.domain_validation.force_revalidation")
async def force_domain_revalidation_endpoint(
    server_name: str | None = Query(None, description="Spezifischer Server für Revalidierung"),
    _: str = Depends(require_auth)
) -> DomainRevalidationResult:
    """Führt sofortige Domain-Revalidierung durch.

    Dieser Endpunkt startet eine manuelle Domain-Revalidierung für alle
    oder einen spezifischen registrierten MCP-Server.

    **Authentifizierung:** Bearer Token erforderlich
    """
    import time

    try:
        start_time = time.time()

        if server_name:
            # Revalidierung für spezifischen Server
            logger.info(f"Starte manuelle Domain-Revalidierung für Server: {server_name}")

            # Prüfe ob Server existiert
            status_data = mcp_registry.get_domain_validation_status(server_name)
            if not status_data["exists"]:
                raise HTTPException(
                    status_code=404,
                    detail=f"Server {server_name} nicht gefunden"
                )

            # Führe Revalidierung durch
            results = await mcp_registry.revalidate_domains_if_needed(0)  # Force revalidation

            # Filtere Ergebnis für spezifischen Server
            server_result = {server_name: results.get(server_name, False)}

        else:
            # Revalidierung für alle Server
            logger.info("Starte manuelle Domain-Revalidierung für alle Server")
            server_result = await _domain_service.force_revalidation()

        duration = time.time() - start_time

        # Statistiken berechnen
        total_servers = len(server_result)
        successful_validations = sum(1 for success in server_result.values() if success)
        failed_validations = total_servers - successful_validations

        logger.info(f"Manuelle Domain-Revalidierung abgeschlossen: "
                   f"{successful_validations}/{total_servers} erfolgreich "
                   f"in {duration:.2f}s")

        return DomainRevalidationResult(
            total_servers=total_servers,
            successful_validations=successful_validations,
            failed_validations=failed_validations,
            results=server_result,
            duration_seconds=duration
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Fehler bei manueller Domain-Revalidierung: {e}")
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler bei Domain-Revalidierung"
        )


@router.post("/domain-validation/config/reload")
@trace_function("api.domain_validation.reload_config")
async def reload_domain_validation_config_endpoint(
    _: str = Depends(require_auth)
) -> dict[str, Any]:
    """Lädt Domain-Validierung-Konfiguration neu.

    Dieser Endpunkt lädt die Domain-Validierung-Konfiguration aus den
    Umgebungsvariablen neu, ohne einen Server-Neustart zu erfordern.

    **Authentifizierung:** Bearer Token erforderlich
    """
    try:
        logger.info("Starte manuellen Config-Reload für Domain-Validierung")

        old_config = get_domain_validation_config()
        new_config = reload_domain_validation_config()

        # Prüfe auf Änderungen
        changes = {}
        if old_config.allowed_domains != new_config.allowed_domains:
            changes["allowed_domains"] = {
                "old_count": len(old_config.allowed_domains),
                "new_count": len(new_config.allowed_domains)
            }

        if old_config.enable_periodic_revalidation != new_config.enable_periodic_revalidation:
            changes["periodic_revalidation"] = {
                "old": old_config.enable_periodic_revalidation,
                "new": new_config.enable_periodic_revalidation
            }

        if old_config.revalidation_interval_hours != new_config.revalidation_interval_hours:
            changes["revalidation_interval_hours"] = {
                "old": old_config.revalidation_interval_hours,
                "new": new_config.revalidation_interval_hours
            }

        logger.info(f"Config-Reload abgeschlossen. Änderungen: {len(changes)}")

        return {
            "success": True,
            "message": "Domain-Validierung-Konfiguration erfolgreich neu geladen",
            "changes": changes,
            "reload_time": time.time()
        }

    except Exception as e:
        logger.exception(f"Fehler beim Config-Reload: {e}")
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler beim Config-Reload"
        )
