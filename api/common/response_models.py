"""Standardisierte Response-Models für einheitliche API-Antworten.

Eliminiert Inkonsistenzen bei Response-Formaten und stellt
wiederverwendbare Response-Strukturen bereit.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from .pagination import PaginationMetaModel, calculate_pagination_meta

# ============================================================================
# GENERIC TYPE VARIABLES
# ============================================================================

T = TypeVar("T")


# ============================================================================
# BASE RESPONSE MODELS
# ============================================================================

class StandardResponse[T](BaseModel):
    """Basis-Response-Model für alle API-Antworten."""

    success: bool = Field(..., description="Erfolgs-Status der Operation")
    data: T | None = Field(None, description="Response-Daten")
    message: str | None = Field(None, description="Optionale Nachricht")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Zeitstempel der Response"
    )
    correlation_id: str | None = Field(None, description="Korrelations-ID für Request-Tracking")


class SuccessResponse(StandardResponse[T]):
    """Response-Model für erfolgreiche Operationen."""

    success: bool = Field(True, description="Erfolgs-Status (immer True)")

    def __init__(self, data: T, message: str | None = None, **kwargs):
        """Initialisiert Success-Response.

        Args:
            data: Response-Daten
            message: Optionale Erfolgs-Nachricht
            **kwargs: Zusätzliche Felder
        """
        super().__init__(data=data, message=message, **kwargs)


class ErrorResponse(StandardResponse[None]):
    """Response-Model für Fehler-Antworten."""

    success: bool = Field(False, description="Erfolgs-Status (immer False)")
    error_code: str = Field(..., description="Interner Fehlercode")
    error_details: dict[str, Any] | None = Field(None, description="Zusätzliche Fehlerdetails")

    def __init__(
        self,
        error_code: str,
        message: str,
        error_details: dict[str, Any] | None = None,
        **kwargs
    ):
        """Initialisiert Error-Response.

        Args:
            error_code: Interner Fehlercode
            message: Fehlermeldung
            error_details: Zusätzliche Fehlerdetails
            **kwargs: Zusätzliche Felder
        """
        super().__init__(
            message=message,
            error_code=error_code,
            error_details=error_details,
            **kwargs
        )


# ============================================================================
# SPECIALIZED RESPONSE MODELS
# ============================================================================

class PaginatedResponse(StandardResponse[list[T]]):
    """Response-Model für paginierte Daten."""

    pagination: PaginationMetaModel = Field(..., description="Pagination-Metadaten")

    def __init__(
        self,
        items: list[T],
        page: int,
        page_size: int,
        total_items: int,
        message: str | None = None,
        **kwargs
    ):
        """Initialisiert paginierte Response.

        Args:
            items: Liste der Items für aktuelle Seite
            page: Aktuelle Seite
            page_size: Items pro Seite
            total_items: Gesamtanzahl Items
            message: Optionale Nachricht
            **kwargs: Zusätzliche Felder
        """
        # Verwende die gemeinsame Pagination-Utility
        pagination_meta = calculate_pagination_meta(page, page_size, total_items)

        # Konvertiere zu Pydantic-Model
        pagination = PaginationMetaModel(
            page=pagination_meta.page,
            page_size=pagination_meta.page_size,
            total_items=pagination_meta.total_items,
            total_pages=pagination_meta.total_pages,
            has_next=pagination_meta.has_next,
            has_previous=pagination_meta.has_previous
        )

        super().__init__(
            data=items,
            message=message,
            pagination=pagination,
            **kwargs
        )


class HealthResponse(StandardResponse[dict[str, Any]]):
    """Response-Model für Health-Check-Endpoints."""

    status: str = Field(..., description="Health-Status (healthy/unhealthy/degraded)")
    version: str | None = Field(None, description="Service-Version")
    uptime: float | None = Field(None, description="Uptime in Sekunden")
    dependencies: dict[str, str] | None = Field(None, description="Status der Abhängigkeiten")


class MetricsResponse(StandardResponse[dict[str, Any]]):
    """Response-Model für Metriken-Endpoints."""

    metrics: dict[str, int | float | str] = Field(..., description="Metriken-Daten")
    collection_time: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Zeitpunkt der Metriken-Sammlung"
    )


# ============================================================================
# RESPONSE FACTORY FUNCTIONS
# ============================================================================

def create_success_response[T](
    data: T,
    message: str | None = None,
    correlation_id: str | None = None
) -> SuccessResponse[T]:
    """Erstellt standardisierte Success-Response.

    Args:
        data: Response-Daten
        message: Optionale Erfolgs-Nachricht
        correlation_id: Korrelations-ID

    Returns:
        Success-Response-Objekt
    """
    return SuccessResponse(
        data=data,
        message=message,
        correlation_id=correlation_id
    )


def create_error_response(
    error_code: str,
    message: str,
    error_details: dict[str, Any] | None = None,
    correlation_id: str | None = None
) -> ErrorResponse:
    """Erstellt standardisierte Error-Response.

    Args:
        error_code: Interner Fehlercode
        message: Fehlermeldung
        error_details: Zusätzliche Fehlerdetails
        correlation_id: Korrelations-ID

    Returns:
        Error-Response-Objekt
    """
    return ErrorResponse(
        error_code=error_code,
        message=message,
        error_details=error_details,
        correlation_id=correlation_id
    )


def create_paginated_response[T](
    items: list[T],
    page: int,
    page_size: int,
    total_items: int,
    message: str | None = None,
    correlation_id: str | None = None
) -> PaginatedResponse[T]:
    """Erstellt paginierte Response.

    Args:
        items: Liste der Items
        page: Aktuelle Seite
        page_size: Items pro Seite
        total_items: Gesamtanzahl Items
        message: Optionale Nachricht
        correlation_id: Korrelations-ID

    Returns:
        Paginierte Response
    """
    return PaginatedResponse(
        items=items,
        page=page,
        page_size=page_size,
        total_items=total_items,
        message=message,
        correlation_id=correlation_id
    )


def create_health_response(
    status: str,
    data: dict[str, Any] | None = None,
    version: str | None = None,
    uptime: float | None = None,
    dependencies: dict[str, str] | None = None,
    correlation_id: str | None = None
) -> HealthResponse:
    """Erstellt Health-Check-Response.

    Args:
        status: Health-Status
        data: Zusätzliche Health-Daten
        version: Service-Version
        uptime: Uptime in Sekunden
        dependencies: Status der Abhängigkeiten
        correlation_id: Korrelations-ID

    Returns:
        Health-Response-Objekt
    """
    return HealthResponse(
        success=True,
        data=data or {},
        status=status,
        version=version,
        uptime=uptime,
        dependencies=dependencies,
        correlation_id=correlation_id
    )


def create_metrics_response(
    metrics: dict[str, int | float | str],
    correlation_id: str | None = None
) -> MetricsResponse:
    """Erstellt Metriken-Response.

    Args:
        metrics: Metriken-Daten
        correlation_id: Korrelations-ID

    Returns:
        Metriken-Response-Objekt
    """
    return MetricsResponse(
        success=True,
        data=metrics,
        metrics=metrics,
        correlation_id=correlation_id
    )
