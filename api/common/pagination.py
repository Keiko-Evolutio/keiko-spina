"""Gemeinsame Pagination-Utilities für API-Endpoints.

Eliminiert duplizierte Pagination-Logik und stellt wiederverwendbare
Pagination-Funktionen für alle API-Komponenten bereit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, Field

from .constants import (
    DEFAULT_PAGE,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    MIN_PAGE_SIZE,
    PAGINATION_LIMITS,
)
from .error_handlers import ValidationError

T = TypeVar("T")


# ============================================================================
# PAGINATION MODELS
# ============================================================================

@dataclass(frozen=True)
class PaginationParams:
    """Immutable Pagination-Parameter."""

    page: int
    page_size: int
    max_page_size: int

    def __post_init__(self) -> None:
        """Validiert Pagination-Parameter nach Initialisierung."""
        if self.page < 1:
            raise ValidationError(
                "Seitennummer muss mindestens 1 sein",
                field="page",
                value=self.page
            )

        if self.page_size < MIN_PAGE_SIZE:
            raise ValidationError(
                f"Seitengröße muss mindestens {MIN_PAGE_SIZE} sein",
                field="page_size",
                value=self.page_size
            )

        if self.page_size > self.max_page_size:
            raise ValidationError(
                f"Seitengröße darf maximal {self.max_page_size} sein",
                field="page_size",
                value=self.page_size
            )

    @property
    def offset(self) -> int:
        """Berechnet Offset für Datenbankabfragen."""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Alias für page_size für Datenbankabfragen."""
        return self.page_size


@dataclass(frozen=True)
class PaginationMeta:
    """Immutable Pagination-Metadaten für Responses."""

    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool

    @classmethod
    def calculate(
        cls,
        page: int,
        page_size: int,
        total_items: int
    ) -> PaginationMeta:
        """Berechnet Pagination-Metadaten.

        Args:
            page: Aktuelle Seite
            page_size: Items pro Seite
            total_items: Gesamtanzahl Items

        Returns:
            Berechnete Pagination-Metadaten
        """
        total_pages = max(1, (total_items + page_size - 1) // page_size)
        has_next = page < total_pages
        has_previous = page > 1

        return cls(
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
            has_next=has_next,
            has_previous=has_previous
        )


class PaginationMetaModel(BaseModel):
    """Pydantic-Model für Pagination-Metadaten in API-Responses."""

    page: int = Field(..., ge=1, description="Aktuelle Seite")
    page_size: int = Field(..., ge=1, description="Items pro Seite")
    total_items: int = Field(..., ge=0, description="Gesamtanzahl Items")
    total_pages: int = Field(..., ge=1, description="Gesamtanzahl Seiten")
    has_next: bool = Field(..., description="Weitere Seiten verfügbar")
    has_previous: bool = Field(..., description="Vorherige Seiten verfügbar")


# ============================================================================
# PAGINATION CALCULATOR
# ============================================================================

class PaginationCalculator:
    """Zentrale Klasse für Pagination-Berechnungen."""

    def __init__(self, default_max_page_size: int = MAX_PAGE_SIZE) -> None:
        """Initialisiert Pagination-Calculator.

        Args:
            default_max_page_size: Standard-Maximum für Seitengröße
        """
        self.default_max_page_size = default_max_page_size

    def validate_params(
        self,
        page: int | None = None,
        page_size: int | None = None,
        endpoint_type: str = "default"
    ) -> PaginationParams:
        """Validiert und normalisiert Pagination-Parameter.

        Args:
            page: Seitennummer (1-basiert)
            page_size: Anzahl Items pro Seite
            endpoint_type: Typ des Endpoints für spezifische Limits

        Returns:
            Validierte Pagination-Parameter

        Raises:
            ValidationError: Bei ungültigen Parametern
        """
        # Standard-Werte setzen
        normalized_page = page if page is not None else DEFAULT_PAGE
        normalized_page_size = page_size if page_size is not None else DEFAULT_PAGE_SIZE

        # Endpoint-spezifisches Maximum bestimmen
        max_page_size = PAGINATION_LIMITS.get(endpoint_type, self.default_max_page_size)

        return PaginationParams(
            page=normalized_page,
            page_size=normalized_page_size,
            max_page_size=max_page_size
        )

    def calculate_meta(
        self,
        params: PaginationParams,
        total_items: int
    ) -> PaginationMeta:
        """Berechnet Pagination-Metadaten.

        Args:
            params: Validierte Pagination-Parameter
            total_items: Gesamtanzahl Items

        Returns:
            Berechnete Pagination-Metadaten
        """
        return PaginationMeta.calculate(
            page=params.page,
            page_size=params.page_size,
            total_items=total_items
        )

    def paginate_list(
        self,
        items: list[T],
        page: int | None = None,
        page_size: int | None = None,
        endpoint_type: str = "default"
    ) -> tuple[list[T], PaginationMeta]:
        """Paginiert eine Liste von Items.

        Args:
            items: Liste der zu paginierenden Items
            page: Seitennummer
            page_size: Items pro Seite
            endpoint_type: Endpoint-Typ für Limits

        Returns:
            Tuple von (paginierte_items, pagination_meta)
        """
        params = self.validate_params(page, page_size, endpoint_type)
        total_items = len(items)

        # Items für aktuelle Seite extrahieren
        start_index = params.offset
        end_index = start_index + params.page_size
        paginated_items = items[start_index:end_index]

        # Metadaten berechnen
        meta = self.calculate_meta(params, total_items)

        return paginated_items, meta


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Globaler Calculator für einfache Verwendung
_default_calculator = PaginationCalculator()


def validate_pagination_params(
    page: int | None = None,
    page_size: int | None = None,
    max_page_size: int = MAX_PAGE_SIZE
) -> tuple[int, int]:
    """Validiert Pagination-Parameter (Legacy-Kompatibilität).

    Args:
        page: Seitennummer (1-basiert)
        page_size: Anzahl Items pro Seite
        max_page_size: Maximale Seitengröße

    Returns:
        Tuple von (page, page_size)

    Raises:
        ValidationError: Bei ungültigen Parametern
    """
    calculator = PaginationCalculator(max_page_size)
    params = calculator.validate_params(page, page_size)
    return params.page, params.page_size


def calculate_pagination_meta(
    page: int,
    page_size: int,
    total_items: int
) -> PaginationMeta:
    """Berechnet Pagination-Metadaten (Convenience-Funktion).

    Args:
        page: Aktuelle Seite
        page_size: Items pro Seite
        total_items: Gesamtanzahl Items

    Returns:
        Berechnete Pagination-Metadaten
    """
    return PaginationMeta.calculate(page, page_size, total_items)


def paginate_items[T](
    items: list[T],
    page: int | None = None,
    page_size: int | None = None
) -> tuple[list[T], PaginationMeta]:
    """Paginiert eine Liste von Items (Convenience-Funktion).

    Args:
        items: Liste der Items
        page: Seitennummer
        page_size: Items pro Seite

    Returns:
        Tuple von (paginierte_items, pagination_meta)
    """
    return _default_calculator.paginate_list(items, page, page_size)
