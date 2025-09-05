"""Rate Limiter Interfaces und Typen für DI.

Definiert das Backend-Interface und das Ergebnis-Datenmodell für
Rate Limiting, getrennt von konkreten Implementierungen.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._base import UtilityService

if TYPE_CHECKING:
    from config.unified_rate_limiting import RateLimitPolicy

    from ._types import OperationResult, OptionalTimeout, ServiceResult


@dataclass(frozen=True)
class RateLimitResult:
    """Ergebnis einer Rate Limit Prüfung.

    Immutable Dataclass für Thread-Safety und bessere Performance.
    """

    allowed: bool
    remaining: int
    reset_time: int
    retry_after: int | None = None

    # Zusätzliche Informationen
    current_usage: int = 0
    limit: int = 0
    window_start: float = 0.0

    # Burst-Informationen (Token Bucket)
    tokens_remaining: float | None = None
    bucket_capacity: int | None = None

    # Soft-Limit-Warnung
    soft_limit_exceeded: bool = False


class RateLimiterService(UtilityService):
    """Abstraktes Backend-Interface für Rate Limiting.

    Utility-Service für Request-Rate-Limiting und Traffic-Kontrolle.
    """

    @abstractmethod
    async def check_rate_limit(
        self,
        key: str,
        policy: RateLimitPolicy,
        current_time: OptionalTimeout = None,
    ) -> RateLimitResult:
        """Prüft Rate Limit für gegebenen Key.

        Args:
            key: Eindeutiger Schlüssel für das Rate Limit.
            policy: Rate Limit Policy mit Parametern.
            current_time: Optionaler Zeitstempel.

        Returns:
            Ergebnis der Rate-Limit-Prüfung.

        Raises:
            ValueError: Bei ungültigen Parametern.
            RuntimeError: Bei Backend-Fehlern.
        """

    @abstractmethod
    async def reset_rate_limit(self, key: str) -> OperationResult:
        """Setzt Rate Limit für Key zurück.

        Args:
            key: Schlüssel des zurückzusetzenden Rate Limits.

        Returns:
            True bei erfolgreichem Reset.

        Raises:
            ValueError: Bei ungültigem Key.
        """

    @abstractmethod
    async def get_rate_limit_info(self, key: str) -> ServiceResult:
        """Gibt Rate Limit Informationen für Key zurück.

        Args:
            key: Schlüssel für die Informationsabfrage.

        Returns:
            Rate Limit Informationen oder leeres Dict wenn nicht gefunden.

        Raises:
            ValueError: Bei ungültigem Key.
        """

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Bereinigt abgelaufene Einträge und liefert Anzahl gelöschter Keys.

        Returns:
            Anzahl der gelöschten Einträge.

        Raises:
            RuntimeError: Bei Backend-Fehlern.
        """

    @abstractmethod
    async def get_statistics(self) -> ServiceResult:
        """Liefert Rate-Limiting-Statistiken.

        Returns:
            Statistiken über Rate-Limiting-Aktivität.
        """


# Backward compatibility alias
RateLimiterBackend = RateLimiterService
