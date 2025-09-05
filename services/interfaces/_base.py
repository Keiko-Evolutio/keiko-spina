"""Gemeinsame ABC-Basis für Services mit Lifecycle und erweiterte Funktionalität."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class ServiceStatus(Enum):
    """Status-Enum für Service-Zustand."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class LifecycleService(ABC):
    """Abstrakte Basis für Services mit Lebenszyklus.

    Alle Services implementieren mindestens `initialize()` und `shutdown()`.
    Erweitert um Health-Checks und Status-Management.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialisiert den Service und stellt Betriebsbereitschaft her."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Führt einen geordneten Shutdown durch und gibt Ressourcen frei."""

    async def health_check(self) -> dict[str, Any]:
        """Führt Health-Check durch und liefert Status-Informationen.

        Returns:
            Dictionary mit Health-Status und Metadaten.
        """
        return {
            "status": "healthy",
            "service": self.__class__.__name__,
        }

    def get_status(self) -> ServiceStatus:
        """Liefert aktuellen Service-Status.

        Returns:
            Aktueller ServiceStatus.
        """
        return ServiceStatus.RUNNING


class CoreService(LifecycleService, ABC):
    """Basis für Kern-Services (Agent, Bus, Stream)."""


class InfrastructureService(LifecycleService, ABC):
    """Basis für Infrastructure-Services (ServiceManager, DomainRevalidation)."""


class FeatureService(LifecycleService, ABC):
    """Basis für Feature-Services (Voice, Webhook)."""


class UtilityService(LifecycleService, ABC):
    """Basis für Utility-Services (RateLimiter)."""
