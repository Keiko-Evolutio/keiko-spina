"""Factory-Funktionen für Capabilities-Modul.

Konsolidiert duplizierte Factory-Patterns und stellt einheitliche
Erstellungs-Interfaces bereit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic

from ..models.capability_models import APICapability, CapabilityCategory
from ..models.feature_flag_models import FeatureFlag, FeatureScope, FeatureStatus
from .constants import (
    DEFAULT_API_VERSION,
)

T = TypeVar("T")


class BaseFactory(Generic[T], ABC):
    """Abstrakte Basis-Factory für einheitliche Factory-Patterns.

    Eliminiert Code-Duplikation zwischen verschiedenen Factory-Implementierungen.
    """

    @abstractmethod
    def create(self, **kwargs) -> T:
        """Erstellt Objekt mit gegebenen Parametern.

        Args:
            **kwargs: Erstellungsparameter

        Returns:
            Erstelltes Objekt
        """

    def _validate_required_params(self, params: dict[str, Any], required: list[str]) -> None:
        """Validiert erforderliche Parameter.

        Args:
            params: Parameter-Dictionary
            required: Liste erforderlicher Parameter-Namen

        Raises:
            ValueError: Wenn erforderliche Parameter fehlen
        """
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

    def _apply_defaults(self, params: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
        """Wendet Standard-Werte auf Parameter an.

        Args:
            params: Parameter-Dictionary
            defaults: Standard-Werte

        Returns:
            Parameter mit angewendeten Standard-Werten
        """
        result = defaults.copy()
        result.update(params)
        return result


class FeatureFlagFactory(BaseFactory[FeatureFlag]):
    """Factory für FeatureFlag-Erstellung.

    Konsolidiert alle FeatureFlag-Erstellungslogik und eliminiert
    Code-Duplikation zwischen verschiedenen Factory-Funktionen.
    """

    def create(
        self,
        name: str,
        description: str,
        status: FeatureStatus | None = None,
        scope: FeatureScope | None = None,
        enabled_by_default: bool = False,
        **kwargs
    ) -> FeatureFlag:
        """Erstellt FeatureFlag mit gegebenen Parametern.

        Args:
            name: Feature-Name
            description: Feature-Beschreibung
            status: Feature-Status (Standard: EXPERIMENTAL)
            scope: Feature-Scope (Standard: GLOBAL)
            enabled_by_default: Standard-Aktivierungsstatus
            **kwargs: Zusätzliche Parameter

        Returns:
            FeatureFlag-Objekt
        """
        # Validierung
        self._validate_required_params(
            {"name": name, "description": description},
            ["name", "description"]
        )

        # Standard-Werte anwenden
        defaults = {
            "status": status or FeatureStatus.EXPERIMENTAL,
            "scope": scope or FeatureScope.GLOBAL,
            "enabled_by_default": enabled_by_default,
            "introduced_in_version": DEFAULT_API_VERSION
        }

        params = self._apply_defaults(kwargs, defaults)

        return FeatureFlag(
            name=name,
            description=description,
            status=params["status"],
            scope=params["scope"],
            enabled_by_default=params["enabled_by_default"],
            enabled_for_clients=params.get("enabled_for_clients", set()),
            enabled_for_servers=params.get("enabled_for_servers", set()),
            introduced_in_version=params["introduced_in_version"],
            deprecated_in_version=params.get("deprecated_in_version"),
            removal_planned_version=params.get("removal_planned_version"),
            dependencies=params.get("dependencies", []),
            metadata=params.get("metadata", {})
        )

    def create_global_feature(
        self,
        name: str,
        description: str,
        enabled: bool = False
    ) -> FeatureFlag:
        """Erstellt globales Feature-Flag.

        Args:
            name: Feature-Name
            description: Feature-Beschreibung
            enabled: Ob Feature aktiviert ist

        Returns:
            Globales FeatureFlag-Objekt
        """
        return self.create(
            name=name,
            description=description,
            status=FeatureStatus.STABLE,
            scope=FeatureScope.GLOBAL,
            enabled_by_default=enabled
        )

    def create_client_feature(
        self,
        name: str,
        description: str,
        enabled_clients: list[str] | None = None
    ) -> FeatureFlag:
        """Erstellt client-spezifisches Feature-Flag.

        Args:
            name: Feature-Name
            description: Feature-Beschreibung
            enabled_clients: Liste aktivierter Clients

        Returns:
            Client-spezifisches FeatureFlag-Objekt
        """
        feature = self.create(
            name=name,
            description=description,
            status=FeatureStatus.STABLE,
            scope=FeatureScope.PER_CLIENT,
            enabled_by_default=False
        )

        if enabled_clients:
            feature.enabled_for_clients.update(enabled_clients)

        return feature


class CapabilityFactory(BaseFactory[APICapability]):
    """Factory für APICapability-Erstellung.

    Konsolidiert alle APICapability-Erstellungslogik und eliminiert
    Code-Duplikation zwischen verschiedenen Factory-Funktionen.
    """

    def create(
        self,
        name: str,
        description: str,
        category: CapabilityCategory,
        endpoints: list[str],
        version: str | None = None,
        **kwargs
    ) -> APICapability:
        """Erstellt APICapability mit gegebenen Parametern.

        Args:
            name: Capability-Name
            description: Capability-Beschreibung
            category: Capability-Kategorie
            endpoints: Liste der Endpoints
            version: Capability-Version (Standard: DEFAULT_API_VERSION)
            **kwargs: Zusätzliche Parameter

        Returns:
            APICapability-Objekt
        """
        # Validierung
        self._validate_required_params(
            {"name": name, "description": description, "endpoints": endpoints},
            ["name", "description", "endpoints"]
        )

        # Standard-Werte anwenden
        defaults = {
            "version": version or DEFAULT_API_VERSION,
            "feature_flags": [],
            "requirements": [],
            "documentation_url": None,
            "examples": []
        }

        params = self._apply_defaults(kwargs, defaults)

        return APICapability(
            name=name,
            description=description,
            category=category,
            version=params["version"],
            endpoints=endpoints,
            feature_flags=params["feature_flags"],
            requirements=params["requirements"],
            documentation_url=params["documentation_url"],
            examples=params["examples"]
        )

    def create_core_capability(
        self,
        name: str,
        description: str,
        endpoints: list[str],
        version: str | None = None
    ) -> APICapability:
        """Erstellt Core-Capability.

        Args:
            name: Capability-Name
            description: Capability-Beschreibung
            endpoints: Liste der Endpoints
            version: Capability-Version

        Returns:
            Core-APICapability-Objekt
        """
        return self.create(
            name=name,
            description=description,
            category=CapabilityCategory.CORE,
            endpoints=endpoints,
            version=version
        )

    def create_experimental_capability(
        self,
        name: str,
        description: str,
        endpoints: list[str],
        feature_flags: list[str] | None = None,
        version: str | None = None
    ) -> APICapability:
        """Erstellt experimentelle Capability.

        Args:
            name: Capability-Name
            description: Capability-Beschreibung
            endpoints: Liste der Endpoints
            feature_flags: Erforderliche Feature-Flags
            version: Capability-Version

        Returns:
            Experimentelle APICapability-Objekt
        """
        return self.create(
            name=name,
            description=description,
            category=CapabilityCategory.EXPERIMENTAL,
            endpoints=endpoints,
            version=version,
            feature_flags=feature_flags or []
        )


# ============================================================================
# SINGLETON FACTORY INSTANCES
# ============================================================================

# Globale Factory-Instanzen für einfache Verwendung
feature_flag_factory = FeatureFlagFactory()
capability_factory = CapabilityFactory()
