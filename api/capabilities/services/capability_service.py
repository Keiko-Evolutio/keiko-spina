"""Service für API-Capabilities-Management.

Extrahiert Business-Logic aus der monolithischen capabilities.py
und stellt testbare, injectable Service-Komponente bereit.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from kei_logging import get_logger

from ..models.capability_models import (
    APICapability,
    CapabilitiesResponse,
    CapabilityCategory,
    CapabilityResponse,
    ServerInfo,
)
from ..utils.constants import (
    DEFAULT_API_VERSION,
    DEFAULT_CAPABILITIES,
    DEFAULT_SERVER_CONFIG,
    ENV_VARS,
    LOG_MESSAGES,
    SUPPORTED_API_VERSIONS,
)
from ..utils.converters import model_converter, response_converter
from ..utils.validators import capability_validator

if TYPE_CHECKING:
    from .feature_flag_service import FeatureFlagService


class CapabilityService:
    """Service für API-Capabilities-Management.

    Eliminiert globale Variablen und stellt testbare,
    injectable Service-Komponente bereit.
    """

    def __init__(self, feature_flag_service: FeatureFlagService) -> None:
        """Initialisiert Capability-Service.

        Args:
            feature_flag_service: Feature-Flag-Service für Abhängigkeiten
        """
        self.logger = get_logger(self.__class__.__name__)
        self.feature_flag_service = feature_flag_service
        self._capabilities: dict[str, APICapability] = {}
        self._load_default_capabilities()

    def _load_default_capabilities(self) -> None:
        """Lädt Standard-Capabilities aus Konfiguration."""
        default_capabilities = model_converter.dicts_to_capabilities(DEFAULT_CAPABILITIES)

        for capability in default_capabilities:
            self._capabilities[capability.name] = capability

        self.logger.info(
            LOG_MESSAGES["loaded_capabilities"].format(count=len(default_capabilities))
        )

    def _get_enabled_feature_names(
        self,
        client_id: str | None = None,
        server_name: str | None = None
    ) -> list[str]:
        """Holt aktivierte Feature-Namen (eliminiert Code-Duplikation).

        Args:
            client_id: Client-ID (optional)
            server_name: Server-Name (optional)

        Returns:
            Liste aktivierter Feature-Namen
        """
        return [
            f.name for f in self.feature_flag_service.get_enabled_features(
                client_id, server_name
            )
        ]

    def get_available_capabilities(
        self,
        client_id: str | None = None,
        server_name: str | None = None
    ) -> list[CapabilityResponse]:
        """Gibt verfügbare Capabilities zurück.

        Args:
            client_id: Client-ID (optional)
            server_name: Server-Name (optional)

        Returns:
            Liste verfügbarer Capability-Responses
        """
        available_capabilities = []
        enabled_features = self._get_enabled_feature_names(client_id, server_name)
        available_capability_names = list(self._capabilities.keys())

        for capability in self._capabilities.values():
            available = capability_validator.validate_capability_availability(
                capability, enabled_features, available_capability_names
            )

            available_capabilities.append(
                response_converter.capability_to_response(capability, available)
            )

        return available_capabilities

    def get_capabilities_response(
        self,
        client_id: str | None = None,
        server_name: str | None = None,
        api_version: str | None = None
    ) -> CapabilitiesResponse:
        """Erstellt vollständige Capabilities-Response.

        Args:
            client_id: Client-ID (optional)
            server_name: Server-Name (optional)
            api_version: API-Version (optional)

        Returns:
            Vollständige CapabilitiesResponse
        """
        # Verfügbare Capabilities abrufen
        capabilities = self.get_available_capabilities(client_id, server_name)

        # Aktivierte Feature-Flags abrufen
        feature_flags = self.feature_flag_service.get_enabled_features(
            client_id, server_name
        )

        # Server-Informationen aus Konfiguration
        server_info = self._create_server_info()

        return CapabilitiesResponse(
            api_version=api_version or DEFAULT_API_VERSION,
            supported_versions=SUPPORTED_API_VERSIONS,
            capabilities=capabilities,
            feature_flags=feature_flags,
            server_info=server_info,
            timestamp=datetime.now(UTC).isoformat()
        )

    def _create_server_info(self) -> ServerInfo:
        """Erstellt Server-Info aus Konfiguration.

        Returns:
            ServerInfo-Objekt
        """
        return ServerInfo(
            name=DEFAULT_SERVER_CONFIG["name"],
            environment=os.getenv(ENV_VARS["environment"], DEFAULT_SERVER_CONFIG["environment"]),
            region=os.getenv(ENV_VARS["region"], DEFAULT_SERVER_CONFIG["region"]),
            instance_id=os.getenv(ENV_VARS["instance_id"], DEFAULT_SERVER_CONFIG["instance_id"])
        )

    def is_capability_available(
        self,
        capability_name: str,
        client_id: str | None = None,
        server_name: str | None = None
    ) -> bool:
        """Prüft ob Capability verfügbar ist.

        Args:
            capability_name: Name der Capability
            client_id: Client-ID (optional)
            server_name: Server-Name (optional)

        Returns:
            True wenn Capability verfügbar ist
        """
        if capability_name not in self._capabilities:
            return False

        capability = self._capabilities[capability_name]
        enabled_features = self._get_enabled_feature_names(client_id, server_name)
        available_capability_names = list(self._capabilities.keys())

        return capability_validator.validate_capability_availability(
            capability, enabled_features, available_capability_names
        )

    def add_capability(self, capability: APICapability) -> bool:
        """Fügt neue Capability hinzu.

        Args:
            capability: APICapability-Objekt

        Returns:
            True wenn erfolgreich hinzugefügt
        """
        if capability.name in self._capabilities:
            self.logger.warning(f"Capability '{capability.name}' already exists")
            return False

        self._capabilities[capability.name] = capability
        self.logger.info(f"Added capability: {capability.name}")
        return True

    def remove_capability(self, capability_name: str) -> bool:
        """Entfernt Capability.

        Args:
            capability_name: Name der Capability

        Returns:
            True wenn erfolgreich entfernt
        """
        if capability_name not in self._capabilities:
            self.logger.warning(f"Cannot remove unknown capability: {capability_name}")
            return False

        del self._capabilities[capability_name]
        self.logger.info(f"Removed capability: {capability_name}")
        return True

    def get_capability(self, capability_name: str) -> APICapability | None:
        """Gibt Capability zurück.

        Args:
            capability_name: Name der Capability

        Returns:
            APICapability-Objekt oder None
        """
        return self._capabilities.get(capability_name)

    def get_capability_names(self) -> list[str]:
        """Gibt alle Capability-Namen zurück.

        Returns:
            Liste aller Capability-Namen
        """
        return list(self._capabilities.keys())

    def get_capabilities_by_category(
        self,
        category: CapabilityCategory
    ) -> list[APICapability]:
        """Gibt Capabilities nach Kategorie zurück.

        Args:
            category: Capability-Kategorie

        Returns:
            Liste der Capabilities in der Kategorie
        """
        return [
            capability for capability in self._capabilities.values()
            if capability.category == category
        ]

    def get_stats(self) -> dict[str, int]:
        """Gibt Statistiken über Capabilities zurück.

        Returns:
            Dictionary mit Statistiken
        """
        stats = {
            "total_capabilities": len(self._capabilities),
            "core": 0,
            "tools": 0,
            "resources": 0,
            "prompts": 0,
            "monitoring": 0,
            "security": 0,
            "experimental": 0
        }

        for capability in self._capabilities.values():
            category_key = capability.category.value
            if category_key in stats:
                stats[category_key] += 1

        return stats
