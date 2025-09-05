"""Validierungs-Utilities für Capabilities-Modul.

Konsolidiert wiederkehrende Validierungslogik und eliminiert Code-Duplikation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ..models.feature_flag_models import FeatureFlag, FeatureStatus
from .constants import VALIDATION_RULES

if TYPE_CHECKING:
    from ..models.capability_models import APICapability


class BaseValidator(ABC):
    """Abstrakte Basis-Klasse für Validatoren.

    Stellt gemeinsame Validierungslogik bereit und eliminiert Code-Duplikation.
    """

    def __init__(self) -> None:
        """Initialisiert Validator."""
        self.validation_rules = VALIDATION_RULES

    @abstractmethod
    def validate(self, obj: Any) -> bool:
        """Validiert Objekt.

        Args:
            obj: Zu validierendes Objekt

        Returns:
            True wenn Objekt valide ist

        Raises:
            ValueError: Bei Validierungsfehlern
        """

    def _validate_string_length(
        self,
        value: str,
        field_name: str,
        max_length: int
    ) -> None:
        """Validiert String-Länge.

        Args:
            value: String-Wert
            field_name: Feld-Name für Fehlermeldung
            max_length: Maximale Länge

        Raises:
            ValueError: Wenn String zu lang ist
        """
        if len(value) > max_length:
            raise ValueError(
                f"{field_name} exceeds maximum length of {max_length} characters"
            )

    def _validate_list_length(
        self,
        value: list[Any],
        field_name: str,
        max_length: int
    ) -> None:
        """Validiert Listen-Länge.

        Args:
            value: Listen-Wert
            field_name: Feld-Name für Fehlermeldung
            max_length: Maximale Länge

        Raises:
            ValueError: Wenn Liste zu lang ist
        """
        if len(value) > max_length:
            raise ValueError(
                f"{field_name} exceeds maximum length of {max_length} items"
            )

    def _validate_not_empty(self, value: str, field_name: str) -> None:
        """Validiert dass String nicht leer ist.

        Args:
            value: String-Wert
            field_name: Feld-Name für Fehlermeldung

        Raises:
            ValueError: Wenn String leer ist
        """
        if not value or not value.strip():
            raise ValueError(f"{field_name} cannot be empty")


class FeatureValidator(BaseValidator):
    """Validator für FeatureFlag-Objekte.

    Konsolidiert alle FeatureFlag-Validierungslogik.
    """

    def validate(self, feature: FeatureFlag) -> bool:
        """Validiert FeatureFlag-Objekt.

        Args:
            feature: FeatureFlag-Objekt

        Returns:
            True wenn FeatureFlag valide ist

        Raises:
            ValueError: Bei Validierungsfehlern
        """
        self._validate_basic_fields(feature)
        self._validate_version_consistency(feature)
        self._validate_dependencies(feature)
        return True

    def _validate_basic_fields(self, feature: FeatureFlag) -> None:
        """Validiert Basis-Felder.

        Args:
            feature: FeatureFlag-Objekt

        Raises:
            ValueError: Bei Validierungsfehlern
        """
        # Name validieren
        self._validate_not_empty(feature.name, "Feature name")
        self._validate_string_length(
            feature.name,
            "Feature name",
            self.validation_rules["max_feature_name_length"]
        )

        # Beschreibung validieren
        self._validate_not_empty(feature.description, "Feature description")
        self._validate_string_length(
            feature.description,
            "Feature description",
            self.validation_rules["max_description_length"]
        )

    def _validate_version_consistency(self, feature: FeatureFlag) -> None:
        """Validiert Versions-Konsistenz.

        Args:
            feature: FeatureFlag-Objekt

        Raises:
            ValueError: Bei Versions-Inkonsistenzen
        """
        if feature.status == FeatureStatus.DEPRECATED and not feature.deprecated_in_version:
            raise ValueError(
                "Deprecated features must have deprecated_in_version set"
            )

        if feature.removal_planned_version and not feature.deprecated_in_version:
            raise ValueError(
                "Features with removal_planned_version must be deprecated first"
            )

    def _validate_dependencies(self, feature: FeatureFlag) -> None:
        """Validiert Feature-Abhängigkeiten.

        Args:
            feature: FeatureFlag-Objekt

        Raises:
            ValueError: Bei zirkulären Abhängigkeiten
        """
        if feature.name in feature.dependencies:
            raise ValueError("Feature cannot depend on itself")

    def validate_feature_availability(
        self,
        feature: FeatureFlag,
        client_id: str | None = None,
        server_name: str | None = None
    ) -> bool:
        """Validiert Feature-Verfügbarkeit für gegebenen Kontext.

        Args:
            feature: FeatureFlag-Objekt
            client_id: Client-ID (optional)
            server_name: Server-Name (optional)

        Returns:
            True wenn Feature verfügbar ist
        """
        if feature.status == FeatureStatus.DISABLED:
            return False

        # Scope-spezifische Validierung
        from ..models.feature_flag_models import FeatureScope

        if feature.scope == FeatureScope.PER_CLIENT:
            return feature.is_enabled_for_client(client_id)
        if feature.scope == FeatureScope.PER_SERVER:
            return feature.is_enabled_for_server(server_name)

        # Globale Features
        return feature.is_enabled_for_client(None)


class CapabilityValidator(BaseValidator):
    """Validator für APICapability-Objekte.

    Konsolidiert alle APICapability-Validierungslogik.
    """

    def validate(self, capability: APICapability) -> bool:
        """Validiert APICapability-Objekt.

        Args:
            capability: APICapability-Objekt

        Returns:
            True wenn APICapability valide ist

        Raises:
            ValueError: Bei Validierungsfehlern
        """
        self._validate_basic_fields(capability)
        self._validate_endpoints(capability)
        self._validate_dependencies(capability)
        return True

    def _validate_basic_fields(self, capability: APICapability) -> None:
        """Validiert Basis-Felder.

        Args:
            capability: APICapability-Objekt

        Raises:
            ValueError: Bei Validierungsfehlern
        """
        # Name validieren
        self._validate_not_empty(capability.name, "Capability name")
        self._validate_string_length(
            capability.name,
            "Capability name",
            self.validation_rules["max_feature_name_length"]
        )

        # Beschreibung validieren
        self._validate_not_empty(capability.description, "Capability description")
        self._validate_string_length(
            capability.description,
            "Capability description",
            self.validation_rules["max_description_length"]
        )

    def _validate_endpoints(self, capability: APICapability) -> None:
        """Validiert Endpoints.

        Args:
            capability: APICapability-Objekt

        Raises:
            ValueError: Bei Endpoint-Validierungsfehlern
        """
        if not capability.endpoints:
            raise ValueError("Capability must have at least one endpoint")

        self._validate_list_length(
            capability.endpoints,
            "Endpoints",
            self.validation_rules["max_endpoints_per_capability"]
        )

        # Validiere Endpoint-Format
        for endpoint in capability.endpoints:
            if not endpoint.strip():
                raise ValueError("Endpoints cannot be empty")

            # Basis-HTTP-Method-Validierung
            if not any(endpoint.startswith(method) for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]):
                raise ValueError(f"Invalid endpoint format: {endpoint}")

    def _validate_dependencies(self, capability: APICapability) -> None:
        """Validiert Capability-Abhängigkeiten.

        Args:
            capability: APICapability-Objekt

        Raises:
            ValueError: Bei Abhängigkeits-Validierungsfehlern
        """
        # Feature-Flags validieren
        self._validate_list_length(
            capability.feature_flags,
            "Feature flags",
            self.validation_rules["max_feature_flags_per_capability"]
        )

        # Requirements validieren
        self._validate_list_length(
            capability.requirements,
            "Requirements",
            self.validation_rules["max_requirements_per_capability"]
        )

        # Selbst-Abhängigkeit prüfen
        if capability.name in capability.requirements:
            raise ValueError("Capability cannot depend on itself")

    def validate_capability_availability(
        self,
        capability: APICapability,
        enabled_features: list[str],
        available_capabilities: list[str]
    ) -> bool:
        """Validiert Capability-Verfügbarkeit für gegebenen Kontext.

        Args:
            capability: APICapability-Objekt
            enabled_features: Liste aktivierter Features
            available_capabilities: Liste verfügbarer Capabilities

        Returns:
            True wenn Capability verfügbar ist
        """
        # Feature-Flag-Abhängigkeiten prüfen
        if not capability.has_required_features(enabled_features):
            return False

        # Capability-Abhängigkeiten prüfen
        return capability.has_required_capabilities(available_capabilities)


# ============================================================================
# SINGLETON VALIDATOR INSTANCES
# ============================================================================

# Globale Validator-Instanzen für einfache Verwendung
feature_validator = FeatureValidator()
capability_validator = CapabilityValidator()
