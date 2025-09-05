"""Konvertierungs-Utilities für Capabilities-Modul.

Konsolidiert duplizierte Response-Model-Konvertierungslogik.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from ..models.capability_models import APICapability, CapabilityResponse
from ..models.feature_flag_models import FeatureFlag, FeatureFlagResponse

T = TypeVar("T")
R = TypeVar("R")


class BaseConverter(ABC, Generic[T, R]):
    """Abstrakte Basis-Klasse für Model-Konverter.

    Eliminiert Code-Duplikation zwischen verschiedenen Konvertierungs-Implementierungen.
    """

    @abstractmethod
    def convert(self, source: T, **kwargs) -> R:
        """Konvertiert Source-Objekt zu Target-Objekt.

        Args:
            source: Source-Objekt
            **kwargs: Zusätzliche Konvertierungsparameter

        Returns:
            Konvertiertes Target-Objekt
        """

    def convert_list(self, sources: list[T], **kwargs) -> list[R]:
        """Konvertiert Liste von Source-Objekten.

        Args:
            sources: Liste von Source-Objekten
            **kwargs: Zusätzliche Konvertierungsparameter

        Returns:
            Liste konvertierter Target-Objekte
        """
        return [self.convert(source, **kwargs) for source in sources]


class FeatureFlagConverter(BaseConverter[FeatureFlag, FeatureFlagResponse]):
    """Konverter für FeatureFlag zu FeatureFlagResponse.

    Konsolidiert die duplizierte Konvertierungslogik aus from_feature_flag().
    """

    def convert(
        self,
        feature_flag: FeatureFlag,
        enabled: bool | None = None,
        **kwargs
    ) -> FeatureFlagResponse:
        """Konvertiert FeatureFlag zu FeatureFlagResponse.

        Args:
            feature_flag: FeatureFlag-Objekt
            enabled: Ob Feature aktiviert ist (optional)
            **kwargs: Zusätzliche Parameter

        Returns:
            FeatureFlagResponse-Objekt
        """
        # Enabled-Status bestimmen
        if enabled is None:
            enabled = feature_flag.enabled_by_default

        return FeatureFlagResponse(
            name=feature_flag.name,
            description=feature_flag.description,
            status=feature_flag.status,
            scope=feature_flag.scope,
            enabled=enabled,
            introduced_in_version=feature_flag.introduced_in_version,
            deprecated_in_version=feature_flag.deprecated_in_version,
            removal_planned_version=feature_flag.removal_planned_version,
            dependencies=feature_flag.dependencies.copy(),
            metadata=feature_flag.metadata.copy()
        )


class CapabilityConverter(BaseConverter[APICapability, CapabilityResponse]):
    """Konverter für APICapability zu CapabilityResponse.

    Konsolidiert die duplizierte Konvertierungslogik aus from_api_capability().
    """

    def convert(
        self,
        capability: APICapability,
        available: bool = True,
        **kwargs
    ) -> CapabilityResponse:
        """Konvertiert APICapability zu CapabilityResponse.

        Args:
            capability: APICapability-Objekt
            available: Ob Capability verfügbar ist
            **kwargs: Zusätzliche Parameter

        Returns:
            CapabilityResponse-Objekt
        """
        return CapabilityResponse(
            name=capability.name,
            description=capability.description,
            category=capability.category,
            version=capability.version,
            available=available,
            endpoints=capability.endpoints.copy(),
            feature_flags=capability.feature_flags.copy(),
            requirements=capability.requirements.copy(),
            documentation_url=capability.documentation_url
        )


class ResponseConverter:
    """Zentrale Response-Konvertierungs-Klasse.

    Stellt einheitliche Schnittstelle für alle Response-Konvertierungen bereit
    und eliminiert Code-Duplikation.
    """

    def __init__(self) -> None:
        """Initialisiert Response-Converter."""
        self.feature_flag_converter = FeatureFlagConverter()
        self.capability_converter = CapabilityConverter()

    def feature_flag_to_response(
        self,
        feature_flag: FeatureFlag,
        enabled: bool | None = None
    ) -> FeatureFlagResponse:
        """Konvertiert FeatureFlag zu Response.

        Args:
            feature_flag: FeatureFlag-Objekt
            enabled: Ob Feature aktiviert ist

        Returns:
            FeatureFlagResponse-Objekt
        """
        return self.feature_flag_converter.convert(feature_flag, enabled=enabled)

    def feature_flags_to_responses(
        self,
        feature_flags: list[FeatureFlag],
        enabled_flags: list[str] | None = None
    ) -> list[FeatureFlagResponse]:
        """Konvertiert FeatureFlag-Liste zu Response-Liste.

        Args:
            feature_flags: Liste von FeatureFlag-Objekten
            enabled_flags: Liste aktivierter Feature-Namen

        Returns:
            Liste von FeatureFlagResponse-Objekten
        """
        enabled_set = set(enabled_flags or [])
        return [
            self.feature_flag_to_response(
                flag,
                enabled=flag.name in enabled_set
            )
            for flag in feature_flags
        ]

    def capability_to_response(
        self,
        capability: APICapability,
        available: bool = True
    ) -> CapabilityResponse:
        """Konvertiert APICapability zu Response.

        Args:
            capability: APICapability-Objekt
            available: Ob Capability verfügbar ist

        Returns:
            CapabilityResponse-Objekt
        """
        return self.capability_converter.convert(capability, available=available)

    def capabilities_to_responses(
        self,
        capabilities: list[APICapability],
        availability_map: dict[str, bool] | None = None
    ) -> list[CapabilityResponse]:
        """Konvertiert APICapability-Liste zu Response-Liste.

        Args:
            capabilities: Liste von APICapability-Objekten
            availability_map: Mapping von Capability-Namen zu Verfügbarkeit

        Returns:
            Liste von CapabilityResponse-Objekten
        """
        availability = availability_map or {}
        return [
            self.capability_to_response(
                cap,
                available=availability.get(cap.name, True)
            )
            for cap in capabilities
        ]


class ModelConverter:
    """Zentrale Model-Konvertierungs-Klasse.

    Stellt Utilities für komplexe Model-Konvertierungen bereit.
    """

    def __init__(self) -> None:
        """Initialisiert Model-Converter."""
        self.response_converter = ResponseConverter()

    def dict_to_feature_flag(self, data: dict[str, Any]) -> FeatureFlag:
        """Konvertiert Dictionary zu FeatureFlag.

        Args:
            data: Feature-Flag-Daten

        Returns:
            FeatureFlag-Objekt
        """
        from .factories import feature_flag_factory

        return feature_flag_factory.create(
            name=data["name"],
            description=data["description"],
            enabled_by_default=data.get("enabled", False),
            **{k: v for k, v in data.items() if k not in ["name", "description", "enabled"]}
        )

    def dict_to_capability(self, data: dict[str, Any]) -> APICapability:
        """Konvertiert Dictionary zu APICapability.

        Args:
            data: Capability-Daten

        Returns:
            APICapability-Objekt
        """
        from ..models.capability_models import CapabilityCategory
        from .factories import capability_factory

        category_str = data.get("category", "core")
        category = CapabilityCategory(category_str)

        return capability_factory.create(
            name=data["name"],
            description=data["description"],
            category=category,
            endpoints=data["endpoints"],
            **{k: v for k, v in data.items() if k not in ["name", "description", "category", "endpoints"]}
        )

    def dicts_to_feature_flags(self, data_list: list[dict[str, Any]]) -> list[FeatureFlag]:
        """Konvertiert Dictionary-Liste zu FeatureFlag-Liste.

        Args:
            data_list: Liste von Feature-Flag-Daten

        Returns:
            Liste von FeatureFlag-Objekten
        """
        return [self.dict_to_feature_flag(data) for data in data_list]

    def dicts_to_capabilities(self, data_list: list[dict[str, Any]]) -> list[APICapability]:
        """Konvertiert Dictionary-Liste zu APICapability-Liste.

        Args:
            data_list: Liste von Capability-Daten

        Returns:
            Liste von APICapability-Objekten
        """
        return [self.dict_to_capability(data) for data in data_list]


# ============================================================================
# SINGLETON CONVERTER INSTANCES
# ============================================================================

# Globale Converter-Instanzen für einfache Verwendung
response_converter = ResponseConverter()
model_converter = ModelConverter()
