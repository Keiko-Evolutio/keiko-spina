"""Service für Feature-Flag-Management.

Extrahiert Business-Logic aus der monolithischen capabilities.py
und stellt testbare, injectable Service-Komponente bereit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kei_logging import get_logger

from ..models.feature_flag_models import (
    FeatureFlag,
    FeatureFlagResponse,
    FeatureScope,
)
from ..utils.constants import DEFAULT_FEATURE_FLAGS, LOG_MESSAGES
from ..utils.converters import response_converter
from ..utils.factories import feature_flag_factory
from ..utils.validators import feature_validator


class FeatureFlagService:
    """Service für Feature-Flag-Management.

    Eliminiert globale Variablen und stellt testbare,
    injectable Service-Komponente bereit.
    """

    def __init__(self) -> None:
        """Initialisiert Feature-Flag-Service."""
        self.logger = get_logger(self.__class__.__name__)
        self._feature_flags: dict[str, FeatureFlag] = {}
        self._load_default_features()

    def _load_default_features(self) -> None:
        """Lädt Standard-Feature-Flags aus Konfiguration."""
        default_features = [
            self._create_feature_from_config(config)
            for config in DEFAULT_FEATURE_FLAGS
        ]

        for feature in default_features:
            self._feature_flags[feature.name] = feature

        self.logger.info(
            LOG_MESSAGES["loaded_features"].format(count=len(default_features))
        )

    def _create_feature_from_config(self, config: dict[str, Any]) -> FeatureFlag:
        """Erstellt FeatureFlag aus Konfiguration.

        Args:
            config: Feature-Konfiguration

        Returns:
            FeatureFlag-Objekt
        """
        scope_str = config.get("scope", "global")
        if scope_str == "global":
            return feature_flag_factory.create_global_feature(
                config["name"],
                config["description"],
                config.get("enabled", False)
            )
        if scope_str == "per_client":
            return feature_flag_factory.create_client_feature(
                config["name"],
                config["description"]
            )
        return feature_flag_factory.create(
            config["name"],
            config["description"],
            scope=FeatureScope(scope_str),
            enabled_by_default=config.get("enabled", False)
        )

    def is_feature_enabled(
        self,
        feature_name: str,
        client_id: str | None = None,
        server_name: str | None = None
    ) -> bool:
        """Prüft ob Feature aktiviert ist.

        Args:
            feature_name: Name des Features
            client_id: Client-ID (optional)
            server_name: Server-Name (optional)

        Returns:
            True wenn Feature aktiviert ist
        """
        if feature_name not in self._feature_flags:
            self.logger.warning(
                LOG_MESSAGES["unknown_feature"].format(feature_name=feature_name)
            )
            return False

        feature = self._feature_flags[feature_name]

        # Client-spezifische Prüfung
        return feature_validator.validate_feature_availability(
            feature, client_id, server_name
        )

    def get_enabled_features(
        self,
        client_id: str | None = None,
        server_name: str | None = None
    ) -> list[FeatureFlagResponse]:
        """Gibt alle aktivierten Features zurück.

        Args:
            client_id: Client-ID (optional)
            server_name: Server-Name (optional)

        Returns:
            Liste aktivierter Feature-Flag-Responses
        """
        enabled_features = []

        for feature_name, feature in self._feature_flags.items():
            if self.is_feature_enabled(feature_name, client_id, server_name):
                enabled_features.append(
                    response_converter.feature_flag_to_response(feature, enabled=True)
                )

        return enabled_features

    def get_all_features(self) -> list[FeatureFlagResponse]:
        """Gibt alle Features zurück (aktiviert und deaktiviert).

        Returns:
            Liste aller Feature-Flag-Responses
        """
        all_features = []

        for feature in self._feature_flags.values():
            all_features.append(
                response_converter.feature_flag_to_response(
                    feature,
                    enabled=feature.enabled_by_default
                )
            )

        return all_features

    def _execute_feature_operation(
        self,
        feature_name: str,
        operation_func: Callable,
        success_message_key: str,
        error_message_key: str,
        *args
    ) -> bool:
        """Führt Feature-Operation mit einheitlicher Fehlerbehandlung aus.

        Args:
            feature_name: Name des Features
            operation_func: Auszuführende Operation
            success_message_key: Schlüssel für Erfolgs-Nachricht
            error_message_key: Schlüssel für Fehler-Nachricht
            *args: Zusätzliche Argumente für operation_func

        Returns:
            True wenn Operation erfolgreich
        """
        if feature_name not in self._feature_flags:
            self.logger.warning(
                LOG_MESSAGES[error_message_key].format(feature_name=feature_name)
            )
            return False

        feature = self._feature_flags[feature_name]
        operation_func(feature, *args)

        self.logger.info(
            LOG_MESSAGES[success_message_key].format(
                feature_name=feature_name,
                client_id=args[0] if args else None,
                server_name=args[0] if args else None
            )
        )
        return True

    def enable_feature_for_client(self, feature_name: str, client_id: str) -> bool:
        """Aktiviert Feature für spezifischen Client.

        Args:
            feature_name: Name des Features
            client_id: Client-ID

        Returns:
            True wenn erfolgreich aktiviert
        """
        return self._execute_feature_operation(
            feature_name,
            lambda f, cid: f.enable_for_client(cid),
            "enabled_for_client",
            "cannot_enable_unknown",
            client_id
        )

    def disable_feature_for_client(self, feature_name: str, client_id: str) -> bool:
        """Deaktiviert Feature für spezifischen Client.

        Args:
            feature_name: Name des Features
            client_id: Client-ID

        Returns:
            True wenn erfolgreich deaktiviert
        """
        return self._execute_feature_operation(
            feature_name,
            lambda f, cid: f.disable_for_client(cid),
            "disabled_for_client",
            "cannot_disable_unknown",
            client_id
        )

    def enable_feature_for_server(self, feature_name: str, server_name: str) -> bool:
        """Aktiviert Feature für spezifischen Server.

        Args:
            feature_name: Name des Features
            server_name: Server-Name

        Returns:
            True wenn erfolgreich aktiviert
        """
        return self._execute_feature_operation(
            feature_name,
            lambda f, sn: f.enable_for_server(sn),
            "enabled_for_server",
            "cannot_enable_unknown",
            server_name
        )

    def disable_feature_for_server(self, feature_name: str, server_name: str) -> bool:
        """Deaktiviert Feature für spezifischen Server.

        Args:
            feature_name: Name des Features
            server_name: Server-Name

        Returns:
            True wenn erfolgreich deaktiviert
        """
        return self._execute_feature_operation(
            feature_name,
            lambda f, sn: f.disable_for_server(sn),
            "disabled_for_server",
            "cannot_disable_unknown",
            server_name
        )

    def add_feature_flag(self, feature: FeatureFlag) -> bool:
        """Fügt neues Feature-Flag hinzu.

        Args:
            feature: FeatureFlag-Objekt

        Returns:
            True wenn erfolgreich hinzugefügt
        """
        if feature.name in self._feature_flags:
            self.logger.warning(
                LOG_MESSAGES["feature_exists"].format(feature_name=feature.name)
            )
            return False

        # Validiere Feature vor dem Hinzufügen
        feature_validator.validate(feature)

        self._feature_flags[feature.name] = feature
        self.logger.info(
            LOG_MESSAGES["added_feature"].format(feature_name=feature.name)
        )
        return True

    def remove_feature_flag(self, feature_name: str) -> bool:
        """Entfernt Feature-Flag.

        Args:
            feature_name: Name des Features

        Returns:
            True wenn erfolgreich entfernt
        """
        if feature_name not in self._feature_flags:
            self.logger.warning(
                LOG_MESSAGES["cannot_remove_unknown"].format(feature_name=feature_name)
            )
            return False

        del self._feature_flags[feature_name]
        self.logger.info(
            LOG_MESSAGES["removed_feature"].format(feature_name=feature_name)
        )
        return True

    def get_feature_flag(self, feature_name: str) -> FeatureFlag | None:
        """Gibt Feature-Flag zurück.

        Args:
            feature_name: Name des Features

        Returns:
            FeatureFlag-Objekt oder None
        """
        return self._feature_flags.get(feature_name)

    def get_feature_names(self) -> list[str]:
        """Gibt alle Feature-Namen zurück.

        Returns:
            Liste aller Feature-Namen
        """
        return list(self._feature_flags.keys())

    def get_stats(self) -> dict[str, int]:
        """Gibt Statistiken über Feature-Flags zurück.

        Returns:
            Dictionary mit Statistiken
        """
        stats = self._initialize_stats()

        for feature in self._feature_flags.values():
            self._update_stats_for_feature(stats, feature)

        return stats

    def _initialize_stats(self) -> dict[str, int]:
        """Initialisiert Statistik-Dictionary.

        Returns:
            Initialisiertes Statistik-Dictionary
        """
        return {
            "total_features": len(self._feature_flags),
            "enabled_by_default": 0,
            "experimental": 0,
            "stable": 0,
            "deprecated": 0,
            "disabled": 0
        }

    def _update_stats_for_feature(self, stats: dict[str, int], feature: FeatureFlag) -> None:
        """Aktualisiert Statistiken für ein Feature.

        Args:
            stats: Statistik-Dictionary
            feature: FeatureFlag-Objekt
        """
        if feature.enabled_by_default:
            stats["enabled_by_default"] += 1

        # Status-spezifische Zählung
        status_key = feature.status.value
        if status_key in stats:
            stats[status_key] += 1
