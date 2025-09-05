"""Basis-Klasse für alle Spezifikations-Generatoren.

Enthält gemeinsame Funktionalitäten für File I/O, Error-Handling,
Konfiguration und Logging.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

from kei_logging import LogLinkedError, get_logger, with_log_links

from .constants import (
    ContactInfo,
    DirectoryNames,
    SpecConstants,
    get_contact_info,
    get_license_info,
)

logger = get_logger(__name__)


class BaseSpecificationError(LogLinkedError):
    """Basis-Exception für Spezifikations-Generierung."""

    def __init__(
        self,
        message: str,
        spec_type: str | None = None,
        cause: Exception | None = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.spec_type = spec_type
        self.cause = cause


class BaseSpecGenerator(ABC):
    """Basis-Klasse für alle Spezifikations-Generatoren.

    Stellt gemeinsame Funktionalitäten zur Verfügung:
    - File I/O Operationen
    - Error-Handling
    - Konfiguration
    - Logging
    - Validierung
    """

    def __init__(self, specs_dir: str = DirectoryNames.SPECS_DIR):
        """Initialisiert Basis-Generator.

        Args:
            specs_dir: Verzeichnis für Spezifikations-Dateien
        """
        self.specs_dir = Path(specs_dir)
        self.specs_dir.mkdir(exist_ok=True)

        # Erstelle Standard-Unterverzeichnisse
        self._create_directories()

        # Initialisiere Konfiguration
        self._init_configuration()

    def _create_directories(self) -> None:
        """Erstellt alle benötigten Unterverzeichnisse."""
        directories = [
            DirectoryNames.OPENAPI_DIR,
            DirectoryNames.ASYNCAPI_DIR,
            DirectoryNames.MCP_DIR,
            DirectoryNames.GENERATED_DIR,
            DirectoryNames.TEMPLATES_DIR
        ]

        for directory in directories:
            (self.specs_dir / directory).mkdir(exist_ok=True)

    def _init_configuration(self) -> None:
        """Initialisiert Standard-Konfiguration."""
        self.config = {
            "api_version": SpecConstants.API_VERSION,
            "openapi_version": SpecConstants.OPENAPI_VERSION,
            "asyncapi_version": SpecConstants.ASYNCAPI_VERSION,
            "contact_info": get_contact_info(),
            "license_info": get_license_info(),
            "encoding": SpecConstants.DEFAULT_ENCODING
        }

    @with_log_links
    def save_spec(
        self,
        spec: dict[str, Any],
        subdirectory: str,
        filename: str,
        format_type: str = "yaml"
    ) -> Path:
        """Speichert Spezifikation in Datei.

        Args:
            spec: Spezifikations-Dictionary
            subdirectory: Unterverzeichnis (z.B. "openapi", "asyncapi")
            filename: Dateiname
            format_type: Format ("yaml" oder "json")

        Returns:
            Pfad zur gespeicherten Datei

        Raises:
            BaseSpecificationError: Bei Speicher-Fehlern
        """
        try:
            output_dir = self.specs_dir / subdirectory
            output_dir.mkdir(exist_ok=True)

            output_path = output_dir / filename

            with open(output_path, "w", encoding=self.config["encoding"]) as f:
                if format_type.lower() == "json":
                    json.dump(spec, f, ensure_ascii=False, indent=2)
                else:
                    yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True)

            logger.info(
                f"Spezifikation gespeichert: {output_path}",
                extra={
                    "spec_type": subdirectory,
                    "filename": filename,
                    "format": format_type,
                    "size_bytes": output_path.stat().st_size
                }
            )

            return output_path

        except Exception as e:
            raise BaseSpecificationError(
                message=f"Fehler beim Speichern der Spezifikation: {e}",
                spec_type=subdirectory,
                cause=e
            ) from e

    @with_log_links
    def load_spec(
        self,
        subdirectory: str,
        filename: str
    ) -> dict[str, Any]:
        """Lädt Spezifikation aus Datei.

        Args:
            subdirectory: Unterverzeichnis
            filename: Dateiname

        Returns:
            Spezifikations-Dictionary

        Raises:
            BaseSpecificationError: Bei Lade-Fehlern
        """
        try:
            spec_path = self.specs_dir / subdirectory / filename

            if not spec_path.exists():
                raise BaseSpecificationError(
                    message=f"Spezifikations-Datei nicht gefunden: {spec_path}",
                    spec_type=subdirectory
                )

            with open(spec_path, encoding=self.config["encoding"]) as f:
                spec = json.load(f) if filename.endswith(".json") else yaml.safe_load(f)

            logger.info(
                f"Spezifikation geladen: {spec_path}",
                extra={
                    "spec_type": subdirectory,
                    "filename": filename
                }
            )

            return spec

        except Exception as e:
            raise BaseSpecificationError(
                message=f"Fehler beim Laden der Spezifikation: {e}",
                spec_type=subdirectory,
                cause=e
            ) from e

    def validate_spec_structure(
        self,
        spec: dict[str, Any],
        required_fields: list[str],
        spec_type: str
    ) -> None:
        """Validiert Basis-Struktur einer Spezifikation.

        Args:
            spec: Spezifikations-Dictionary
            required_fields: Liste erforderlicher Felder
            spec_type: Typ der Spezifikation (für Logging)

        Raises:
            BaseSpecificationError: Bei Validierungs-Fehlern
        """
        for field in required_fields:
            if field not in spec:
                raise BaseSpecificationError(
                    message=f"Erforderliches Feld '{field}' fehlt in {spec_type}-Spezifikation",
                    spec_type=spec_type,
                    field=field
                )

    def get_enhanced_info_section(
        self,
        title: str,
        description: str,
        version: str | None = None
    ) -> dict[str, Any]:
        """Erstellt erweiterte Info-Sektion für Spezifikationen.

        Args:
            title: Titel der API
            description: Beschreibung der API
            version: Version (Standard: aus Konfiguration)

        Returns:
            Info-Sektion Dictionary
        """
        return {
            "title": title,
            "version": version or self.config["api_version"],
            "description": description,
            "contact": self.config["contact_info"],
            "license": self.config["license_info"],
            "termsOfService": ContactInfo.TERMS_OF_SERVICE
        }

    def get_standard_servers(self) -> list[dict[str, str]]:
        """Gibt Standard-Server-Konfiguration zurück.

        Returns:
            Liste von Server-Definitionen
        """
        return [
            {
                "url": SpecConstants.LOCAL_DEV_SERVER,
                "description": "Lokale Entwicklungsumgebung"
            },
            {
                "url": SpecConstants.PRODUCTION_SERVER,
                "description": "Produktionsumgebung"
            }
        ]

    @abstractmethod
    def generate_spec(self, **kwargs) -> dict[str, Any]:
        """Generiert Spezifikation.

        Muss von abgeleiteten Klassen implementiert werden.

        Returns:
            Generierte Spezifikation
        """

    @abstractmethod
    def validate_spec(self, spec: dict[str, Any]) -> None:
        """Validiert generierte Spezifikation.

        Muss von abgeleiteten Klassen implementiert werden.

        Args:
            spec: Zu validierende Spezifikation
        """
