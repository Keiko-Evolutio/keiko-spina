"""JSON-Schema-Validierung für KEI-MCP Tool-Parameter.

Dieses Modul implementiert umfassende JSON-Schema-Validierung für Tool-Parameter
vor der Weiterleitung an externe MCP Server, um Sicherheit und Datenintegrität
zu gewährleisten.
"""

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import jsonschema
from jsonschema import Draft7Validator, Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError, ValidationError

from kei_logging import get_logger
from observability import trace_function

from .core.constants import DEFAULT_CACHE_TTL_SECONDS, DEFAULT_MAX_CACHE_SIZE

logger = get_logger(__name__)


class SchemaValidationError(Exception):
    """Fehler bei JSON-Schema-Validierung."""

    def __init__(self, message: str, field_path: str = "", validation_errors: list[str] | None = None):
        """Initialisiert Schema-Validierungsfehler.

        Args:
            message: Hauptfehlermeldung
            field_path: Pfad zum fehlerhaften Feld
            validation_errors: Liste detaillierter Validierungsfehler
        """
        super().__init__(message)
        self.field_path = field_path
        self.validation_errors = validation_errors or []


class SchemaFormat(Enum):
    """Unterstützte JSON-Schema-Formate."""

    DRAFT_7 = "draft7"
    DRAFT_2019_09 = "draft2019-09"
    DRAFT_2020_12 = "draft2020-12"


@dataclass
class ValidationResult:
    """Ergebnis einer Schema-Validierung."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    field_errors: dict[str, list[str]] = field(default_factory=dict)
    validation_time_ms: float = 0.0
    schema_format: str | None = None
    sanitized_value: Any = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CachedSchema:
    """Gecachtes JSON-Schema mit Metadaten."""

    schema: dict[str, Any]
    validator: jsonschema.protocols.Validator
    cached_at: datetime
    ttl_seconds: int = 3600  # 1 Stunde default
    access_count: int = 0

    def is_expired(self) -> bool:
        """Prüft ob das gecachte Schema abgelaufen ist."""
        return datetime.now() - self.cached_at > timedelta(seconds=self.ttl_seconds)

    def touch(self):
        """Markiert Schema als verwendet."""
        self.access_count += 1


class KEIMCPSchemaValidator:
    """JSON-Schema-Validator für KEI-MCP Tool-Parameter."""

    def __init__(self, cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS, max_cache_size: int = DEFAULT_MAX_CACHE_SIZE):
        """Initialisiert den Schema-Validator.

        Args:
            cache_ttl_seconds: Cache-TTL für Schemas in Sekunden
            max_cache_size: Maximale Anzahl gecachter Schemas
        """
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_cache_size = max_cache_size
        self._schema_cache: dict[str, CachedSchema] = {}

        # Custom Format Checker für erweiterte Validierung
        self.format_checker = FormatChecker()
        self._register_custom_formats()

        logger.info(f"Schema-Validator initialisiert - Cache TTL: {cache_ttl_seconds}s, Max Size: {max_cache_size}")

    def _register_custom_formats(self):
        """Registriert custom Format-Validatoren."""

        @self.format_checker.checks("kei-server-name")
        def check_server_name(instance: str) -> bool:
            """Validiert KEI-MCP Server-Namen."""
            if not isinstance(instance, str):
                return False
            # Server-Namen: alphanumerisch, Bindestriche, Unterstriche, 3-63 Zeichen
            return bool(re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$", instance))

        @self.format_checker.checks("kei-tool-name")
        def check_tool_name(instance: str) -> bool:
            """Validiert KEI-MCP Tool-Namen."""
            if not isinstance(instance, str):
                return False
            # Tool-Namen: alphanumerisch, Unterstriche, 1-100 Zeichen
            return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9_]{0,99}$", instance))

        @self.format_checker.checks("iso8601-duration")
        def check_iso8601_duration(instance: str) -> bool:
            """Validiert ISO 8601 Duration Format."""
            if not isinstance(instance, str):
                return False
            # Vereinfachte ISO 8601 Duration: P[n]Y[n]M[n]DT[n]H[n]M[n]S
            pattern = r"^P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$"
            return bool(re.match(pattern, instance))

        @self.format_checker.checks("semantic-version")
        def check_semantic_version(instance: str) -> bool:
            """Validiert Semantic Versioning Format."""
            if not isinstance(instance, str):
                return False
            # SemVer: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
            pattern = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
            return bool(re.match(pattern, instance))

    def _detect_schema_format(self, schema: dict[str, Any]) -> SchemaFormat:
        """Erkennt das JSON-Schema-Format.

        Args:
            schema: JSON-Schema

        Returns:
            Erkanntes Schema-Format
        """
        schema_version = schema.get("$schema", "")

        if "draft/2020-12" in schema_version:
            return SchemaFormat.DRAFT_2020_12
        if "draft/2019-09" in schema_version:
            return SchemaFormat.DRAFT_2019_09
        # Default zu Draft 7 für Kompatibilität
        return SchemaFormat.DRAFT_7

    def _create_validator(self, schema: dict[str, Any], schema_format: SchemaFormat) -> jsonschema.protocols.Validator:
        """Erstellt einen JSON-Schema-Validator.

        Args:
            schema: JSON-Schema
            schema_format: Schema-Format

        Returns:
            Konfigurierter Validator
        """
        if schema_format == SchemaFormat.DRAFT_2020_12:
            validator_class = Draft202012Validator
        else:
            # Draft 7 als Default (auch für 2019-09 kompatibel)
            validator_class = Draft7Validator

        # Validator mit custom Format Checker erstellen
        return validator_class(schema, format_checker=self.format_checker)

    @staticmethod
    def _generate_cache_key(server_name: str, tool_name: str) -> str:
        """Generiert Cache-Schlüssel für Schema.

        Args:
            server_name: Name des MCP Servers
            tool_name: Name des Tools

        Returns:
            Cache-Schlüssel
        """
        return f"{server_name}:{tool_name}"

    def _cleanup_cache(self):
        """Bereinigt abgelaufene Cache-Einträge."""
        datetime.now()
        expired_keys = []

        for key, cached_schema in self._schema_cache.items():
            if cached_schema.is_expired():
                expired_keys.append(key)

        for key in expired_keys:
            del self._schema_cache[key]
            logger.debug(f"Abgelaufenes Schema aus Cache entfernt: {key}")

        # Cache-Größe begrenzen (LRU-ähnlich basierend auf access_count)
        if len(self._schema_cache) > self.max_cache_size:
            # Sortiere nach access_count (aufsteigend) und entferne die am wenigsten verwendeten
            sorted_items = sorted(
                self._schema_cache.items(),
                key=lambda x: x[1].access_count
            )

            items_to_remove = len(self._schema_cache) - self.max_cache_size
            for i in range(items_to_remove):
                key_to_remove = sorted_items[i][0]
                del self._schema_cache[key_to_remove]
                logger.debug(f"Schema aus vollem Cache entfernt: {key_to_remove}")

    @trace_function("schema_validator.cache_schema")
    def cache_schema(self, server_name: str, tool_name: str, schema: dict[str, Any]) -> bool:
        """Cached ein JSON-Schema für ein Tool.

        Args:
            server_name: Name des MCP Servers
            tool_name: Name des Tools
            schema: JSON-Schema für Tool-Parameter

        Returns:
            True wenn erfolgreich gecacht
        """
        try:
            # Schema-Format erkennen
            schema_format = self._detect_schema_format(schema)

            # Validator erstellen und Schema validieren
            validator = self._create_validator(schema, schema_format)
            validator.check_schema(schema)  # Validiert das Schema selbst

            # Cache-Schlüssel generieren
            cache_key = KEIMCPSchemaValidator._generate_cache_key(server_name, tool_name)

            # Schema cachen
            cached_schema = CachedSchema(
                schema=schema,
                validator=validator,
                cached_at=datetime.now(),
                ttl_seconds=self.cache_ttl_seconds
            )

            self._schema_cache[cache_key] = cached_schema

            # Cache bereinigen falls nötig
            self._cleanup_cache()

            logger.debug(f"Schema gecacht für {server_name}:{tool_name} (Format: {schema_format.value})")
            return True

        except SchemaError as e:
            logger.exception(f"Ungültiges JSON-Schema für {server_name}:{tool_name}: {e}")
            return False
        except Exception as e:
            logger.exception(f"Fehler beim Cachen des Schemas für {server_name}:{tool_name}: {e}")
            return False

    @trace_function("schema_validator.validate_parameters")
    def validate_parameters(
        self,
        server_name: str,
        tool_name: str,
        parameters: dict[str, Any],
        schema: dict[str, Any] | None = None
    ) -> ValidationResult:
        """Validiert Tool-Parameter gegen JSON-Schema.

        Args:
            server_name: Name des MCP Servers
            tool_name: Name des Tools
            parameters: Zu validierende Parameter
            schema: Optionales Schema (falls nicht gecacht)

        Returns:
            Validierungsergebnis
        """
        start_time = time.time()

        try:
            # Schema abrufen (Cache oder Parameter)
            if schema:
                # Schema direkt verwenden
                schema_format = self._detect_schema_format(schema)
                validator = self._create_validator(schema, schema_format)
            else:
                # Aus Cache abrufen
                cache_key = KEIMCPSchemaValidator._generate_cache_key(server_name, tool_name)
                cached_schema = self._schema_cache.get(cache_key)

                if not cached_schema or cached_schema.is_expired():
                    return ValidationResult(
                        valid=False,
                        errors=["Schema nicht verfügbar oder abgelaufen"],
                        validation_time_ms=(time.time() - start_time) * 1000
                    )

                validator = cached_schema.validator
                cached_schema.touch()

            # Parameter validieren
            validation_errors = []
            field_errors = {}

            try:
                validator.validate(parameters)

                # Validierung erfolgreich
                result = ValidationResult(
                    valid=True,
                    validation_time_ms=(time.time() - start_time) * 1000,
                    schema_format=getattr(validator, "META_SCHEMA", {}).get("$id", "unknown")
                )

                logger.debug(f"Parameter-Validierung erfolgreich für {server_name}:{tool_name}")
                return result

            except ValidationError:
                # Detaillierte Validierungsfehler sammeln
                for error in validator.iter_errors(parameters):
                    field_path = ".".join(str(p) for p in error.absolute_path) if error.absolute_path else "root"
                    error_message = f"{error.message}"

                    if field_path not in field_errors:
                        field_errors[field_path] = []
                    field_errors[field_path].append(error_message)
                    validation_errors.append(f"{field_path}: {error_message}")

                result = ValidationResult(
                    valid=False,
                    errors=validation_errors,
                    field_errors=field_errors,
                    validation_time_ms=(time.time() - start_time) * 1000,
                    schema_format=getattr(validator, "META_SCHEMA", {}).get("$id", "unknown")
                )

                logger.warning(f"Parameter-Validierung fehlgeschlagen für {server_name}:{tool_name}: {len(validation_errors)} Fehler")
                return result

        except Exception as e:
            logger.exception(f"Unerwarteter Fehler bei Parameter-Validierung für {server_name}:{tool_name}: {e}")
            return ValidationResult(
                valid=False,
                errors=[f"Validierungsfehler: {e!s}"],
                validation_time_ms=(time.time() - start_time) * 1000
            )

    def get_cache_stats(self) -> dict[str, Any]:
        """Gibt Cache-Statistiken zurück.

        Returns:
            Cache-Statistiken
        """
        total_schemas = len(self._schema_cache)
        expired_schemas = sum(1 for schema in self._schema_cache.values() if schema.is_expired())
        total_access_count = sum(schema.access_count for schema in self._schema_cache.values())

        return {
            "total_schemas": total_schemas,
            "expired_schemas": expired_schemas,
            "active_schemas": total_schemas - expired_schemas,
            "cache_utilization": total_schemas / self.max_cache_size if self.max_cache_size > 0 else 0,
            "total_access_count": total_access_count,
            "average_access_count": total_access_count / total_schemas if total_schemas > 0 else 0
        }

    def clear_cache(self, server_name: str | None = None, tool_name: str | None = None):
        """Bereinigt Schema-Cache.

        Args:
            server_name: Optionaler Server-Name (nur diesen Server bereinigen)
            tool_name: Optionaler Tool-Name (nur dieses Tool bereinigen)
        """
        if server_name and tool_name:
            # Spezifisches Schema entfernen
            cache_key = KEIMCPSchemaValidator._generate_cache_key(server_name, tool_name)
            if cache_key in self._schema_cache:
                del self._schema_cache[cache_key]
                logger.info(f"Schema-Cache bereinigt für {server_name}:{tool_name}")
        elif server_name:
            # Alle Schemas für einen Server entfernen
            keys_to_remove = [key for key in self._schema_cache if key.startswith(f"{server_name}:")]
            for key in keys_to_remove:
                del self._schema_cache[key]
            logger.info(f"Schema-Cache bereinigt für Server {server_name} ({len(keys_to_remove)} Schemas)")
        else:
            # Kompletten Cache leeren
            cache_size = len(self._schema_cache)
            self._schema_cache.clear()
            logger.info(f"Kompletter Schema-Cache bereinigt ({cache_size} Schemas)")

    def validate_schema(self, schema: dict[str, Any]) -> ValidationResult:
        """Validiert ein JSON-Schema.

        Args:
            schema: JSON-Schema-Dictionary

        Returns:
            ValidationResult mit Validierungsergebnissen
        """
        errors = []
        warnings = []
        field_errors = {}

        try:
            # Schema-Format erkennen
            schema_format = self._detect_schema_format(schema)

            # Validator erstellen
            if schema_format == SchemaFormat.DRAFT_7:
                validator_class = Draft7Validator
            elif schema_format == SchemaFormat.DRAFT_2020_12:
                validator_class = Draft202012Validator
            else:
                validator_class = Draft7Validator  # Fallback

            validator = validator_class(schema)

            # Schema selbst validieren
            validator.check_schema(schema)

        except SchemaError as e:
            errors.append(f"Ungültiges JSON-Schema: {e.message}")
        except Exception as e:
            errors.append(f"Schema-Validierung fehlgeschlagen: {e!s}")

        return ValidationResult(
            valid=len(errors) == 0,
            sanitized_value=schema,
            errors=errors,
            warnings=warnings,
            field_errors=field_errors,
            metadata={
                "schema_format": schema_format.value if "schema_format" in locals() else "unknown",
                "validation_time_ms": 0.0
            }
        )

    def validate_tool_schema(self, schema: dict[str, Any]) -> ValidationResult:
        """Validiert Tool-Schema-Definition für Discovery.

        Args:
            schema: Tool-Schema-Dictionary

        Returns:
            ValidationResult mit Validierungsergebnissen
        """
        errors = []
        warnings = []
        field_errors = {}

        try:
            # Basis-Schema-Validierung
            base_result = self.validate_schema(schema)
            errors.extend(base_result.errors)
            warnings.extend(base_result.warnings)
            field_errors.update(base_result.field_errors)

            # Tool-spezifische Validierungen
            if "properties" in schema:
                properties = schema["properties"]

                # Prüfe auf sinnvolle Parameter-Namen
                for prop_name in properties:
                    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", prop_name):
                        warnings.append(f"Parameter-Name '{prop_name}' entspricht nicht den Konventionen")

                # Prüfe auf Beispiele
                if "examples" not in schema or not schema["examples"]:
                    warnings.append("Schema enthält keine Beispiele")

            # Prüfe auf Beschreibungen
            if "description" not in schema or not schema["description"].strip():
                warnings.append("Schema-Beschreibung fehlt oder ist leer")

        except Exception as e:
            errors.append(f"Tool-Schema-Validierung fehlgeschlagen: {e!s}")

        return ValidationResult(
            valid=len(errors) == 0,
            sanitized_value=schema,
            errors=errors,
            warnings=warnings,
            metadata={
                "schema_format": SchemaFormat.DRAFT_7,
                "validation_time_ms": 0.0,
                "field_errors": field_errors
            }
        )

    def validate_resource_metadata(self, metadata: dict[str, Any]) -> ValidationResult:
        """Validiert Resource-Metadaten für Discovery.

        Args:
            metadata: Metadaten-Dictionary

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        field_errors = {}

        # Erforderliche Felder prüfen
        required_fields = ["resource_type", "access_level"]
        for required_field in required_fields:
            if required_field not in metadata:
                errors.append(f"Erforderliches Metadaten-Feld '{required_field}' fehlt")
                field_errors[field] = "Feld ist erforderlich"

        # Content-Type validieren
        if "content_type" in metadata:
            content_type = metadata["content_type"]
            if not KEIMCPSchemaValidator._is_valid_content_type(content_type):
                warnings.append(f"Ungewöhnlicher Content-Type: {content_type}")

        # Checksum validieren
        if "checksum" in metadata:
            checksum = metadata["checksum"]
            if not KEIMCPSchemaValidator._is_valid_checksum(checksum):
                warnings.append(f"Ungültiges Checksum-Format: {checksum}")

        return ValidationResult(
            valid=len(errors) == 0,
            sanitized_value=metadata,
            errors=errors,
            warnings=warnings,
            metadata={
                "schema_format": SchemaFormat.DRAFT_7,
                "validation_time_ms": 0.0,
                "field_errors": field_errors
            }
        )

    @staticmethod
    def validate_prompt_template(template: str, parameters: list[dict[str, Any]]) -> ValidationResult:
        """Validiert Prompt-Template und Parameter für Discovery.

        Args:
            template: Prompt-Template-String
            parameters: Liste der Parameter-Definitionen

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        field_errors = {}

        # Template-Syntax validieren
        open_braces = template.count("{")
        close_braces = template.count("}")

        if open_braces != close_braces:
            errors.append("Ungeschlossene geschweifte Klammern im Template")
            field_errors["template"] = "Syntax-Fehler"

        # Prüfe auf leere Platzhalter
        empty_placeholders = re.findall(r"{\s*}", template)
        if empty_placeholders:
            warnings.append("Leere Platzhalter im Template gefunden")

        # Parameter-Konsistenz prüfen
        template_params = set(re.findall(r"{(\w+)}", template))
        defined_params = {param.get("name", "") for param in parameters if param.get("name")}

        # Prüfe auf undefinierte Parameter im Template
        undefined_params = template_params - defined_params
        for param in undefined_params:
            errors.append(f"Parameter '{param}' im Template verwendet aber nicht definiert")

        # Prüfe auf ungenutzte Parameter-Definitionen
        unused_params = defined_params - template_params
        for param in unused_params:
            warnings.append(f"Parameter '{param}' definiert aber nicht im Template verwendet")

        # Template-Qualität bewerten
        if len(template) < 10:
            warnings.append("Template ist sehr kurz")
        elif len(template) > 5000:
            warnings.append("Template ist sehr lang")

        return ValidationResult(
            valid=len(errors) == 0,
            sanitized_value=template,
            errors=errors,
            warnings=warnings,
            metadata={
                "schema_format": SchemaFormat.DRAFT_7,
                "validation_time_ms": 0.0,
                "field_errors": field_errors
            }
        )

    @staticmethod
    def _is_valid_content_type(content_type: str) -> bool:
        """Prüft, ob Content-Type gültig ist."""
        pattern = r"^[a-zA-Z0-9][a-zA-Z0-9!#$&\-\^_]*\/[a-zA-Z0-9][a-zA-Z0-9!#$&\-\^_]*$"
        return bool(re.match(pattern, content_type))

    @staticmethod
    def _is_valid_checksum(checksum: str) -> bool:
        """Prüft, ob Checksum gültig ist."""
        patterns = [
            r"^[a-fA-F0-9]{32}$",  # MD5
            r"^[a-fA-F0-9]{40}$",  # SHA1
            r"^[a-fA-F0-9]{64}$",  # SHA256
        ]
        return any(re.match(pattern, checksum) for pattern in patterns)


# Globale Schema-Validator-Instanz
schema_validator = KEIMCPSchemaValidator()


__all__ = [
    "CachedSchema",
    "KEIMCPSchemaValidator",
    "SchemaFormat",
    "SchemaValidationError",
    "ValidationResult",
    "schema_validator"
]
