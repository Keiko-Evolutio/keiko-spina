#!/usr/bin/env python3
"""Platform Schema Registry für Issue #56 Messaging-first Architecture
Implementiert Schema-Validierung für Platform-interne Events

ARCHITEKTUR-COMPLIANCE:
- Nur für Platform-interne Event-Schemas
- Keine SDK-Dependencies oder -Exports
- Vollständig isoliert von externen Schema-Registries
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from jsonschema import ValidationError, validate

from kei_logging import get_logger

logger = get_logger(__name__)

class SchemaCompatibility(Enum):
    """Schema-Kompatibilitäts-Modi"""
    BACKWARD = "backward"
    FORWARD = "forward"
    FULL = "full"
    NONE = "none"

@dataclass
class PlatformEventSchema:
    """Platform-interne Event-Schema Definition"""
    event_type: str
    version: str
    schema: dict[str, Any]
    compatibility: SchemaCompatibility = SchemaCompatibility.BACKWARD
    description: str | None = None
    created_at: datetime | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(UTC)

@dataclass
class SchemaValidationResult:
    """Ergebnis der Schema-Validierung"""
    valid: bool
    errors: list[str]
    warnings: list[str]
    schema_version: str

class PlatformSchemaRegistry:
    """Platform-interne Schema Registry für Event-Validierung"""

    def __init__(self):
        # Schema-Speicher (in-memory für Platform-interne Nutzung)
        self.schemas: dict[str, dict[str, PlatformEventSchema]] = {}  # event_type -> version -> schema
        self.latest_versions: dict[str, str] = {}  # event_type -> latest_version

        # Validierungs-Cache
        self.validation_cache: dict[str, SchemaValidationResult] = {}
        self.cache_max_size = 1000

        # Metriken
        self.validations_performed = 0
        self.validation_errors = 0
        self.cache_hits = 0

        # Status
        self.initialized = False

    async def initialize(self):
        """Initialisiert die Platform Schema Registry"""
        try:
            logger.info("Initialisiere Platform Schema Registry...")

            # Basis-Schemas laden
            await self._load_base_schemas()

            self.initialized = True
            logger.info("Platform Schema Registry erfolgreich initialisiert")

        except Exception as e:
            logger.error(f"Fehler beim Initialisieren der Platform Schema Registry: {e}")
            raise

    async def register_schema(self, schema: PlatformEventSchema) -> bool:
        """Registriert ein neues Platform Event Schema"""
        try:
            event_type = schema.event_type
            version = schema.version

            # Prüfe ob Event-Typ bereits existiert
            if event_type not in self.schemas:
                self.schemas[event_type] = {}
                self.latest_versions[event_type] = version

            # Prüfe Schema-Kompatibilität
            if version in self.schemas[event_type]:
                logger.warning(f"Schema {event_type} v{version} bereits registriert - überschreibe")

            if await self._validate_schema_compatibility(schema):
                # Schema registrieren
                self.schemas[event_type][version] = schema

                # Latest Version aktualisieren
                if self._is_newer_version(version, self.latest_versions[event_type]):
                    self.latest_versions[event_type] = version

                # Cache invalidieren
                self._invalidate_cache(event_type)

                logger.info(f"Platform Schema registriert: {event_type} v{version}")
                return True
            logger.error(f"Schema-Kompatibilität fehlgeschlagen: {event_type} v{version}")
            return False

        except Exception as e:
            logger.error(f"Fehler beim Registrieren des Platform Schemas: {e}")
            return False

    async def get_schema(self, event_type: str, version: str | None = None) -> PlatformEventSchema | None:
        """Gibt Platform Event Schema zurück"""
        try:
            if event_type not in self.schemas:
                return None

            # Verwende latest Version falls nicht spezifiziert
            if version is None:
                version = self.latest_versions.get(event_type)
                if version is None:
                    return None

            return self.schemas[event_type].get(version)

        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Platform Schemas: {e}")
            return None

    async def validate_event(self, event) -> bool:
        """Validiert Platform Event gegen registriertes Schema"""
        try:
            self.validations_performed += 1

            # Cache-Key erstellen
            cache_key = f"{event.event_type}:{event.version}:{hash(json.dumps(event.data, sort_keys=True))}"

            # Prüfe Cache
            if cache_key in self.validation_cache:
                self.cache_hits += 1
                result = self.validation_cache[cache_key]
                return result.valid

            # Schema abrufen
            schema = await self.get_schema(event.event_type, event.version)
            if schema is None:
                logger.warning(f"Kein Schema gefunden für Platform Event: {event.event_type} v{event.version}")
                return False

            # Validierung durchführen
            result = await self._validate_against_schema(event.data, schema)

            # Cache aktualisieren
            if len(self.validation_cache) < self.cache_max_size:
                self.validation_cache[cache_key] = result

            if not result.valid:
                self.validation_errors += 1
                logger.error(f"Platform Event Validierung fehlgeschlagen: {event.event_type} - {result.errors}")

            return result.valid

        except Exception as e:
            self.validation_errors += 1
            logger.error(f"Fehler bei Platform Event Validierung: {e}")
            return False

    async def _validate_against_schema(self, data: dict[str, Any], schema: PlatformEventSchema) -> SchemaValidationResult:
        """Validiert Daten gegen JSON Schema"""
        errors = []
        warnings = []

        try:
            # JSON Schema Validierung
            validate(instance=data, schema=schema.schema)

            return SchemaValidationResult(
                valid=True,
                errors=errors,
                warnings=warnings,
                schema_version=schema.version
            )

        except ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
            return SchemaValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                schema_version=schema.version
            )
        except Exception as e:
            errors.append(f"Validation error: {e!s}")
            return SchemaValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                schema_version=schema.version
            )

    async def _validate_schema_compatibility(self, new_schema: PlatformEventSchema) -> bool:
        """Prüft Schema-Kompatibilität mit bestehenden Versionen"""
        try:
            event_type = new_schema.event_type

            # Wenn erstes Schema für Event-Typ, dann kompatibel
            if event_type not in self.schemas or not self.schemas[event_type]:
                return True

            # Hole aktuellste Version
            latest_version = self.latest_versions.get(event_type)
            if latest_version is None:
                return True

            latest_schema = self.schemas[event_type].get(latest_version)
            if latest_schema is None:
                return True

            # Kompatibilitäts-Prüfung basierend auf Modus
            compatibility_mode = new_schema.compatibility

            if compatibility_mode == SchemaCompatibility.NONE:
                return True
            if compatibility_mode == SchemaCompatibility.BACKWARD:
                return await self._check_backward_compatibility(latest_schema, new_schema)
            if compatibility_mode == SchemaCompatibility.FORWARD:
                return await self._check_forward_compatibility(latest_schema, new_schema)
            if compatibility_mode == SchemaCompatibility.FULL:
                backward_ok = await self._check_backward_compatibility(latest_schema, new_schema)
                forward_ok = await self._check_forward_compatibility(latest_schema, new_schema)
                return backward_ok and forward_ok

            return True

        except Exception as e:
            logger.error(f"Fehler bei Schema-Kompatibilitätsprüfung: {e}")
            return False

    async def _check_backward_compatibility(self, old_schema: PlatformEventSchema, new_schema: PlatformEventSchema) -> bool:
        """Prüft Backward-Kompatibilität (neue Version kann alte Daten lesen)"""
        try:
            old_required = set(old_schema.schema.get("required", []))
            new_required = set(new_schema.schema.get("required", []))

            # Neue Version darf keine zusätzlichen Required-Felder haben
            if not new_required.issubset(old_required):
                logger.warning(f"Backward compatibility violation: new required fields {new_required - old_required}")
                return False

            # Weitere Kompatibilitätsprüfungen können hier hinzugefügt werden
            return True

        except Exception as e:
            logger.error(f"Fehler bei Backward-Kompatibilitätsprüfung: {e}")
            return False

    async def _check_forward_compatibility(self, old_schema: PlatformEventSchema, new_schema: PlatformEventSchema) -> bool:
        """Prüft Forward-Kompatibilität (alte Version kann neue Daten lesen)"""
        try:
            old_properties = set(old_schema.schema.get("properties", {}).keys())
            new_properties = set(new_schema.schema.get("properties", {}).keys())

            # Neue Version darf keine Felder entfernen
            if not old_properties.issubset(new_properties):
                logger.warning(f"Forward compatibility violation: removed fields {old_properties - new_properties}")
                return False

            return True

        except Exception as e:
            logger.error(f"Fehler bei Forward-Kompatibilitätsprüfung: {e}")
            return False

    def _is_newer_version(self, version1: str, version2: str) -> bool:
        """Prüft ob version1 neuer als version2 ist"""
        try:
            # Einfache Semantic Versioning Vergleich
            v1_parts = [int(x) for x in version1.split(".")]
            v2_parts = [int(x) for x in version2.split(".")]

            # Pad mit Nullen falls unterschiedliche Länge
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))

            return v1_parts > v2_parts

        except Exception:
            # Fallback: String-Vergleich
            return version1 > version2

    def _invalidate_cache(self, event_type: str):
        """Invalidiert Validierungs-Cache für Event-Typ"""
        keys_to_remove = [key for key in self.validation_cache.keys() if key.startswith(f"{event_type}:")]
        for key in keys_to_remove:
            del self.validation_cache[key]

    async def _load_base_schemas(self):
        """Lädt Basis-Schemas für Platform Events"""
        try:
            # Base Event Schema
            base_event_schema = PlatformEventSchema(
                event_type="platform.base.event",
                version="1.0",
                schema={
                    "type": "object",
                    "properties": {
                        "event_type": {"type": "string"},
                        "event_id": {"type": "string"},
                        "source_service": {"type": "string"},
                        "data": {"type": "object"},
                        "correlation_id": {"type": "string"},
                        "timestamp": {"type": "string", "format": "date-time"},
                        "version": {"type": "string"}
                    },
                    "required": ["event_type", "event_id", "source_service", "data"]
                },
                description="Basis-Schema für alle Platform Events"
            )

            await self.register_schema(base_event_schema)

            logger.debug("Platform Basis-Schemas geladen")

        except Exception as e:
            logger.error(f"Fehler beim Laden der Platform Basis-Schemas: {e}")

    def get_all_schemas(self) -> dict[str, list[PlatformEventSchema]]:
        """Gibt alle registrierten Schemas zurück"""
        result = {}
        for event_type, versions in self.schemas.items():
            result[event_type] = list(versions.values())
        return result

    def get_metrics(self) -> dict[str, Any]:
        """Gibt Schema Registry Metriken zurück"""
        total_schemas = sum(len(versions) for versions in self.schemas.values())

        return {
            "initialized": self.initialized,
            "total_event_types": len(self.schemas),
            "total_schemas": total_schemas,
            "validations_performed": self.validations_performed,
            "validation_errors": self.validation_errors,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.validation_cache),
            "cache_hit_rate": self.cache_hits / max(self.validations_performed, 1) * 100
        }
