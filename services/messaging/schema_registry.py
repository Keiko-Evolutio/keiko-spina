"""Schema-Registry für KEI-Bus (JSON/Avro/Protobuf) mit einfacher Persistenz.

Funktionen:
- JSON Schema (Draft 2020-12) Validierung
- Avro Schema Validierung via fastavro (Syntax Check)
- (Platzhalter) Protobuf Schema (Descriptor) Hinterlegung
- Persistenz als Dateiablage unter `./schemas` (versioniert nach URI)
- Kompatibilitäts-Regeln (einfach): backward/forward (JSON/Avro minimal)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, validate

from kei_logging import get_logger

logger = get_logger(__name__)

try:  # Optional: fastavro ist nicht in jeder Testumgebung installiert
    from fastavro.schema import parse_schema as avro_parse_schema  # type: ignore
    _FASTAVRO_AVAILABLE = True
except ImportError:  # pragma: no cover - Fallback, wenn fastavro fehlt
    _FASTAVRO_AVAILABLE = False
    def avro_parse_schema(schema):  # type: ignore
        """Fallback-Parser der Schema unverändert zurückgibt, wenn fastavro fehlt."""
        return schema
except Exception as e:  # pragma: no cover - Unerwarteter Import-Fehler
    logger.debug(f"Unerwarteter Fehler beim Import von fastavro: {e}")
    _FASTAVRO_AVAILABLE = False
    def avro_parse_schema(schema):  # type: ignore
        """Fallback-Parser der Schema unverändert zurückgibt, wenn fastavro fehlt."""
        return schema


class SchemaRegistry:
    """In-Memory-Registry für JSON Schemas."""

    def __init__(self) -> None:
        self._schemas: dict[str, dict[str, Any]] = {}
        self._types: dict[str, str] = {}  # uri -> type: json|avro|protobuf
        self._versions: dict[str, int] = {}  # uri -> latest version
        self._root = Path(os.getenv("KEI_SCHEMA_ROOT", "schemas")).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def register(self, uri: str, schema: dict[str, Any], schema_type: str = "json", compatibility: str = "backward") -> int:
        """Registriert ein Schema unter gegebener URI.

        Returns:
            Neue Versionsnummer
        """
        schema_type = schema_type.lower()
        if schema_type not in {"json", "avro", "protobuf"}:
            from core.exceptions import KeikoValidationError
            raise KeikoValidationError("Nicht unterstützter Schema-Typ", details={"type": str(schema_type)})

        # Kompatibilität prüfen falls Version existiert
        prev_schema = self._schemas.get(uri)
        prev_type = self._types.get(uri)
        if prev_schema is not None and prev_type == schema_type:
            self._check_compatibility(prev_schema, schema, schema_type, mode=compatibility)

        # Syntax-Validierung
        self._validate_syntax(schema, schema_type)

        # Persistenz
        version = self._versions.get(uri, 0) + 1
        self._versions[uri] = version
        self._schemas[uri] = schema
        self._types[uri] = schema_type
        self._persist_schema(uri, version, schema, schema_type)
        return version

    def get(self, uri: str) -> dict[str, Any] | None:
        """Gibt aktuelles Schema zur URI zurück."""
        return self._schemas.get(uri)

    def get_with_type(self, uri: str) -> tuple[dict[str, Any] | None, str | None]:
        """Gibt Schema und Typ zurück."""
        return self._schemas.get(uri), self._types.get(uri)

    def validate_payload(self, uri: str | None, payload: dict[str, Any]) -> None:
        """Validiert Payload gegen Schema-URI (falls angegeben)."""
        if not uri:
            return
        schema, s_type = self.get_with_type(uri)
        if not schema or not s_type:
            logger.warning(f"Schema nicht gefunden: {uri}")
            return
        if s_type == "json":
            Draft202012Validator.check_schema(schema)
            validate(instance=payload, schema=schema)
        elif s_type == "avro":
            # Vereinfachte Prüfung: Nur parse; wenn fastavro fehlt, wird Fallback verwendet
            avro_parse_schema(schema)  # raises bei Syntaxfehlern, falls verfügbar
        elif s_type == "protobuf":
            # Platzhalter: Protobuf-Descriptor/Validation später
            pass

    def _validate_syntax(self, schema: dict[str, Any], schema_type: str) -> None:
        if schema_type == "json":
            Draft202012Validator.check_schema(schema)
        elif schema_type == "avro":
            avro_parse_schema(schema)
        elif schema_type == "protobuf":
            # Placeholder: accept
            return

    def _persist_schema(self, uri: str, version: int, schema: dict[str, Any], schema_type: str) -> None:
        safe_uri = uri.replace("://", "_").replace("/", "_")
        out_dir = self._root / safe_uri
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {"uri": uri, "version": version, "type": schema_type}
        with open(out_dir / f"v{version}.{schema_type}.json", "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "schema": schema}, f, ensure_ascii=False, indent=2)

    def _check_compatibility(self, old: dict[str, Any], new: dict[str, Any], schema_type: str, mode: str = "backward") -> None:
        """Sehr einfache Kompatibilitäts-Prüfung (Demo):
        - JSON: backward: neue required Felder verboten; forward: Entfernen required Felder verboten
        - Avro: Platzhalter (nur Syntax geprüft).
        """
        if schema_type == "json":
            old_req = set(old.get("required") or [])
            new_req = set(new.get("required") or [])
            if mode == "backward":
                # Neue required Felder wären nicht backward-kompatibel
                added_required = new_req - old_req
                if added_required:
                    from core.exceptions import KeikoValidationError
                    raise KeikoValidationError("JSON Schema nicht backward-kompatibel", details={"added_required": list(added_required)})
            elif mode == "forward":
                removed_required = old_req - new_req
                if removed_required:
                    from core.exceptions import KeikoValidationError
                    raise KeikoValidationError("JSON Schema nicht forward-kompatibel", details={"removed_required": list(removed_required)})
        elif schema_type == "avro":
            # Für Avro könnte man mit Schema-Evolution-Regeln (Default-Werte etc.) prüfen
            return
        elif schema_type == "protobuf":
            return

    def list_all(self) -> list[dict[str, Any]]:
        """Listet alle registrierten Schemata aus dem Persistenz-Verzeichnis."""
        items: list[dict[str, Any]] = []
        try:
            for child in self._root.iterdir():
                if not child.is_dir():
                    continue
                versions: list[tuple[int, Path]] = []
                for f in child.iterdir():
                    if not f.is_file():
                        continue
                    name = f.name
                    if name.startswith("v") and name.endswith(".json"):
                        try:
                            ver = int(name.split(".")[0][1:])
                            versions.append((ver, f))
                        except Exception:
                            continue
                if not versions:
                    continue
                versions.sort(key=lambda x: x[0])
                last_ver, last_file = versions[-1]
                try:
                    content = json.loads(last_file.read_text(encoding="utf-8"))
                    meta = content.get("meta") or {}
                    items.append({
                        "uri": meta.get("uri") or child.name,
                        "version": meta.get("version") or last_ver,
                        "type": meta.get("type") or "json",
                    })
                except Exception:
                    continue
        except Exception:
            return items
        return items


_global_registry: SchemaRegistry | None = None


def get_schema_registry() -> SchemaRegistry:
    """Gibt globale SchemaRegistry-Instanz zurück."""
    global _global_registry
    if _global_registry is None:
        _global_registry = SchemaRegistry()
    return _global_registry
