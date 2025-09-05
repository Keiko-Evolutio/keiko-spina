"""Utility-Funktionen für das specs-Modul.

Enthält wiederverwendbare Funktionen für File I/O, Validierung und
andere gemeinsame Operationen.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import yaml

from kei_logging import get_logger

from .constants import SpecConstants

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


def write_spec_file(
    data: dict[str, Any],
    output_path: Path,
    format_type: str = "yaml",
    encoding: str = SpecConstants.DEFAULT_ENCODING
) -> None:
    """Schreibt Spezifikations-Daten in Datei.

    Args:
        data: Zu schreibende Daten
        output_path: Ausgabe-Pfad
        format_type: Format ("yaml" oder "json")
        encoding: Datei-Encoding

    Raises:
        IOError: Bei Schreibfehlern
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding=encoding) as f:
        if format_type.lower() == "json":
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    logger.debug(
        f"Datei geschrieben: {output_path}",
        extra={
            "format": format_type,
            "size_bytes": output_path.stat().st_size
        }
    )


def read_spec_file(
    input_path: Path,
    encoding: str = SpecConstants.DEFAULT_ENCODING
) -> dict[str, Any]:
    """Liest Spezifikations-Daten aus Datei.

    Args:
        input_path: Eingabe-Pfad
        encoding: Datei-Encoding

    Returns:
        Geladene Daten

    Raises:
        FileNotFoundError: Wenn Datei nicht existiert
        IOError: Bei Lesefehlern
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {input_path}")

    with open(input_path, encoding=encoding) as f:
        data = json.load(f) if input_path.suffix.lower() == ".json" else yaml.safe_load(f)

    logger.debug(f"Datei gelesen: {input_path}")
    return data


def ensure_directory(directory_path: Path) -> None:
    """Stellt sicher, dass Verzeichnis existiert.

    Args:
        directory_path: Verzeichnis-Pfad
    """
    directory_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Verzeichnis erstellt/überprüft: {directory_path}")


def validate_required_fields(
    data: dict[str, Any],
    required_fields: list[str],
    context: str = "data"
) -> None:
    """Validiert erforderliche Felder in Dictionary.

    Args:
        data: Zu validierende Daten
        required_fields: Liste erforderlicher Felder
        context: Kontext für Fehlermeldungen

    Raises:
        ValueError: Wenn erforderliche Felder fehlen
    """
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        raise ValueError(
            f"Erforderliche Felder fehlen in {context}: {', '.join(missing_fields)}"
        )


def merge_dictionaries(
    base_dict: dict[str, Any],
    *update_dicts: dict[str, Any]
) -> dict[str, Any]:
    """Führt mehrere Dictionaries zusammen.

    Args:
        base_dict: Basis-Dictionary
        *update_dicts: Weitere Dictionaries zum Zusammenführen

    Returns:
        Zusammengeführtes Dictionary
    """
    result = base_dict.copy()

    for update_dict in update_dicts:
        result.update(update_dict)

    return result


def sanitize_filename(filename: str) -> str:
    """Bereinigt Dateinamen von ungültigen Zeichen.

    Args:
        filename: Ursprünglicher Dateiname

    Returns:
        Bereinigter Dateiname
    """
    # Entferne/ersetze ungültige Zeichen
    invalid_chars = '<>:"/\\|?*'
    sanitized = filename

    for char in invalid_chars:
        sanitized = sanitized.replace(char, "_")

    # Entferne führende/nachfolgende Leerzeichen und Punkte
    sanitized = sanitized.strip(" .")

    # Stelle sicher, dass Dateiname nicht leer ist
    if not sanitized:
        sanitized = "unnamed"

    return sanitized


def get_file_size_mb(file_path: Path) -> float:
    """Gibt Dateigröße in MB zurück.

    Args:
        file_path: Pfad zur Datei

    Returns:
        Dateigröße in MB
    """
    if not file_path.exists():
        return 0.0

    size_bytes = file_path.stat().st_size
    return size_bytes / (1024 * 1024)


def create_backup_file(file_path: Path) -> Path:
    """Erstellt Backup einer Datei.

    Args:
        file_path: Pfad zur Original-Datei

    Returns:
        Pfad zur Backup-Datei

    Raises:
        FileNotFoundError: Wenn Original-Datei nicht existiert
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Original-Datei nicht gefunden: {file_path}")

    backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")

    # Kopiere Datei-Inhalt
    with open(file_path, "rb") as src, open(backup_path, "wb") as dst:
        dst.write(src.read())

    logger.info(f"Backup erstellt: {backup_path}")
    return backup_path


def cleanup_temp_files(directory: Path, pattern: str = "*.tmp") -> int:
    """Bereinigt temporäre Dateien in Verzeichnis.

    Args:
        directory: Verzeichnis zum Bereinigen
        pattern: Datei-Pattern (Standard: "*.tmp")

    Returns:
        Anzahl gelöschter Dateien
    """
    if not directory.exists():
        return 0

    deleted_count = 0

    for temp_file in directory.glob(pattern):
        try:
            temp_file.unlink()
            deleted_count += 1
            logger.debug(f"Temporäre Datei gelöscht: {temp_file}")
        except OSError as e:
            logger.warning(f"Konnte temporäre Datei nicht löschen {temp_file}: {e}")

    if deleted_count > 0:
        logger.info(f"{deleted_count} temporäre Dateien bereinigt in {directory}")

    return deleted_count


def format_spec_for_output(
    spec: dict[str, Any],
    format_type: str = "yaml",
    indent: int = 2
) -> str:
    """Formatiert Spezifikation für Ausgabe.

    Args:
        spec: Spezifikations-Dictionary
        format_type: Ausgabe-Format ("yaml" oder "json")
        indent: Einrückung für JSON

    Returns:
        Formatierte Spezifikation als String
    """
    if format_type.lower() == "json":
        return json.dumps(spec, ensure_ascii=False, indent=indent)
    return yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)


def extract_spec_metadata(spec: dict[str, Any]) -> dict[str, Any]:
    """Extrahiert Metadaten aus Spezifikation.

    Args:
        spec: Spezifikations-Dictionary

    Returns:
        Metadaten-Dictionary
    """
    metadata = {}

    # Basis-Informationen
    if "info" in spec:
        info = spec["info"]
        metadata.update({
            "title": info.get("title", "Unknown"),
            "version": info.get("version", "Unknown"),
            "description": info.get("description", "")
        })

    # OpenAPI-spezifische Metadaten
    if "openapi" in spec:
        metadata["spec_type"] = "openapi"
        metadata["spec_version"] = spec["openapi"]
        metadata["paths_count"] = len(spec.get("paths", {}))

    # AsyncAPI-spezifische Metadaten
    elif "asyncapi" in spec:
        metadata["spec_type"] = "asyncapi"
        metadata["spec_version"] = spec["asyncapi"]
        metadata["channels_count"] = len(spec.get("channels", {}))

    return metadata
