"""Enterprise-Grade Prompty Parser.

Refactored und optimierter Prompty-Parser mit verbesserter Code-Qualität,
klarer Separation of Concerns und umfassendem Error-Handling.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles
import yaml

from kei_logging import get_logger

# Voice-spezifische Imports für Legacy-Kompatibilität
from ..common.constants import (
    CONFIG_CATEGORY_KEY,
    CONFIG_CONTENT_KEY,
    CONFIG_DEFAULT_KEY,
    CONFIG_ID_KEY,
    CONFIG_NAME_KEY,
    CONFIG_TOOLS_KEY,
    INLINE_PATH_NAME,
)
from .constants import (
    ALLOWED_TEMPLATE_CATEGORIES,
    DEFAULT_ENCODING,
    DEFAULT_TEMPLATE_CATEGORY,
    ERROR_TEMPLATE_TOO_LARGE,
    ERROR_TEMPLATE_TOO_SMALL,
    ERROR_YAML_PARSING,
    MAX_TEMPLATE_SIZE_BYTES,
    MIN_TEMPLATE_SIZE_BYTES,
    MIN_YAML_PARTS,
    OPTIONAL_METADATA_FIELDS,
    REQUIRED_METADATA_FIELDS,
    YAML_DELIMITER,
)
from .exceptions import (
    PromptyParsingError,
    PromptyValidationError,
    TemplateNotFoundError,
    YAMLParsingError,
    wrap_exception,
)

logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class ParseResult:
    """Ergebnis des Prompty-Parsing-Prozesses.

    Enthält alle extrahierten Informationen aus einer Prompty-Datei
    mit klarer Struktur und Type-Safety.
    """

    # Metadata
    name: str
    category: str
    version: str
    author: str | None = None
    description: str | None = None

    # Content
    template_content: str = ""

    # Configuration
    tools: list[dict[str, Any]] = None
    parameters: dict[str, Any] = None

    # Source Information
    source_path: str | None = None
    is_default: bool = False

    def __post_init__(self) -> None:
        """Post-Initialisierung für Mutable Defaults."""
        if self.tools is None:
            object.__setattr__(self, "tools", [])
        if self.parameters is None:
            object.__setattr__(self, "parameters", {})

    def to_voice_config_dict(self) -> dict[str, Any]:
        """Konvertiert ParseResult zu Voice-spezifischem Configuration-Dictionary.

        Für Legacy-Kompatibilität mit common/prompty_parser.py API.

        Returns:
            Configuration-Dictionary im Voice-Format
        """
        return {
            CONFIG_ID_KEY: self.name,
            CONFIG_NAME_KEY: self.name,
            CONFIG_CATEGORY_KEY: self.category,
            CONFIG_DEFAULT_KEY: self.is_default,
            CONFIG_CONTENT_KEY: self.template_content,
            CONFIG_TOOLS_KEY: self.tools,
        }


@dataclass(frozen=True)
class ValidationResult:
    """Ergebnis der Template-Validation.

    Enthält Validation-Status und detaillierte Fehlerinformationen.
    """

    is_valid: bool
    errors: list[str] = None
    warnings: list[str] = None

    def __post_init__(self) -> None:
        """Post-Initialisierung für Mutable Defaults."""
        if self.errors is None:
            object.__setattr__(self, "errors", [])
        if self.warnings is None:
            object.__setattr__(self, "warnings", [])


# =============================================================================
# YAML Parser
# =============================================================================

class YAMLParser:
    """Optimierter YAML-Parser für Prompty-Templates.

    Bietet robustes YAML-Parsing mit umfassendem Error-Handling
    und Performance-Optimierungen.
    """

    def __init__(self, *, strict_mode: bool = False) -> None:
        """Initialisiert YAML-Parser.

        Args:
            strict_mode: Aktiviert strenge Validation
        """
        self._strict_mode = strict_mode
        self._logger = logger

    def parse_yaml_content(
        self,
        content: str,
        *,
        source: str = "unknown",
    ) -> dict[str, Any]:
        """Parsed YAML-Content mit robustem Error-Handling.

        Args:
            content: YAML-Content zum Parsen
            source: Quelle für Error-Messages

        Returns:
            Geparste YAML-Daten

        Raises:
            YAMLParsingError: Bei YAML-Syntax-Fehlern
        """
        if not content.strip():
            return {}

        try:
            parsed = yaml.safe_load(content)

            if parsed is None:
                return {}

            if not isinstance(parsed, dict):
                if self._strict_mode:
                    raise YAMLParsingError(
                        f"YAML muss Dictionary sein, erhalten: {type(parsed).__name__}"
                    )
                self._logger.warning(f"YAML in {source} ist kein Dictionary")
                return {}

            return parsed

        except yaml.YAMLError as e:
            raise YAMLParsingError(
                ERROR_YAML_PARSING.format(error=str(e)),
                cause=e
            )
        except Exception as e:
            raise wrap_exception(
                e,
                YAMLParsingError,
                f"Unerwarteter YAML-Fehler in {source}: {e}"
            )

    def extract_front_matter(
        self,
        content: str,
        *,
        source: str = "unknown",
    ) -> tuple[dict[str, Any], str]:
        """Extrahiert YAML Front Matter aus Template-Content.

        Args:
            content: Template-Content mit Front Matter
            source: Quelle für Error-Messages

        Returns:
            Tuple aus (Metadata Dictionary, Template Content)

        Raises:
            PromptyParsingError: Bei ungültigem Format
        """
        content = content.strip()

        if not content.startswith(YAML_DELIMITER):
            return {}, content

        parts = content.split(YAML_DELIMITER, 2)
        if len(parts) < MIN_YAML_PARTS:
            raise PromptyParsingError(
                f"Ungültiges Front Matter-Format. Erwartet mindestens {MIN_YAML_PARTS} Teile, erhalten: {len(parts)}",
                context={"source": source, "parts_found": len(parts)}
            )

        yaml_content = parts[1].strip()
        template_content = parts[2].strip()

        # Parse YAML Front Matter
        metadata = self.parse_yaml_content(
            yaml_content,
            source=f"{source} (Front Matter)"
        )

        return metadata, template_content


# =============================================================================
# Template Validator
# =============================================================================

class TemplateValidator:
    """Validator für Prompty-Templates und Metadata.

    Implementiert umfassende Validation-Rules für Template-Struktur,
    Metadata und Content mit konfigurierbaren Validation-Levels.
    """

    def __init__(self, *, strict_mode: bool = False) -> None:
        """Initialisiert Template-Validator.

        Args:
            strict_mode: Aktiviert strenge Validation
        """
        self._strict_mode = strict_mode
        self._logger = logger

    def validate_metadata(
        self,
        metadata: dict[str, Any],
        *,
        source: str = "unknown",
    ) -> ValidationResult:
        """Validiert Template-Metadata.

        Args:
            metadata: Metadata-Dictionary
            source: Quelle für Error-Messages

        Returns:
            Validation-Ergebnis
        """
        errors = []
        warnings = []

        # Required Fields prüfen
        for field in REQUIRED_METADATA_FIELDS:
            if field not in metadata:
                errors.append(f"Erforderliches Feld fehlt: {field}")

        # Category validieren
        category = metadata.get("category", DEFAULT_TEMPLATE_CATEGORY)
        if category not in ALLOWED_TEMPLATE_CATEGORIES:
            errors.append(
                f"Ungültige Kategorie '{category}'. "
                f"Erlaubt: {', '.join(ALLOWED_TEMPLATE_CATEGORIES)}"
            )

        # Optional Fields prüfen (nur Warnings)
        for field in metadata:
            if field not in REQUIRED_METADATA_FIELDS | OPTIONAL_METADATA_FIELDS:
                warnings.append(f"Unbekanntes Metadata-Feld: {field}")

        # Name validieren
        name = metadata.get("name", "")
        if isinstance(name, str) and len(name.strip()) == 0:
            errors.append("Template-Name darf nicht leer sein")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def validate_template_content(
        self,
        content: str,
        *,
        source: str = "unknown",
    ) -> ValidationResult:
        """Validiert Template-Content.

        Args:
            content: Template-Content
            source: Quelle für Error-Messages

        Returns:
            Validation-Ergebnis
        """
        errors = []
        warnings = []

        # Content-Länge prüfen
        content_length = len(content.encode("utf-8"))

        if content_length > MAX_TEMPLATE_SIZE_BYTES:
            errors.append(
                ERROR_TEMPLATE_TOO_LARGE.format(
                    size=content_length,
                    max_size=MAX_TEMPLATE_SIZE_BYTES
                )
            )

        if content_length < MIN_TEMPLATE_SIZE_BYTES:
            errors.append(
                ERROR_TEMPLATE_TOO_SMALL.format(
                    size=content_length,
                    min_size=MIN_TEMPLATE_SIZE_BYTES
                )
            )

        # Content-Qualität prüfen
        if not content.strip():
            errors.append("Template-Content darf nicht leer sein")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


# =============================================================================
# Prompty Parser
# =============================================================================

class PromptyParser:
    """Enterprise-Grade Prompty-Parser.

    Hauptklasse für das Parsing von Prompty-Templates mit optimierter
    Performance, umfassendem Error-Handling und klarer API.
    """

    def __init__(
        self,
        *,
        yaml_parser: YAMLParser | None = None,
        validator: TemplateValidator | None = None,
        strict_mode: bool = False,
    ) -> None:
        """Initialisiert Prompty-Parser.

        Args:
            yaml_parser: YAML-Parser-Instanz
            validator: Template-Validator-Instanz
            strict_mode: Aktiviert strenge Validation
        """
        self._yaml_parser = yaml_parser or YAMLParser(strict_mode=strict_mode)
        self._validator = validator or TemplateValidator(strict_mode=strict_mode)
        self._strict_mode = strict_mode
        self._logger = logger

    def parse_content(
        self,
        content: str,
        *,
        source: str = "inline",
        is_default: bool = False,
    ) -> ParseResult:
        """Parsed Prompty-Content zu ParseResult.

        Args:
            content: Prompty-Template-Content
            source: Quelle des Contents
            is_default: Ob es sich um Default-Template handelt

        Returns:
            Parse-Ergebnis mit allen extrahierten Daten

        Raises:
            PromptyParsingError: Bei Parsing-Fehlern
            PromptyValidationError: Bei Validation-Fehlern
        """
        try:
            # Front Matter extrahieren
            metadata, template_content = self._yaml_parser.extract_front_matter(
                content,
                source=source
            )

            # Metadata validieren
            validation_result = self._validator.validate_metadata(
                metadata,
                source=source
            )

            if not validation_result.is_valid:
                raise PromptyValidationError(
                    f"Metadata-Validation fehlgeschlagen: {'; '.join(validation_result.errors)}",
                    context={"source": source, "errors": validation_result.errors}
                )

            # Template-Content validieren
            content_validation = self._validator.validate_template_content(
                template_content,
                source=source
            )

            if not content_validation.is_valid:
                raise PromptyValidationError(
                    f"Content-Validation fehlgeschlagen: {'; '.join(content_validation.errors)}",
                    context={"source": source, "errors": content_validation.errors}
                )

            # Tools extrahieren
            tools = self._extract_tools(metadata)

            # Parameters extrahieren
            parameters = metadata.get("parameters", {})

            # ParseResult erstellen
            return ParseResult(
                name=metadata.get("name", Path(source).stem),
                category=metadata.get("category", DEFAULT_TEMPLATE_CATEGORY),
                version=metadata.get("version", "1.0.0"),
                author=metadata.get("author"),
                description=metadata.get("description"),
                template_content=template_content,
                tools=tools,
                parameters=parameters,
                source_path=source,
                is_default=is_default,
            )

        except (PromptyParsingError, PromptyValidationError):
            # Re-raise bekannte Exceptions
            raise
        except Exception as e:
            raise wrap_exception(
                e,
                PromptyParsingError,
                f"Unerwarteter Parsing-Fehler in {source}: {e}",
                context={"source": source}
            )

    async def parse_file(
        self,
        file_path: Path,
        *,
        is_default: bool = False,
    ) -> ParseResult:
        """Parsed Prompty-Datei zu ParseResult.

        Args:
            file_path: Pfad zur Prompty-Datei
            is_default: Ob es sich um Default-Template handelt

        Returns:
            Parse-Ergebnis

        Raises:
            TemplateNotFoundError: Wenn Datei nicht existiert
            PromptyParsingError: Bei Parsing-Fehlern
        """
        if not file_path.exists():
            raise TemplateNotFoundError(
                file_path.name,
                search_paths=[str(file_path.parent)]
            )

        # File-Size-Check
        file_size = file_path.stat().st_size
        if file_size > MAX_TEMPLATE_SIZE_BYTES:
            raise PromptyParsingError(
                ERROR_TEMPLATE_TOO_LARGE.format(
                    size=file_size,
                    max_size=MAX_TEMPLATE_SIZE_BYTES
                ),
                context={"file_path": str(file_path), "file_size": file_size}
            )

        try:
            async with aiofiles.open(file_path, encoding=DEFAULT_ENCODING) as f:
                content = await f.read()

            return self.parse_content(
                content,
                source=str(file_path),
                is_default=is_default
            )

        except (TemplateNotFoundError, PromptyParsingError):
            # Re-raise bekannte Exceptions
            raise
        except Exception as e:
            raise wrap_exception(
                e,
                PromptyParsingError,
                f"Fehler beim Lesen der Datei {file_path}: {e}",
                context={"file_path": str(file_path)}
            )

    def _extract_tools(self, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """Extrahiert Tools aus Metadata.

        Args:
            metadata: Template-Metadata

        Returns:
            Liste der Tools
        """
        tools = metadata.get("tools", [])

        if not isinstance(tools, list):
            self._logger.warning(f"Tools-Feld ist kein Array: {type(tools)}")
            return []

        # Normalisiere Tools zu Dictionary-Format
        normalized_tools = []
        for tool in tools:
            if isinstance(tool, str):
                # String zu Dictionary konvertieren
                normalized_tools.append({"name": tool, "description": ""})
            elif isinstance(tool, dict):
                # Bereits Dictionary-Format
                normalized_tools.append(tool)
            else:
                self._logger.warning(f"Ungültiges Tool-Format: {type(tool)}")

        return normalized_tools


# =============================================================================
# Factory Functions
# =============================================================================

def create_parser(*, strict_mode: bool = False) -> PromptyParser:
    """Factory für Prompty-Parser.

    Args:
        strict_mode: Aktiviert strenge Validation

    Returns:
        Konfigurierter PromptyParser
    """
    return PromptyParser(strict_mode=strict_mode)


# =============================================================================
# Utility Functions
# =============================================================================

def parse_prompty_file(file_path: Path, *, is_default: bool = False) -> ParseResult:
    """Utility-Funktion für synchrones Parsing einer Prompty-Datei.

    Args:
        file_path: Pfad zur Prompty-Datei
        is_default: Ob es sich um Default-Template handelt

    Returns:
        Parse-Ergebnis
    """
    parser = create_parser()
    return asyncio.run(parser.parse_file(file_path, is_default=is_default))


def extract_metadata(content: str, *, source: str = "inline") -> dict[str, Any]:
    """Extrahiert nur Metadata aus Prompty-Content.

    Args:
        content: Prompty-Content
        source: Quelle für Error-Messages

    Returns:
        Metadata-Dictionary
    """
    yaml_parser = YAMLParser()
    metadata, _ = yaml_parser.extract_front_matter(content, source=source)
    return metadata


def validate_prompty_content(content: str, *, source: str = "inline") -> ValidationResult:
    """Validiert Prompty-Content ohne vollständiges Parsing.

    Args:
        content: Prompty-Content
        source: Quelle für Error-Messages

    Returns:
        Validation-Ergebnis
    """
    try:
        parser = create_parser(strict_mode=True)
        parser.parse_content(content, source=source)
        return ValidationResult(is_valid=True)
    except PromptyValidationError as e:
        return ValidationResult(
            is_valid=False,
            errors=[str(e)]
        )
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            errors=[f"Unerwarteter Fehler: {e}"]
        )


# =============================================================================
# Legacy Compatibility Functions (Voice-spezifisch)
# =============================================================================

def extract_config_from_content(
    content: str,
    *,
    source: str = INLINE_PATH_NAME,
    is_default: bool = False,
) -> dict[str, Any]:
    """Legacy-Kompatibilität für common/prompty_parser.py API.

    Args:
        content: Prompty-Content
        source: Quelle des Contents
        is_default: Ob es sich um Default-Configuration handelt

    Returns:
        Configuration-Dictionary im Voice-Format
    """
    parser = create_parser()
    result = parser.parse_content(content, source=source, is_default=is_default)
    return result.to_voice_config_dict()


async def load_from_file(
    file_path: Path,
    *,
    is_default: bool = False,
) -> dict[str, Any]:
    """Legacy-Kompatibilität für common/prompty_parser.py API.

    Args:
        file_path: Pfad zur Prompty-Datei
        is_default: Ob es sich um Default-Configuration handelt

    Returns:
        Configuration-Dictionary im Voice-Format
    """
    parser = create_parser()
    result = await parser.parse_file(file_path, is_default=is_default)
    return result.to_voice_config_dict()


# Legacy-Kompatibilitätsfunktionen für common/prompty_parser.py
def _extract_config_from_prompty(
    content: str,
    file_path: Path,
    is_default: bool = False,
) -> dict[str, Any] | None:
    """Legacy-Kompatibilität für _extract_config_from_prompty().

    Deprecated: Verwende extract_config_from_content() stattdessen.
    """
    try:
        return extract_config_from_content(
            content,
            source=str(file_path),
            is_default=is_default
        )
    except Exception:
        return None


async def load_prompty_file(
    file_path: Path,
    default: bool = False,
) -> dict[str, Any] | None:
    """Legacy-Kompatibilität für load_prompty_file().

    Deprecated: Verwende load_from_file() stattdessen.
    """
    try:
        return await load_from_file(file_path, is_default=default)
    except Exception:
        return None


def load_prompty_config(content: str) -> dict[str, Any] | None:
    """Legacy-Kompatibilität für load_prompty_config().

    Deprecated: Verwende extract_config_from_content() stattdessen.
    """
    try:
        return extract_config_from_content(content)
    except Exception:
        return None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data Classes
    "ParseResult",
    "PromptyParser",
    "TemplateValidator",
    "ValidationResult",
    # Classes
    "YAMLParser",
    "_extract_config_from_prompty",
    # Factory Functions
    "create_parser",
    # Legacy Compatibility (Voice-spezifisch)
    "extract_config_from_content",
    "extract_metadata",
    "load_from_file",
    "load_prompty_config",
    "load_prompty_file",
    # Utility Functions
    "parse_prompty_file",
    "validate_prompty_content",
]
