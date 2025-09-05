# backend/voice/common/prompty_parser.py
"""Legacy-Adapter für Prompty-File-Parsing.

DEPRECATED: Diese Datei ist ein Legacy-Adapter für die konsolidierte
Prompty-Parser-Implementierung in voice/prompty/parser.py.

Für neue Implementierungen verwende:
    from voice.prompty.parser import PromptyParser, create_parser

Dieser Adapter stellt Backward-Compatibility für bestehenden Code bereit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

# Importiere die konsolidierte Implementierung
from ..prompty.parser import (
    TemplateValidator as ModernTemplateValidator,
)
from ..prompty.parser import (
    YAMLParser as ModernYAMLParser,
)
from ..prompty.parser import (
    _extract_config_from_prompty as modern_extract_config_from_prompty,
)
from ..prompty.parser import (
    create_parser as create_modern_parser,
)
from ..prompty.parser import (
    extract_config_from_content as modern_extract_config,
)
from ..prompty.parser import (
    load_from_file as modern_load_from_file,
)
from ..prompty.parser import (
    load_prompty_config as modern_load_prompty_config,
)
from ..prompty.parser import (
    load_prompty_file as modern_load_prompty_file,
)

# Legacy-Konstanten für Kompatibilität
from .constants import (
    ERROR_PROMPTY_PARSE,
    ERROR_PROMPTY_READ,
    INLINE_PATH_NAME,
)
from .exceptions import (
    ValidationError,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

# =============================================================================
# Legacy-Adapter-Klassen
# =============================================================================

class YAMLParser:
    """Legacy-Adapter für YAMLParser.

    DEPRECATED: Verwende voice.prompty.parser.YAMLParser stattdessen.
    """

    def __init__(self) -> None:
        """Initialisiert Legacy-Adapter."""
        self._modern_parser = ModernYAMLParser()

    @staticmethod
    def parse_yaml_content(
        content: str,
        *,
        source: str = "unknown",
        strict: bool = False,
    ) -> dict[str, Any]:
        """Legacy-Adapter für parse_yaml_content."""
        parser = ModernYAMLParser(strict_mode=strict)
        return parser.parse_yaml_content(content, source=source)

    @staticmethod
    def extract_front_matter(content: str, *, source: str = "unknown") -> tuple[dict[str, Any], str]:
        """Legacy-Adapter für extract_front_matter."""
        parser = ModernYAMLParser()
        return parser.extract_front_matter(content, source=source)

class PromptyValidator:
    """Legacy-Adapter für TemplateValidator.

    DEPRECATED: Verwende voice.prompty.parser.TemplateValidator stattdessen.
    """

    def __init__(self) -> None:
        """Initialisiert Legacy-Adapter."""
        self._modern_validator = ModernTemplateValidator()

    def validate_front_matter(
        self,
        front_matter: dict[str, Any],
        *,
        source: str = "unknown",
        strict: bool = False,
    ) -> None:
        """Legacy-Adapter für validate_metadata."""
        result = self._modern_validator.validate_metadata(front_matter, source=source)
        if not result.is_valid:
            raise ValidationError(f"Validation fehlgeschlagen: {'; '.join(result.errors)}")

    def validate_configuration_data(self, config_data: dict[str, Any]) -> None:
        """Legacy-Adapter für Configuration-Validation."""
        # Basis-Validation für Configuration-Dictionary
        required_keys = ["id", "name", "category", "is_default", "content", "tools"]
        for key in required_keys:
            if key not in config_data:
                raise ValidationError(f"Erforderlicher Key fehlt: {key}")

class PromptyParser:
    """Legacy-Adapter für PromptyParser.

    DEPRECATED: Verwende voice.prompty.parser.PromptyParser stattdessen.
    """

    def __init__(
        self,
        *,
        yaml_parser: YAMLParser | None = None,
        validator: PromptyValidator | None = None,
        strict_validation: bool = False,
    ) -> None:
        """Initialisiert Legacy-Adapter."""
        self._modern_parser = create_modern_parser()
        self._strict_validation = strict_validation
        self._logger = logger

    def extract_config_from_content(
        self,
        content: str,
        *,
        source: str = INLINE_PATH_NAME,
        is_default: bool = False,
    ) -> dict[str, Any]:
        """Legacy-Adapter für extract_config_from_content."""
        return modern_extract_config(content, source=source, is_default=is_default)

    async def load_from_file(
        self,
        file_path: Path,
        *,
        is_default: bool = False,
    ) -> dict[str, Any]:
        """Legacy-Adapter für load_from_file."""
        return await modern_load_from_file(file_path, is_default=is_default)

# =============================================================================
# Factory Functions
# =============================================================================

def create_prompty_parser(*, strict_validation: bool = False) -> PromptyParser:
    """Factory für Legacy Prompty Parser.

    Args:
        strict_validation: Strict-Validation aktivieren

    Returns:
        Legacy PromptyParser-Adapter
    """
    return PromptyParser(strict_validation=strict_validation)

# =============================================================================
# Legacy Compatibility Functions
# =============================================================================

def _extract_config_from_prompty(
    content: str,
    file_path: Path,
    is_default: bool = False,
) -> dict[str, Any] | None:
    """Legacy-Kompatibilität für _extract_config_from_prompty()."""
    try:
        return modern_extract_config_from_prompty(content, file_path, is_default)
    except Exception as e:
        logger.exception(ERROR_PROMPTY_PARSE, file_path, e)
        return None

async def load_prompty_file(
    file_path: Path,
    default: bool = False,
) -> dict[str, Any] | None:
    """Legacy-Kompatibilität für load_prompty_file()."""
    try:
        return await modern_load_prompty_file(file_path, default)
    except Exception as e:
        logger.exception(ERROR_PROMPTY_READ, file_path, e)
        return None

def load_prompty_config(content: str) -> dict[str, Any] | None:
    """Legacy-Kompatibilität für load_prompty_config()."""
    try:
        return modern_load_prompty_config(content)
    except Exception as e:
        logger.exception(ERROR_PROMPTY_PARSE, INLINE_PATH_NAME, e)
        return None

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PromptyParser",
    "PromptyValidator",
    # Classes
    "YAMLParser",
    # Legacy Compatibility
    "_extract_config_from_prompty",
    # Factory Functions
    "create_prompty_parser",
    "load_prompty_config",
    "load_prompty_file",
]
