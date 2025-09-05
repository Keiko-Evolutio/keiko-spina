# backend/voice/common/exceptions.py
"""Voice-spezifische Exception-Hierarchie.

Strukturierte Exceptions für bessere Fehlerbehandlung und Debugging
im Voice-Modul.
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# Base Exceptions
# =============================================================================

class VoiceError(Exception):
    """Basis-Exception für alle Voice-spezifischen Fehler.

    Stellt gemeinsame Funktionalität für alle Voice-Exceptions bereit.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None
    ) -> None:
        """Initialisiert Voice-Error.

        Args:
            message: Fehlermeldung
            error_code: Eindeutiger Error-Code
            details: Zusätzliche Fehler-Details
            original_error: Ursprünglicher Fehler (falls Wrapper)
        """
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.original_error = original_error

    def __str__(self) -> str:
        """String-Repräsentation des Fehlers."""
        parts = [super().__str__()]

        if self.error_code:
            parts.append(f"Code: {self.error_code}")

        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {details_str}")

        return " | ".join(parts)


# =============================================================================
# Configuration Exceptions
# =============================================================================

class ConfigurationError(VoiceError):
    """Fehler bei Configuration-Management."""

    def __init__(
        self,
        message: str,
        *,
        config_id: str | None = None,
        **kwargs: Any
    ) -> None:
        """Initialisiert Configuration-Error.

        Args:
            message: Fehlermeldung
            config_id: ID der betroffenen Konfiguration
            **kwargs: Weitere Argumente für VoiceError
        """
        super().__init__(message, **kwargs)
        self.config_id = config_id
        if config_id:
            self.details["config_id"] = config_id


class ConfigurationNotFoundError(ConfigurationError):
    """Konfiguration wurde nicht gefunden."""

    def __init__(self, config_id: str, **kwargs: Any) -> None:
        """Initialisiert Configuration-Not-Found-Error.

        Args:
            config_id: ID der nicht gefundenen Konfiguration
            **kwargs: Weitere Argumente für ConfigurationError
        """
        message = f"Konfiguration nicht gefunden: {config_id}"
        super().__init__(
            message,
            config_id=config_id,
            error_code="CONFIG_NOT_FOUND",
            **kwargs
        )


class DefaultConfigurationError(ConfigurationError):
    """Fehler beim Laden der Standard-Konfiguration."""

    def __init__(self, reason: str, **kwargs: Any) -> None:
        """Initialisiert Default-Configuration-Error.

        Args:
            reason: Grund für den Fehler
            **kwargs: Weitere Argumente für ConfigurationError
        """
        message = f"Standard-Konfiguration konnte nicht geladen werden: {reason}"
        super().__init__(
            message,
            error_code="DEFAULT_CONFIG_ERROR",
            **kwargs
        )


# =============================================================================
# File Processing Exceptions
# =============================================================================

class FileProcessingError(VoiceError):
    """Fehler bei File-Processing."""

    def __init__(
        self,
        message: str,
        *,
        file_path: str | None = None,
        **kwargs: Any
    ) -> None:
        """Initialisiert File-Processing-Error.

        Args:
            message: Fehlermeldung
            file_path: Pfad der betroffenen Datei
            **kwargs: Weitere Argumente für VoiceError
        """
        super().__init__(message, **kwargs)
        self.file_path = file_path
        if file_path:
            self.details["file_path"] = file_path


class PromptyFileNotFoundError(FileProcessingError):
    """Prompty-Datei wurde nicht gefunden."""

    def __init__(self, file_path: str, **kwargs: Any) -> None:
        """Initialisiert Prompty-File-Not-Found-Error.

        Args:
            file_path: Pfad der nicht gefundenen Datei
            **kwargs: Weitere Argumente für FileProcessingError
        """
        message = f"Prompty-Datei nicht gefunden: {file_path}"
        super().__init__(
            message,
            file_path=file_path,
            error_code="PROMPTY_FILE_NOT_FOUND",
            **kwargs
        )


class PromptyParsingError(FileProcessingError):
    """Fehler beim Parsen von Prompty-Dateien."""

    def __init__(
        self,
        file_path: str,
        reason: str,
        *,
        line_number: int | None = None,
        **kwargs: Any
    ) -> None:
        """Initialisiert Prompty-Parsing-Error.

        Args:
            file_path: Pfad der Datei
            reason: Grund für den Parsing-Fehler
            line_number: Zeilennummer des Fehlers (falls bekannt)
            **kwargs: Weitere Argumente für FileProcessingError
        """
        message = f"Fehler beim Parsen der Prompty-Datei {file_path}: {reason}"
        super().__init__(
            message,
            file_path=file_path,
            error_code="PROMPTY_PARSING_ERROR",
            **kwargs
        )
        self.line_number = line_number
        if line_number:
            self.details["line_number"] = line_number


class YAMLParsingError(FileProcessingError):
    """Fehler beim YAML-Parsing."""

    def __init__(
        self,
        content_source: str,
        reason: str,
        **kwargs: Any
    ) -> None:
        """Initialisiert YAML-Parsing-Error.

        Args:
            content_source: Quelle des YAML-Contents (Datei oder "inline")
            reason: Grund für den Parsing-Fehler
            **kwargs: Weitere Argumente für FileProcessingError
        """
        message = f"YAML-Parsing-Fehler in {content_source}: {reason}"
        super().__init__(
            message,
            error_code="YAML_PARSING_ERROR",
            **kwargs
        )
        self.content_source = content_source
        self.details["content_source"] = content_source


# =============================================================================
# Database Exceptions
# =============================================================================

class DatabaseError(VoiceError):
    """Fehler bei Datenbank-Operationen."""

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        **kwargs: Any
    ) -> None:
        """Initialisiert Database-Error.

        Args:
            message: Fehlermeldung
            operation: Name der fehlgeschlagenen Operation
            **kwargs: Weitere Argumente für VoiceError
        """
        super().__init__(message, **kwargs)
        self.operation = operation
        if operation:
            self.details["operation"] = operation


class CosmosDBConnectionError(DatabaseError):
    """Fehler bei Cosmos DB-Verbindung."""

    def __init__(self, reason: str, **kwargs: Any) -> None:
        """Initialisiert Cosmos DB-Connection-Error.

        Args:
            reason: Grund für den Verbindungsfehler
            **kwargs: Weitere Argumente für DatabaseError
        """
        message = f"Cosmos DB-Verbindung fehlgeschlagen: {reason}"
        super().__init__(
            message,
            operation="connect",
            error_code="COSMOSDB_CONNECTION_ERROR",
            **kwargs
        )


class CosmosDBQueryError(DatabaseError):
    """Fehler bei Cosmos DB-Queries."""

    def __init__(
        self,
        query: str,
        reason: str,
        **kwargs: Any
    ) -> None:
        """Initialisiert Cosmos DB-Query-Error.

        Args:
            query: Fehlgeschlagene Query
            reason: Grund für den Query-Fehler
            **kwargs: Weitere Argumente für DatabaseError
        """
        message = f"Cosmos DB-Query fehlgeschlagen: {reason}"
        super().__init__(
            message,
            operation="query",
            error_code="COSMOSDB_QUERY_ERROR",
            **kwargs
        )
        self.query = query
        self.details["query"] = query


# =============================================================================
# Validation Exceptions
# =============================================================================

class ValidationError(VoiceError):
    """Fehler bei Daten-Validierung."""

    def __init__(
        self,
        message: str,
        *,
        field_name: str | None = None,
        field_value: Any | None = None,
        **kwargs: Any
    ) -> None:
        """Initialisiert Validation-Error.

        Args:
            message: Fehlermeldung
            field_name: Name des invaliden Feldes
            field_value: Wert des invaliden Feldes
            **kwargs: Weitere Argumente für VoiceError
        """
        super().__init__(message, **kwargs)
        self.field_name = field_name
        self.field_value = field_value
        if field_name:
            self.details["field_name"] = field_name
        if field_value is not None:
            self.details["field_value"] = str(field_value)


class InvalidCategoryError(ValidationError):
    """Ungültige Kategorie."""

    def __init__(self, category: str, valid_categories: set[str], **kwargs: Any) -> None:
        """Initialisiert Invalid-Category-Error.

        Args:
            category: Ungültige Kategorie
            valid_categories: Set gültiger Kategorien
            **kwargs: Weitere Argumente für ValidationError
        """
        valid_list = ", ".join(sorted(valid_categories))
        message = f"Ungültige Kategorie '{category}'. Gültig: {valid_list}"
        super().__init__(
            message,
            field_name="category",
            field_value=category,
            error_code="INVALID_CATEGORY",
            **kwargs
        )
        self.valid_categories = valid_categories
        self.details["valid_categories"] = valid_list


# =============================================================================
# Utility Functions
# =============================================================================

def wrap_exception(
    original_error: Exception,
    voice_error_class: type[VoiceError],
    message: str,
    **kwargs: Any
) -> VoiceError:
    """Wraps eine Standard-Exception in eine Voice-Exception.

    Args:
        original_error: Ursprüngliche Exception
        voice_error_class: Voice-Exception-Klasse
        message: Neue Fehlermeldung
        **kwargs: Weitere Argumente für die Voice-Exception

    Returns:
        Voice-Exception mit Original-Error als Context
    """
    return voice_error_class(
        message,
        original_error=original_error,
        **kwargs
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration Exceptions
    "ConfigurationError",
    "ConfigurationNotFoundError",
    "CosmosDBConnectionError",
    "CosmosDBQueryError",
    # Database Exceptions
    "DatabaseError",
    "DefaultConfigurationError",
    # File Processing Exceptions
    "FileProcessingError",
    "InvalidCategoryError",
    "PromptyFileNotFoundError",
    "PromptyParsingError",
    # Validation Exceptions
    "ValidationError",
    # Base Exceptions
    "VoiceError",
    "YAMLParsingError",
    # Utilities
    "wrap_exception",
]
