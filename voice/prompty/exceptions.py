"""Exception-Hierarchie für das Prompty-Modul.

Definiert eine klare Exception-Hierarchie für alle Prompty-bezogenen
Fehler mit aussagekräftigen Fehlermeldungen und Kontext-Informationen.
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# Base Exception
# =============================================================================

class PromptyError(Exception):
    """Basis-Exception für alle Prompty-bezogenen Fehler.

    Bietet erweiterte Funktionalität für Fehler-Kontext und -Metadaten.
    """

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialisiert Prompty-Error.

        Args:
            message: Fehlermeldung
            context: Zusätzlicher Kontext für Debugging
            cause: Ursprüngliche Exception (falls vorhanden)
        """
        super().__init__(message)
        self.context = context or {}
        self.cause = cause

    def __str__(self) -> str:
        """Gibt formatierte Fehlermeldung zurück."""
        base_message = super().__str__()

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_message += f" (Kontext: {context_str})"

        if self.cause:
            base_message += f" (Ursache: {self.cause})"

        return base_message


# =============================================================================
# Template-bezogene Exceptions
# =============================================================================

class PromptyTemplateError(PromptyError):
    """Basis-Exception für Template-bezogene Fehler."""

    def __init__(
        self,
        message: str,
        *,
        template_name: str | None = None,
        template_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialisiert Template-Error.

        Args:
            message: Fehlermeldung
            template_name: Name des betroffenen Templates
            template_path: Pfad des betroffenen Templates
            **kwargs: Zusätzliche Argumente für PromptyError
        """
        context = kwargs.get("context", {})
        if template_name:
            context["template_name"] = template_name
        if template_path:
            context["template_path"] = template_path
        kwargs["context"] = context

        super().__init__(message, **kwargs)
        self.template_name = template_name
        self.template_path = template_path


class TemplateNotFoundError(PromptyTemplateError):
    """Exception für nicht gefundene Templates."""

    def __init__(
        self,
        template_name: str,
        search_paths: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialisiert TemplateNotFoundError.

        Args:
            template_name: Name des gesuchten Templates
            search_paths: Liste der durchsuchten Pfade
            **kwargs: Zusätzliche Argumente
        """
        message = f"Template '{template_name}' nicht gefunden"
        if search_paths:
            message += f" in Pfaden: {', '.join(search_paths)}"

        context = kwargs.get("context", {})
        context["search_paths"] = search_paths or []
        kwargs["context"] = context

        super().__init__(message, template_name=template_name, **kwargs)
        self.search_paths = search_paths or []


class InvalidTemplateError(PromptyTemplateError):
    """Exception für ungültige Template-Formate."""

    def __init__(
        self,
        template_name: str,
        reason: str,
        **kwargs: Any,
    ) -> None:
        """Initialisiert InvalidTemplateError.

        Args:
            template_name: Name des ungültigen Templates
            reason: Grund für die Ungültigkeit
            **kwargs: Zusätzliche Argumente
        """
        message = f"Ungültiges Template '{template_name}': {reason}"

        context = kwargs.get("context", {})
        context["validation_reason"] = reason
        kwargs["context"] = context

        super().__init__(message, template_name=template_name, **kwargs)
        self.reason = reason


class RenderingError(PromptyTemplateError):
    """Exception für Template-Rendering-Fehler."""

    def __init__(
        self,
        template_name: str,
        rendering_error: str,
        *,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialisiert RenderingError.

        Args:
            template_name: Name des Templates
            rendering_error: Beschreibung des Rendering-Fehlers
            parameters: Parameter, die beim Rendering verwendet wurden
            **kwargs: Zusätzliche Argumente
        """
        message = f"Rendering-Fehler für Template '{template_name}': {rendering_error}"

        context = kwargs.get("context", {})
        context["rendering_error"] = rendering_error
        if parameters:
            context["parameters"] = parameters
        kwargs["context"] = context

        super().__init__(message, template_name=template_name, **kwargs)
        self.rendering_error = rendering_error
        self.parameters = parameters or {}


# =============================================================================
# Parsing-bezogene Exceptions
# =============================================================================

class PromptyParsingError(PromptyError):
    """Basis-Exception für Parsing-Fehler."""

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        line_number: int | None = None,
        column_number: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialisiert Parsing-Error.

        Args:
            message: Fehlermeldung
            source: Quelle des Parsing-Fehlers
            line_number: Zeilennummer des Fehlers
            column_number: Spaltennummer des Fehlers
            **kwargs: Zusätzliche Argumente
        """
        context = kwargs.get("context", {})
        if source:
            context["source"] = source
        if line_number:
            context["line_number"] = line_number
        if column_number:
            context["column_number"] = column_number
        kwargs["context"] = context

        super().__init__(message, **kwargs)
        self.source = source
        self.line_number = line_number
        self.column_number = column_number


class YAMLParsingError(PromptyParsingError):
    """Exception für YAML-Parsing-Fehler."""

    def __init__(
        self,
        yaml_error: str,
        **kwargs: Any,
    ) -> None:
        """Initialisiert YAML-Parsing-Error.

        Args:
            yaml_error: YAML-spezifische Fehlermeldung
            **kwargs: Zusätzliche Argumente
        """
        message = f"YAML-Parsing fehlgeschlagen: {yaml_error}"

        context = kwargs.get("context", {})
        context["yaml_error"] = yaml_error
        kwargs["context"] = context

        super().__init__(message, **kwargs)
        self.yaml_error = yaml_error


# =============================================================================
# Validation-bezogene Exceptions
# =============================================================================

class PromptyValidationError(PromptyError):
    """Basis-Exception für Validation-Fehler."""

    def __init__(
        self,
        message: str,
        *,
        field_name: str | None = None,
        field_value: Any | None = None,
        validation_rule: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialisiert Validation-Error.

        Args:
            message: Fehlermeldung
            field_name: Name des validierten Feldes
            field_value: Wert des validierten Feldes
            validation_rule: Verletzte Validation-Regel
            **kwargs: Zusätzliche Argumente
        """
        context = kwargs.get("context", {})
        if field_name:
            context["field_name"] = field_name
        if field_value is not None:
            context["field_value"] = field_value
        if validation_rule:
            context["validation_rule"] = validation_rule
        kwargs["context"] = context

        super().__init__(message, **kwargs)
        self.field_name = field_name
        self.field_value = field_value
        self.validation_rule = validation_rule


class MetadataError(PromptyValidationError):
    """Exception für Metadata-Validation-Fehler."""

    def __init__(
        self,
        field_name: str,
        reason: str,
        **kwargs: Any,
    ) -> None:
        """Initialisiert Metadata-Error.

        Args:
            field_name: Name des Metadata-Feldes
            reason: Grund für den Validation-Fehler
            **kwargs: Zusätzliche Argumente
        """
        message = f"Metadata-Fehler für Feld '{field_name}': {reason}"

        super().__init__(
            message,
            field_name=field_name,
            validation_rule="metadata_validation",
            **kwargs
        )
        self.reason = reason


# =============================================================================
# Utility Functions
# =============================================================================

def wrap_exception(
    original_exception: Exception,
    new_exception_class: type[PromptyError],
    message: str,
    **kwargs: Any,
) -> PromptyError:
    """Wraps eine Exception in eine Prompty-spezifische Exception.

    Args:
        original_exception: Ursprüngliche Exception
        new_exception_class: Neue Exception-Klasse
        message: Neue Fehlermeldung
        **kwargs: Zusätzliche Argumente für neue Exception

    Returns:
        Neue Prompty-Exception mit ursprünglicher Exception als Ursache
    """
    kwargs["cause"] = original_exception
    return new_exception_class(message, **kwargs)


def create_context_from_exception(exception: Exception) -> dict[str, Any]:
    """Erstellt Kontext-Dictionary aus Exception.

    Args:
        exception: Exception für Kontext-Extraktion

    Returns:
        Kontext-Dictionary mit Exception-Informationen
    """
    context = {
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
    }

    # Zusätzliche Attribute extrahieren
    for attr in ["args", "errno", "filename", "lineno"]:
        if hasattr(exception, attr):
            context[f"exception_{attr}"] = getattr(exception, attr)

    return context
