# backend/kei_logging/__init__.py
"""KEI Logging Package – Erweiterte Logging-Funktionalitäten mit Log-Links.

Stellt einen komfortablen Formatter mit Emojis, Farben und klickbaren IDE-Links
für Log-Ausgaben bereit sowie erweiterte Funktionalitäten für Log-Links,
Enhanced Error-Handling und verbesserte PII-Redaction.
"""

import logging
import os
from typing import Optional

from .clickable_logging_formatter import ClickableLoggingFormatter, get_logger

# Training Logger
try:
    from .training_logger import (
        TrainingLogger,
        get_training_logger,
        log_orchestrator_step,
        training_trace,
    )
    TRAINING_LOGGER_AVAILABLE = True
except ImportError:
    TrainingLogger = None
    get_training_logger = None
    training_trace = None
    log_orchestrator_step = None
    TRAINING_LOGGER_AVAILABLE = False

# Erweiterte Logging-Funktionalitäten (optional import)
try:
    from .log_links import (
        LogEntry,
        LogLinkConfig,
        LogLinkFilter,
        LogLinkFormat,
        LogLinkRegistry,
        create_log_link,
        get_log_entry_by_id,
        get_log_link_registry,
        search_log_entries,
        setup_log_links,
    )
    LOG_LINKS_AVAILABLE = True
except ImportError:
    # Fallback-Definitionen für fehlende Imports
    LogEntry = None
    LogLinkConfig = None
    LogLinkFilter = None
    LogLinkFormat = None
    LogLinkRegistry = None
    create_log_link = None
    get_log_entry_by_id = None
    get_log_link_registry = None
    search_log_entries = None
    setup_log_links = None
    LOG_LINKS_AVAILABLE = False

try:
    from .enhanced_error_handling import (
        AuthenticationError,
        AuthorizationError,
        BusinessLogicError,
        ErrorContext,
        ExternalServiceError,
        KeiSystemError,
        LogLinkedError,
        ValidationError,
        create_error_response,
        enhance_exception_with_log_link,
        log_and_raise,
        with_log_links,
    )
    ENHANCED_ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ENHANCED_ERROR_HANDLING_AVAILABLE = False

try:
    from .enhanced_pii_redaction import (
        DataType,
        EnhancedPIIRedactionConfig,
        EnhancedPIIRedactionFilter,
        EnhancedPIIRedactor,
        RedactionLevel,
        RedactionPattern,
        enhanced_redact_structure,
        enhanced_redact_text,
        get_enhanced_pii_redactor,
        setup_enhanced_pii_redaction,
    )
    ENHANCED_PII_REDACTION_AVAILABLE = True
except ImportError:
    # Fallback-Definitionen für fehlende Imports
    DataType = None
    EnhancedPIIRedactionConfig = None
    EnhancedPIIRedactionFilter = None
    EnhancedPIIRedactor = None
    RedactionLevel = None
    RedactionPattern = None
    enhanced_redact_structure = None
    enhanced_redact_text = None
    get_enhanced_pii_redactor = None
    setup_enhanced_pii_redaction = None
    ENHANCED_PII_REDACTION_AVAILABLE = False

def structured_msg(message: str, **fields: object) -> str:
    """Erzeugt konsistente strukturierte Log-Nachrichten als JSON-ähnliche Zeichenkette.

    Hinweis: Formatter/Filter können zusätzliche Redaction übernehmen; dies stellt
    sicher, dass Diagnosefelder wie `correlation_id`, `causation_id`, `tenant`,
    `subject`, `type`, `message_id` einheitlich enthalten sind.
    """
    try:
        import json
        payload = {"message": message}
        payload.update(fields)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        # Fallback: einfache Formatierung
        extras = " ".join(f"{k}={v}" for k, v in fields.items())
        return f"{message} {extras}".strip()


def setup_enhanced_logging(
    log_links_config: Optional["LogLinkConfig"] = None,
    pii_redaction_config: Optional["EnhancedPIIRedactionConfig"] = None,
    enable_log_links: bool = True,
    enable_enhanced_pii_redaction: bool = True,
    enable_training_mode: bool = None
) -> bool:
    """Setzt erweiterte Logging-Funktionalitäten auf.

    Args:
        log_links_config: Konfiguration für Log-Links
        pii_redaction_config: Konfiguration für erweiterte PII-Redaction
        enable_log_links: Log-Links aktivieren
        enable_enhanced_pii_redaction: Erweiterte PII-Redaction aktivieren
        enable_training_mode: Training-Modus aktivieren (None = aus Environment)

    Returns:
        True wenn erfolgreich konfiguriert
    """
    success = True

    try:
        # Setup Log-Links
        if enable_log_links and LOG_LINKS_AVAILABLE:
            setup_log_links(log_links_config)
            logger = get_logger(__name__)
            logger.info("Log-Links erfolgreich aktiviert")

        # Setup Enhanced PII-Redaction
        if enable_enhanced_pii_redaction and ENHANCED_PII_REDACTION_AVAILABLE:
            setup_enhanced_pii_redaction(pii_redaction_config)
            logger = get_logger(__name__)
            logger.info("Erweiterte PII-Redaction erfolgreich aktiviert")

        # Setup Training Mode
        if enable_training_mode is not None and TRAINING_LOGGER_AVAILABLE:
            import os
            os.environ["LOGGING_TRAIN"] = "true" if enable_training_mode else "false"

            if enable_training_mode:
                training_logger = get_training_logger("system")
                training_logger.logger.train("🎓 Training-Modus aktiviert")
            else:
                logger = get_logger(__name__)
                logger.info("Training-Modus deaktiviert")

    except Exception as e:
        logger = get_logger(__name__)
        logger.exception(f"Setup erweiterte Logging-Funktionalitäten fehlgeschlagen: {e}")
        success = False

    return success


def get_enhanced_logging_status() -> dict:
    """Gibt Status der erweiterten Logging-Funktionalitäten zurück.

    Returns:
        Status-Dictionary
    """
    status = {
        "log_links": {
            "available": LOG_LINKS_AVAILABLE,
            "active": False
        },
        "enhanced_error_handling": {
            "available": ENHANCED_ERROR_HANDLING_AVAILABLE,
            "active": ENHANCED_ERROR_HANDLING_AVAILABLE
        },
        "enhanced_pii_redaction": {
            "available": ENHANCED_PII_REDACTION_AVAILABLE,
            "active": False
        },
        "training_mode": {
            "available": TRAINING_LOGGER_AVAILABLE,
            "active": os.getenv("LOGGING_TRAIN", "false").lower() in ("true", "1", "yes", "on"),
            "environment_variable": "LOGGING_TRAIN"
        }
    }

    # Prüfe aktive Status
    if LOG_LINKS_AVAILABLE:
        try:
            registry = get_log_link_registry()
            stats = registry.get_statistics()
            status["log_links"]["active"] = True
            status["log_links"]["entries_created"] = stats["entries_created"]
        except:
            pass

    if ENHANCED_PII_REDACTION_AVAILABLE:
        try:
            redactor = get_enhanced_pii_redactor()
            stats = redactor.get_statistics()
            status["enhanced_pii_redaction"]["active"] = True
            status["enhanced_pii_redaction"]["redactions_performed"] = stats["redactions_performed"]
        except:
            pass

    return status


# Alle verfügbaren Exports
__all__ = [
    # Basis-Logging
    "ClickableLoggingFormatter",
    "get_enhanced_logging_status",
    "get_logger",
    # Setup-Funktionen
    "setup_enhanced_logging",
    "structured_msg",
]

# Training Logger Exports hinzufügen
if TRAINING_LOGGER_AVAILABLE:
    __all__.extend([
        "TrainingLogger",
        "get_training_logger",
        "log_orchestrator_step",
        "training_trace"
    ])

# Erweitere __all__ basierend auf verfügbaren Komponenten
if LOG_LINKS_AVAILABLE:
    __all__.extend([
        "LogEntry",
        "LogLinkConfig",
        "LogLinkFilter",
        "LogLinkFormat",
        "LogLinkRegistry",
        "create_log_link",
        "get_log_entry_by_id",
        "get_log_link_registry",
        "search_log_entries",
        "setup_log_links"
    ])

if ENHANCED_ERROR_HANDLING_AVAILABLE:
    __all__.extend([
        "AuthenticationError",
        "AuthorizationError",
        "BusinessLogicError",
        "ErrorContext",
        "ExternalServiceError",
        "KeiSystemError",
        "LogLinkedError",
        "ValidationError",
        "create_error_response",
        "enhance_exception_with_log_link",
        "log_and_raise",
        "with_log_links"
    ])

if ENHANCED_PII_REDACTION_AVAILABLE:
    __all__.extend([
        "DataType",
        "EnhancedPIIRedactionConfig",
        "EnhancedPIIRedactionFilter",
        "EnhancedPIIRedactor",
        "RedactionLevel",
        "RedactionPattern",
        "enhanced_redact_structure",
        "enhanced_redact_text",
        "get_enhanced_pii_redactor",
        "setup_enhanced_pii_redaction"
    ])

def configure_loki_logging(service_name: str = "keiko-backend", level: str | None = None) -> None:
    """Konfiguriert optionalen Loki JSON Log‑Export per STDOUT.

    Diese Funktion stellt sicher, dass Logs strukturiert im JSON‑Format
    ausgegeben werden, sodass Promtail sie zuverlässig parsen und nach Loki
    shippen kann. Correlation‑IDs bleiben durchgehend erhalten.

    Args:
        service_name: Log‑Label für den Service
        level: Optionales Log‑Level (z. B. "INFO", "DEBUG")
    """
    # Level aus Settings/ENV ableiten
    log_level = (level or os.getenv("KEIKO_LOG_LEVEL") or "INFO").upper()

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level, logging.INFO))

    # JSON Formatter für STDOUT
    class _JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            import json
            payload = {
                "service": service_name,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "timestamp": int(record.created * 1000),
            }
            # Extra Felder aus LogRecord um Correlation/Tracing zu erhalten
            for key in ("correlation_id", "causation_id", "tenant", "trace_id"):
                val = getattr(record, key, None)
                if val is not None:
                    payload[key] = val
            try:
                return json.dumps(payload, ensure_ascii=False)
            except Exception:
                return f"{record.levelname} {record.name} {record.getMessage()}"

    # Vorhandene Handler ersetzen durch STDOUT JSON (nicht doppeln)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)

__all__: list[str] = ["ClickableLoggingFormatter", "configure_loki_logging", "get_logger", "structured_msg"]

# Package Metadaten
__version__ = "1.0.0"
__author__ = "Keiko Development Team"
