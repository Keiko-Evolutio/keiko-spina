"""Produktionsreife Pydantic Logfire-Integration für Keiko Personal Assistant.

Enterprise-Grade-Integration mit vollständiger Instrumentierung und nahtloser
Integration in die bestehende OpenTelemetry/Jaeger/Prometheus-Infrastruktur.

Features:
- Vollständige Logfire-Instrumentierung (30+ Integrationen)
- Enterprise-Sicherheit mit PII-Redaction
- Performance-Monitoring und Alerting
- Fallback-Mechanismen und Fehlerbehandlung
- Nahtlose Integration mit bestehender Observability
"""

from __future__ import annotations

import hashlib
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None

from kei_logging import get_logger

from .logfire_config import (
    LogfireMode,
    LogfirePIIRedactionConfig,
    LogfireSettings,
    get_logfire_settings,
    validate_logfire_config,
)

# String-Konstanten zur Vermeidung von Duplikationen
ALREADY_INSTRUMENTED_MSG = "already instrumented"

logger = get_logger(__name__)


@dataclass
class LogfireMetrics:
    """Metriken für die Logfire-Integration selbst."""

    total_logs: int = 0
    total_spans: int = 0
    total_errors: int = 0
    fallback_activations: int = 0
    pii_redactions: int = 0
    last_error_time: float | None = None
    initialization_time: float | None = None

    def record_log(self) -> None:
        """Registriert einen gesendeten Log."""
        self.total_logs += 1

    def record_span(self) -> None:
        """Registriert einen erstellten Span."""
        self.total_spans += 1

    def record_error(self) -> None:
        """Registriert einen Fehler."""
        self.total_errors += 1
        self.last_error_time = time.time()

    def record_fallback(self) -> None:
        """Registriert eine Fallback-Aktivierung."""
        self.fallback_activations += 1

    def record_pii_redaction(self) -> None:
        """Registriert eine PII-Redaction."""
        self.pii_redactions += 1


class LogfirePIIRedactor:
    """Enterprise-Grade PII-Redaction für Logfire-Daten.

    Implementiert umfassende Datenschutz-Compliance durch:
    - Feldbasierte Redaction (Delete, Hash, Mask)
    - Pattern-basierte String-Redaction
    - Verschachtelte Datenstruktur-Unterstützung
    - Konfigurierbare Redaction-Regeln
    """

    def __init__(self, config: LogfirePIIRedactionConfig):
        self.config = config
        self._compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in config.string_patterns.items()
        }

    def redact_data(self, data: Any) -> Any:
        """Redaktiert PII-Daten aus beliebigen Datenstrukturen.

        Args:
            data: Zu redaktierende Daten

        Returns:
            Redaktierte Daten
        """
        try:
            return self._redact_recursive(data)
        except Exception as e:
            logger.warning(f"PII-Redaction fehlgeschlagen: {e}")
            return "[REDACTED_ERROR]"

    def _redact_recursive(self, data: Any) -> Any:
        """Rekursive Redaction für verschachtelte Strukturen."""
        if isinstance(data, dict):
            return self._redact_dict(data)
        if isinstance(data, (list, tuple)):
            return type(data)(self._redact_recursive(item) for item in data)
        if isinstance(data, str):
            return self._redact_string(data)
        return data

    def _redact_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redaktiert Dictionary-Daten."""
        result = {}

        for key, value in data.items():
            key_lower = key.lower()

            # Felder komplett entfernen
            if any(field in key_lower for field in self.config.delete_fields):
                continue

            # Felder hashen
            if any(field in key_lower for field in self.config.hash_fields):
                if isinstance(value, str):
                    result[key] = self._hash_value(value)
                else:
                    result[key] = self._hash_value(str(value))

            # Felder maskieren
            elif any(field in key_lower for field in self.config.mask_fields):
                if isinstance(value, str):
                    result[key] = self._mask_value(value)
                else:
                    result[key] = self._mask_value(str(value))

            # Rekursive Redaction
            else:
                result[key] = self._redact_recursive(value)

        return result

    def _redact_string(self, text: str) -> str:
        """Redaktiert String-Inhalte basierend auf Patterns."""
        result = text

        for pattern_name, compiled_pattern in self._compiled_patterns.items():
            replacement = self.config.replacements.get(pattern_name, "[REDACTED]")
            result = compiled_pattern.sub(replacement, result)

        return result

    def _hash_value(self, value: str) -> str:
        """Erstellt einen Hash-Wert für sensible Daten."""
        return hashlib.sha256(value.encode()).hexdigest()[:16]

    def _mask_value(self, value: str) -> str:
        """Maskiert einen Wert teilweise."""
        if len(value) <= 4:
            return "*" * len(value)

        visible_chars = 2
        masked_chars = len(value) - (2 * visible_chars)

        if masked_chars <= 0:
            return value

        return value[:visible_chars] + "*" * masked_chars + value[-visible_chars:]


class LogfireManager:
    """Enterprise-Grade Logfire-Manager für Keiko Personal Assistant.

    Verwaltet die vollständige Logfire-Integration mit:
    - Automatischer Instrumentierung aller verfügbaren Services
    - Enterprise-Sicherheit und PII-Redaction
    - Performance-Monitoring und Fallback-Mechanismen
    - Nahtlose Integration mit bestehender Observability
    """

    def __init__(self, settings: LogfireSettings | None = None):
        self.settings = settings or get_logfire_settings()
        self.metrics = LogfireMetrics()
        self.pii_redactor = LogfirePIIRedactor(self.settings.pii_config) if self.settings.enable_pii_redaction else None
        self._initialized = False
        self._fallback_active = False
        self._instrumentation_status: dict[str, bool] = {}

    def initialize(self) -> bool:
        """Initialisiert die Logfire-Integration.

        Returns:
            bool: True wenn erfolgreich initialisiert
        """
        # Verhindere mehrfache Initialisierung
        if self._initialized:
            logger.debug("Logfire bereits initialisiert - überspringe")
            return True

        start_time = time.time()

        try:
            # Prüfe Verfügbarkeit
            if not LOGFIRE_AVAILABLE:
                logger.warning("Logfire SDK nicht verfügbar")
                return self._activate_fallback()

            # Prüfe Konfiguration
            if not validate_logfire_config(self.settings):
                logger.error("Ungültige Logfire-Konfiguration")
                return self._activate_fallback()

            # Disabled-Modus
            if self.settings.mode == LogfireMode.DISABLED:
                logger.info("Logfire ist deaktiviert")
                return True

            # Local-Only-Modus
            if self.settings.mode == LogfireMode.LOCAL_ONLY:
                logger.info("Logfire im LOCAL_ONLY Modus - nur lokale OpenTelemetry-Exporte")
                # Keine Logfire-Konfiguration oder -Instrumentierung
                self.metrics.initialization_time = time.time() - start_time
                self._initialized = True
                return True

            # Konfiguriere Logfire (nur für Cloud-Modi)
            self._configure_logfire()

            # Aktiviere Instrumentierungen (nur für Cloud-Modi)
            self._setup_instrumentations()

            # Initialisierung abgeschlossen
            self.metrics.initialization_time = time.time() - start_time
            self._initialized = True

            logger.info(f"Logfire erfolgreich initialisiert (Modus: {self.settings.mode})")
            return True

        except Exception as e:
            logger.error(f"Logfire-Initialisierung fehlgeschlagen: {e}")
            self.metrics.record_error()
            return self._activate_fallback()

    def _configure_logfire(self) -> None:
        """Konfiguriert Logfire basierend auf den Settings."""
        # LOCAL_ONLY Modus: Keine Logfire-Konfiguration
        if self.settings.mode == LogfireMode.LOCAL_ONLY:
            logger.info("Logfire im LOCAL_ONLY Modus - keine Cloud-Konfiguration")
            return

        config_kwargs = {
            "service_name": self.settings.service_name,
            "service_version": self.settings.service_version,
            "environment": self.settings.environment.value,
        }

        # Token für Cloud-Modi
        if self.settings.mode in [LogfireMode.CLOUD_ONLY, LogfireMode.DUAL_EXPORT]:
            config_kwargs["token"] = self.settings.token

        # Sampling-Konfiguration
        if self.settings.trace_sample_rate < 1.0:
            config_kwargs["trace_sample_rate"] = self.settings.trace_sample_rate

        # Console-Konfiguration (vereinfacht)
        if self.settings.console_enabled:
            config_kwargs["console"] = {}

        try:
            logfire.configure(**config_kwargs)
        except Exception as e:
            # Behandle bekannte Konfigurationsfehler
            error_msg = str(e).lower()
            if "shutdown can only be called once" in error_msg:
                logger.debug("⚠️ Logfire bereits konfiguriert (shutdown-Fehler erwartet)")
            elif ALREADY_INSTRUMENTED_MSG in error_msg:
                logger.debug("⚠️ Logfire bereits instrumentiert")
            else:
                raise  # Unbekannte Fehler weiterwerfen

    def _setup_instrumentations(self) -> None:
        """Aktiviert alle konfigurierten Instrumentierungen."""
        # LOCAL_ONLY Modus: Keine Logfire-Instrumentierungen
        if self.settings.mode == LogfireMode.LOCAL_ONLY:
            logger.info("Logfire im LOCAL_ONLY Modus - keine Logfire-Instrumentierungen")
            return

        # Instrumentierungen ohne Parameter
        simple_instrumentations = {
            "openai": (self.settings.instrument_openai, self._instrument_openai),
            "anthropic": (self.settings.instrument_anthropic, self._instrument_anthropic),
            "httpx": (self.settings.instrument_httpx, self._instrument_httpx),
            "requests": (self.settings.instrument_requests, self._instrument_requests),
            "sqlalchemy": (self.settings.instrument_sqlalchemy, self._instrument_sqlalchemy),
            "pydantic": (self.settings.instrument_pydantic, self._instrument_pydantic),
            "system_metrics": (self.settings.instrument_system_metrics, self._instrument_system_metrics),
        }

        for name, (enabled, instrument_func) in simple_instrumentations.items():
            if enabled:
                # Prüfe, ob bereits instrumentiert
                if self._instrumentation_status.get(name, False):
                    logger.debug(f"⚠️ {name.title()}-Instrumentierung bereits aktiv - überspringe")
                    continue

                try:
                    instrument_func()
                    self._instrumentation_status[name] = True
                    logger.debug(f"✅ {name.title()}-Instrumentierung aktiviert")
                except Exception as e:
                    error_msg = str(e).lower()
                    if ALREADY_INSTRUMENTED_MSG in error_msg:
                        self._instrumentation_status[name] = True
                        logger.debug(f"⚠️ {name.title()}-Instrumentierung bereits aktiv")
                    else:
                        self._instrumentation_status[name] = False
                        logger.warning(f"⚠️ {name.title()}-Instrumentierung fehlgeschlagen: {e}")

    def _instrument_openai(self) -> None:
        """Instrumentiert OpenAI API-Calls."""
        try:
            logfire.instrument_openai()
        except Exception as e:
            if ALREADY_INSTRUMENTED_MSG not in str(e).lower():
                raise

    def _instrument_anthropic(self) -> None:
        """Instrumentiert Anthropic API-Calls."""
        try:
            import anthropic  # noqa: F401
            logfire.instrument_anthropic()
        except ImportError:
            raise ImportError("Anthropic package nicht installiert. Installieren Sie mit: pip install anthropic")
        except Exception as e:
            if ALREADY_INSTRUMENTED_MSG not in str(e).lower():
                raise

    def _instrument_httpx(self) -> None:
        """Instrumentiert HTTPX HTTP-Client."""
        try:
            logfire.instrument_httpx()
        except Exception as e:
            if ALREADY_INSTRUMENTED_MSG not in str(e).lower():
                raise

    def _instrument_requests(self) -> None:
        """Instrumentiert Requests HTTP-Client."""
        try:
            logfire.instrument_requests()
        except Exception as e:
            if ALREADY_INSTRUMENTED_MSG not in str(e).lower():
                raise

    def _instrument_sqlalchemy(self) -> None:
        """Instrumentiert SQLAlchemy Database-Queries."""
        try:
            import sqlalchemy  # noqa: F401
            logfire.instrument_sqlalchemy()
        except ImportError:
            raise ImportError("SQLAlchemy package nicht installiert. Installieren Sie mit: pip install 'logfire[sqlalchemy]'")
        except Exception as e:
            if ALREADY_INSTRUMENTED_MSG not in str(e).lower():
                raise

    def _instrument_pydantic(self) -> None:
        """Instrumentiert Pydantic Validierung."""
        try:
            logfire.instrument_pydantic()
        except Exception as e:
            if ALREADY_INSTRUMENTED_MSG not in str(e).lower():
                raise

    def _instrument_system_metrics(self) -> None:
        """Instrumentiert System-Metriken (CPU, Memory, Disk)."""
        try:
            # Prüfe ob das System-Metrics-Package verfügbar ist
            from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor  # noqa: F401
            logfire.instrument_system_metrics()
        except ImportError:
            raise ImportError("System-Metrics package nicht installiert. Installieren Sie mit: pip install 'logfire[system-metrics]'")
        except Exception as e:
            if ALREADY_INSTRUMENTED_MSG not in str(e).lower():
                raise

    def instrument_fastapi_app(self, app) -> None:
        """Instrumentiert FastAPI-Anwendung (benötigt app-Parameter)."""
        if self.settings.instrument_fastapi and self.is_available():
            # Prüfe, ob bereits instrumentiert
            if self._instrumentation_status.get("fastapi", False):
                logger.debug("⚠️ FastAPI-Instrumentierung bereits aktiv - überspringe")
                return

            try:
                logfire.instrument_fastapi(app)
                self._instrumentation_status["fastapi"] = True
                logger.debug("✅ FastAPI-Instrumentierung aktiviert")
            except Exception as e:
                error_msg = str(e).lower()
                if ALREADY_INSTRUMENTED_MSG in error_msg:
                    self._instrumentation_status["fastapi"] = True
                    logger.debug("⚠️ FastAPI-Instrumentierung bereits aktiv")
                else:
                    self._instrumentation_status["fastapi"] = False
                    logger.warning(f"⚠️ FastAPI-Instrumentierung fehlgeschlagen: {e}")

    def _activate_fallback(self) -> bool:
        """Aktiviert Fallback-Modus."""
        if self.settings.enable_fallback:
            self._fallback_active = True
            self.metrics.record_fallback()
            logger.info("Logfire-Fallback aktiviert - verwende OpenTelemetry")
            return True
        return False

    def is_available(self) -> bool:
        """Prüft ob Logfire verfügbar ist."""
        return (
            LOGFIRE_AVAILABLE and
            self._initialized and
            not self._fallback_active and
            self.settings.mode != LogfireMode.DISABLED
        )

    def log_info(self, message: str, **kwargs) -> None:
        """Sendet einen Info-Log."""
        self._log("info", message, **kwargs)

    def log_warning(self, message: str, **kwargs) -> None:
        """Sendet einen Warning-Log."""
        self._log("warning", message, **kwargs)

    def log_error(self, message: str, **kwargs) -> None:
        """Sendet einen Error-Log."""
        self._log("error", message, **kwargs)

    def _log(self, level: str, message: str, **kwargs) -> None:
        """Interne Log-Methode mit PII-Redaction."""
        try:
            if not self.is_available():
                return

            # PII-Redaction
            if self.pii_redactor:
                kwargs = self.pii_redactor.redact_data(kwargs)
                self.metrics.record_pii_redaction()

            # Sende Log
            getattr(logfire, level)(message, **kwargs)
            self.metrics.record_log()

        except Exception as e:
            logger.warning(f"Logfire-Log fehlgeschlagen: {e}")
            self.metrics.record_error()

    @contextmanager
    def span(self, name: str, **kwargs):
        """Erstellt einen Logfire-Span mit PII-Redaction."""
        if not self.is_available():
            yield None
            return

        try:
            # PII-Redaction
            if self.pii_redactor:
                kwargs = self.pii_redactor.redact_data(kwargs)
                self.metrics.record_pii_redaction()

            with logfire.span(name, **kwargs) as span:
                self.metrics.record_span()
                yield span

        except Exception as e:
            logger.warning(f"Logfire-Span fehlgeschlagen: {e}")
            self.metrics.record_error()
            yield None

    def get_metrics(self) -> dict[str, Any]:
        """Gibt Metriken der Logfire-Integration zurück."""
        return {
            "total_logs": self.metrics.total_logs,
            "total_spans": self.metrics.total_spans,
            "total_errors": self.metrics.total_errors,
            "fallback_activations": self.metrics.fallback_activations,
            "pii_redactions": self.metrics.pii_redactions,
            "initialization_time": self.metrics.initialization_time,
            "fallback_active": self._fallback_active,
            "instrumentation_status": self._instrumentation_status.copy(),
            "settings": {
                "mode": self.settings.mode.value,
                "environment": self.settings.environment.value,
                "service_name": self.settings.service_name,
                "trace_sample_rate": self.settings.trace_sample_rate,
            }
        }


# Globaler Manager (Singleton)
_logfire_manager: LogfireManager | None = None
_shutdown_in_progress: bool = False
_shutdown_completed: bool = False


def get_logfire_manager() -> LogfireManager:
    """Gibt den globalen Logfire-Manager zurück (Singleton)."""
    global _logfire_manager
    if _logfire_manager is None:
        _logfire_manager = LogfireManager()
    return _logfire_manager


def initialize_logfire(settings: LogfireSettings | None = None) -> bool:
    """Initialisiert die globale Logfire-Integration.

    Args:
        settings: Optionale Logfire-Settings

    Returns:
        bool: True wenn erfolgreich initialisiert
    """
    global _logfire_manager

    # Prüfe, ob bereits initialisiert
    if _logfire_manager and _logfire_manager._initialized:
        logger.debug("Logfire bereits global initialisiert - überspringe")
        return True

    _logfire_manager = LogfireManager(settings)
    return _logfire_manager.initialize()


def shutdown_logfire() -> None:
    """Beendet die Logfire-Integration und OpenTelemetry MeterProvider gracefully."""
    global _logfire_manager, _shutdown_in_progress, _shutdown_completed

    # Verhindere mehrfache Shutdown-Aufrufe
    if _shutdown_in_progress or _shutdown_completed:
        logger.debug("⚠️ Logfire shutdown bereits in Bearbeitung oder abgeschlossen - überspringe")
        return

    if not _logfire_manager:
        _shutdown_completed = True
        return

    _shutdown_in_progress = True

    try:
        # Proper Logfire shutdown with provider cleanup
        if LOGFIRE_AVAILABLE and hasattr(logfire, 'shutdown'):
            try:
                logfire.shutdown()
                logger.debug("✅ Logfire SDK shutdown erfolgreich")
            except Exception as e:
                # Spezielle Behandlung für bekannte Shutdown-Fehler
                error_msg = str(e).lower()
                if "shutdown can only be called once" in error_msg:
                    logger.debug("⚠️ Logfire SDK bereits beendet (erwartet bei mehrfachem Aufruf)")
                elif "deadline already exceeded" in error_msg:
                    logger.debug("⚠️ Logfire SDK shutdown deadline exceeded (erwartet)")
                elif "meterprovider.shutdown failed" in error_msg:
                    logger.debug("⚠️ Logfire MeterProvider shutdown fehlgeschlagen (erwartet)")
                else:
                    logger.debug(f"⚠️ Logfire SDK shutdown fehlgeschlagen: {e}")

        # Clean up our manager reference
        _logfire_manager = None
        logger.info("✅ Logfire-Integration beendet")

    except Exception as e:
        logger.warning(f"⚠️ Fehler beim Logfire shutdown: {e}")
        _logfire_manager = None
    finally:
        _shutdown_completed = True
        _shutdown_in_progress = False


def is_logfire_shutdown_completed() -> bool:
    """Prüft, ob der Logfire-Shutdown bereits abgeschlossen ist."""
    return _shutdown_completed


def reset_logfire_shutdown_state() -> None:
    """Setzt den Shutdown-Status zurück (nur für Tests)."""
    global _shutdown_in_progress, _shutdown_completed
    _shutdown_in_progress = False
    _shutdown_completed = False
