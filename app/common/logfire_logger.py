"""Logfire-Logger-Adapter für die Keiko Personal Assistant Anwendung.

Erweitert das bestehende Logging-System um Logfire-Integration
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

from kei_logging import get_logger

# Type Variable für Decorator
F = TypeVar("F", bound=Callable[..., Any])

# Globale Logfire-Integration
_logfire_manager: Any | None = None
_llm_tracker: Any | None = None
_logfire_available = False


def initialize_logfire_logging() -> bool:
    """Initialisiert die Logfire-Logging-Integration.

    Returns:
        bool: True wenn erfolgreich initialisiert
    """
    global _logfire_manager, _llm_tracker, _logfire_available

    try:
        from observability import (
            LOGFIRE_INTEGRATION_AVAILABLE,
            get_llm_tracker,
            get_logfire_manager,
        )

        if not LOGFIRE_INTEGRATION_AVAILABLE:
            return False

        _logfire_manager = get_logfire_manager()
        _llm_tracker = get_llm_tracker()
        _logfire_available = _logfire_manager.is_available()

        return _logfire_available

    except Exception:
        _logfire_available = False
        return False


class LogfireLogger:
    """Logfire-Logger-Adapter für nahtlose Integration.

    Erweitert bestehende Logger um Logfire-Funktionalität ohne
    die bestehende Logging-Infrastruktur zu beeinträchtigen.
    """

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.standard_logger = get_logger(module_name)
        self._ensure_logfire_initialized()

    def _ensure_logfire_initialized(self) -> None:
        """Stellt sicher, dass Logfire initialisiert ist."""
        global _logfire_available
        if not _logfire_available:
            initialize_logfire_logging()

    def _get_logfire_manager(self) -> Any | None:
        """Gibt den Logfire-Manager zurück falls verfügbar."""
        global _logfire_manager, _logfire_available
        if _logfire_available and _logfire_manager:
            return _logfire_manager
        return None

    def info(self, message: str, **kwargs) -> None:
        """Sendet Info-Log an Standard-Logger und Logfire."""
        # Standard-Logger (immer)
        self.standard_logger.info(message)

        # Logfire (falls verfügbar)
        manager = self._get_logfire_manager()
        if manager:
            try:
                manager.log_info(
                    message,
                    module=self.module_name,
                    **kwargs
                )
            except Exception:
                pass  # Fehler in Logfire sollen Standard-Logging nicht beeinträchtigen

    def warning(self, message: str, **kwargs) -> None:
        """Sendet Warning-Log an Standard-Logger und Logfire."""
        # Standard-Logger (immer)
        self.standard_logger.warning(message)

        # Logfire (falls verfügbar)
        manager = self._get_logfire_manager()
        if manager:
            try:
                manager.log_warning(
                    message,
                    module=self.module_name,
                    **kwargs
                )
            except Exception:
                pass

    def error(self, message: str, **kwargs) -> None:
        """Sendet Error-Log an Standard-Logger und Logfire."""
        # Standard-Logger (immer)
        self.standard_logger.error(message)

        # Logfire (falls verfügbar)
        manager = self._get_logfire_manager()
        if manager:
            try:
                manager.log_error(
                    message,
                    module=self.module_name,
                    **kwargs
                )
            except Exception:
                pass

    def debug(self, message: str, **_kwargs) -> None:
        """Sendet Debug-Log an Standard-Logger."""
        # Debug-Logs nur an Standard-Logger (zu viel für Logfire)
        self.standard_logger.debug(message)

    @contextmanager
    def span(self, name: str, **kwargs):
        """Erstellt einen Logfire-Span falls verfügbar."""
        manager = self._get_logfire_manager()
        if manager:
            try:
                with manager.span(f"{self.module_name}.{name}", **kwargs) as span:
                    yield span
            except Exception:
                yield None
        else:
            yield None

    def log_execution_time(self, level: str = "info") -> Callable[[F], F]:
        """Decorator um Ausführungszeit zu loggen (mit Logfire-Span)."""
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                func_name = f"{getattr(func, '__module__', 'unknown')}.{func.__name__}"

                with self.span(f"execution.{func.__name__}") as _span:
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        execution_time = time.time() - start_time

                        # Log Erfolg
                        log_message = f"{func_name} executed successfully in {execution_time:.3f}s"
                        if level == "info":
                            self.info(log_message,
                                    function=func_name,
                                    execution_time_ms=execution_time * 1000,
                                    success=True)
                        elif level == "debug":
                            self.debug(log_message)

                        return result

                    except Exception as e:
                        execution_time = time.time() - start_time

                        # Log Fehler
                        error_message = f"{func_name} failed after {execution_time:.3f}s: {e}"
                        self.error(error_message,
                                 function=func_name,
                                 execution_time_ms=execution_time * 1000,
                                 success=False,
                                 error_type=type(e).__name__,
                                 error_message=str(e))
                        raise

            return wrapper
        return decorator

    def log_llm_call(self, model: str, provider: str, **kwargs) -> Callable[[F], F]:
        """Decorator für LLM-Call-Tracking."""
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs_inner: Any) -> Any:
                global _llm_tracker, _logfire_available

                if not _logfire_available or not _llm_tracker:
                    # Fallback auf Standard-Ausführung
                    return func(*args, **kwargs_inner)

                try:
                    with _llm_tracker.track_llm_call(
                        model=model,
                        provider=provider,
                        **kwargs
                    ) as context:
                        result = func(*args, **kwargs_inner)

                        # Extrahiere Token-Usage falls verfügbar
                        if hasattr(result, "usage"):
                            usage = result.usage
                            context.update_tokens(
                                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                                completion_tokens=getattr(usage, "completion_tokens", 0),
                                total_tokens=getattr(usage, "total_tokens", 0)
                            )

                        return result

                except Exception:
                    # Bei Logfire-Fehlern: Fallback auf Standard-Ausführung
                    return func(*args, **kwargs_inner)

            return wrapper
        return decorator


def get_logfire_logger(module_name: str) -> LogfireLogger:
    """Erstellt einen Logfire-Logger für das angegebene Modul.

    Args:
        module_name: Name des Moduls (normalerweise __name__)

    Returns:
        LogfireLogger-Instanz für das Modul
    """
    return LogfireLogger(module_name)


def log_service_startup(service_name: str, **kwargs) -> None:
    """Loggt Service-Startup-Event.

    Args:
        service_name: Name des Services
        **kwargs: Zusätzliche Metadaten
    """
    logger = get_logfire_logger("service_startup")
    logger.info(
        f"🚀 Service gestartet: {service_name}",
        service_name=service_name,
        event_type="service_startup",
        **kwargs
    )


def log_service_shutdown(service_name: str, **kwargs) -> None:
    """Loggt Service-Shutdown-Event.

    Args:
        service_name: Name des Services
        **kwargs: Zusätzliche Metadaten
    """
    logger = get_logfire_logger("service_shutdown")
    logger.info(
        f"🛑 Service beendet: {service_name}",
        service_name=service_name,
        event_type="service_shutdown",
        **kwargs
    )


def log_user_interaction(user_id: str, interaction_type: str, **kwargs) -> None:
    """Loggt User-Interaktion.

    Args:
        user_id: Benutzer-ID
        interaction_type: Art der Interaktion
        **kwargs: Zusätzliche Metadaten
    """
    logger = get_logfire_logger("user_interaction")
    logger.info(
        f"👤 User-Interaktion: {interaction_type}",
        user_id=user_id,
        interaction_type=interaction_type,
        event_type="user_interaction",
        **kwargs
    )


def log_agent_activity(agent_id: str, activity_type: str, **kwargs) -> None:
    """Loggt Agent-Aktivität.

    Args:
        agent_id: Agent-ID
        activity_type: Art der Aktivität
        **kwargs: Zusätzliche Metadaten
    """
    logger = get_logfire_logger("agent_activity")
    logger.info(
        f"🤖 Agent-Aktivität: {activity_type}",
        agent_id=agent_id,
        activity_type=activity_type,
        event_type="agent_activity",
        **kwargs
    )


# Convenience-Funktionen für häufige Use Cases
def log_api_request(method: str, path: str, status_code: int, duration_ms: float, **kwargs) -> None:
    """Loggt API-Request nur über kei_logging."""
    try:
        from kei_logging import get_logger
        kei_logger = get_logger("api_request")
        kei_logger.debug(f"🌐 API Request: {method} {path} -> {status_code}")
    except ImportError:
        pass

    # Logfire-Logging deaktiviert für Console-Output
    # logger = get_logfire_logger("api_request")
    # logger.info(
    #     f"🌐 API Request: {method} {path} -> {status_code}",
    #     method=method,
    #     path=path,
    #     status_code=status_code,
    #     duration_ms=duration_ms,
    #     event_type="api_request",
    #     **kwargs
    # )


def log_database_operation(operation: str, table: str, duration_ms: float, **kwargs) -> None:
    """Loggt Database-Operation."""
    logger = get_logfire_logger("database_operation")
    logger.info(
        f"🗄️ Database: {operation} on {table}",
        operation=operation,
        table=table,
        duration_ms=duration_ms,
        event_type="database_operation",
        **kwargs
    )


def log_cache_operation(operation: str, key: str, hit: bool, **kwargs) -> None:
    """Loggt Cache-Operation."""
    logger = get_logfire_logger("cache_operation")
    logger.info(
        f"💾 Cache: {operation} {key} -> {'HIT' if hit else 'MISS'}",
        operation=operation,
        key=key,
        hit=hit,
        event_type="cache_operation",
        **kwargs
    )
