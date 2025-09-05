# backend/kei_logging/training_logger.py
"""Training Logger für detaillierte Funktionsschritte-Verfolgung."""

from __future__ import annotations

import functools
import inspect
import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

from .clickable_logging_formatter import get_logger

# Type Variables für Decorator
F = TypeVar("F", bound=Callable[..., Any])

# Training-Konfiguration
TRAINING_SHOW_PARAMETERS = os.getenv("TRAINING_SHOW_PARAMETERS", "true").lower() in ("true", "1", "yes", "on")
TRAINING_SHOW_RETURN_VALUES = os.getenv("TRAINING_SHOW_RETURN_VALUES", "true").lower() in ("true", "1", "yes", "on")
TRAINING_SHOW_EXECUTION_TIME = os.getenv("TRAINING_SHOW_EXECUTION_TIME", "true").lower() in ("true", "1", "yes", "on")
TRAINING_SHOW_CALL_STACK = os.getenv("TRAINING_SHOW_CALL_STACK", "false").lower() in ("true", "1", "yes", "on")
TRAINING_MAX_PARAM_LENGTH = int(os.getenv("TRAINING_MAX_PARAM_LENGTH", "200"))
TRAINING_INDENT_LEVEL = int(os.getenv("TRAINING_INDENT_LEVEL", "2"))


class TrainingLogger:
    """Training Logger für detaillierte Funktionsschritte-Verfolgung."""

    def __init__(self, logger_name: str = "training"):
        """Initialisiert Training Logger."""
        self.logger = get_logger(f"🎓.{logger_name}")

        # Stelle sicher, dass die train-Methode verfügbar ist
        from .clickable_logging_formatter import _add_train_method_to_logger
        _add_train_method_to_logger(self.logger)
        self.call_depth = 0
        self.execution_stack: list[dict[str, Any]] = []

        # Training-spezifische Emojis
        self.training_emojis = {
            "function_enter": "🔍",
            "function_exit": "✅",
            "function_error": "❌",
            "parameter": "📥",
            "return_value": "📤",
            "execution_time": "⏱️",
            "llm_call": "🧠",
            "agent_call": "🤖",
            "task_decomposition": "🔧",
            "orchestration": "🎼",
            "saga_step": "🔄",
            "policy_check": "🛡️",
            "performance_metric": "📊",
            "state_change": "🔄",
            "event_publish": "📢",
            "event_receive": "📨"
        }

    def is_training_enabled(self) -> bool:
        """Prüft ob Training-Modus aktiviert ist."""
        return os.getenv("LOGGING_TRAIN", "false").lower() in ("true", "1", "yes", "on")

    def _format_value(self, value: Any, max_length: int = TRAINING_MAX_PARAM_LENGTH) -> str:
        """Formatiert Werte für Training-Logs."""
        try:
            if value is None:
                return "None"
            if isinstance(value, (str, int, float, bool)):
                str_value = str(value)
            elif isinstance(value, (list, tuple)):
                str_value = f"[{len(value)} items]" if len(value) > 3 else str(value)
            elif isinstance(value, dict):
                str_value = f"{{...{len(value)} keys...}}" if len(value) > 3 else str(value)
            else:
                str_value = f"<{type(value).__name__}>"

            # Kürze zu lange Werte
            if len(str_value) > max_length:
                str_value = str_value[:max_length-3] + "..."

            return str_value
        except Exception:
            return f"<{type(value).__name__}>"

    def _get_indent(self) -> str:
        """Gibt Einrückung basierend auf Call-Depth zurück."""
        return "  " * (self.call_depth * TRAINING_INDENT_LEVEL)

    def _get_caller_context(self) -> dict[str, str]:
        """Ermittelt Kontext des Aufrufers."""
        frame = inspect.currentframe()
        try:
            # Gehe 3 Frames zurück (current -> decorator -> actual function)
            caller_frame = frame.f_back.f_back.f_back
            if caller_frame:
                return {
                    "file": os.path.basename(caller_frame.f_code.co_filename),
                    "function": caller_frame.f_code.co_name,
                    "line": str(caller_frame.f_lineno)
                }
        except:
            pass
        finally:
            del frame

        return {"file": "unknown", "function": "unknown", "line": "0"}

    def log_function_enter(
        self,
        function_name: str,
        args: tuple = (),
        kwargs: dict[str, Any] = None,
        context: dict[str, Any] | None = None
    ) -> None:
        """Loggt Funktions-Eintritt."""
        if not self.is_training_enabled():
            return

        kwargs = kwargs or {}
        context = context or {}

        indent = self._get_indent()
        emoji = self.training_emojis["function_enter"]

        # Basis-Log-Message
        message = f"{indent}{emoji} ENTER: {function_name}()"

        # Parameter-Details
        if TRAINING_SHOW_PARAMETERS and (args or kwargs):
            params = []

            # Positionelle Argumente
            for i, arg in enumerate(args):
                params.append(f"arg{i}={self._format_value(arg)}")

            # Keyword-Argumente
            for key, value in kwargs.items():
                params.append(f"{key}={self._format_value(value)}")

            if params:
                param_emoji = self.training_emojis["parameter"]
                param_str = ", ".join(params)
                message += f"\n{indent}  {param_emoji} Parameters: {param_str}"

        # Kontext-Informationen
        if context:
            for key, value in context.items():
                message += f"\n{indent}  📋 {key}: {self._format_value(value)}"

        # Call-Stack-Info
        if TRAINING_SHOW_CALL_STACK:
            caller = self._get_caller_context()
            message += f"\n{indent}  📍 Called from: {caller['file']}:{caller['line']} in {caller['function']}()"

        self.logger.train(message)
        self.call_depth += 1

        # Execution-Stack für Tracking
        self.execution_stack.append({
            "function": function_name,
            "start_time": time.time(),
            "context": context
        })

    def log_function_exit(
        self,
        function_name: str,
        return_value: Any = None,
        execution_time_ms: float | None = None,
        context: dict[str, Any] | None = None
    ) -> None:
        """Loggt Funktions-Austritt."""
        if not self.is_training_enabled():
            return

        self.call_depth = max(0, self.call_depth - 1)
        indent = self._get_indent()
        emoji = self.training_emojis["function_exit"]

        # Basis-Log-Message
        message = f"{indent}{emoji} EXIT: {function_name}()"

        # Rückgabewert
        if TRAINING_SHOW_RETURN_VALUES and return_value is not None:
            return_emoji = self.training_emojis["return_value"]
            message += f"\n{indent}  {return_emoji} Returns: {self._format_value(return_value)}"

        # Ausführungszeit
        if TRAINING_SHOW_EXECUTION_TIME and execution_time_ms is not None:
            time_emoji = self.training_emojis["execution_time"]
            message += f"\n{indent}  {time_emoji} Execution time: {execution_time_ms:.2f}ms"

        # Kontext-Änderungen
        if context:
            for key, value in context.items():
                message += f"\n{indent}  🔄 {key}: {self._format_value(value)}"

        self.logger.train(message)

        # Entferne von Execution-Stack
        if self.execution_stack:
            self.execution_stack.pop()

    def log_function_error(
        self,
        function_name: str,
        error: Exception,
        execution_time_ms: float | None = None
    ) -> None:
        """Loggt Funktions-Fehler."""
        if not self.is_training_enabled():
            return

        self.call_depth = max(0, self.call_depth - 1)
        indent = self._get_indent()
        emoji = self.training_emojis["function_error"]

        message = f"{indent}{emoji} ERROR in {function_name}(): {type(error).__name__}: {error!s}"

        if TRAINING_SHOW_EXECUTION_TIME and execution_time_ms is not None:
            time_emoji = self.training_emojis["execution_time"]
            message += f"\n{indent}  {time_emoji} Time until error: {execution_time_ms:.2f}ms"

        self.logger.error(message)

        # Entferne von Execution-Stack
        if self.execution_stack:
            self.execution_stack.pop()

    def log_orchestrator_step(
        self,
        step_name: str,
        step_type: str,
        details: dict[str, Any] | None = None
    ) -> None:
        """Loggt Orchestrator-spezifische Schritte."""
        if not self.is_training_enabled():
            return

        details = details or {}
        indent = self._get_indent()
        emoji = self.training_emojis.get(step_type, "🔧")

        message = f"{indent}{emoji} {step_type.upper()}: {step_name}"

        for key, value in details.items():
            message += f"\n{indent}  📋 {key}: {self._format_value(value)}"

        self.logger.train(message)


# Global Training Logger Instance
_training_logger: TrainingLogger | None = None


def get_training_logger(logger_name: str = "training") -> TrainingLogger:
    """Holt globale Training Logger Instanz."""
    global _training_logger
    if _training_logger is None:
        _training_logger = TrainingLogger(logger_name)
    return _training_logger


def training_trace(
    context: dict[str, Any] | None = None,
    show_params: bool = True,
    show_return: bool = True
) -> Callable[[F], F]:
    """Decorator für automatisches Training-Logging von Funktionen."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            train_enabled = os.getenv("LOGGING_TRAIN", "false").lower() in ("true", "1", "yes", "on")
            if not train_enabled:
                return func(*args, **kwargs)

            training_logger = get_training_logger()
            function_name = f"{func.__module__}.{func.__qualname__}"  # type: ignore[attr-defined]
            start_time = time.time()

            # Log function enter
            training_logger.log_function_enter(
                function_name,
                args if show_params else (),
                kwargs if show_params else {},
                context
            )

            try:
                result = func(*args, **kwargs)
                execution_time_ms = (time.time() - start_time) * 1000

                # Log function exit
                training_logger.log_function_exit(
                    function_name,
                    result if show_return else None,
                    execution_time_ms
                )

                return result

            except Exception as e:
                execution_time_ms = (time.time() - start_time) * 1000
                training_logger.log_function_error(function_name, e, execution_time_ms)
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            train_enabled = os.getenv("LOGGING_TRAIN", "false").lower() in ("true", "1", "yes", "on")
            if not train_enabled:
                return await func(*args, **kwargs)

            training_logger = get_training_logger()
            function_name = f"{func.__module__}.{func.__qualname__}"  # type: ignore[attr-defined]
            start_time = time.time()

            # Log function enter
            training_logger.log_function_enter(
                function_name,
                args if show_params else (),
                kwargs if show_params else {},
                context
            )

            try:
                result = await func(*args, **kwargs)
                execution_time_ms = (time.time() - start_time) * 1000

                # Log function exit
                training_logger.log_function_exit(
                    function_name,
                    result if show_return else None,
                    execution_time_ms
                )

                return result

            except Exception as e:
                execution_time_ms = (time.time() - start_time) * 1000
                training_logger.log_function_error(function_name, e, execution_time_ms)
                raise

        # Wähle passenden Wrapper basierend auf Funktion
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def log_orchestrator_step(
    step_name: str,
    step_type: str = "orchestration",
    **details
) -> None:
    """Convenience-Funktion für Orchestrator-Logging."""
    # Prüfe ob TRAIN Level aktiviert ist
    train_enabled = os.getenv("LOGGING_TRAIN", "false").lower() in ("true", "1", "yes", "on")
    if train_enabled:
        training_logger = get_training_logger()
        training_logger.log_orchestrator_step(step_name, step_type, details)
