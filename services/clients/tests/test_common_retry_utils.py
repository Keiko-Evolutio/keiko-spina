# backend/services/clients/tests/test_common_retry_utils.py
"""Tests für Common Retry Utilities.

Deutsche Docstrings, englische Identifiers, isolierte Tests mit Mocking.
"""

from unittest.mock import AsyncMock, patch

import pytest

from services.clients.common.retry_utils import (
    RetryableClient,
    RetryConfig,
    RetryExhaustedException,
    create_content_safety_retry_config,
    create_http_retry_config,
    create_image_generation_retry_config,
    retry_with_backoff,
    with_retry,
)


class TestRetryConfig:
    """Tests für RetryConfig Dataclass."""

    def test_retry_config_default_values(self) -> None:
        """Prüft, dass RetryConfig korrekte Standardwerte hat."""
        config = RetryConfig()

        assert config.max_retries == 2
        assert config.initial_delay == 0.5
        assert config.backoff_multiplier == 2.0
        assert config.max_delay == 60.0
        assert config.exceptions == (Exception,)

    def test_retry_config_custom_values(self) -> None:
        """Prüft, dass RetryConfig benutzerdefinierte Werte akzeptiert."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=1.0,
            backoff_multiplier=1.5,
            max_delay=120.0,
            exceptions=(ValueError, RuntimeError)
        )

        assert config.max_retries == 5
        assert config.initial_delay == 1.0
        assert config.backoff_multiplier == 1.5
        assert config.max_delay == 120.0
        assert config.exceptions == (ValueError, RuntimeError)

    def test_retry_config_slots(self) -> None:
        """Prüft, dass RetryConfig __slots__ verwendet."""
        config = RetryConfig()

        # __slots__ verhindert dynamische Attribute
        with pytest.raises(AttributeError):
            config.dynamic_attribute = "test"  # type: ignore


class TestRetryExhaustedException:
    """Tests für RetryExhaustedException."""

    def test_retry_exhausted_exception_creation(self) -> None:
        """Prüft, dass RetryExhaustedException korrekt erstellt wird."""
        original_error = ValueError("Original error")
        exception = RetryExhaustedException(3, original_error)

        assert exception.attempts == 3
        assert exception.last_error is original_error
        assert "Retry nach 3 Versuchen fehlgeschlagen" in str(exception)
        assert "Original error" in str(exception)

    def test_retry_exhausted_exception_inheritance(self) -> None:
        """Prüft, dass RetryExhaustedException von Exception erbt."""
        original_error = RuntimeError("Test error")
        exception = RetryExhaustedException(2, original_error)

        assert isinstance(exception, Exception)


class TestRetryWithBackoff:
    """Tests für retry_with_backoff Funktion."""

    @pytest.mark.asyncio
    async def test_successful_function_no_retry_needed(self) -> None:
        """Prüft, dass erfolgreiche Funktionen ohne Retry ausgeführt werden."""
        mock_func = AsyncMock(return_value="success")

        result = await retry_with_backoff(mock_func, "arg1", kwarg1="value1")

        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    @pytest.mark.asyncio
    async def test_retry_on_failure_then_success(self) -> None:
        """Prüft, dass Retry bei Fehlern funktioniert."""
        mock_func = AsyncMock(side_effect=[
            RuntimeError("First failure"),
            RuntimeError("Second failure"),
            "success"
        ])

        config = RetryConfig(max_retries=3, initial_delay=0.01)

        result = await retry_with_backoff(mock_func, config=config)

        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_exception(self) -> None:
        """Prüft, dass RetryExhaustedException bei erschöpften Versuchen geworfen wird."""
        mock_func = AsyncMock(side_effect=RuntimeError("Persistent error"))

        config = RetryConfig(max_retries=2, initial_delay=0.01)

        with pytest.raises(RetryExhaustedException) as exc_info:
            await retry_with_backoff(mock_func, config=config)

        assert exc_info.value.attempts == 3  # max_retries + 1
        assert isinstance(exc_info.value.last_error, RuntimeError)
        assert "Persistent error" in str(exc_info.value.last_error)

    @pytest.mark.asyncio
    async def test_retry_respects_exception_filter(self) -> None:
        """Prüft, dass nur konfigurierte Exceptions zu Retry führen."""
        mock_func = AsyncMock(side_effect=ValueError("Not retryable"))

        config = RetryConfig(
            max_retries=2,
            initial_delay=0.01,
            exceptions=(RuntimeError,)  # Nur RuntimeError wird wiederholt
        )

        with pytest.raises(ValueError):
            await retry_with_backoff(mock_func, config=config)

        # Sollte nur einmal aufgerufen werden, da ValueError nicht in exceptions ist
        mock_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_exponential_backoff_delay(self) -> None:
        """Prüft, dass Exponential Backoff korrekt implementiert ist."""
        mock_func = AsyncMock(side_effect=[
            RuntimeError("Error 1"),
            RuntimeError("Error 2"),
            "success"
        ])

        config = RetryConfig(
            max_retries=2,
            initial_delay=0.1,
            backoff_multiplier=2.0
        )

        with patch("asyncio.sleep") as mock_sleep:
            result = await retry_with_backoff(mock_func, config=config)

        assert result == "success"

        # Prüfe, dass sleep mit exponentiell steigenden Delays aufgerufen wurde
        expected_delays = [0.1, 0.2]  # initial_delay * backoff_multiplier^attempt
        actual_delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays

    @pytest.mark.asyncio
    async def test_max_delay_is_respected(self) -> None:
        """Prüft, dass max_delay respektiert wird."""
        mock_func = AsyncMock(side_effect=[
            RuntimeError("Error 1"),
            RuntimeError("Error 2"),
            RuntimeError("Error 3"),
            "success"
        ])

        config = RetryConfig(
            max_retries=3,
            initial_delay=10.0,
            backoff_multiplier=10.0,
            max_delay=5.0  # Kleiner als initial_delay * backoff_multiplier
        )

        with patch("asyncio.sleep") as mock_sleep:
            result = await retry_with_backoff(mock_func, config=config)

        assert result == "success"

        # Alle Delays sollten max_delay nicht überschreiten
        actual_delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert all(delay <= config.max_delay for delay in actual_delays)


class TestWithRetryDecorator:
    """Tests für with_retry Decorator."""

    @pytest.mark.asyncio
    async def test_decorator_with_default_config(self) -> None:
        """Prüft, dass der Decorator mit Standard-Konfiguration funktioniert."""
        @with_retry()
        async def test_function() -> str:
            return "decorated_success"

        result = await test_function()
        assert result == "decorated_success"

    @pytest.mark.asyncio
    async def test_decorator_with_custom_config(self) -> None:
        """Prüft, dass der Decorator mit benutzerdefinierter Konfiguration funktioniert."""
        call_count = 0

        @with_retry(RetryConfig(max_retries=2, initial_delay=0.01))
        async def test_function() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError(f"Attempt {call_count}")
            return "success_after_retries"

        result = await test_function()
        assert result == "success_after_retries"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self) -> None:
        """Prüft, dass der Decorator Funktions-Metadaten erhält."""
        @with_retry()
        async def documented_function() -> str:
            """Eine dokumentierte Funktion."""
            return "result"

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "Eine dokumentierte Funktion."


class TestRetryableClient:
    """Tests für RetryableClient Base-Klasse."""

    def test_retryable_client_initialization(self) -> None:
        """Prüft, dass RetryableClient korrekt initialisiert wird."""
        config = RetryConfig(max_retries=5)
        client = RetryableClient(config)

        assert client._retry_config is config

    def test_retryable_client_default_config(self) -> None:
        """Prüft, dass RetryableClient Standard-Konfiguration verwendet."""
        client = RetryableClient()

        assert isinstance(client._retry_config, RetryConfig)
        assert client._retry_config.max_retries == 2  # Standard-Wert

    @pytest.mark.asyncio
    async def test_execute_with_retry_method(self) -> None:
        """Prüft, dass _execute_with_retry korrekt funktioniert."""
        client = RetryableClient(RetryConfig(max_retries=1, initial_delay=0.01))

        mock_func = AsyncMock(side_effect=[RuntimeError("Error"), "success"])

        result = await client._execute_with_retry(mock_func, "arg1", kwarg1="value1")

        assert result == "success"
        assert mock_func.call_count == 2


class TestRetryConfigFactories:
    """Tests für Retry-Konfiguration Factory-Funktionen."""

    def test_create_content_safety_retry_config(self) -> None:
        """Prüft, dass Content Safety Retry-Konfiguration korrekt erstellt wird."""
        config = create_content_safety_retry_config()

        assert isinstance(config, RetryConfig)
        assert config.max_retries == 2
        assert config.initial_delay == 0.5
        assert config.exceptions == (Exception,)

    def test_create_image_generation_retry_config(self) -> None:
        """Prüft, dass Image Generation Retry-Konfiguration korrekt erstellt wird."""
        config = create_image_generation_retry_config()

        assert isinstance(config, RetryConfig)
        assert config.max_retries == 2
        assert config.initial_delay == 0.5
        assert config.exceptions == (Exception,)

    def test_create_http_retry_config(self) -> None:
        """Prüft, dass HTTP Retry-Konfiguration korrekt erstellt wird."""
        config = create_http_retry_config()

        assert isinstance(config, RetryConfig)
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.exceptions == (Exception,)

    def test_factory_configs_are_independent(self) -> None:
        """Prüft, dass Factory-Konfigurationen unabhängig sind."""
        config1 = create_content_safety_retry_config()
        config2 = create_image_generation_retry_config()
        config3 = create_http_retry_config()

        # Verschiedene Instanzen
        assert config1 is not config2
        assert config2 is not config3
        assert config1 is not config3

        # Verschiedene Werte für HTTP
        assert config3.max_retries != config1.max_retries
        assert config3.initial_delay != config1.initial_delay


class TestRetryLogging:
    """Tests für Retry-Logging."""

    @pytest.mark.asyncio
    async def test_retry_logging_on_attempts(self) -> None:
        """Prüft, dass Retry-Versuche geloggt werden."""
        mock_func = AsyncMock(side_effect=[RuntimeError("Error"), "success"])

        config = RetryConfig(max_retries=1, initial_delay=0.01)

        with patch("services.clients.common.retry_utils.logger") as mock_logger:
            result = await retry_with_backoff(mock_func, config=config)

        assert result == "success"

        # Prüfe, dass Debug-Logs für Versuche erstellt wurden
        debug_calls = list(mock_logger.debug.call_args_list)
        assert len(debug_calls) >= 2  # Mindestens 2 Versuche geloggt

        # Prüfe, dass Warning-Log für fehlgeschlagenen Versuch erstellt wurde
        warning_calls = list(mock_logger.warning.call_args_list)
        assert len(warning_calls) >= 1

    @pytest.mark.asyncio
    async def test_retry_logging_on_success_after_retry(self) -> None:
        """Prüft, dass erfolgreiche Retry-Versuche geloggt werden."""
        mock_func = AsyncMock(side_effect=[RuntimeError("Error"), "success"])

        config = RetryConfig(max_retries=1, initial_delay=0.01)

        with patch("services.clients.common.retry_utils.logger") as mock_logger:
            result = await retry_with_backoff(mock_func, config=config)

        assert result == "success"

        # Prüfe, dass Info-Log für erfolgreichen Retry erstellt wurde
        info_calls = list(mock_logger.info.call_args_list)
        assert len(info_calls) >= 1

        # Prüfe Log-Inhalt
        log_data = info_calls[0][0][0]
        assert log_data["event"] == "retry_success"
