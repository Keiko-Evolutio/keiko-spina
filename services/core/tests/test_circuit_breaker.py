# backend/services/core/tests/test_circuit_breaker.py
"""Tests für Circuit Breaker Implementierung.

Testet alle Zustände, Übergänge und Edge Cases des Circuit Breakers.
"""

import asyncio
import time

import pytest

from services.core.circuit_breaker import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_OPEN_TIMEOUT_SECONDS,
    CircuitBreaker,
    CircuitPolicy,
    CircuitState,
)


class TestCircuitPolicy:
    """Tests für CircuitPolicy Konfiguration."""

    def test_default_values(self):
        """Testet Standard-Konfigurationswerte."""
        policy = CircuitPolicy()

        assert policy.failure_threshold == DEFAULT_FAILURE_THRESHOLD
        assert policy.open_timeout_seconds == DEFAULT_OPEN_TIMEOUT_SECONDS
        assert policy.half_open_max_concurrent == 1
        assert policy.recovery_backoff_base == 1.5
        assert policy.recovery_backoff_max_seconds == 30.0

    def test_custom_values(self):
        """Testet benutzerdefinierte Konfigurationswerte."""
        policy = CircuitPolicy(
            failure_threshold=3,
            open_timeout_seconds=10.0,
            half_open_max_concurrent=2,
            recovery_backoff_base=2.0,
            recovery_backoff_max_seconds=60.0
        )

        assert policy.failure_threshold == 3
        assert policy.open_timeout_seconds == 10.0
        assert policy.half_open_max_concurrent == 2
        assert policy.recovery_backoff_base == 2.0
        assert policy.recovery_backoff_max_seconds == 60.0


class TestCircuitBreaker:
    """Tests für CircuitBreaker Implementierung."""

    @pytest.fixture
    def circuit_breaker(self):
        """Circuit Breaker mit Test-Konfiguration."""
        policy = CircuitPolicy(
            failure_threshold=2,
            open_timeout_seconds=0.1,  # Kurze Timeouts für Tests
            recovery_backoff_max_seconds=0.5
        )
        return CircuitBreaker("test-circuit", policy)

    @pytest.fixture
    def failing_function(self):
        """Mock-Funktion die immer fehlschlägt."""
        async def fail():
            raise ValueError("Test error")
        return fail

    @pytest.fixture
    def succeeding_function(self):
        """Mock-Funktion die immer erfolgreich ist."""
        async def succeed():
            return "success"
        return succeed

    def test_initial_state(self, circuit_breaker):
        """Testet initialen Zustand des Circuit Breakers."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.name == "test-circuit"
        assert circuit_breaker._failures == 0

    @pytest.mark.asyncio
    async def test_successful_call_in_closed_state(self, circuit_breaker, succeeding_function):
        """Testet erfolgreichen Aufruf im CLOSED Zustand."""
        result = await circuit_breaker.call(succeeding_function)

        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker._failures == 0

    @pytest.mark.asyncio
    async def test_failure_in_closed_state(self, circuit_breaker, failing_function):
        """Testet Fehler im CLOSED Zustand ohne Schwellenwert-Überschreitung."""
        with pytest.raises(ValueError, match="Test error"):
            await circuit_breaker.call(failing_function)

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker._failures == 1

    @pytest.mark.asyncio
    async def test_transition_to_open_on_threshold(self, circuit_breaker, failing_function):
        """Testet Übergang zu OPEN bei Überschreitung der Fehlerschwelle."""
        # Erste Fehler (unter Schwellenwert)
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_function)
        assert circuit_breaker.state == CircuitState.CLOSED

        # Zweiter Fehler (erreicht Schwellenwert)
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_function)
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker._failures == 2

    @pytest.mark.asyncio
    async def test_circuit_open_blocks_calls(self, circuit_breaker, succeeding_function):
        """Testet dass OPEN Circuit weitere Aufrufe blockiert."""
        # Circuit öffnen durch Fehler
        circuit_breaker._state = CircuitState.OPEN
        circuit_breaker._opened_at = time.time()

        with pytest.raises(RuntimeError, match="Circuit 'test-circuit' is OPEN"):
            await circuit_breaker.call(succeeding_function)

    @pytest.mark.asyncio
    async def test_recovery_after_timeout(self, circuit_breaker, succeeding_function):
        """Testet Recovery nach Timeout."""
        # Circuit öffnen
        circuit_breaker._state = CircuitState.OPEN
        circuit_breaker._opened_at = time.time() - 0.2  # Vor 0.2s geöffnet

        # Sollte jetzt Recovery versuchen
        result = await circuit_breaker.call(succeeding_function)

        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker._failures == 0

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self, circuit_breaker, succeeding_function):
        """Testet dass erfolgreicher HALF_OPEN Aufruf Circuit schließt."""
        # Manuell in HALF_OPEN setzen
        circuit_breaker._state = CircuitState.HALF_OPEN
        circuit_breaker._failures = 2

        result = await circuit_breaker.call(succeeding_function)

        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker._failures == 0

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self, circuit_breaker, failing_function):
        """Testet dass fehlgeschlagener HALF_OPEN Aufruf Circuit wieder öffnet."""
        # Manuell in HALF_OPEN setzen
        circuit_breaker._state = CircuitState.HALF_OPEN
        circuit_breaker._failures = 2

        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_function)

        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker._failures == 3

    @pytest.mark.asyncio
    async def test_concurrent_half_open_calls(self, circuit_breaker):
        """Testet Concurrency Control im HALF_OPEN Zustand."""
        circuit_breaker._state = CircuitState.HALF_OPEN

        call_count = 0
        first_call_started = asyncio.Event()

        async def slow_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                first_call_started.set()
                await asyncio.sleep(0.1)  # Erste Funktion dauert länger
            return "success"

        # Starte mehrere gleichzeitige Aufrufe
        async def make_call():
            return await circuit_breaker.call(slow_function)

        # Erste Task starten
        task1 = asyncio.create_task(make_call())

        # Warten bis erste Task das Semaphore hat
        await first_call_started.wait()

        # Weitere Tasks starten - diese sollten blockiert werden
        task2 = asyncio.create_task(make_call())
        task3 = asyncio.create_task(make_call())

        # Kurz warten um sicherzustellen dass Tasks 2&3 am Semaphore warten
        await asyncio.sleep(0.05)

        # Erste Task sollte erfolgreich sein und Circuit schließen
        result1 = await task1
        assert result1 == "success"
        assert circuit_breaker.state == CircuitState.CLOSED

        # Andere Tasks sollten jetzt auch durchgehen (Circuit ist CLOSED)
        result2 = await task2
        result3 = await task3

        assert result2 == "success"
        assert result3 == "success"

        # Aber nur ein Aufruf sollte im HALF_OPEN Zustand gewesen sein
        # Die anderen liefen im CLOSED Zustand nach dem ersten Erfolg

    def test_can_attempt_recovery_logic(self, circuit_breaker):
        """Testet Recovery-Logik mit verschiedenen Szenarien."""
        # Nicht OPEN -> kein Recovery
        circuit_breaker._state = CircuitState.CLOSED
        assert not circuit_breaker._can_attempt_recovery()

        # OPEN aber zu früh
        circuit_breaker._state = CircuitState.OPEN
        circuit_breaker._opened_at = time.time()
        assert not circuit_breaker._can_attempt_recovery()

        # OPEN und genug Zeit vergangen
        circuit_breaker._opened_at = time.time() - 0.2
        assert circuit_breaker._can_attempt_recovery()

    @pytest.mark.asyncio
    async def test_exponential_backoff(self, circuit_breaker):
        """Testet exponentielles Backoff bei wiederholten Fehlern."""
        policy = CircuitPolicy(
            failure_threshold=1,
            open_timeout_seconds=0.1,
            recovery_backoff_base=2.0,
            recovery_backoff_max_seconds=1.0
        )
        cb = CircuitBreaker("backoff-test", policy)

        # Simuliere mehrere Fehler
        cb._state = CircuitState.OPEN
        cb._failures = 3  # 2 über Schwellenwert
        cb._opened_at = time.time() - 0.05  # Vor 0.05s

        # Sollte noch nicht recovern können (exponentielles Backoff)
        assert not cb._can_attempt_recovery()

        # Nach längerer Zeit sollte Recovery möglich sein
        cb._opened_at = time.time() - 1.0
        assert cb._can_attempt_recovery()

    @pytest.mark.asyncio
    async def test_reset_failures_on_success(self, circuit_breaker, succeeding_function, failing_function):
        """Testet dass Fehler bei Erfolg zurückgesetzt werden."""
        # Ein Fehler
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_function)
        assert circuit_breaker._failures == 1

        # Erfolgreicher Aufruf sollte Fehler zurücksetzen
        await circuit_breaker.call(succeeding_function)
        assert circuit_breaker._failures == 0
