"""Tests für Type-Definitionen."""

from __future__ import annotations

from typing import get_type_hints

import pytest

from services.interfaces._types import (
    AgentId,
    CapabilityConfig,
    ChannelName,
    EventData,
    EventHandler,
    HealthStatus,
    MessageHandler,
    OperationResult,
    OptionalConfig,
    OptionalQueue,
    OptionalTimeout,
    ServiceConfig,
    ServiceId,
    ServiceResult,
    SubjectName,
    TaskPayload,
)


class TestTypeAliases:
    """Tests für Type-Aliases."""

    def test_string_aliases(self) -> None:
        """Testet String-basierte Type-Aliases."""
        # Diese sollten alle str sein
        string_types = [ServiceId, ChannelName, SubjectName, AgentId]

        for type_alias in string_types:
            assert type_alias == str, f"{type_alias} sollte str sein"

    def test_dict_aliases(self) -> None:
        """Testet Dictionary-basierte Type-Aliases."""
        from typing import Any

        # Diese sollten alle Dict[str, Any] sein
        dict_types = [
            TaskPayload,
            EventData,
            HealthStatus,
            ServiceResult,
            ServiceConfig,
            CapabilityConfig,
        ]

        for type_alias in dict_types:
            # Type-Aliases sind zur Laufzeit identisch mit ihrem Ziel-Typ
            assert type_alias == dict[str, Any], f"{type_alias} sollte Dict[str, Any] sein"

    def test_union_types(self) -> None:
        """Testet Union-Type-Aliases."""
        from typing import Any, Union

        # OperationResult sollte Union[bool, Dict[str, Any]] sein
        assert OperationResult == Union[bool, dict[str, Any]]

    def test_optional_types(self) -> None:
        """Testet Optional-Type-Aliases."""
        from typing import Optional

        optional_types = [
            (OptionalConfig, Optional[ServiceConfig]),
            (OptionalTimeout, Optional[float]),
            (OptionalQueue, Optional[str]),
        ]

        for type_alias, expected in optional_types:
            assert type_alias == expected, f"{type_alias} sollte {expected} sein"

    def test_callable_types(self) -> None:
        """Testet Callable-Type-Aliases."""
        from collections.abc import Awaitable, Callable
        from typing import Any

        # EventHandler sollte Callable[[bytes], Awaitable[None]] sein
        assert EventHandler == Callable[[bytes], Awaitable[None]]

        # MessageHandler sollte Callable[[EventData], Awaitable[None]] sein
        assert MessageHandler == Callable[[dict[str, Any]], Awaitable[None]]


class TestTypeUsage:
    """Tests für Type-Verwendung in realen Szenarien."""

    def test_service_id_usage(self) -> None:
        """Testet ServiceId Verwendung."""
        service_id: ServiceId = "test-service-123"
        assert isinstance(service_id, str)
        assert service_id == "test-service-123"

    def test_task_payload_usage(self) -> None:
        """Testet TaskPayload Verwendung."""
        payload: TaskPayload = {
            "action": "process",
            "data": {"key": "value"},
            "timeout": 30.0
        }

        assert isinstance(payload, dict)
        assert payload["action"] == "process"
        assert isinstance(payload["data"], dict)

    def test_event_data_usage(self) -> None:
        """Testet EventData Verwendung."""
        event: EventData = {
            "type": "user_action",
            "timestamp": "2025-08-20T10:00:00Z",
            "payload": {"user_id": "123"}
        }

        assert isinstance(event, dict)
        assert event["type"] == "user_action"

    def test_health_status_usage(self) -> None:
        """Testet HealthStatus Verwendung."""
        health: HealthStatus = {
            "status": "healthy",
            "service": "test-service",
            "uptime": 3600,
            "memory_usage": 0.75
        }

        assert isinstance(health, dict)
        assert health["status"] == "healthy"

    def test_service_result_usage(self) -> None:
        """Testet ServiceResult Verwendung."""
        result: ServiceResult = {
            "success": True,
            "data": {"result": "processed"},
            "metadata": {"duration": 0.5}
        }

        assert isinstance(result, dict)
        assert result["success"] is True

    def test_operation_result_usage(self) -> None:
        """Testet OperationResult Verwendung."""
        # Bool-Variante
        bool_result: OperationResult = True
        assert isinstance(bool_result, bool)

        # Dict-Variante
        dict_result: OperationResult = {"success": True, "message": "OK"}
        assert isinstance(dict_result, dict)

    def test_optional_types_usage(self) -> None:
        """Testet Optional-Types Verwendung."""
        # None-Werte
        no_config: OptionalConfig = None
        no_timeout: OptionalTimeout = None
        no_queue: OptionalQueue = None

        assert no_config is None
        assert no_timeout is None
        assert no_queue is None

        # Echte Werte
        config: OptionalConfig = {"key": "value"}
        timeout: OptionalTimeout = 30.0
        queue: OptionalQueue = "test-queue"

        assert isinstance(config, dict)
        assert isinstance(timeout, float)
        assert isinstance(queue, str)


class TestTypeCompatibility:
    """Tests für Type-Kompatibilität."""

    def test_backward_compatibility(self) -> None:
        """Testet Backward Compatibility der Types."""
        # Alte Dict[str, Any] sollte mit neuen Types kompatibel sein
        from typing import Any

        old_payload: dict[str, Any] = {"key": "value"}
        new_payload: TaskPayload = old_payload

        assert new_payload == old_payload

    def test_type_annotations_work(self) -> None:
        """Testet, dass Type-Annotations funktionieren."""
        def process_task(agent_id: AgentId, payload: TaskPayload) -> ServiceResult:
            """Test-Funktion mit Type-Annotations."""
            return {
                "agent_id": agent_id,
                "processed": True,
                "payload_size": len(payload)
            }

        # Type Hints sollten verfügbar sein
        hints = get_type_hints(process_task)
        assert "agent_id" in hints
        assert "payload" in hints
        assert "return" in hints

    def test_mypy_compatibility(self) -> None:
        """Testet MyPy-Kompatibilität (soweit möglich zur Laufzeit)."""
        # Diese Tests validieren, dass die Types zur Laufzeit korrekt sind
        # MyPy-Validierung erfolgt separat

        def handler_function(data: bytes) -> None:
            """Test-Handler-Funktion."""

        async def async_handler_function(data: bytes) -> None:
            """Test-Async-Handler-Funktion."""

        # Diese sollten ohne Fehler funktionieren
        assert callable(handler_function)
        assert callable(async_handler_function)


if __name__ == "__main__":
    pytest.main([__file__])
