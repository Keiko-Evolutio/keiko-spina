"""Unit-Tests für state_utils.py."""

from unittest.mock import MagicMock, patch

import pytest

from agents.state.state_utils import (
    SerializationMixin,
    create_max_reducer,
    create_optional_replace_reducer,
    create_replace_reducer,
    deserialize_state_safely,
    handle_workflow_operation,
    serialize_state_safely,
    validate_workflow_config,
    validate_workflow_name,
)


class TestReducerFactories:
    """Tests für Reducer-Factory-Funktionen."""

    def test_create_replace_reducer(self):
        """Testet generischen Replace-Reducer."""
        reducer = create_replace_reducer()

        # String-Test
        assert reducer("old", "new") == "new"

        # Integer-Test
        assert reducer(1, 2) == 2

        # Boolean-Test
        assert reducer(True, False) is False

        # Dict-Test
        old_dict = {"a": 1}
        new_dict = {"b": 2}
        assert reducer(old_dict, new_dict) == new_dict

    def test_create_optional_replace_reducer(self):
        """Testet Optional-Replace-Reducer."""
        reducer = create_optional_replace_reducer()

        # None-sicherer Test
        assert reducer("old", None) == "old"
        assert reducer(None, "new") == "new"
        assert reducer("old", "new") == "new"
        assert reducer(None, None) is None

    def test_create_max_reducer(self):
        """Testet Max-Reducer."""
        reducer = create_max_reducer()

        # Integer-Test
        assert reducer(1, 2) == 2
        assert reducer(5, 3) == 5

        # Float-Test
        assert reducer(1.5, 2.7) == 2.7
        assert reducer(3.14, 2.71) == 3.14


class TestSerializationMixin:
    """Tests für SerializationMixin."""

    def test_serialization_mixin(self):
        """Testet Mixin-Funktionalität."""
        from dataclasses import dataclass

        @dataclass
        class TestState(SerializationMixin):
            name: str
            value: int

        # Test to_dict
        state = TestState(name="test", value=42)
        data = state.to_dict()
        assert data == {"name": "test", "value": 42}

        # Test from_dict
        restored = TestState.from_dict(data)
        assert restored.name == "test"
        assert restored.value == 42


class TestSerializationFunctions:
    """Tests für Serialisierungs-Funktionen."""

    def test_serialize_state_safely_success(self):
        """Testet erfolgreiche State-Serialisierung."""
        mock_state = MagicMock()
        mock_state.to_dict.return_value = {"test": "data"}

        with patch("agents.state.state_utils.trace_span"):
            result = serialize_state_safely(mock_state)

        assert result == {"test": "data"}
        mock_state.to_dict.assert_called_once()

    def test_serialize_state_safely_error(self):
        """Testet Serialisierungs-Fehler."""
        mock_state = MagicMock()
        mock_state.to_dict.side_effect = Exception("Serialization error")

        with patch("agents.state.state_utils.trace_span"):
            with pytest.raises(ValueError) as exc_info:
                serialize_state_safely(mock_state)

        assert "Ungültiger State" in str(exc_info.value)

    def test_deserialize_state_safely_success(self):
        """Testet erfolgreiche State-Deserialisierung."""
        mock_class = MagicMock()
        mock_instance = MagicMock()
        mock_class.from_dict.return_value = mock_instance

        data = {"test": "data"}

        with patch("agents.state.state_utils.trace_span"):
            result = deserialize_state_safely(mock_class, data)

        assert result == mock_instance
        mock_class.from_dict.assert_called_once_with(data)

    def test_deserialize_state_safely_invalid_data(self):
        """Testet Deserialisierung mit ungültigen Daten."""
        mock_class = MagicMock()

        with patch("agents.state.state_utils.trace_span"):
            with pytest.raises(ValueError) as exc_info:
                deserialize_state_safely(mock_class, "not a dict")

        assert "Data muss ein Dict sein" in str(exc_info.value)

    def test_deserialize_state_safely_error(self):
        """Testet Deserialisierungs-Fehler."""
        mock_class = MagicMock()
        mock_class.from_dict.side_effect = Exception("Deserialization error")

        with patch("agents.state.state_utils.trace_span"):
            with pytest.raises(ValueError) as exc_info:
                deserialize_state_safely(mock_class, {"test": "data"})

        assert "Ungültiger State" in str(exc_info.value)


class TestWorkflowOperationHandler:
    """Tests für Workflow-Operation-Handler."""

    @pytest.mark.asyncio
    async def test_handle_workflow_operation_success(self):
        """Testet erfolgreiche Workflow-Operation."""
        async def mock_operation():
            return "success"

        with patch("agents.state.state_utils.trace_span"):
            result = await handle_workflow_operation(
                mock_operation,
                "test_context",
                test_attr="value"
            )

        assert result == "success"

    @pytest.mark.asyncio
    async def test_handle_workflow_operation_error(self):
        """Testet Workflow-Operation-Fehler."""
        async def mock_operation():
            raise ValueError("Operation failed")

        with patch("agents.state.state_utils.trace_span"):
            with pytest.raises(ValueError) as exc_info:
                await handle_workflow_operation(
                    mock_operation,
                    "test_context"
                )

        assert "Operation failed" in str(exc_info.value)


class TestValidationFunctions:
    """Tests für Validierungs-Funktionen."""

    def test_validate_workflow_config_valid(self):
        """Testet gültige Workflow-Konfiguration."""
        config = {
            "configurable": {
                "thread_id": "test-thread"
            }
        }

        # Sollte keine Exception werfen
        validate_workflow_config(config)

    def test_validate_workflow_config_invalid_type(self):
        """Testet ungültigen Konfigurationstyp."""
        with pytest.raises(ValueError) as exc_info:
            validate_workflow_config("not a dict")

        assert "Config muss ein Dict sein" in str(exc_info.value)

    def test_validate_workflow_config_missing_configurable(self):
        """Testet fehlende configurable-Sektion."""
        config = {"other": "value"}

        with pytest.raises(ValueError) as exc_info:
            validate_workflow_config(config)

        assert "thread_id ist erforderlich" in str(exc_info.value)

    def test_validate_workflow_config_invalid_thread_id(self):
        """Testet ungültige Thread-ID."""
        config = {
            "configurable": {
                "thread_id": None
            }
        }

        with pytest.raises(ValueError) as exc_info:
            validate_workflow_config(config)

        assert "thread_id ist erforderlich" in str(exc_info.value)

    def test_validate_workflow_name_valid(self):
        """Testet gültigen Workflow-Namen."""
        # Sollte keine Exception werfen
        validate_workflow_name("valid_workflow_name")

    def test_validate_workflow_name_invalid_type(self):
        """Testet ungültigen Workflow-Namen-Typ."""
        with pytest.raises(ValueError) as exc_info:
            validate_workflow_name("123")  # String statt int für Type-Safety

        assert "nicht-leerer String" in str(exc_info.value)

    def test_validate_workflow_name_empty(self):
        """Testet leeren Workflow-Namen."""
        with pytest.raises(ValueError) as exc_info:
            validate_workflow_name("")

        assert "nicht-leerer String" in str(exc_info.value)

    def test_validate_workflow_name_whitespace_only(self):
        """Testet Workflow-Namen mit nur Whitespace."""
        with pytest.raises(ValueError) as exc_info:
            validate_workflow_name("   ")

        assert "nicht leer sein" in str(exc_info.value)
