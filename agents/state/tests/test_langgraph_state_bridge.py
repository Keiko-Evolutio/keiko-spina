"""Unit-Tests für langgraph_state_bridge.py."""

from unittest.mock import patch

import pytest

from agents.state.langgraph_state_bridge import (
    WorkflowState,
    max_step,
    replace_bool,
    replace_dict,
    replace_int,
    replace_message,
)
from agents.state.state_constants import (
    DEFAULT_EMPTY_MESSAGE,
    DEFAULT_STEP_VALUE,
)


class TestReducerFunctions:
    """Tests für konsolidierte Reducer-Funktionen."""

    def test_replace_message(self):
        """Testet Message-Replace-Reducer."""
        result = replace_message("old message", "new message")
        assert result == "new message"

    def test_replace_bool(self):
        """Testet Boolean-Replace-Reducer."""
        result = replace_bool(True, False)
        assert result is False

        result = replace_bool(False, True)
        assert result is True

    def test_replace_int(self):
        """Testet Integer-Replace-Reducer."""
        result = replace_int(10, 20)
        assert result == 20

    def test_replace_dict(self):
        """Testet Dict-Replace-Reducer."""
        old_dict = {"a": 1, "b": 2}
        new_dict = {"c": 3, "d": 4}
        result = replace_dict(old_dict, new_dict)
        assert result == new_dict

    def test_max_step(self):
        """Testet Max-Step-Reducer."""
        # Normale Max-Operation
        result = max_step(5, 10)
        assert result == 10

        result = max_step(15, 8)
        assert result == 15

        # Gleiche Werte
        result = max_step(7, 7)
        assert result == 7


class TestWorkflowState:
    """Tests für WorkflowState-Klasse."""

    def test_workflow_state_creation(self):
        """Testet WorkflowState-Erstellung."""
        state = WorkflowState(message="test message", step=5)

        assert state.message == "test message"
        assert state.step == 5

    def test_workflow_state_defaults(self):
        """Testet WorkflowState-Default-Werte."""
        state = WorkflowState(message="test")

        assert state.message == "test"
        assert state.step == DEFAULT_STEP_VALUE

    def test_workflow_state_to_dict_success(self):
        """Testet erfolgreiche to_dict-Serialisierung."""
        state = WorkflowState(message="test message", step=3)

        with patch("agents.state.langgraph_state_bridge.serialize_state_safely") as mock_serialize:
            mock_serialize.return_value = {"message": "test message", "step": 3}

            result = state.to_dict()

            assert result == {"message": "test message", "step": 3}
            mock_serialize.assert_called_once_with(state)

    def test_workflow_state_to_dict_error(self):
        """Testet to_dict-Serialisierungs-Fehler."""
        state = WorkflowState(message="test", step=1)

        with patch("agents.state.langgraph_state_bridge.serialize_state_safely") as mock_serialize:
            mock_serialize.side_effect = ValueError("Serialization failed")

            with pytest.raises(ValueError) as exc_info:
                state.to_dict()

            assert "Serialization failed" in str(exc_info.value)

    def test_workflow_state_from_dict_success(self):
        """Testet erfolgreiche from_dict-Deserialisierung."""
        data = {"message": "restored message", "step": 7}

        state = WorkflowState.from_dict(data)

        assert state.message == "restored message"
        assert state.step == 7

    def test_workflow_state_from_dict_partial_data(self):
        """Testet from_dict mit partiellen Daten."""
        # Nur message
        data = {"message": "only message"}
        state = WorkflowState.from_dict(data)

        assert state.message == "only message"
        assert state.step == DEFAULT_STEP_VALUE

        # Nur step
        data = {"step": 42}
        state = WorkflowState.from_dict(data)

        assert state.message == DEFAULT_EMPTY_MESSAGE
        assert state.step == 42

    def test_workflow_state_from_dict_empty_data(self):
        """Testet from_dict mit leeren Daten."""
        data = {}
        state = WorkflowState.from_dict(data)

        assert state.message == DEFAULT_EMPTY_MESSAGE
        assert state.step == DEFAULT_STEP_VALUE

    def test_workflow_state_from_dict_invalid_data(self):
        """Testet from_dict mit ungültigen Daten."""
        with pytest.raises(ValueError) as exc_info:
            WorkflowState.from_dict("not a dict")

        assert "Data muss ein Dict sein" in str(exc_info.value)

    def test_workflow_state_from_dict_type_conversion(self):
        """Testet Type-Konvertierung in from_dict."""
        # String-zu-Int-Konvertierung für step
        data = {"message": "test", "step": "42"}
        state = WorkflowState.from_dict(data)

        assert state.message == "test"
        assert state.step == 42
        assert isinstance(state.step, int)

        # Verschiedene Typen für message
        data = {"message": 123, "step": 1}
        state = WorkflowState.from_dict(data)

        assert state.message == "123"
        assert isinstance(state.message, str)

    def test_workflow_state_serialization_roundtrip(self):
        """Testet vollständigen Serialisierungs-Roundtrip."""
        original = WorkflowState(message="roundtrip test", step=99)

        # Mock serialize_state_safely für to_dict
        with patch("agents.state.langgraph_state_bridge.serialize_state_safely") as mock_serialize:
            mock_serialize.return_value = {"message": "roundtrip test", "step": 99}

            # Serialisierung
            data = original.to_dict()

            # Deserialisierung
            restored = WorkflowState.from_dict(data)

            assert restored.message == original.message
            assert restored.step == original.step

    def test_workflow_state_inheritance(self):
        """Testet SerializationMixin-Vererbung."""
        state = WorkflowState(message="test", step=1)

        # Prüfe dass SerializationMixin-Methoden verfügbar sind
        assert hasattr(state, "to_dict")

        # Test der geerbten from_dict-Funktionalität (falls verfügbar)
        from agents.state.state_utils import SerializationMixin
        assert isinstance(state, SerializationMixin)


class TestWorkflowStateAnnotations:
    """Tests für Annotated-Type-Verhalten."""

    def test_message_annotation(self):
        """Testet Message-Annotation mit replace_message."""
        # Diese Tests prüfen das Verhalten der Annotated-Types
        # In der Praxis werden diese von LangGraph verwendet

        state = WorkflowState(message="initial", step=1)
        assert state.message == "initial"

        # Simuliere LangGraph-Reducer-Verhalten
        new_message = replace_message(state.message, "updated")
        assert new_message == "updated"

    def test_step_annotation(self):
        """Testet Step-Annotation mit max_step."""
        state = WorkflowState(message="test", step=5)
        assert state.step == 5

        # Simuliere LangGraph-Reducer-Verhalten
        new_step = max_step(state.step, 3)  # Sollte 5 bleiben (max)
        assert new_step == 5

        new_step = max_step(state.step, 10)  # Sollte 10 werden (max)
        assert new_step == 10


class TestWorkflowStateEdgeCases:
    """Tests für Edge-Cases und Fehlerbehandlung."""

    def test_workflow_state_extreme_values(self):
        """Testet extreme Werte."""
        # Sehr lange Message
        long_message = "x" * 10000
        state = WorkflowState(message=long_message, step=999999)

        assert state.message == long_message
        assert state.step == 999999

    def test_workflow_state_unicode_message(self):
        """Testet Unicode-Messages."""
        unicode_message = "🚀 Workflow mit Emojis und Ümläuten äöü"
        state = WorkflowState(message=unicode_message, step=1)

        assert state.message == unicode_message

        # Roundtrip-Test
        with patch("agents.state.langgraph_state_bridge.serialize_state_safely") as mock_serialize:
            mock_serialize.return_value = {"message": unicode_message, "step": 1}

            data = state.to_dict()
            restored = WorkflowState.from_dict(data)

            assert restored.message == unicode_message

    def test_workflow_state_negative_step(self):
        """Testet negative Step-Werte."""
        state = WorkflowState(message="test", step=-5)
        assert state.step == -5

        # Max-Reducer sollte trotzdem funktionieren
        result = max_step(-5, -2)
        assert result == -2
