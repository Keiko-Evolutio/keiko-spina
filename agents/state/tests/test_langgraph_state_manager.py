"""Unit-Tests für langgraph_state_manager.py."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.state.langgraph_state_bridge import WorkflowState
from agents.state.langgraph_state_manager import (
    LangGraphStateManager,
    ManagedWorkflow,
)


class MockCheckpointSaver:
    """Mock-Implementierung für CheckpointSaver."""

    def __init__(self):
        self.checkpoints = {}

    def get(self, config: dict[str, Any]) -> dict[str, Any]:
        thread_id = config.get("configurable", {}).get("thread_id")
        return self.checkpoints.get(thread_id)

    def put(self, config: dict[str, Any], value: dict[str, Any], metadata=None) -> dict[str, Any]:
        thread_id = config.get("configurable", {}).get("thread_id")
        self.checkpoints[thread_id] = {"state": value, "metadata": metadata}
        return {"checkpoint_id": f"cp_{thread_id}", "thread_id": thread_id}


class TestManagedWorkflow:
    """Tests für ManagedWorkflow-Dataclass."""

    def test_managed_workflow_creation(self):
        """Testet ManagedWorkflow-Erstellung."""
        mock_graph = MagicMock()
        mock_checkpointer = MockCheckpointSaver()

        workflow = ManagedWorkflow(
            graph=mock_graph,
            checkpointer=mock_checkpointer,
            name="test_workflow"
        )

        assert workflow.graph == mock_graph
        assert workflow.checkpointer == mock_checkpointer
        assert workflow.name == "test_workflow"


class TestLangGraphStateManager:
    """Tests für LangGraphStateManager-Klasse."""

    def setup_method(self):
        """Setup für jeden Test."""
        self.manager = LangGraphStateManager()
        self.mock_graph = MagicMock()
        self.mock_checkpointer = MockCheckpointSaver()

    def test_manager_initialization(self):
        """Testet Manager-Initialisierung."""
        manager = LangGraphStateManager()
        assert manager._workflows == {}

    @patch("agents.state.langgraph_state_manager.validate_workflow_name")
    def test_register_workflow_success(self, mock_validate):
        """Testet erfolgreiche Workflow-Registrierung."""
        with patch("agents.state.langgraph_state_manager.logger") as mock_logger:
            self.manager.register(
                name="test_workflow",
                graph=self.mock_graph,
                checkpointer=self.mock_checkpointer
            )

        # Validierung wurde aufgerufen
        mock_validate.assert_called_once_with("test_workflow")

        # Workflow wurde registriert
        assert "test_workflow" in self.manager._workflows
        workflow = self.manager._workflows["test_workflow"]
        assert workflow.graph == self.mock_graph
        assert workflow.checkpointer == self.mock_checkpointer
        assert workflow.name == "test_workflow"

        # Logging wurde aufgerufen
        mock_logger.info.assert_called_once()

    @patch("agents.state.langgraph_state_manager.validate_workflow_name")
    def test_register_workflow_with_default_checkpointer(self, _mock_validate):
        """Testet Registrierung mit Default-Checkpointer."""
        with patch("agents.memory.langgraph_cosmos_checkpointer.CosmosCheckpointSaver") as mock_cosmos:
            mock_cosmos_instance = MagicMock()
            mock_cosmos.return_value = mock_cosmos_instance

            self.manager.register(
                name="test_workflow",
                graph=self.mock_graph
            )

        # Default-Checkpointer wurde erstellt
        mock_cosmos.assert_called_once()
        workflow = self.manager._workflows["test_workflow"]
        assert workflow.checkpointer == mock_cosmos_instance

    def test_register_workflow_invalid_name(self):
        """Testet Registrierung mit ungültigem Namen."""
        with patch("agents.state.langgraph_state_manager.validate_workflow_name") as mock_validate:
            mock_validate.side_effect = ValueError("Invalid name")

            with pytest.raises(ValueError):
                self.manager.register("", self.mock_graph)

    @pytest.mark.asyncio
    async def test_start_workflow_success(self):
        """Testet erfolgreichen Workflow-Start."""
        # Setup
        self.manager.register("test_workflow", self.mock_graph, self.mock_checkpointer)

        # Mock graph.ainvoke
        self.mock_graph.ainvoke = AsyncMock(return_value={"result": "success"})

        state = WorkflowState(message="test", step=1)

        with patch("agents.state.langgraph_state_manager.handle_workflow_operation") as mock_handler:
            mock_handler.return_value = {
                "state": {"result": "success"},
                "checkpoint": {"checkpoint_id": "cp_test"}
            }

            result = await self.manager.start("test_workflow", state, thread_id="test_thread")

        # Prüfe Ergebnis
        assert result["state"]["result"] == "success"
        assert result["checkpoint"]["checkpoint_id"] == "cp_test"

        # Prüfe dass handle_workflow_operation aufgerufen wurde
        mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_workflow_not_found(self):
        """Testet Start eines nicht registrierten Workflows."""
        state = WorkflowState(message="test", step=1)

        with patch("core.exceptions.KeikoNotFoundError") as mock_error:
            mock_error_instance = Exception("Not found")
            mock_error.side_effect = mock_error_instance

            with pytest.raises(Exception):
                await self.manager.start("nonexistent", state, thread_id="test")

    @pytest.mark.asyncio
    async def test_start_workflow_invalid_name(self):
        """Testet Start mit ungültigem Workflow-Namen."""
        state = WorkflowState(message="test", step=1)

        with patch("agents.state.langgraph_state_manager.validate_workflow_name") as mock_validate:
            mock_validate.side_effect = ValueError("Invalid name")

            with pytest.raises(ValueError):
                await self.manager.start("", state, thread_id="test")

    @pytest.mark.asyncio
    async def test_resume_workflow_success(self):
        """Testet erfolgreichen Workflow-Resume."""
        # Setup
        self.manager.register("test_workflow", self.mock_graph, self.mock_checkpointer)

        # Mock graph.ainvoke
        self.mock_graph.ainvoke = AsyncMock(return_value={"result": "resumed"})

        with patch("agents.state.langgraph_state_manager.handle_workflow_operation") as mock_handler:
            mock_handler.return_value = {
                "state": {"result": "resumed"},
                "checkpoint": {"checkpoint_id": "cp_test"}
            }

            result = await self.manager.resume("test_workflow", thread_id="test_thread")

        # Prüfe Ergebnis
        assert result["state"]["result"] == "resumed"
        assert result["checkpoint"]["checkpoint_id"] == "cp_test"

        # Prüfe dass handle_workflow_operation aufgerufen wurde
        mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_workflow_not_found(self):
        """Testet Resume eines nicht registrierten Workflows."""
        with patch("core.exceptions.KeikoNotFoundError") as mock_error:
            mock_error_instance = Exception("Not found")
            mock_error.side_effect = mock_error_instance

            with pytest.raises(Exception):
                await self.manager.resume("nonexistent", thread_id="test")

    def test_list_workflows(self):
        """Testet Workflow-Auflistung."""
        # Registriere mehrere Workflows
        self.manager.register("workflow1", MagicMock(), self.mock_checkpointer)
        self.manager.register("workflow2", MagicMock(), self.mock_checkpointer)

        workflows = self.manager.list_workflows()

        assert len(workflows) == 2
        assert "workflow1" in workflows
        assert "workflow2" in workflows
        assert workflows["workflow1"] == "Workflow: workflow1"
        assert workflows["workflow2"] == "Workflow: workflow2"

    def test_list_workflows_empty(self):
        """Testet Workflow-Auflistung bei leerem Manager."""
        workflows = self.manager.list_workflows()
        assert workflows == {}

    def test_is_registered(self):
        """Testet Workflow-Registrierungs-Check."""
        # Vor Registrierung
        assert not self.manager.is_registered("test_workflow")

        # Nach Registrierung
        self.manager.register("test_workflow", self.mock_graph, self.mock_checkpointer)
        assert self.manager.is_registered("test_workflow")

        # Anderer Workflow
        assert not self.manager.is_registered("other_workflow")


class TestCheckpointSaverProtocol:
    """Tests für CheckpointSaver-Protocol."""

    def test_mock_checkpointer_implements_protocol(self):
        """Testet dass MockCheckpointSaver das Protocol implementiert."""
        checkpointer = MockCheckpointSaver()

        # Prüfe dass alle Protocol-Methoden vorhanden sind
        assert hasattr(checkpointer, "get")
        assert hasattr(checkpointer, "put")
        assert callable(checkpointer.get)
        assert callable(checkpointer.put)

    def test_mock_checkpointer_functionality(self):
        """Testet MockCheckpointSaver-Funktionalität."""
        checkpointer = MockCheckpointSaver()

        config = {"configurable": {"thread_id": "test_thread"}}
        value = {"state": "test"}
        metadata = {"meta": "data"}

        # Test put
        result = checkpointer.put(config, value, metadata)
        assert result["thread_id"] == "test_thread"
        assert "checkpoint_id" in result

        # Test get
        retrieved = checkpointer.get(config)
        assert retrieved["state"] == value
        assert retrieved["metadata"] == metadata


class TestIntegrationScenarios:
    """Integration-Tests für komplexere Szenarien."""

    def setup_method(self):
        """Setup für jeden Test."""
        self.manager = LangGraphStateManager()

    @pytest.mark.asyncio
    async def test_full_workflow_lifecycle(self):
        """Testet vollständigen Workflow-Lebenszyklus."""
        # Setup
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock()
        checkpointer = MockCheckpointSaver()

        # Registrierung
        self.manager.register("lifecycle_test", mock_graph, checkpointer)
        assert self.manager.is_registered("lifecycle_test")

        # Start
        state = WorkflowState(message="start", step=1)
        mock_graph.ainvoke.return_value = {"message": "started", "step": 2}

        with patch("agents.state.langgraph_state_manager.handle_workflow_operation") as mock_handler:
            mock_handler.return_value = {
                "state": {"message": "started", "step": 2},
                "checkpoint": {"checkpoint_id": "cp_1"}
            }

            start_result = await self.manager.start("lifecycle_test", state, thread_id="lifecycle")

        assert start_result["state"]["message"] == "started"

        # Resume
        mock_graph.ainvoke.return_value = {"message": "resumed", "step": 3}

        with patch("agents.state.langgraph_state_manager.handle_workflow_operation") as mock_handler:
            mock_handler.return_value = {
                "state": {"message": "resumed", "step": 3},
                "checkpoint": {"checkpoint_id": "cp_2"}
            }

            resume_result = await self.manager.resume("lifecycle_test", thread_id="lifecycle")

        assert resume_result["state"]["message"] == "resumed"
