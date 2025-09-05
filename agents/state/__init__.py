"""State-Management-Module für LangGraph-Workflows.

Stellt typisierte State-Management-Funktionalitäten bereit:
- Serialisierung und Persistierung
- Einheitliches Exception-Handling
- Konsolidierte Reducer-Funktionen
"""

from __future__ import annotations

from .langgraph_state_bridge import (
    WorkflowState,
    max_step,
    replace_bool,
    replace_dict,
    replace_int,
    replace_message,
)
from .langgraph_state_manager import (
    CheckpointSaver,
    LangGraphStateManager,
    ManagedWorkflow,
)
from .state_constants import (
    DEFAULT_EMPTY_MESSAGE,
    DEFAULT_STEP_VALUE,
    DEFAULT_THREAD_ID,
    INVALID_CONFIG_ERROR,
    INVALID_STATE_ERROR,
    MAX_MESSAGE_LENGTH,
    MAX_THREAD_ID_LENGTH,
    MAX_WORKFLOW_NAME_LENGTH,
    TRACE_STATE_DESERIALIZATION,
    TRACE_STATE_SERIALIZATION,
    TRACE_WORKFLOW_RESUME,
    TRACE_WORKFLOW_START,
    WORKFLOW_NOT_FOUND_ERROR,
    WORKFLOW_RESUME_ERROR,
    WORKFLOW_START_ERROR,
)
from .state_utils import (
    SerializableState,
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

__all__ = [
    # Constants
    "DEFAULT_EMPTY_MESSAGE",
    "DEFAULT_STEP_VALUE",
    "DEFAULT_THREAD_ID",
    "INVALID_CONFIG_ERROR",
    "INVALID_STATE_ERROR",
    "MAX_MESSAGE_LENGTH",
    "MAX_THREAD_ID_LENGTH",
    "MAX_WORKFLOW_NAME_LENGTH",
    "TRACE_WORKFLOW_RESUME",
    "TRACE_WORKFLOW_START",
    "WORKFLOW_NOT_FOUND_ERROR",
    "WORKFLOW_RESUME_ERROR",
    "WORKFLOW_START_ERROR",
    # Protocols
    "CheckpointSaver",
    "LangGraphStateManager",
    "ManagedWorkflow",
    "SerializableState",
    "SerializationMixin",
    # Core State-Management
    "WorkflowState",
    "create_max_reducer",
    "create_optional_replace_reducer",
    "create_replace_reducer",
    "deserialize_state_safely",
    "handle_workflow_operation",
    "max_step",
    "replace_bool",
    "replace_dict",
    "replace_int",
    "replace_message",
    "serialize_state_safely",
    "validate_workflow_config",
    "validate_workflow_name",
]


