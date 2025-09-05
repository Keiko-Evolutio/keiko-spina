"""Modelle und Typen für n8n-Interaktionen."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ExecutionStatus(str, Enum):
    """Ausführungsstatus für n8n Executions."""

    running = "running"
    success = "success"
    error = "error"
    unknown = "unknown"


@dataclass(slots=True)
class TriggerResult:
    """Ergebnis eines Workflow-Triggerings."""

    execution_id: str | None
    started: bool
    raw: dict[str, Any]


@dataclass(slots=True)
class ExecutionResult:
    """Ausführungsresultat/Status einer n8n-Execution."""

    status: ExecutionStatus
    finished: bool
    raw: dict[str, Any]


__all__ = ["ExecutionResult", "ExecutionStatus", "TriggerResult"]
