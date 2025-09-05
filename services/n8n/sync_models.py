"""Modelle für die n8n Workflow-Synchronisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(slots=True)
class SyncContext:
    """Kontext für eine laufende n8n-Execution."""

    execution_id: str
    connection_id: str | None
    agent_id: str | None
    started_at: datetime
    paused: bool = False


__all__ = ["SyncContext"]
