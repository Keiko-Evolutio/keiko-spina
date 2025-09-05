"""Namenskonventionen für KEI-Bus Subjects.

Enthält Hilfsfunktionen zur Bildung stabiler Subject-Namen für Events, RPC, A2A und Tasks.
"""

from __future__ import annotations


def subject_for_event(
    *,
    tenant: str,
    bounded_context: str,
    aggregate: str,
    event: str,
    version: int = 1,
) -> str:
    """Erzeugt Subject nach `kei.{tenant}.{bounded_context}.{aggregate}.{event}.vN`."""
    return f"kei.{tenant}.{bounded_context}.{aggregate}.{event}.v{version}"


def subject_for_rpc(
    *,
    service: str,
    method: str,
    version: int = 1,
    tenant: str | None = None,
) -> str:
    """Erzeugt RPC Subject nach `kei.rpc.{service}.{method}.vN`, optional mit Tenant-Präfix."""
    base = f"kei.rpc.{service}.{method}.v{version}"
    return f"kei.{tenant}.{base}" if tenant else base


def subject_for_a2a(*, to_agent_id: str, version: int = 1, tenant: str | None = None) -> str:
    """Bildet Subject für A2A-Direktnachrichten an einen Ziel-Agenten."""
    base = f"kei.a2a.v{version}"
    if tenant:
        return f"{base}.tenant.{tenant}.agent.{to_agent_id}"
    return f"{base}.agent.{to_agent_id}"


def subject_for_tasks(*, queue: str, version: int = 1, tenant: str | None = None) -> str:
    """Bildet Subject für Task-Queues (Pull Scheduling)."""
    base = f"kei.tasks.v{version}"
    if tenant:
        return f"{base}.tenant.{tenant}.queue.{queue}"
    return f"{base}.queue.{queue}"


__all__ = [
    "subject_for_a2a",
    "subject_for_event",
    "subject_for_rpc",
    "subject_for_tasks",
]
