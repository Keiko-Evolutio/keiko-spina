"""Bidirektionale State-Synchronisation zwischen Keiko Agent-State und n8n Workflow-Variablen.

Diese Implementierung bietet:
- Echtzeit-Synchronisation über periodisches Polling (best effort)
- Konfliktauflösung mittels versionsbasiertem Merge (Last-Write-Wins mit Policy)
- Korrelation zwischen LangGraph-Checkpoints (thread_id) und n8n Executions (execution_id)
- OpenTelemetry-Tracing-Integration
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from kei_logging import get_logger
from observability import add_span_attributes, record_exception_in_span, trace_function, trace_span

from .n8n_client import DEFAULT_POLL_INTERVAL, N8nClient
from .workflow_poller import run_polling_loop

if TYPE_CHECKING:
    from collections.abc import Callable

try:  # pragma: no cover - optional Cosmos
    from azure.cosmos.aio import ContainerProxy
    COSMOS_AVAILABLE = True
except Exception:  # pragma: no cover - tests/runtime ohne cosmos
    # Fallback-Typ für bessere IDE-Unterstützung
    class ContainerProxy:  # type: ignore
        async def upsert_item(self, body: dict[str, Any], partition_key: str, **kwargs: Any) -> Any: ...
        async def read_item(self, item: str, partition_key: str, **kwargs: Any) -> Any: ...
        async def query_items(self, query: str, **kwargs: Any) -> Any: ...
        async def delete_item(self, item: str, partition_key: str, **kwargs: Any) -> None: ...

    COSMOS_AVAILABLE = False

try:  # pragma: no cover - optional storage helper
    from storage.cache.redis_cache import get_cached_cosmos_container
except Exception:  # pragma: no cover
    get_cached_cosmos_container = None  # type: ignore


logger = get_logger(__name__)


@dataclass(slots=True)
class VersionedState:
    """Versionierter State-Snapshot mit Quelle für Konfliktauflösung.

    Attributes:
        version: Monoton steigende Version
        source: Quelle des letzten Updates ("agent" | "n8n")
        data: Zustandsdaten
        updated_at: Zeitpunkt der letzten Aktualisierung in UTC
    """

    version: int
    source: Literal["agent", "n8n"]
    data: dict[str, Any]
    updated_at: datetime


class ConflictResolutionPolicy:
    """Strategien für Konfliktauflösung."""

    LAST_WRITE_WINS = "last_write_wins"
    PREFER_AGENT = "prefer_agent"
    PREFER_N8N = "prefer_n8n"


class N8nStateSynchronizer:
    """Synchronisiert Agent-State mit n8n-Workflow-Variablen.

    Diese Klasse kapselt Polling und Push/Pull-Mechanismen sowie die Korrelation
    von LangGraph-Checkpoints mit n8n-Executions.

    Hinweis: n8n bietet keine offizielle API zum direkten Mutieren laufender
    Execution-Variablen. Diese Implementierung nutzt deshalb folgenden Ansatz:
    - Pull: Periodisches Lesen des Execution-Status (`/rest/executions/{id}`)
            und Extraktion von Variablen aus `data`/`raw` (best effort)
    - Push: Optionaler Webhook-Fallback (`/webhook/{path}`) für state_update,
            falls ein entsprechender Workflow-Hook konfiguriert ist
    """

    def __init__(
        self,
        *,
        state_extractor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        state_update_webhook_path: str | None = None,
        conflict_policy: str = ConflictResolutionPolicy.LAST_WRITE_WINS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        """Initialisiert den Synchronizer.

        Args:
            state_extractor: Funktion, die aus n8n-Execution-RAW einen Variablen-Dict extrahiert
            state_update_webhook_path: Optionaler Webhook-Pfad, der State-Updates von Agent an n8n sendet
            conflict_policy: Konfliktauflösungsstrategie
            poll_interval_seconds: Intervall für Polling
        """
        self._state_extractor = state_extractor or self._default_state_extractor
        self._webhook_path = state_update_webhook_path
        self._conflict_policy = conflict_policy
        self._poll_interval_seconds = poll_interval_seconds

        self._current: VersionedState | None = None
        self._sync_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Default Extractor
    # ------------------------------------------------------------------
    @staticmethod
    def _default_state_extractor(raw: dict[str, Any]) -> dict[str, Any]:
        """Extrahiert Variablen aus n8n RAW-Struktur.

        Erwartet ein Dictionary mit optionalen Schlüsseln wie `data`, `variables` oder
        `context`. Fällt auf leeres Dict zurück, wenn nichts extrahierbar ist.
        """
        try:
            if not isinstance(raw, dict):
                return {}
            if "variables" in raw and isinstance(raw["variables"], dict):
                return dict(raw["variables"])
            if "data" in raw and isinstance(raw["data"], dict):
                return dict(raw["data"])
            if "context" in raw and isinstance(raw["context"], dict):
                return dict(raw["context"])
            # Fallback: Leeres Dictionary zurückgeben
            return {}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @trace_function("n8n.sync.start")
    async def start(
        self,
        *,
        execution_id: str,
        thread_id: str | None,
        initial_agent_state: dict[str, Any],
    ) -> None:
        """Startet die bidirektionale Synchronisation.

        - Initialisiert lokalen VersionedState
        - Startet Polling-Loop gegen n8n (best effort)
        - Korreliert Checkpoint/Thread mit Execution in Cosmos (falls verfügbar)
        """
        self._current = VersionedState(
            version=1,
            source="agent",
            data=dict(initial_agent_state),
            updated_at=datetime.now(UTC),
        )

        add_span_attributes({
            "n8n.execution_id": execution_id,
            "langgraph.thread_id": thread_id or "",
            "sync.conflict_policy": self._conflict_policy,
        })

        # Fire-and-forget Polling starten
        self._stop_event.clear()
        self._sync_task = asyncio.create_task(self._run_sync_loop(execution_id))

        # Korrelation schreiben (best effort)
        if thread_id:
            await self._persist_correlation(thread_id=thread_id, execution_id=execution_id)

    @trace_function("n8n.sync.stop")
    async def stop(self) -> None:
        """Beendet die Synchronisation und wartet auf Task-Abschluss."""
        self._stop_event.set()
        if self._sync_task is not None:
            try:
                await self._sync_task
            except Exception as exc:
                logger.debug(f"Sync-Task Stop mit Fehler: {exc}")
            finally:
                self._sync_task = None

    @trace_function("n8n.sync.push_agent_state")
    async def push_agent_state(self, state: dict[str, Any]) -> None:
        """Aktualisiert lokalen State (Quelle Agent) und pusht optional an n8n.

        Hinweis: Für das Pushen wird ein optionaler Webhook verwendet.
        """
        self._apply_local_update(source="agent", new_data=state)
        await self._push_to_n8n_if_configured(state)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    async def _run_sync_loop(self, execution_id: str) -> None:
        """Interner Loop: pollt n8n und merged State bis Stop-Event gesetzt ist."""

        async def handle(res: dict[str, Any]) -> bool:
            try:
                with trace_span("n8n.sync.handle_poll", {"execution_id": execution_id}):
                    variables = self._state_extractor(res.get("raw", {}))
                    if variables:
                        self._apply_local_update(source="n8n", new_data=variables)
            except Exception as exc:
                record_exception_in_span(exc)
                logger.debug(f"Fehler beim Poll-Handle: {exc}")
            return self._stop_event.is_set() or bool(res.get("finished"))

        try:
            await run_polling_loop(execution_id, handle_result=handle, sleep_seconds=self._poll_interval_seconds)
        except Exception as exc:
            record_exception_in_span(exc)
            logger.warning(f"Sync-Polling Fehlgeschlagen: {exc}")

    def _apply_local_update(self, *, source: Literal["agent", "n8n"], new_data: dict[str, Any]) -> None:
        """Wendet lokale Aktualisierung mit Konfliktauflösung an."""
        now = datetime.now(UTC)
        if self._current is None:
            self._current = VersionedState(version=1, source=source, data=dict(new_data), updated_at=now)
            return

        merged = self._merge_states(self._current, VersionedState(
            version=self._current.version + 1,
            source=source,
            data=dict(new_data),
            updated_at=now,
        ))
        self._current = merged

    def _merge_states(self, left: VersionedState, right: VersionedState) -> VersionedState:
        """Führt zwei VersionedStates zusammen unter Berücksichtigung der Policy."""
        # Gleiche Version -> Policy entscheidet
        if right.version == left.version:
            return self._resolve_conflict(left, right)

        # Höhere Version gewinnt
        return right if right.version > left.version else left

    def _resolve_conflict(self, a: VersionedState, b: VersionedState) -> VersionedState:
        """Konfliktauflösung zwischen zwei gleichwertigen Versionen."""
        policy = self._conflict_policy
        if policy == ConflictResolutionPolicy.PREFER_AGENT:
            return a if a.source == "agent" else b
        if policy == ConflictResolutionPolicy.PREFER_N8N:
            return a if a.source == "n8n" else b

        # Last-Write-Wins anhand updated_at
        if a.updated_at == b.updated_at:
            # Stabil: Bevorzuge Agent, falls exakt gleich
            return a if a.source == "agent" else b
        return a if a.updated_at > b.updated_at else b

    async def _push_to_n8n_if_configured(self, state: dict[str, Any]) -> None:
        """Push an n8n über optionalen Webhook (best effort)."""
        if not self._webhook_path:
            return
        try:
            client = N8nClient()
            url_path = self._webhook_path.rstrip("/") + "/state_update"
            with trace_span("n8n.sync.push_webhook", {"webhook_path": url_path}):
                await client._request_with_retry(  # noqa: SLF001 - bewusste Nutzung der internen Methode
                    "POST",
                    f"{client.base_url}/webhook/{url_path.lstrip('/')}",
                    headers=client._build_headers(),  # noqa: SLF001
                    json={"state": state},
                )
        except Exception as exc:
            # Graceful Degradation – nur loggen
            record_exception_in_span(exc)
            logger.debug(f"n8n Webhook Push fehlgeschlagen: {exc}")

    async def _persist_correlation(self, *, thread_id: str, execution_id: str) -> None:
        """Persistiert Korrelation zwischen LangGraph Thread und n8n Execution in Cosmos (best effort)."""
        if get_cached_cosmos_container is None:  # pragma: no cover
            return
        try:
            async with get_cached_cosmos_container() as cached:  # type: ignore[operator]
                if not cached:
                    return
                container: ContainerProxy = cached.container  # type: ignore[attr-defined]
                item = {
                    "id": f"corr-{thread_id}-{execution_id}",
                    "category": "n8n_langgraph_correlation",
                    "thread_id": thread_id,
                    "execution_id": execution_id,
                    "created_at": datetime.now(UTC).isoformat(),
                }
                with trace_span("n8n.sync.persist_correlation", {"thread_id": thread_id, "execution_id": execution_id}):
                    await container.upsert_item(body=item, partition_key="n8n_langgraph_correlation")
        except Exception as exc:  # pragma: no cover - best effort, Cosmos optional
            logger.debug(f"Korrelation Persist fehlgeschlagen: {exc}")


__all__ = [
    "ConflictResolutionPolicy",
    "N8nStateSynchronizer",
    "VersionedState",
]
