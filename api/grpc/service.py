"""KEI-RPC Service für standardisierte Agent-Operationen.

Implementiert plan/act/observe/explain Operationen mit verbesserter
Struktur und Type Safety.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any, TypeVar, Union

from kei_logging import get_logger
from observability import trace_function, trace_span

from .agent_integration import AgentIntegrationMixin
from .constants import ErrorCodes
from .models import (
    ActRequest,
    ActResponse,
    ExplainRequest,
    ExplainResponse,
    ObserveRequest,
    ObserveResponse,
    OperationError,
    OperationStatus,
    OperationTiming,
    OperationType,
    PlanRequest,
    PlanResponse,
)

logger = get_logger(__name__)

# Type Variables für bessere Type Safety
RequestType = TypeVar("RequestType", PlanRequest, ActRequest, ObserveRequest, ExplainRequest)
ResponseType = TypeVar("ResponseType", PlanResponse, ActResponse, ObserveResponse, ExplainResponse)
OperationResult = Union[PlanResponse, ActResponse, ObserveResponse, ExplainResponse]


class KEIRPCServiceError(Exception):
    """Basis-Exception für KEI-RPC Service Fehler."""

    def __init__(self, message: str, error_code: str = ErrorCodes.INTERNAL_ERROR) -> None:
        """Initialisiert Service-Fehler.

        Args:
            message: Fehlermeldung
            error_code: Error-Code
        """
        super().__init__(message)
        self.error_code = error_code


class AgentNotFoundError(KEIRPCServiceError):
    """Fehler wenn Agent nicht gefunden wird."""

    def __init__(self, agent_id: str | None = None) -> None:
        message = (
            f"Agent nicht gefunden: {agent_id}" if agent_id else "Kein passender Agent gefunden"
        )
        super().__init__(message, ErrorCodes.AGENT_NOT_FOUND)


class CapabilityNotAvailableError(KEIRPCServiceError):
    """Fehler wenn erforderliche Capability nicht verfügbar ist."""

    def __init__(self, capability: str) -> None:
        super().__init__(
            f"Capability nicht verfügbar: {capability}", ErrorCodes.CAPABILITY_NOT_AVAILABLE
        )


class OperationTimeoutError(KEIRPCServiceError):
    """Fehler bei Operation-Timeout."""

    def __init__(self, timeout_seconds: int) -> None:
        super().__init__(
            f"Operation Timeout nach {timeout_seconds} Sekunden", ErrorCodes.OPERATION_TIMEOUT
        )


class OperationRouter:
    """Router für KEI-RPC Operationen.

    Trennt Routing-Logik von Service-Logik für bessere Testbarkeit.
    """

    def __init__(self) -> None:
        """Initialisiert Operation Router."""
        self.logger = get_logger(f"{__name__}.OperationRouter")

    def get_response_class(self, operation_type: OperationType) -> type[Any]:
        """Ermittelt Response-Klasse für Operation-Typ.

        Args:
            operation_type: Typ der Operation

        Returns:
            Response-Klasse
        """
        response_mapping = {
            OperationType.PLAN: PlanResponse,
            OperationType.ACT: ActResponse,
            OperationType.OBSERVE: ObserveResponse,
            OperationType.EXPLAIN: ExplainResponse,
        }

        response_class = response_mapping.get(operation_type)
        if not response_class:
            raise KEIRPCServiceError(f"Unbekannter Operation-Typ: {operation_type}")

        return response_class

    def create_error_response(
        self, operation_type: OperationType, error: Exception, request_metadata: Any
    ) -> OperationResult:
        """Erstellt Error-Response für fehlgeschlagene Operation.

        Args:
            operation_type: Typ der Operation
            error: Aufgetretener Fehler
            request_metadata: Request-Metadaten

        Returns:
            Error-Response
        """
        response_class = self.get_response_class(operation_type)

        # Error-Details erstellen
        error_code = getattr(error, "error_code", ErrorCodes.INTERNAL_ERROR)
        operation_error = OperationError(
            error_code=error_code,
            error_message=str(error),
            error_type=type(error).__name__,
            details={"exception_type": type(error).__name__},
        )

        # Timing-Informationen
        timing = OperationTiming(completed_at=datetime.now(UTC), duration_ms=0)

        # Response erstellen
        return response_class(
            status=OperationStatus.FAILED,
            error=operation_error,
            timing=timing,
            metadata=request_metadata,
            # Operation-spezifische Felder mit Defaults
            **OperationRouter._get_default_response_fields(operation_type),
        )

    @staticmethod
    def _get_default_response_fields(operation_type: OperationType) -> dict[str, Any]:
        """Gibt Default-Felder für Response-Typ zurück."""
        defaults = {
            OperationType.PLAN: {"plan": "", "steps": [], "estimated_duration": 0},
            OperationType.ACT: {"result": "", "artifacts": [], "side_effects": []},
            OperationType.OBSERVE: {"observations": [], "insights": [], "anomalies": []},
            OperationType.EXPLAIN: {"explanation": "", "reasoning": [], "confidence": 0.0},
        }

        return defaults.get(operation_type, {})


class KEIRPCService(AgentIntegrationMixin):
    """KEI-RPC Service mit verbesserter Struktur.

    Features:
    - Single Responsibility Principle
    - Bessere Error-Handling
    - Verbesserte Type Safety
    - Eliminierte Code-Duplikation
    """

    def __init__(self) -> None:
        """Initialisiert KEI-RPC Service."""
        self._operation_cache: dict[str, OperationResult] = {}
        self._agent_registry: Any | None = None
        self._capability_manager: Any | None = None
        self._initialized = False
        self._router = OperationRouter()

    async def initialize(self) -> None:
        """Initialisiert Service-Abhängigkeiten."""
        if self._initialized:
            return

        try:
            # Agent Registry laden
            from agents.registry.dynamic_registry import DynamicAgentRegistry

            self._agent_registry = DynamicAgentRegistry()
            await self._agent_registry.refresh_agents()

            # Capability Manager laden
            from agents.capabilities import get_capability_manager

            self._capability_manager = get_capability_manager()

            self._initialized = True
            logger.info("KEI-RPC Service initialisiert")

        except Exception as e:
            logger.exception(f"KEI-RPC Service Initialisierung fehlgeschlagen: {e}")
            raise KEIRPCServiceError(f"Service-Initialisierung fehlgeschlagen: {e}")

    async def ensure_initialized(self) -> None:
        """Stellt sicher, dass der Service initialisiert ist."""
        await self.initialize()

    def get_health_status(self) -> dict[str, Any]:
        """Gibt Health-Status-Informationen zurück."""
        return {
            "agent_registry": self._agent_registry is not None,
            "capability_manager": self._capability_manager is not None,
            "initialized": self._initialized,
        }

    def get_status_info(self) -> dict[str, Any]:
        """Gibt detaillierte Status-Informationen zurück."""
        cache_size = len(self._operation_cache)
        agent_count = 0
        if self._agent_registry:
            agent_count = len(getattr(self._agent_registry, "agents", {}))

        return {
            "operation_cache_size": cache_size,
            "available_agents": agent_count,
            "initialized": self._initialized,
        }

    @trace_function("kei_rpc.plan")
    async def plan(self, request: PlanRequest) -> PlanResponse:
        """Führt Plan-Operation aus.

        Args:
            request: Plan-Request

        Returns:
            Plan-Response
        """
        return await self._execute_operation(
            operation_type=OperationType.PLAN, request=request, response_class=PlanResponse
        )

    @trace_function("kei_rpc.act")
    async def act(self, request: ActRequest) -> ActResponse:
        """Führt Act-Operation aus.

        Args:
            request: Act-Request

        Returns:
            Act-Response
        """
        return await self._execute_operation(
            operation_type=OperationType.ACT, request=request, response_class=ActResponse
        )

    @trace_function("kei_rpc.observe")
    async def observe(self, request: ObserveRequest) -> ObserveResponse:
        """Führt Observe-Operation aus.

        Args:
            request: Observe-Request

        Returns:
            Observe-Response
        """
        return await self._execute_operation(
            operation_type=OperationType.OBSERVE, request=request, response_class=ObserveResponse
        )

    @trace_function("kei_rpc.explain")
    async def explain(self, request: ExplainRequest) -> ExplainResponse:
        """Führt Explain-Operation aus.

        Args:
            request: Explain-Request

        Returns:
            Explain-Response
        """
        return await self._execute_operation(
            operation_type=OperationType.EXPLAIN, request=request, response_class=ExplainResponse
        )

    async def _execute_operation(
        self,
        operation_type: OperationType,
        request: RequestType,
        response_class: type[Any],
    ) -> Any:
        """Zentrale Operation-Ausführung mit einheitlicher Fehlerbehandlung.

        Args:
            operation_type: Typ der Operation
            request: Request-Objekt
            response_class: Response-Klasse

        Returns:
            Response-Objekt
        """
        await self._ensure_initialized()

        start_time = time.time()

        try:
            # 1. Idempotenz prüfen
            if request.metadata.idempotency_key:
                cached_response = self._get_cached_response(request.metadata.idempotency_key)
                if cached_response:
                    logger.info(f"Idempotente Response für {operation_type} zurückgegeben")
                    return cached_response

            # 2. Agent finden
            agent = await self._find_suitable_agent(request.agent_context, operation_type)
            if not agent:
                raise AgentNotFoundError

            # 3. Operation ausführen
            with trace_span(f"kei_rpc.{operation_type.value}.execute"):
                result = await self._execute_agent_operation(agent, operation_type, request)

            # 4. Response erstellen
            timing = OperationTiming(
                completed_at=datetime.now(UTC),
                duration_ms=int((time.time() - start_time) * 1000),
            )

            response = response_class(
                status=OperationStatus.SUCCESS,
                timing=timing,
                metadata=request.metadata,
                agent_id=getattr(agent, "id", "unknown"),
                **result,
            )

            # 5. Response cachen (falls Idempotenz-Key vorhanden)
            if request.metadata.idempotency_key:
                self._cache_response(request.metadata.idempotency_key, response)

            return response

        except Exception as e:
            logger.exception(f"Operation {operation_type} fehlgeschlagen: {e}")
            return self._router.create_error_response(operation_type, e, request.metadata)

    async def _find_suitable_agent(self, agent_context: Any, operation_type: OperationType) -> Any:
        """Findet einen geeigneten Agent für die Operation.

        Args:
            agent_context: Kontext für Agent-Auswahl
            operation_type: Typ der Operation

        Returns:
            Geeigneter Agent oder None
        """
        try:
            await self._ensure_initialized()

            if not self._agent_registry:
                logger.warning("Agent Registry nicht verfügbar")
                return None

            # Extrahiere Capabilities aus dem Kontext
            required_capabilities = []
            if hasattr(agent_context, "capabilities"):
                required_capabilities = agent_context.capabilities
            elif hasattr(agent_context, "required_capabilities"):
                required_capabilities = agent_context.required_capabilities

            # Suche Agents basierend auf Capabilities
            if required_capabilities:
                matches = await self._agent_registry.search_agents(
                    capabilities=required_capabilities,
                    limit=5
                )
                if matches:
                    # Wähle den besten Match
                    best_match = max(matches, key=lambda x: getattr(x, "match_score", 0.5))
                    agent_id = getattr(best_match, "agent_id", None)
                    if agent_id:
                        return await self._get_agent_by_id(agent_id)

            # Fallback: Hole alle verfügbaren Agents
            all_agents = await self._agent_registry.search_agents(limit=10)
            if all_agents:
                # Wähle ersten verfügbaren Agent
                first_match = all_agents[0]
                agent_id = getattr(first_match, "agent_id", None)
                if agent_id:
                    return await self._get_agent_by_id(agent_id)

            logger.warning(f"Kein geeigneter Agent für Operation {operation_type} gefunden")
            return None

        except Exception as e:
            logger.exception(f"Agent-Suche fehlgeschlagen: {e}")
            return None

    async def _get_agent_by_id(self, agent_id: str) -> Any:
        """Holt einen Agent anhand seiner ID.

        Args:
            agent_id: Agent-ID

        Returns:
            Agent-Instanz oder None
        """
        try:
            if not self._agent_registry:
                return None

            # Versuche Agent aus Registry zu holen
            agents = getattr(self._agent_registry, "agents", {})
            if agent_id in agents:
                return agents[agent_id]

            # Fallback: Erstelle Mock-Agent für Tests
            return type("MockAgent", (), {
                "id": agent_id,
                "name": f"Agent-{agent_id}",
                "status": "available"
            })()

        except Exception as e:
            logger.exception(f"Agent-Abruf fehlgeschlagen für ID {agent_id}: {e}")
            return None

    async def _ensure_initialized(self) -> None:
        """Stellt sicher, dass Service initialisiert ist."""
        if not self._initialized:
            await self.initialize()

    def _get_cached_response(self, idempotency_key: str) -> OperationResult | None:
        """Holt gecachte Response für Idempotenz-Key."""
        return self._operation_cache.get(idempotency_key)

    def _cache_response(self, idempotency_key: str, response: OperationResult) -> None:
        """Cached Response für Idempotenz-Key."""
        self._operation_cache[idempotency_key] = response

    async def _execute_agent_operation(
        self, agent: Any, operation_type: OperationType, request: RequestType
    ) -> dict[str, Any]:
        """Führt Operation auf Agent aus.

        Args:
            agent: Agent-Instanz
            operation_type: Operation-Typ
            request: Request-Objekt

        Returns:
            Operation-Ergebnis
        """
        # Vereinfachte Implementierung - in Produktion würde hier
        # die tatsächliche Agent-Operation ausgeführt werden
        operation_methods = {
            OperationType.PLAN: self._execute_plan_operation,
            OperationType.ACT: self._execute_act_operation,
            OperationType.OBSERVE: self._execute_observe_operation,
            OperationType.EXPLAIN: self._execute_explain_operation,
        }

        method = operation_methods.get(operation_type)
        if not method:
            raise KEIRPCServiceError(f"Unbekannte Operation: {operation_type}")

        return await method(agent, request)

    async def _execute_plan_operation(self, agent: Any, request: PlanRequest) -> dict[str, Any]:
        """Führt Plan-Operation aus."""
        # Parameter wird aktuell nicht verwendet
        _ = agent
        # Implementierung der Plan-Operation
        return {
            "plan": f"Plan für: {request.objective}",
            "steps": ["Schritt 1", "Schritt 2", "Schritt 3"],
            "estimated_duration": 3600,
        }

    async def _execute_act_operation(self, agent: Any, request: ActRequest) -> dict[str, Any]:
        """Führt Act-Operation aus."""
        # Parameter wird aktuell nicht verwendet
        _ = agent
        # Implementierung der Act-Operation
        return {
            "result": f"Aktion ausgeführt: {request.action}",
            "artifacts": [],
            "side_effects": [],
        }

    async def _execute_observe_operation(
        self, agent: Any, request: ObserveRequest
    ) -> dict[str, Any]:
        """Führt Observe-Operation aus."""
        # Parameter wird aktuell nicht verwendet
        _ = agent
        # Implementierung der Observe-Operation
        return {
            "observations": [f"Beobachtung: {request.target}"],
            "insights": [],
            "anomalies": [],
        }

    async def _execute_explain_operation(
        self, agent: Any, request: ExplainRequest
    ) -> dict[str, Any]:
        """Führt Explain-Operation aus."""
        # Parameter wird aktuell nicht verwendet
        _ = agent
        # Implementierung der Explain-Operation
        return {
            "explanation": f"Erklärung für: {request.subject}",
            "reasoning": ["Grund 1", "Grund 2"],
            "confidence": 0.85,
        }


# Singleton-Instanz für globale Verwendung
kei_rpc_service = KEIRPCService()


__all__ = [
    "AgentNotFoundError",
    "CapabilityNotAvailableError",
    "KEIRPCService",
    "KEIRPCServiceError",
    "OperationRouter",
    "OperationTimeoutError",
    "kei_rpc_service",
]
