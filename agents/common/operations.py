"""Framework-spezifische Agent-Operationen."""

import time
from typing import Any

from agents.common.adapters import FRAMEWORKS, get_adapter
from agents.common.circuit_breaker import CircuitBreakerException, get_circuit_breaker
from agents.common.utils import create_error_result, extract_token_usage
from agents.tracing.langsmith_callback_handler import LangSmithAgentCallbackHandler
from kei_logging import get_logger
from monitoring.custom_metrics import MetricsCollector
from observability.tracing import record_exception_in_span, trace_span

try:
    from agents.circuit_breaker.interfaces import (
        AgentCallContext,
        AgentType,
        FailureType,
        IAgentCircuitBreakerService,
    )
    from core.container import get_container
    AGENT_CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    get_container = None
    IAgentCircuitBreakerService = None
    AgentCallContext = None
    AgentType = None
    FailureType = None
    AGENT_CIRCUIT_BREAKER_AVAILABLE = False

logger = get_logger(__name__)





def get_agent_circuit_breaker_service() -> Any | None:
    """Holt Agent Circuit Breaker Service aus DI Container."""
    if not AGENT_CIRCUIT_BREAKER_AVAILABLE:
        return None

    try:
        container = get_container()
        return container.resolve(IAgentCircuitBreakerService)
    except Exception as e:
        logger.debug(f"Could not resolve agent circuit breaker service: {e}")
        return None


def create_agent_call_context(framework: str, agent_id: str, task: str, **kwargs) -> Any:
    """Erstellt Agent Call Context für Circuit Breaker."""
    if not AGENT_CIRCUIT_BREAKER_AVAILABLE:
        return None

    # Bestimme Agent-Type basierend auf Framework
    agent_type = AgentType.CUSTOM_AGENT
    if "voice" in framework.lower():
        agent_type = AgentType.VOICE_AGENT
    elif "tool" in framework.lower():
        agent_type = AgentType.TOOL_AGENT
    elif "workflow" in framework.lower():
        agent_type = AgentType.WORKFLOW_AGENT
    elif "orchestrator" in framework.lower():
        agent_type = AgentType.ORCHESTRATOR_AGENT

    # Extrahiere Voice-Workflow-Kontext aus kwargs
    voice_workflow_id = kwargs.get("voice_workflow_id")
    user_id = kwargs.get("user_id")
    session_id = kwargs.get("session_id")
    request_id = kwargs.get("request_id")

    return AgentCallContext(
        agent_id=agent_id,
        agent_type=agent_type,
        framework=framework,
        task=task,
        voice_workflow_id=voice_workflow_id,
        user_id=user_id,
        session_id=session_id,
        request_id=request_id
    )






async def get_agents(framework: str) -> dict[str, Any]:
    """Lädt Agents für Framework."""
    adapter = await get_adapter(framework)
    if not adapter or not hasattr(adapter, "get_agents"):
        return {}

    try:
        return await adapter.get_agents()
    except Exception as e:
        logger.error(f"Fehler beim Laden von {framework} Agents: {e}")
        return {}


async def create_agent(framework: str, config: dict[str, Any]) -> Any | None:
    """Erstellt Agent im Framework."""
    adapter = await get_adapter(framework)
    if not adapter:
        return None

    try:

        if hasattr(adapter, "create_agent_from_config"):
            return await adapter.create_agent_from_config(config)
        if hasattr(adapter, "create_agent"):
            return await adapter.create_agent(**config)
        return None
    except Exception as e:
        logger.error(f"Fehler beim Erstellen von {framework} Agent: {e}")
        return None


async def execute_agent_task(
    agent_id: str, task: str, framework: str | None = None, **kwargs
) -> dict[str, Any]:
    """Führt Task mit Agent aus.

    Args:
        agent_id: Agent-ID (kann framework_agentid Format haben)
        task: Auszuführender Task
        framework: Optional explizites Framework
        **kwargs: Framework-spezifische Parameter

    Returns:
        Ausführungsergebnis
    """
    if not framework and "_" in agent_id:
        framework, agent_id = agent_id.split("_", 1)

    if not framework:
        return {"error": "Framework nicht ermittelbar", "success": False}

    adapter = await get_adapter(framework)
    if not adapter or not hasattr(adapter, "execute_task"):
        return {"error": f"Adapter {framework} nicht verfügbar", "success": False}


    callback = LangSmithAgentCallbackHandler(agent_id)



    await callback.on_start(task, {"framework": framework})

    try:

        circuit_breaker = get_circuit_breaker(f"{framework}_{agent_id}")

        async def agent_execution():
            _metrics = MetricsCollector()
            start_t = time.time()
            with trace_span(
                "agent.execute", {"framework": framework, "agent_id": agent_id, "kind": "execute_task"}
            ):
                execution_result = await adapter.execute_task(agent_id, task, **kwargs)
            prompt_tokens, completion_tokens = extract_token_usage(execution_result)
            await callback.on_end(execution_result, prompt_tokens, completion_tokens)


            duration = time.time() - start_t
            _metrics.record_histogram(
                "agent.execute.duration_seconds",
                duration,
                tags={"framework": framework, "agent_id": agent_id, "status": "success"},
            )
            return execution_result


        result = await circuit_breaker.call(agent_execution)
        return result
    except CircuitBreakerException as e:

        logger.warning(f"Circuit Breaker blockiert {framework}_{agent_id}: {e}")
        await callback.on_error(e)
        return create_error_result("circuit_breaker_open", circuit_state=e.state.value)
    except Exception as e:
        record_exception_in_span(
            e, attributes={"framework": framework, "agent_id": agent_id, "kind": "execute_task"}
        )
        logger.error(f"Ausführungsfehler {framework}: {e}")
        await callback.on_error(e)
        return create_error_result(str(e))



async def get_foundry_agents() -> dict[str, Any]:
    """Lädt Foundry Agents."""
    return await get_agents("foundry")


async def create_foundry_thread(agent_id: str) -> dict[str, Any]:
    """Erstellt Foundry Thread."""
    adapter = await get_adapter("foundry")
    if not adapter or not hasattr(adapter, "create_thread"):
        return {"error": "Foundry nicht verfügbar"}

    try:
        return await adapter.create_thread(agent_id)
    except Exception as e:
        return {"error": str(e)}


async def execute_foundry_agent(agent_id: str, thread_id: str, message: str) -> dict[str, Any]:
    """Führt Foundry Agent aus."""
    adapter = await get_adapter("foundry")
    if not adapter or not hasattr(adapter, "execute_agent"):
        return {"error": "Foundry nicht verfügbar"}

    try:
        return await adapter.execute_agent(agent_id, thread_id, message)
    except Exception as e:
        return {"error": str(e)}


async def create_thread_message(thread_id: str, content: str, role: str = "user") -> dict[str, Any]:
    """Erstellt Thread-Nachricht."""
    adapter = await get_adapter("foundry")
    if not adapter or not hasattr(adapter, "create_message"):
        return {"error": "Foundry nicht verfügbar"}

    try:
        return await adapter.create_message(thread_id, content, role)
    except Exception as e:
        return {"error": str(e)}



async def get_all_agents() -> dict[str, Any]:
    """Lädt alle Agents aller Frameworks.

    Returns:
        Dict mit allen Agents, Keys im Format framework_agentid
    """
    all_agents = {}

    for framework in FRAMEWORKS:
        agents = await get_agents(framework)
        for agent_id, agent_data in agents.items():
            key = f"{framework}_{agent_id}"
            all_agents[key] = {**agent_data, "framework": framework}

    return all_agents


async def find_best_agent_for_task(
    task: str, frameworks: list[str] | None = None
) -> tuple[str, dict[str, Any]]:
    """Findet besten Agent für Task

    Args:
        task: Aufgabenbeschreibung
        frameworks: Optional Liste der zu prüfenden Frameworks

    Returns:
        Tuple (agent_id, agent_info) oder (None, {}) wenn keiner gefunden
    """
    frameworks = frameworks or FRAMEWORKS
    best_agent = (None, {})
    best_score = 0

    for framework in frameworks:
        agents = await get_agents(framework)

        for agent_id, agent_info in agents.items():
            score = _calculate_task_score(agent_info, task.lower())

            if score > best_score:
                best_score = score
                best_agent = (f"{framework}_{agent_id}", agent_info)

    return best_agent


def _calculate_task_score(agent_info: dict[str, Any], task_lower: str) -> int:
    """Berechnet Matching-Score zwischen Agent und Task"""
    score = 0

    # Prüfe Capabilities
    if "capabilities" in agent_info:
        capabilities = agent_info["capabilities"]
        if isinstance(capabilities, list):
            for cap in capabilities:
                if cap.lower() in task_lower:
                    score += 10

    # Prüfe Name
    if "name" in agent_info:
        name = agent_info["name"]
        if isinstance(name, str):
            if name.lower() in task_lower or task_lower in name.lower():
                score += 5

    # Prüfe Description
    if "description" in agent_info:
        desc = agent_info["description"]
        if isinstance(desc, str):
            if any(word in task_lower for word in desc.lower().split()):
                score += 3

    return score


# Semantic Kernel Operationen (für Test-Kompatibilität)
async def execute_sk_orchestration(_config: dict[str, Any], _instruction: str) -> dict[str, Any]:
    """Führt Semantic Kernel Orchestrierung aus (Test-Platzhalter)."""
    return {"result": "SK orchestration completed", "status": "success"}


async def create_sk_agent(config: dict[str, Any]) -> Any | None:
    """Erstellt Semantic Kernel Agent (Test-Platzhalter)."""
    return {"id": "sk_agent", "type": "foundry", "config": config}
