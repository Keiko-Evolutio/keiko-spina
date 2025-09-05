"""Agent-Ausführungsmodul für Orchestrator-Integration."""

from collections.abc import Awaitable, Callable
from typing import Any

from agents.common.operations import execute_foundry_agent as _execute_foundry
from agents.common.utils import extract_instruction_from_kwargs, extract_token_usage
from agents.tracing.langsmith_callback_handler import LangSmithAgentCallbackHandler
from data_models import AgentUpdateEvent, Function
from kei_logging import get_logger
from observability.tracing import record_exception_in_span, trace_span

logger = get_logger(__name__)








async def execute_agent(
    framework: str,
    agent_id: str,
    notify: Callable[[str, str, str], Awaitable[None]] | AgentUpdateEvent | None = None,
    **kwargs,
) -> Any:
    """Einheitliche Agent-Ausführung für alle Frameworks."""
    callback = LangSmithAgentCallbackHandler(agent_id)
    instruction = extract_instruction_from_kwargs(kwargs)

    await callback.on_start(instruction, {"framework": framework})

    try:
        with trace_span("agent.execute", {"framework": framework, "agent_id": agent_id}):

            if framework == "foundry":
                result = await _execute_foundry_framework(agent_id, kwargs)
            elif framework == "autogen":
                result = await _execute_autogen_framework(kwargs)
            elif framework == "semantic_kernel":
                result = await _execute_sk_framework(kwargs)
            else:
                from core.exceptions import KeikoValidationError

                raise KeikoValidationError(
                    "Unbekanntes Framework", details={"framework": str(framework)}
                )


        if notify and callable(notify):
            try:
                await notify(agent_id, "success", f"{framework} Ausführung abgeschlossen")
            except Exception as notify_error:
                logger.warning(f"Benachrichtigungsfehler: {notify_error}")

        prompt_tokens, completion_tokens = extract_token_usage(result)
        await callback.on_end(result, prompt_tokens, completion_tokens)
        return result

    except Exception as e:
        record_exception_in_span(e, attributes={"framework": framework, "agent_id": agent_id})
        logger.error(f"{framework} Agent {agent_id} Fehler: {e}")


        if notify and callable(notify):
            try:
                await notify(agent_id, "error", str(e))
            except Exception as notify_error:
                logger.warning(f"Benachrichtigungsfehler: {notify_error}")
        try:
            await callback.on_error(e)
        finally:
            pass
        raise


async def _execute_foundry_framework(agent_id: str, kwargs: dict[str, Any]) -> Any:
    """Foundry-spezifische Ausführung."""
    thread_id = kwargs.get("thread_id", "")
    message = kwargs.get("query", "")

    return await _execute_foundry(agent_id, thread_id, message)


async def _execute_autogen_framework(kwargs: dict[str, Any]) -> Any:
    """AutoGen-spezifische Ausführung"""
    task = kwargs.get("task", "")
    context = kwargs.get("context", {})

    # Tests patchen execute_autogen_workflow in agents.common.execution
    try:
        from agents.common.execution import execute_autogen_workflow as _exec  # type: ignore
    except Exception:
        # Fallback auf Agents-Implementierung, falls Alias nicht verfügbar
        from agents.common.execution import execute_autogen_workflow as _exec  # type: ignore
    config = {"task": task, "context": context}
    return await _exec(config, task)


async def _execute_sk_framework(kwargs: dict[str, Any]) -> Any:
    """Semantic Kernel-spezifische Ausführung"""
    instruction = kwargs.get("instruction", "")
    parameters = kwargs.get("parameters", {})

    try:
        from agents.common.execution import execute_sk_orchestration as _exec  # type: ignore
    except Exception:
        from agents.common.execution import execute_sk_orchestration as _exec  # type: ignore
    config = {"instruction": instruction, "parameters": parameters}
    return await _exec(config, instruction)


# Platzhalter für Tests: werden durch patch() überschrieben
async def execute_autogen_workflow(_config: dict[str, Any], _task: str) -> dict[str, Any]:
    """Führt einen AutoGen-Workflow aus (Test-Platzhalter)."""
    # Nur Dummy-Antwort; Tests patchen diese Funktion
    return {"result": "AutoGen workflow completed", "status": "success"}


async def execute_sk_orchestration(_config: dict[str, Any], _instruction: str) -> dict[str, Any]:
    """Führt eine Semantic Kernel Orchestrierung aus (Test-Platzhalter)."""
    return {"result": "SK orchestration completed", "status": "success"}





# Kompatibilitäts-Wrapper für direkte Framework-Aufrufe
async def execute_foundry_agent(
    agent_id: str,
    query: str,
    *,
    additional_instructions: str = "",
    tools: dict[str, Function] = None,
    notify: AgentUpdateEvent = None,
) -> None:
    """Foundry Agent-Ausführung (Kompatibilität)"""
    return await execute_agent(
        "foundry",
        agent_id,
        notify,
        query=query,
        additional_instructions=additional_instructions,
        tools=tools or {},
    )


async def execute_agent_by_framework(framework: str, agent_id: str, **kwargs) -> Any:
    """Framework-agnostische Agent-Ausführung"""
    return await execute_agent(framework, agent_id, **kwargs)
