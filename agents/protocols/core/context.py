"""Protocol Context Management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from kei_logging import get_logger

from .dataclasses import AgentExecutionContext

logger = get_logger(__name__)


@asynccontextmanager
async def agent_execution_session(
    agent_protocol: Any,  # BaseAgentProtocol
    agent: Any,
    services: Any,
    mcp_config: dict[str, Any] | None = None,
) -> AsyncIterator[AgentExecutionContext]:
    """Context Manager für sichere Agent-Execution mit automatischem Cleanup.

    Args:
        agent_protocol: Agent Protocol-Implementierung
        agent: Agent-Instanz
        services: Service-Container
        mcp_config: MCP-Konfiguration

    Yields:
        AgentExecutionContext: Execution-Context

    Raises:
        Exception: Bei Initialisierungs- oder Cleanup-Fehlern
    """
    context = None
    try:
        context = await agent_protocol.initialize(agent, services, mcp_config)
        if context:
            logger.debug(f"Agent execution session gestartet: {context.session_id}")
        yield context
    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        raise
    finally:
        if context:
            try:
                await agent_protocol.terminate(context)
                logger.debug(f"Agent execution session beendet: {context.session_id}")
            except Exception as cleanup_error:
                logger.error(f"Cleanup error: {cleanup_error}")


def create_agent_context(
    session_id: str,
    agent_id: str,
    **kwargs: Any
) -> AgentExecutionContext:
    """Factory-Function für AgentExecutionContext.

    Args:
        session_id: Session-Identifier
        agent_id: Agent-Identifier
        **kwargs: Zusätzliche Context-Parameter

    Returns:
        AgentExecutionContext: Neuer Execution-Context

    Example:
        >>> context = create_agent_context("sess-123", "agent-456", user_id="user-789")
        >>> assert context.session_id == "sess-123"
        >>> assert context.agent_id == "agent-456"
        >>> assert context.metadata.get("user_id") == "user-789"
    """
    # Extrahiere bekannte Parameter
    thread_id = kwargs.pop("thread_id", None)
    user_id = kwargs.pop("user_id", None)
    trace_id = kwargs.pop("trace_id", None)
    parent_span_id = kwargs.pop("parent_span_id", None)

    # Alle anderen Parameter gehen in metadata
    metadata = kwargs

    return AgentExecutionContext(
        session_id=session_id,
        agent_id=agent_id,
        thread_id=thread_id,
        user_id=user_id,
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        metadata=metadata,
    )


__all__ = [
    "agent_execution_session",
    "create_agent_context",
]
