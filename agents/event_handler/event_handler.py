# backend/agents/event_handler/event_handler.py
"""Azure AI Foundry Event-Handler für Personal Assistant.
Verarbeitet Agent-Events und stellt WebSocket-Integration bereit.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import (
    AsyncAgentEventHandler,
    RequiredFunctionToolCall,
    RunStep,
    SubmitToolOutputsAction,
    ThreadMessage,
    ThreadRun,
    ToolOutput,
)

from data_models import AgentUpdateEvent, Content, Function
from kei_logging import get_logger

from .constants import (
    DEBUG_TOOL_OUTPUTS_SENT,
    DEBUG_UNHANDLED_EVENT,
    DISPLAY_STATUS_MESSAGE_COMPLETED,
    ERROR_AGENT_STREAM,
    ERROR_TOOL_OUTPUT_SUBMISSION,
    INFO_MESSAGE_RESULT_RECEIVED,
    TOOL_EXECUTION_TIMEOUT_SECONDS,
    WARNING_TOOL_NOT_FOUND,
    ContentType,
    MessageStatus,
    RunStatus,
    StepType,
)
from .utils import (
    ErrorContext,
    EventDispatcher,
    EventErrorHandler,
    StructuredLogger,
    ToolExecutor,
)

logger = get_logger(__name__)


class AgentEventHandler(AsyncAgentEventHandler[str]):
    """Event-Handler für Azure AI Foundry Agent-Streams.

    Verarbeitet Events von Azure AI Foundry Agent-Streams, führt Tool-Calls aus
    und benachrichtigt WebSocket-Clients über Status-Updates. Implementiert
    Event-Deduplication für effiziente Stream-Verarbeitung und robustes
    Error-Handling für produktive Umgebungen.

    Attributes:
        _runtime_client: Azure AI Agents Client für Tool-Output-Submission
        _tools: Registry der verfügbaren Tools für Function-Calls
        _notify: WebSocket-Callback für Client-Benachrichtigungen
        _event_dispatcher: Utility für Event-Dispatch mit Deduplication
        _tool_executor: Utility für sichere Tool-Ausführung mit Timeout
        _error_handler: Erweiterte Error-Handling-Komponente
        _structured_logger: Strukturiertes Logging für Monitoring

    Example:
        ```python
        handler = AgentEventHandler(
            runtime_client=agents_client,
            tools=tool_registry,
            notify=websocket_callback
        )

        # Handler wird automatisch von Azure AI Foundry aufgerufen
        async with agents_client.create_stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
            event_handler=handler
        ) as stream:
            await stream.until_done()
        ```
    """

    def __init__(
        self,
        runtime_client: AgentsClient,
        tools: Mapping[str, Function],
        notify: AgentUpdateEvent,
    ) -> None:
        """Initialisiert Event-Handler mit allen erforderlichen Komponenten.

        Erstellt und konfiguriert alle Utility-Komponenten für Event-Processing,
        Tool-Execution und Error-Handling. Der Handler ist nach der Initialisierung
        bereit für die Verwendung mit Azure AI Foundry Agent-Streams.

        Args:
            runtime_client: Azure AI Agents Client für Tool-Output-Submission
                und Stream-Interaktion
            tools: Mapping von Tool-Namen zu ausführbaren Funktionen.
                Jede Funktion muss die Tool-Call-Argumente als Keyword-Parameter
                akzeptieren und kann synchron oder asynchron sein.
            notify: Callback-Funktion für WebSocket-Benachrichtigungen.
                Wird mit Event-Daten als Keyword-Argumente aufgerufen.

        Raises:
            TypeError: Wenn Parameter nicht den erwarteten Typ haben
            ValueError: Wenn Tools-Registry leer ist

        Note:
            Der Handler verwendet automatisch Event-Deduplication und Timeout-basierte
            Tool-Execution für robuste Verarbeitung in produktiven Umgebungen.
        """
        super().__init__()
        self._runtime_client = runtime_client
        self._tools = tools
        self._notify = notify

        # Wrapper für Type-Safety - konvertiert AgentUpdateEvent zu dict
        def notify_wrapper(event_dict: dict[str, Any]) -> Any:
            return self._notify(event_dict)

        self._event_dispatcher = EventDispatcher(callback=notify_wrapper)
        self._tool_executor = ToolExecutor(timeout_seconds=TOOL_EXECUTION_TIMEOUT_SECONDS)
        self._error_handler = EventErrorHandler(component_name="agent_event_handler")
        self._structured_logger = StructuredLogger(component_name="agent")

    @staticmethod
    def _format_status(base_type: str, status: str, object_type: str | None = None) -> str:
        """Formatiert Status für Client-Anzeige.

        Args:
            base_type: Basis-Event-Typ (run, step, message)
            status: Roher Status-String
            object_type: Optionaler Objekt-Typ für spezielle Formatierung

        Returns:
            Formatierter Status-String für Client-Anzeige
        """
        formatted_status = status.replace("_", " ")
        if object_type == StepType.TOOL_CALLS.value:
            return f'{base_type} "{object_type}" {formatted_status}'
        return f"{base_type} {formatted_status}"

    async def _dispatch(
        self,
        *,
        obj: ThreadRun | RunStep | ThreadMessage,
        status: str,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Versendet Event an Client mit Duplikatprüfung.

        Args:
            obj: Event-Objekt mit ID
            status: Display-Status
            extra: Zusätzliche Event-Daten
        """
        await self._event_dispatcher.dispatch(
            event_id=obj.id,
            status=status,
            object_type=obj.object,
            extra=extra
        )

    async def on_thread_run(self, run: ThreadRun) -> None:
        """Verarbeitet Thread-Run-Events.

        Args:
            run: ThreadRun-Objekt mit Event-Daten
        """
        status = AgentEventHandler._format_status("run", run.status)
        await self._dispatch(obj=run, status=status)

        if run.status == RunStatus.REQUIRES_ACTION and isinstance(
            run.required_action, SubmitToolOutputsAction
        ):
            await self._handle_tool_calls(run)

    async def on_run_step(self, step: RunStep) -> None:
        """Verarbeitet Run-Step-Events.

        Args:
            step: RunStep-Objekt mit Event-Daten
        """
        status = AgentEventHandler._format_status("step", step.status, step.type)
        await self._dispatch(obj=step, status=status)

    async def on_thread_message(self, message: ThreadMessage) -> None:
        """Verarbeitet Thread-Message-Events.

        Args:
            message: ThreadMessage-Objekt mit Event-Daten
        """
        if message.status == MessageStatus.COMPLETED:
            items = AgentEventHandler._extract_message_content(message)
            await self._dispatch(
                obj=message,
                status=DISPLAY_STATUS_MESSAGE_COMPLETED,
                extra={
                    "information": INFO_MESSAGE_RESULT_RECEIVED,
                    "content": Content(type=ContentType.TEXT, data={"items": items}),
                    "output": True,
                },
            )
        else:
            status = AgentEventHandler._format_status("message", message.status)
            await self._dispatch(obj=message, status=status)

    @staticmethod
    def _extract_message_content(message: ThreadMessage) -> list[dict[str, Any]]:
        """Extrahiert und formatiert Message-Content.

        Args:
            message: ThreadMessage-Objekt

        Returns:
            Liste von Content-Items
        """
        content_items = []
        for content_item in message.content:
            content_dict = content_item.as_dict()
            content_type = content_dict.get("type", "text")

            # Sichere Extraktion der Content-Daten
            if content_type in content_dict and isinstance(content_dict[content_type], dict):
                content_items.append({
                    "type": content_type,
                    **content_dict[content_type]
                })
            else:
                # Fallback für einfache Content-Strukturen
                content_items.append({
                    "type": content_type,
                    "value": str(content_dict.get(content_type, content_dict))
                })

        return content_items

    async def on_error(self, data: str) -> None:
        """Verarbeitet Stream-Fehler.

        Args:
            data: Error-Daten vom Stream
        """
        logger.error(f"{ERROR_AGENT_STREAM}: %s", data)

    async def on_unhandled_event(self, event_type: str, event_data: Any) -> None:
        """Verarbeitet unbekannte Event-Typen.

        Args:
            event_type: Typ des unbekannten Events
            event_data: Event-Daten
        """
        logger.debug(f"{DEBUG_UNHANDLED_EVENT} %s: %s", event_type, event_data)

    async def _handle_tool_calls(self, run: ThreadRun) -> None:
        """Führt angeforderte Tool-Calls aus und sendet Outputs zurück.

        Args:
            run: ThreadRun mit angeforderten Tool-Calls
        """
        if not isinstance(run.required_action, SubmitToolOutputsAction):
            return

        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        # Type-Safety: Filtere nur RequiredFunctionToolCall
        function_tool_calls = [tc for tc in tool_calls if isinstance(tc, RequiredFunctionToolCall)]
        tool_outputs = await self._process_tool_calls(function_tool_calls)

        if tool_outputs:
            await self._submit_tool_outputs(run, tool_outputs)

    async def _process_tool_calls(self, tool_calls: list[RequiredFunctionToolCall]) -> list[ToolOutput]:
        """Verarbeitet eine Liste von Tool-Calls.

        Args:
            tool_calls: Liste der auszuführenden Tool-Calls

        Returns:
            Liste der Tool-Outputs
        """
        tool_outputs: list[ToolOutput] = []

        for tool_call in tool_calls:
            if not isinstance(tool_call, RequiredFunctionToolCall):
                continue

            output = await self._execute_single_tool_call(tool_call)
            if output:
                tool_outputs.append(output)

        return tool_outputs

    async def _execute_single_tool_call(self, tool_call: RequiredFunctionToolCall) -> ToolOutput | None:
        """Führt einen einzelnen Tool-Call aus.

        Args:
            tool_call: Auszuführender Tool-Call

        Returns:
            Tool-Output oder None bei Fehlern
        """
        func_name = tool_call.function.name

        if func_name not in self._tools:
            logger.warning(f"{WARNING_TOOL_NOT_FOUND}: '%s'", func_name)
            return None

        try:
            func_args = json.loads(tool_call.function.arguments)
            func = self._tools[func_name]

            # Tool mit Timeout und Error-Handling ausführen
            execution_result = await self._tool_executor.execute_with_timeout(
                func=func,
                args=func_args,
                tool_name=func_name
            )

            self._structured_logger.log_tool_execution(
                tool_name=func_name,
                execution_time_ms=execution_result.execution_time_ms,
                success=execution_result.success,
                context={"tool_call_id": tool_call.id}
            )

            if execution_result.success:
                output_str = ToolExecutor.serialize_result(
                    execution_result.result
                )
                return ToolOutput(tool_call_id=tool_call.id, output=output_str)

            error_context = ErrorContext(
                operation="tool_execution",
                component="agent_event_handler",
                additional_data={"tool_call_id": tool_call.id}
            )
            error_msg = self._error_handler.handle_tool_error(
                func_name,
                Exception(execution_result.error),
                error_context
            )
            logger.error(error_msg)
            return ToolOutput(
                tool_call_id=tool_call.id,
                output=execution_result.error or "Unknown error"
            )

        except Exception as e:
            error_context = ErrorContext(
                operation="tool_execution",
                component="agent_event_handler",
                additional_data={"tool_call_id": tool_call.id}
            )
            error_msg = self._error_handler.handle_tool_error(
                func_name, e, error_context
            )
            logger.error(error_msg)
            return ToolOutput(tool_call_id=tool_call.id, output=str(e))



    async def _submit_tool_outputs(self, run: ThreadRun, tool_outputs: list[ToolOutput]) -> None:
        """Sendet Tool-Outputs an Azure AI Foundry.

        Args:
            run: ThreadRun-Objekt
            tool_outputs: Liste der Tool-Outputs
        """
        try:
            await self._runtime_client.submit_tool_outputs_to_run(  # type: ignore
                thread_id=run.thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs,
            )
            logger.debug(f"{DEBUG_TOOL_OUTPUTS_SENT}: %d", len(tool_outputs))
        except Exception as e:
            logger.error(f"{ERROR_TOOL_OUTPUT_SUBMISSION}: %s", e)
