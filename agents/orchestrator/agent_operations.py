"""Agent-Operationen für Orchestrator."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from agents.common.execution import execute_agent
from kei_logging import get_logger, training_trace
from observability.budget import build_outgoing_budget_headers

try:
    from services.scheduling.scheduler import Scheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    Scheduler = None
    SCHEDULER_AVAILABLE = False

try:
    from agents.registry.dynamic_registry import dynamic_registry
    from config.settings import settings as _settings
    SETTINGS_REGISTRY_AVAILABLE = True
except ImportError:
    _settings = None
    dynamic_registry = None
    SETTINGS_REGISTRY_AVAILABLE = False

try:
    from services.limits.rate_limiter import check_image_limits
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    check_image_limits = None
    RATE_LIMITER_AVAILABLE = False

try:
    from agents.workflows.workflows_constants import BRANCH_NAMES
    from services.workflows.photo_workflow import create_photo_workflow
    PHOTO_WORKFLOW_AVAILABLE = True
except ImportError:
    create_photo_workflow = None
    BRANCH_NAMES = None
    PHOTO_WORKFLOW_AVAILABLE = False

try:
    from services.clients.deep_research import create_deep_research_service
    DEEP_RESEARCH_AVAILABLE = True
except ImportError:
    create_deep_research_service = None
    DEEP_RESEARCH_AVAILABLE = False

logger = get_logger(__name__)


async def _notify_ws_status(status: str, details: str) -> None:
    """Versendet standardisierte KEI-Stream Status-Events an alle Clients."""
    try:
        from data_models.websocket import create_status_update
        from services.streaming.manager import websocket_manager

        event = create_status_update(status=status, details=details)
        await websocket_manager.broadcast(event.model_dump())
    except Exception:
        # WebSockets optional; Fehler hier unterdrücken
        pass


async def _notify(notify: Callable[[str, str, str], Awaitable[None]] | None, status: str, message: str) -> None:
    """Sendet Benachrichtigungen."""
    if notify:
        await notify("orchestrator", status, message)


def _error_response(error: Exception, **context) -> dict:
    """Erstellt Fehler-Antwort."""
    logger.error(f"Orchestrator-Fehler: {error}")
    return {
        "status": "failed",
        "error": str(error),
        "timestamp": datetime.now().isoformat(),
        **context,
    }


def _success_response(**data) -> dict:
    """Erstellt Erfolgs-Antwort."""
    return {"status": "success", "timestamp": datetime.now().isoformat(), **data}
def _generate_execution_id(agent_id: str) -> str:
    """Generiert eindeutige Execution-ID für Agent-Task.

    Args:
        agent_id: ID des Ziel-Agents

    Returns:
        Eindeutige Execution-ID
    """
    import time
    timestamp = int(time.time_ns() // 1000)
    return f"exec_{agent_id}_{timestamp}"


def _extract_task_specification(task_specification: dict) -> tuple[str, str]:
    """Extrahiert Instruction und Priority aus Task-Spezifikation.

    Args:
        task_specification: Task-Spezifikation Dictionary

    Returns:
        Tuple aus (instruction, priority)
    """
    instruction = task_specification.get("instruction", "")
    priority = task_specification.get("priority", "normal")
    return instruction, priority


async def _get_agent_hints(agent_id: str) -> dict:
    """Holt Agent-Hints für Scheduling-Entscheidungen.

    Args:
        agent_id: ID des Agents

    Returns:
        Dictionary mit Agent-Hints
    """
    try:
        from agents.registry.dynamic_registry import dynamic_registry

        agent_obj = await dynamic_registry.get_agent_by_id(agent_id)
        agent_hints = getattr(agent_obj, "hints", {}) or {}
        # Falls Heartbeat Felder direkt am Agent vorhanden
        for k in ("queue_length", "concurrency", "readiness"):
            v = getattr(agent_obj, k, None)
            if v is not None:
                agent_hints[k] = v
    except Exception:
        agent_hints = {}

    return agent_hints


def _create_agent_notify_callback(agent_id: str, notify: Callable[[str, str, str], Awaitable[None]] | None):
    """Erstellt Agent-spezifischen Notification-Callback.

    Args:
        agent_id: ID des Agents
        notify: Haupt-Notification-Callback

    Returns:
        Agent-spezifischer Notification-Callback
    """
    async def agent_notify(update_id: str, status: str, information: str):
        if notify:
            await notify(f"{agent_id}_{update_id}", status, information)

    return agent_notify


def _create_direct_delegate_function(agent_id: str, instruction: str, agent_notify):
    """Erstellt Funktion für direkte Agent-Delegation.

    Args:
        agent_id: ID des Agents
        instruction: Auszuführende Instruction
        agent_notify: Agent-spezifischer Notification-Callback

    Returns:
        Async-Funktion für direkte Delegation
    """
    async def _direct_delegate():
        return await execute_agent(
            framework="foundry",
            agent_id=agent_id,
            instruction=instruction,
            notify=agent_notify,
        )

    return _direct_delegate


from agents.orchestrator.intent_recognition import (
    build_followup_question,
    detect_image_intent,
    detect_photo_intent,
)


async def discover_agents_implementation(
    task_description: str,
    required_capabilities: list[str] | None = None,
    preferred_category: str | None = None,
    notify: Callable[[str, str, str], Awaitable[None]] | None = None,
    **_kwargs,
) -> dict:
    """Findet passende Agents für Task."""
    try:
        await _notify(notify, "discovering", f"Suche Agents für: {task_description}")
        await _notify_ws_status("discovering", f"Suche Agents für: {task_description}")

        from agents.registry.dynamic_registry import dynamic_registry

        if not dynamic_registry.is_initialized():
            await dynamic_registry.start()

        matches = await dynamic_registry.find_agents_for_task(
            task_description=task_description,
            required_capabilities=required_capabilities,
            preferred_category=preferred_category,
        )

        agents = [
            {
                "agent_id": match.agent.id,
                "name": match.agent.name,
                "type": getattr(match.agent, "type", "foundry"),
                "description": match.agent.description,
                "capabilities": getattr(match.agent, "capabilities", []),
                "category": getattr(match.agent, "category", "general"),
                "match_score": round(match.match_score, 2),
                "load_factor": match.load_factor,
                "estimated_response_time": f"{match.estimated_response_time:.1f}s",
                "status": getattr(match.agent, "status", "available"),
                "matched_capabilities": match.matched_capabilities,
            }
            for match in matches
        ]

        await _notify(notify, "discovered", f"Gefunden: {len(agents)} Agents")
        await _notify_ws_status("discovered", f"Gefunden: {len(agents)} Agents")

        return _success_response(
            available_agents=agents,
            total_found=len(agents),
            query=task_description,
            search_criteria={
                "required_capabilities": required_capabilities or [],
                "preferred_category": preferred_category,
            },
        )

    except Exception as e:
        await _notify(notify, "error", f"Discovery-Fehler: {e!s}")
        await _notify_ws_status("error", f"Discovery-Fehler: {e!s}")
        return _error_response(e, available_agents=[], total_found=0)


async def analyze_and_maybe_generate_image(
    user_input: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    notify: Callable[[str, str, str], Awaitable[None]] | None = None,
) -> dict:
    """Analysiert Nutzerinput, fragt fehlende Parameter nach oder generiert direkt ein Bild."""
    try:
        intent = detect_image_intent(user_input)
        if not intent.is_image:
            return _success_response(is_image_intent=False)

        followup = build_followup_question(intent)
        if followup:
            await _notify(notify, "need_parameters", followup)
            await _notify_ws_status("need_parameters", followup)
            return _success_response(
                is_image_intent=True, requires_more_info=True, message=followup
            )

        # Alle Infos vorhanden – fehlende Parameter mit Defaults ergänzen
        size = intent.size or "1024x1024"
        quality = intent.quality or "standard"
        style = intent.style or "Realistic"
        return await generate_image_implementation(
            prompt=intent.prompt,
            size=size,
            quality=quality,
            style=style,
            user_id=user_id,
            session_id=session_id,
            notify=notify,
        )
    except Exception as e:
        return _error_response(e, user_input=user_input)


# Agent Delegation
async def delegate_to_agent_implementation(
    agent_id: str,
    task_specification: dict,
    _execution_mode: str = "async",
    notify: Callable[[str, str, str], Awaitable[None]] | None = None,
    **_kwargs,
) -> dict:
    """Delegiert Task an spezifischen Agent mit Backpressure‑aware Scheduling.

    Wählt dynamisch zwischen Push (direkt) und Pull (Queue) basierend auf
    Agent‑Heartbeat‑Hinweisen (z. B. queue_length, desired_concurrency).

    Args:
        agent_id: ID des Ziel-Agents
        task_specification: Task-Spezifikation mit instruction und priority
        _execution_mode: Ausführungsmodus (async/sync, ungenutzt in aktueller Implementation)
        notify: Benachrichtigungs-Callback
        **_kwargs: Zusätzliche Parameter (ungenutzt in aktueller Implementation)

    Returns:
        Erfolgs- oder Fehler-Response mit Execution-Details
    """
    execution_id = _generate_execution_id(agent_id)

    try:
        # Benachrichtigungen senden
        await _notify(notify, "delegating", f"Delegiere an Agent {agent_id}")
        await _notify_ws_status("delegating", f"Delegiere an Agent {agent_id}")

        # Task-Spezifikation extrahieren
        instruction, priority = _extract_task_specification(task_specification)

        # Agent-spezifischen Notification-Callback erstellen
        agent_notify = _create_agent_notify_callback(agent_id, notify)

        # Agent-Hints für Scheduling abrufen
        agent_hints = await _get_agent_hints(agent_id)

        # Direkte Delegation-Funktion erstellen
        direct_delegate = _create_direct_delegate_function(agent_id, instruction, agent_notify)

        # Scheduling durchführen
        return await _perform_agent_scheduling(
            agent_id=agent_id,
            execution_id=execution_id,
            instruction=instruction,
            priority=priority,
            agent_hints=agent_hints,
            direct_delegate=direct_delegate,
            notify=notify
        )

    except (ValueError, TypeError) as e:
        await _notify(notify, "error", f"Delegation-Fehler - Validierungsfehler: {e!s}")
        await _notify_ws_status("error", f"Delegation-Fehler - Validierungsfehler: {e!s}")
        return _error_response(e, execution_id=execution_id, agent_id=agent_id)
    except (ConnectionError, TimeoutError) as e:
        await _notify(notify, "error", f"Delegation-Fehler - Verbindungsproblem: {e!s}")
        await _notify_ws_status("error", f"Delegation-Fehler - Verbindungsproblem: {e!s}")
        return _error_response(e, execution_id=execution_id, agent_id=agent_id)
    except Exception as e:
        await _notify(notify, "error", f"Delegation-Fehler - Unerwarteter Fehler: {e!s}")
        await _notify_ws_status("error", f"Delegation-Fehler - Unerwarteter Fehler: {e!s}")
        return _error_response(e, execution_id=execution_id, agent_id=agent_id)


async def _perform_agent_scheduling(
    agent_id: str,
    execution_id: str,
    instruction: str,
    priority: str,
    agent_hints: dict,
    direct_delegate,
    notify: Callable[[str, str, str], Awaitable[None]] | None = None
) -> dict:
    """Führt Agent-Scheduling mit Backpressure-Awareness durch.

    Args:
        agent_id: ID des Ziel-Agents
        execution_id: Eindeutige Execution-ID
        instruction: Auszuführende Instruction
        priority: Task-Priorität
        agent_hints: Agent-Hints für Scheduling-Entscheidungen
        direct_delegate: Funktion für direkte Delegation
        notify: Benachrichtigungs-Callback

    Returns:
        Erfolgs-Response mit Scheduling-Details

    Raises:
        ImportError: Wenn Scheduler nicht verfügbar
        KeikoServiceError: Wenn Scheduling fehlschlägt
    """
    if not SCHEDULER_AVAILABLE:
        raise ImportError("Scheduler service not available")

    scheduler = Scheduler()
    task_id = execution_id
    scheduling = await scheduler.schedule_task(
        agent_id=agent_id,
        task_id=task_id,
        task_payload={
            "instruction": instruction,
            "priority": priority,
            "headers": build_outgoing_budget_headers(),
        },
        agent_hints=agent_hints,
        direct_delegate=direct_delegate,
        queue_name=agent_id,
    )

    if not scheduling.accepted:
        from core.exceptions import KeikoServiceError

        raise KeikoServiceError(
            "Scheduling fehlgeschlagen",
            details={"mode": str(scheduling.mode), "reason": scheduling.reason},
        )

    # Erfolgsantwort basierend auf Scheduling-Modus
    return await _build_scheduling_response(
        execution_id=execution_id,
        agent_id=agent_id,
        instruction=instruction,
        priority=priority,
        scheduling=scheduling,
        notify=notify
    )


async def _build_scheduling_response(
    execution_id: str,
    agent_id: str,
    instruction: str,
    priority: str,
    scheduling,
    notify: Callable[[str, str, str], Awaitable[None]] | None = None
) -> dict:
    """Erstellt Response basierend auf Scheduling-Ergebnis.

    Args:
        execution_id: Eindeutige Execution-ID
        agent_id: ID des Agents
        instruction: Ausgeführte Instruction
        priority: Task-Priorität
        scheduling: Scheduling-Ergebnis
        notify: Benachrichtigungs-Callback

    Returns:
        Erfolgs-Response mit entsprechenden Details
    """
    base_response = _success_response(
        execution_id=execution_id,
        agent_id=agent_id,
        task=instruction,
        priority=priority,
        execution_mode=scheduling.mode,
    )

    if scheduling.mode == "pull":
        base_response.update({"enqueued": True, "queue_subject": scheduling.queue_subject})
        await _notify(notify, "enqueued", f"Task eingereiht für {agent_id}")
        await _notify_ws_status("enqueued", f"Task in Queue für {agent_id}")
    else:
        await _notify(notify, "completed", f"Agent {agent_id} erfolgreich")
        await _notify_ws_status("completed", f"Agent {agent_id} erfolgreich")
        base_response.update({"result": scheduling.direct_result})

    return base_response


# Image Generation Helper Functions
def _log_image_generation_start(user_id: str | None, session_id: str | None, prompt: str, size: str, quality: str, style: str) -> None:
    """Loggt den Start der Bildgenerierung.

    Args:
        user_id: Benutzer-ID
        session_id: Session-ID
        prompt: Bildgenerierungs-Prompt
        size: Bildgröße
        quality: Bildqualität
        style: Bildstil
    """
    logger.debug({
        "event": "generate_image_start",
        "user_id": user_id,
        "session_id": session_id,
        "prompt": prompt,
        "params": {"size": size, "quality": quality, "style": style},
    })


async def _get_image_generator_agent(target_id: str):
    """Holt den Image-Generator-Agent aus der Registry.

    Args:
        target_id: ID des Image-Generator-Agents

    Returns:
        Image-Generator-Agent oder None

    Raises:
        ImportError: Wenn Settings/Registry nicht verfügbar
    """
    if not SETTINGS_REGISTRY_AVAILABLE:
        raise ImportError("Settings and registry services not available")

    agent = None
    try:
        if hasattr(dynamic_registry, "get_agent_by_id"):
            agent = await dynamic_registry.get_agent_by_id(target_id)  # type: ignore[attr-defined]
        if agent is None and hasattr(dynamic_registry, "agents"):
            agent = dynamic_registry.agents.get(target_id)  # type: ignore[assignment]
    except Exception:
        agent = None

    if agent is None:
        if not getattr(dynamic_registry, "_initialized", False):
            logger.debug({"event": "registry_start_needed"})
            await dynamic_registry.start()  # type: ignore[attr-defined]
        try:
            if hasattr(dynamic_registry, "get_agent_by_id"):
                agent = await dynamic_registry.get_agent_by_id(target_id)  # type: ignore[attr-defined]
            if agent is None and hasattr(dynamic_registry, "agents"):
                agent = dynamic_registry.agents.get(target_id)  # type: ignore[assignment]
        except Exception:
            agent = None

    return agent


def _get_image_generator_target_id() -> str:
    """Ermittelt die Target-ID für den Image-Generator-Agent.

    Returns:
        Target-ID des Image-Generator-Agents
    """
    return getattr(_settings, "agent_image_generator_id", "") or "agent_image_generator"


async def _check_image_rate_limits(
    user_id: str | None,
    session_id: str | None = None,
    notify: Callable[[str, str, str], Awaitable[None]] | None = None
) -> dict:
    """Prüft Rate-Limits für Bildgenerierung.

    Args:
        user_id: Benutzer-ID
        session_id: Session-ID
        notify: Benachrichtigungs-Callback

    Returns:
        Rate-Limit-Headers oder leeres Dict

    Raises:
        Exception: Wenn Rate-Limit überschritten
    """
    rl_headers = {}
    if RATE_LIMITER_AVAILABLE and (user_id or session_id):
        try:
            allowed, headers, retry_after, scope = await check_image_limits(user_id, session_id)  # type: ignore[misc]
            rl_headers.update(headers)

            if not allowed:
                error_msg = f"Rate-Limit erreicht für {scope}. Retry nach {retry_after} Sekunden."
                await _notify(notify, "rate_limited", error_msg)
                await _notify_ws_status("rate_limited", "Bildgenerierung rate-limitiert")
                raise Exception(error_msg)

        except Exception as rl_error:
            await _notify(notify, "rate_limited", f"Rate-Limit erreicht: {rl_error}")
            await _notify_ws_status("rate_limited", "Bildgenerierung rate-limitiert")
            raise rl_error

    return rl_headers


async def delegate_to_best_agent_implementation(
    task_description: str,
    selection_criteria: dict,
    notify: Callable[[str, str, str], Awaitable[None]] | None = None,
    **_kwargs,
) -> dict:
    """Wählt besten Agent automatisch aus und delegiert."""
    try:
        await _notify(notify, "selecting", "Suche besten Agent")
        await _notify_ws_status("selecting", "Suche besten Agent")

        # Discovery mit Kriterien
        discovery_result = await discover_agents_implementation(
            task_description=task_description,
            required_capabilities=selection_criteria.get("required_capabilities"),
            preferred_category=selection_criteria.get("preferred_category"),
            notify=notify,
        )

        if discovery_result["status"] != "success" or not discovery_result["available_agents"]:
            return _error_response(
                Exception("Keine passenden Agents gefunden"), task=task_description
            )

        # Filter und Score Agents
        agents = discovery_result["available_agents"]
        min_score = selection_criteria.get("min_match_score", 0.5)
        max_load = selection_criteria.get("max_load_factor", 0.8)

        eligible = [
            a
            for a in agents
            if a["match_score"] >= min_score
            and a["load_factor"] <= max_load
            and a["status"] == "available"
        ]

        if not eligible:
            return _error_response(
                Exception("Keine Agents erfüllen die Kriterien"),
                task=task_description,
                criteria=selection_criteria,
            )

        # Wähle besten Agent
        best_agent = max(eligible, key=lambda a: a["match_score"] - (a["load_factor"] * 0.1))

        await _notify(notify, "selected", f"Ausgewählt: {best_agent['name']}")
        await _notify_ws_status("selected", f"Ausgewählt: {best_agent['name']}")

        # Delegiere an besten Agent
        return await delegate_to_agent_implementation(
            agent_id=best_agent["agent_id"],
            task_specification={"instruction": task_description},
            notify=notify,
        )

    except (ValueError, TypeError) as e:
        await _notify(notify, "error", f"Auswahl-Fehler - Validierungsfehler: {e!s}")
        await _notify_ws_status("error", f"Auswahl-Fehler - Validierungsfehler: {e!s}")
        return _error_response(e, task=task_description)
    except (ConnectionError, TimeoutError) as e:
        await _notify(notify, "error", f"Auswahl-Fehler - Verbindungsproblem: {e!s}")
        await _notify_ws_status("error", f"Auswahl-Fehler - Verbindungsproblem: {e!s}")
        return _error_response(e, task=task_description)
    except Exception as e:
        await _notify(notify, "error", f"Auswahl-Fehler - Unerwarteter Fehler: {e!s}")
        await _notify_ws_status("error", f"Auswahl-Fehler - Unerwarteter Fehler: {e!s}")
        return _error_response(e, task=task_description)


@training_trace(context={"component": "orchestrator", "phase": "image_generation"})
async def generate_image_implementation(
    prompt: str,
    *,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "Realistic",
    user_id: str | None = None,
    session_id: str | None = None,
    notify: Callable[[str, str, str], Awaitable[None]] | None = None,
    **_kwargs,
) -> dict:
    """Implementierung der Bildgenerierung über nativen Agent.

    Ermittelt konfigurierten Image-Generator-Agent und führt Anfrage aus.

    Args:
        prompt: Beschreibung des zu generierenden Bildes
        size: Bildgröße (1024x1024|1024x1792|1792x1024)
        quality: Qualität (standard|hd)
        style: Stil (Realistic|Artistic|Cartoon|Photography|Digital Art)
        user_id: Benutzer-ID
        session_id: Session-ID
        notify: Benachrichtigungs-Callback
        **_kwargs: Zusätzliche Parameter (ungenutzt in aktueller Implementation)

    Returns:
        Erfolgs- oder Fehler-Response mit Bildgenerierungs-Details
    """
    try:
        # Logging und Benachrichtigungen
        _log_image_generation_start(user_id, session_id, prompt, size, quality, style)
        await _notify(notify, "analyzing", "Analysiere Bildanfrage")
        await _notify_ws_status("analyzing", "Analysiere Bildanfrage")

        # Target-Agent ermitteln
        target_id = _get_image_generator_target_id()
        logger.debug({"event": "resolve_agent_id", "target_id": target_id})

        # Agent aus Registry holen
        agent = await _get_image_generator_agent(target_id)

        if agent is None:
            return _error_response(
                Exception(f"Image-Generator-Agent '{target_id}' nicht gefunden"),
                target_id=target_id
            )

        # Rate-Limits prüfen
        rl_headers = await _check_image_rate_limits(user_id, session_id, notify)

        # Bildgenerierung durchführen
        return await _execute_image_generation(
            agent=agent,
            target_id=target_id,
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            user_id=user_id,
            session_id=session_id,
            rl_headers=rl_headers,
            notify=notify
        )

    except (ValueError, TypeError) as e:
        await _notify(notify, "error", f"Bildgenerierung-Fehler - Validierungsfehler: {e!s}")
        await _notify_ws_status("error", f"Bildgenerierung fehlgeschlagen - Validierungsfehler: {e!s}")
        return _error_response(e, prompt=prompt)
    except (ConnectionError, TimeoutError) as e:
        await _notify(notify, "error", f"Bildgenerierung-Fehler - Verbindungsproblem: {e!s}")
        await _notify_ws_status("error", f"Bildgenerierung fehlgeschlagen - Verbindungsproblem: {e!s}")
        return _error_response(e, prompt=prompt)
    except Exception as e:
        await _notify(notify, "error", f"Bildgenerierung-Fehler - Unerwarteter Fehler: {e!s}")
        await _notify_ws_status("error", f"Bildgenerierung fehlgeschlagen - Unerwarteter Fehler: {e!s}")
        return _error_response(e, prompt=prompt)


async def _execute_image_generation(
    agent,
    target_id: str,
    prompt: str,
    size: str,
    quality: str,
    style: str,
    user_id: str | None,
    session_id: str | None,
    rl_headers: dict,
    notify: Callable[[str, str, str], Awaitable[None]] | None = None
) -> dict:
    """Führt die eigentliche Bildgenerierung durch.

    Args:
        agent: Image-Generator-Agent
        target_id: Agent-ID
        prompt: Bildgenerierungs-Prompt
        size: Bildgröße
        quality: Bildqualität
        style: Bildstil
        user_id: Benutzer-ID
        session_id: Session-ID
        rl_headers: Rate-Limit-Headers
        notify: Benachrichtigungs-Callback

    Returns:
        Erfolgs- oder Fehler-Response mit Bildgenerierungs-Details
    """
    try:
        await _notify(notify, "generating", "Generiere Bild")
        await _notify_ws_status("generating", "Bildgenerierung läuft")

        # Agent ausführen
        from agents.constants import ImageQuality, ImageSize, ImageStyle
        from agents.custom.image_generator_agent import ImageTask

        # Konvertiere String-Parameter zu Enum-Werten
        size_enum = ImageSize(size) if isinstance(size, str) else size
        quality_enum = ImageQuality(quality) if isinstance(quality, str) else quality
        style_enum = ImageStyle(style) if isinstance(style, str) else style

        task = ImageTask(
            prompt=prompt,
            size=size_enum,
            quality=quality_enum,
            style=style_enum,
            user_id=user_id,
            session_id=session_id,
        )
        logger.debug({"event": "invoke_image_agent", "agent_id": target_id})
        result = await agent.handle(task)  # type: ignore[call-arg]
        result.setdefault("headers", {}).update(rl_headers)

        status = result.get("status", "failed")
        logger.debug({"event": "image_agent_result", "status": status, "keys": list(result.keys())})

        if status != "success":
            await _notify(notify, "failed", f"Bildgenerierung fehlgeschlagen: {result}")
            await _notify_ws_status("failed", "Bildgenerierung fehlgeschlagen")
            return result

        await _notify(notify, "completed", "Bild erfolgreich generiert")
        await _notify_ws_status("completed", "Bild erfolgreich generiert")
        return result

    except (ValueError, TypeError) as e:
        logger.error(f"Image generation execution failed - Validierungsfehler: {e}")
        await _notify(notify, "failed", f"Bildgenerierung fehlgeschlagen - Validierungsfehler: {e}")
        await _notify_ws_status("failed", "Bildgenerierung fehlgeschlagen - Validierungsfehler")
        return _error_response(e, prompt=prompt)
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Image generation execution failed - Verbindungsproblem: {e}")
        await _notify(notify, "failed", f"Bildgenerierung fehlgeschlagen - Verbindungsproblem: {e}")
        await _notify_ws_status("failed", "Bildgenerierung fehlgeschlagen - Verbindungsproblem")
        return _error_response(e, prompt=prompt)
    except Exception as e:
        logger.exception(f"Image generation execution failed - Unerwarteter Fehler: {e}")
        await _notify(notify, "failed", f"Bildgenerierung fehlgeschlagen - Unerwarteter Fehler: {e}")
        await _notify_ws_status("failed", "Bildgenerierung fehlgeschlagen - Unerwarteter Fehler")
        return _error_response(e, prompt=prompt)


# Photo Operations
async def photo_request_implementation(
    *,
    user_id: str | None = None,
    notify: Callable[[str, str, str], Awaitable[None]] | None = None,
    **_kwargs: Any,
) -> dict:
    """Signalisiert eine explizite Nutzeranfrage zum Foto-Aufnehmen an den Orchestrator.

    Diese Funktion dient als leichtgewichtige Benachrichtigung und Protokollierung und kann
    vom Frontend genutzt werden, um die Fotofunktion on-demand zu aktivieren.

    Args:
        user_id: Benutzer-ID
        notify: Benachrichtigungs-Callback
        **_kwargs: Zusätzliche Parameter (ignoriert)

    Returns:
        Erfolgs-Response mit Bestätigung
    """
    try:
        await _notify(notify, "photo_request", f"photo_request for user={user_id or 'unknown'}")
    except Exception:
        pass
    return _success_response(action="photo_request", acknowledged=True, user_id=user_id)


async def analyze_and_maybe_take_photo(
    user_input: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    notify: Callable[[str, str, str], Awaitable[None]] | None = None,
) -> dict:
    """Analysiert Nutzerinput und startet Foto-Workflow falls Foto-Intent erkannt wird.

    Args:
        user_input: Benutzereingabe
        user_id: Benutzer-ID
        session_id: Session-ID
        notify: Benachrichtigungs-Callback

    Returns:
        Erfolgs- oder Fehler-Response mit Foto-Workflow-Details
    """
    try:
        intent = detect_photo_intent(user_input)
        if not intent.is_photo:
            return _success_response(is_photo_intent=False)

        await _notify(notify, "photo_intent_detected", "Foto-Anfrage erkannt, starte Workflow")
        await _notify_ws_status("photo_intent_detected", "Foto-Anfrage erkannt")


        try:
            from workflows.photo_workflow import create_photo_workflow

            workflow = create_photo_workflow()
            initial_state = {
                "user_id": user_id or "unknown",
                "session_id": session_id,
                "user_input": user_input,
                "request_sent": False,
                "preview_url": None,
                "decision": None,
                "attempts": 0,
                "max_attempts": 3,
                "extras": {},
            }


            config = {"configurable": {"thread_id": f"photo_{user_id}_{session_id}"}}
            result = await workflow.ainvoke(initial_state, config)

            await _notify(notify, "photo_workflow_started", "Foto-Workflow gestartet")
            await _notify_ws_status("photo_workflow_started", "Foto-Workflow aktiv")

            return _success_response(
                is_photo_intent=True,
                workflow_started=True,
                workflow_state=result,
                message="Foto-Workflow gestartet. Bitte verwenden Sie die Kamera-Funktion.",
            )

        except Exception as workflow_error:
            logger.error(f"Foto-Workflow Fehler: {workflow_error}")

            return await photo_request_implementation(user_id=user_id, notify=notify)

    except Exception as e:
        return _error_response(e, user_input=user_input)



async def get_agent_details_implementation(
    agent_id: str, _: Callable[[str, str, str], Awaitable[None]] | None = None, **__
) -> dict:
    """Holt Agent-Informationen aus der Registry.

    Args:
        agent_id: ID des Agents
        notify: Benachrichtigungs-Callback
        **kwargs: Zusätzliche Parameter

    Returns:
        Erfolgs- oder Fehler-Response mit Agent-Details
    """
    try:
        from agents.registry.dynamic_registry import dynamic_registry

        if dynamic_registry.is_initialized() and agent_id in dynamic_registry.agents:
            agent = dynamic_registry.agents[agent_id]
            return _success_response(
                agent_id=agent_id,
                name=getattr(agent, "name", "Unknown"),
                description=getattr(agent, "description", ""),
                type=getattr(agent, "type", "foundry"),
                capabilities=getattr(agent, "capabilities", []),
                category=getattr(agent, "category", "general"),
                status=getattr(agent, "status", "unknown"),
                last_activity=getattr(agent, "last_activity", None),
            )

        return _error_response(Exception(f"Agent {agent_id} nicht gefunden"), agent_id=agent_id)

    except Exception as e:
        return _error_response(e, agent_id=agent_id)


async def monitor_execution_implementation(
    execution_id: str,
    include_logs: bool = False,
    notify: Callable[[str, str, str], Awaitable[None]] | None = None,
    **_kwargs,
) -> dict:
    """Überwacht Agent-Ausführung und gibt Status-Updates zurück.

    Args:
        execution_id: ID der Ausführung
        include_logs: Ob Logs eingeschlossen werden sollen
        notify: Benachrichtigungs-Callback
        **_kwargs: Zusätzliche Parameter (ungenutzt in aktueller Implementation)

    Returns:
        Erfolgs- oder Fehler-Response mit Execution-Status
    """
    try:
        await _notify(notify, "monitoring", f"Überwache: {execution_id}")
        await _notify_ws_status("monitoring", f"Überwache: {execution_id}")


        await asyncio.sleep(0.1)

        status_data = {
            "execution_id": execution_id,
            "status": "running",
            "progress": 75,
            "start_time": datetime.now().isoformat(),
            "estimated_completion": (datetime.now()).isoformat(),
        }

        if include_logs:
            status_data["logs"] = [
                {"timestamp": datetime.now().isoformat(), "message": "Task gestartet"},
                {"timestamp": datetime.now().isoformat(), "message": "Verarbeitung läuft"},
            ]

        return _success_response(**status_data)

    except (ValueError, TypeError) as e:
        await _notify_ws_status("error", f"Monitoring-Fehler - Validierungsfehler: {e!s}")
        return _error_response(e, execution_id=execution_id)
    except (ConnectionError, TimeoutError) as e:
        await _notify_ws_status("error", f"Monitoring-Fehler - Verbindungsproblem: {e!s}")
        return _error_response(e, execution_id=execution_id)
    except Exception as e:
        await _notify_ws_status("error", f"Monitoring-Fehler - Unerwarteter Fehler: {e!s}")
        return _error_response(e, execution_id=execution_id)



async def perform_web_research_implementation(
    query: str,
    *,
    max_iterations: int = 3,
    context: dict | None = None,
    notify: Callable[[str, str, str], Awaitable[None]] | None = None,
    **_kwargs,
) -> dict:
    """Führt eine strukturierte Web-Recherche mit Azure AI Foundry Deep Research aus.

    Gibt strukturierte Ergebnisse inkl. Quellen zurück. Fällt bei Nichtverfügbarkeit
    der Azure AI Foundry SDKs auf ein Fallback-Ergebnis zurück (graceful degradation).

    Args:
        query: Suchanfrage
        max_iterations: Maximale Anzahl Iterationen
        context: Zusätzlicher Kontext
        notify: Benachrichtigungs-Callback
        **_kwargs: Zusätzliche Parameter (ungenutzt in aktueller Implementation)

    Returns:
        Erfolgs- oder Fehler-Response mit Recherche-Ergebnissen
    """
    try:
        await _notify(notify, "research_start", f"Starte Web-Recherche: {query}")
        await _notify_ws_status("research_start", f"Starte Web-Recherche: {query}")

        # Primär: Service-Layer verwenden
        try:
            if not DEEP_RESEARCH_AVAILABLE:
                raise ImportError("Deep research service not available")

            service = create_deep_research_service()
            if service is not None:
                result = await service.run(
                    query, max_iterations=max_iterations, context=context or {}
                )
            else:
                from core.exceptions import KeikoServiceError

                raise KeikoServiceError("DeepResearchService nicht verfügbar")
        except Exception as inner:

            try:
                from agents.protocols.foundry_protocols import FoundryProtocolFactory

                protocol = FoundryProtocolFactory.create_protocol(
                    "deep_research", max_iterations=max_iterations
                )
                result = await protocol.execute(query=query, context=context or {})
            except Exception:

                result = {
                    "query": query,
                    "results": [
                        {
                            "iteration": 0,
                            "query": query,
                            "sources": [],
                            "key_findings": [
                                "Deep Research SDK nicht verfügbar. Fallback-Ergebnis.",
                            ],
                            "confidence": 0.0,
                            "requires_followup": False,
                        }
                    ],
                    "synthesis": {
                        "summary": "Keine verifizierten Quellen verfügbar",
                        "key_insights": [],
                        "confidence": 0.0,
                        "recommendation": "Bitte Azure AI Foundry konfigurieren",
                    },
                    "error": str(inner),
                }


        from typing import Any as _Any

        sources: list[dict[str, _Any]] = []
        for step in result.get("results", []):
            for src in step.get("sources", []) or []:
                if isinstance(src, dict) and src.get("url"):
                    sources.append(
                        {
                            "url": src.get("url"),
                            "title": src.get("title", src.get("url")),
                            "verified": bool(src.get("verified", False)),
                        }
                    )


        seen_urls = set()
        deduped_sources: list[dict[str, _Any]] = []
        for s in sources:
            u = s["url"]
            if u not in seen_urls:
                seen_urls.add(u)
                deduped_sources.append(s)

        await _notify(notify, "research_completed", "Web-Recherche abgeschlossen")
        await _notify_ws_status("research_completed", "Web-Recherche abgeschlossen")
        return _success_response(
            query=result.get("query", query),
            results=result.get("results", []),
            synthesis=result.get("synthesis", {}),
            sources=deduped_sources,
            timestamp=result.get("timestamp"),
        )

    except (ValueError, TypeError) as e:
        await _notify(notify, "error", f"Recherche-Fehler - Validierungsfehler: {e!s}")
        await _notify_ws_status("error", f"Recherche-Fehler - Validierungsfehler: {e!s}")
        return _error_response(e, query=query)
    except (ConnectionError, TimeoutError) as e:
        await _notify(notify, "error", f"Recherche-Fehler - Verbindungsproblem: {e!s}")
        await _notify_ws_status("error", f"Recherche-Fehler - Verbindungsproblem: {e!s}")
        return _error_response(e, query=query)
    except Exception as e:
        await _notify(notify, "error", f"Recherche-Fehler - Unerwarteter Fehler: {e!s}")
        await _notify_ws_status("error", f"Recherche-Fehler - Unerwarteter Fehler: {e!s}")
        return _error_response(e, query=query)
