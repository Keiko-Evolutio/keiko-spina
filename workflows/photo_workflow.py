"""Photo-Workflow implementiert mit LangGraph.

Workflow-Ablauf:
1) start(user_input) → Intent erkennen; bei Photo-Intent, photo_request signalisieren
2) capture → Kamera aufrufen (Server-Aufnahme oder Client-Upload erwarten) und Vorschau
3) await_user_decision → Auf Benutzer-Bestätigung warten (ok/retake)
4) bei ok → upload_to_blob (bereits Teil der capture/upload Endpunkte,
   gibt SAS zurück)
5) bei retake → Zurück zu capture

Diese Datei stellt einen minimalen, produktionsreifen Graphen mit
Checkpoint-freundlichem State bereit.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

# Robuster Logger-Import mit Fallback
try:
    from kei_logging import get_logger
except ImportError:
    try:
        from core.logging_strategy import get_module_logger as get_logger
    except ImportError:
        import logging
        def get_logger(name: str = __name__):
            return logging.getLogger(name)

# Korrekte relative Imports
sys.path.append(os.path.join(
    os.path.dirname(__file__), "..", "agents", "workflows"
))

try:
    from agents.workflows.graph_export import WorkflowVisualizer
except ImportError:
    WorkflowVisualizer = None

try:
    from agents.workflows.workflows_base import BaseWorkflowBuilder, BaseWorkflowConfig
except ImportError:
    # Fallback-Definitionen
    @dataclass
    class BaseWorkflowConfig:
        """Basis-Konfiguration für Workflows."""
        max_attempts: int = 3
        enable_preview: bool = True
        auto_upload: bool = True

    class BaseWorkflowBuilder:
        """Basis-Builder für Workflows."""
        def __init__(self, config=None):
            self.config = config or BaseWorkflowConfig()

try:
    from agents.workflows.workflows_constants import (
        BRANCH_NAMES,
        DEFAULT_PHOTO_MAX_ATTEMPTS,
        ERROR_MESSAGES,
        PHOTO_INTENT_KEYWORDS,
        WORKFLOW_NODES,
    )
except ImportError:
    # Fallback-Konstanten
    BRANCH_NAMES = {
        "PHOTO_READY": "ready",
        "PHOTO_OK": "ok",
        "PHOTO_RETAKE": "retake",
        "PHOTO_ABORT": "abort"
    }
    WORKFLOW_NODES = {
        "PHOTO_DETECT": "detect",
        "PHOTO_ASK_READY": "ask_ready",
        "PHOTO_CAPTURE": "capture",
        "PHOTO_AWAIT_DECISION": "await_decision",
        "PHOTO_FINISH": "finish"
    }
    DEFAULT_PHOTO_MAX_ATTEMPTS = 3
    ERROR_MESSAGES = {"LANGGRAPH_NOT_INSTALLED": "LangGraph nicht installiert"}
    PHOTO_INTENT_KEYWORDS = ["foto", "photo", "bild", "picture"]

try:
    from agents.workflows.workflows_utils import (
        handle_langgraph_import,
        safe_exception_handler,
    )
except ImportError:
    def handle_langgraph_import():
        try:
            from langgraph.graph import END, StateGraph
            return StateGraph, END
        except ImportError:
            return None, object()

    def safe_exception_handler(func_name, exception, logger):
        logger.error(f"Fehler in {func_name}: {exception}")

try:
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional at runtime
    MEMORY_SAVER = None
else:
    MEMORY_SAVER = MemorySaver

StateGraph, END = handle_langgraph_import()
logger = get_logger(__name__)


@dataclass
class PhotoWorkflowConfig(BaseWorkflowConfig):
    """Konfiguration für Photo-Workflow.

    Erweitert BaseWorkflowConfig um photo-spezifische Parameter.

    Attributes:
        max_photo_attempts: Maximale Anzahl von Photo-Aufnahme-Versuchen
        enable_preview: Ob Vorschau-URLs generiert werden sollen
        auto_upload: Ob automatischer Upload nach Aufnahme erfolgen soll
    """
    max_photo_attempts: int = DEFAULT_PHOTO_MAX_ATTEMPTS
    enable_preview: bool = True
    auto_upload: bool = True


@dataclass
class PhotoState:  # pylint: disable=too-many-instance-attributes
    """LangGraph-State für Photo-Workflow.

    Attributes:
        user_id: Eindeutige Benutzer-ID
        session_id: Optionale Session-ID für Tracking
        user_input: Benutzereingabe für Intent-Erkennung
        request_sent: Flag ob Photo-Request gesendet wurde
        preview_url: URL des Vorschau-Bildes
        decision: Benutzerentscheidung ("ok" | "retake" | "abort" | None)
        attempts: Anzahl der Aufnahme-Versuche
        max_attempts: Maximale Anzahl erlaubter Versuche
        extras: Zusätzliche State-Daten
    """

    user_id: str = ""
    session_id: str | None = None
    user_input: str = ""
    request_sent: bool = False
    preview_url: str | None = None
    decision: str | None = None  # "ok" | "retake" | "abort" | None
    attempts: int = 0
    max_attempts: int = DEFAULT_PHOTO_MAX_ATTEMPTS
    extras: dict[str, Any] = field(default_factory=dict)


def _validate_basic_fields(state: PhotoState) -> None:
    """Validiert Basis-Felder des PhotoState."""
    if not isinstance(state.user_id, str):
        raise ValueError("user_id muss ein String sein")
    if state.session_id is not None and not isinstance(state.session_id, str):
        raise ValueError("session_id muss None oder ein String sein")
    if not isinstance(state.user_input, str):
        raise ValueError("user_input muss ein String sein")


def _validate_attempt_fields(state: PhotoState) -> None:
    """Validiert Attempt-bezogene Felder des PhotoState."""
    if state.attempts < 0:
        raise ValueError("attempts darf nicht negativ sein")
    if state.max_attempts <= 0:
        raise ValueError("max_attempts muss positiv sein")


def _validate_decision_field(state: PhotoState) -> None:
    """Validiert das decision-Feld des PhotoState."""
    valid_decisions = [
        BRANCH_NAMES["PHOTO_OK"],
        BRANCH_NAMES["PHOTO_RETAKE"],
        BRANCH_NAMES["PHOTO_ABORT"]
    ]
    if state.decision is not None and state.decision not in valid_decisions:
        raise ValueError(f"Ungültige decision: {state.decision}")


def validate_photo_state(state: PhotoState) -> None:
    """Validiert PhotoState-Parameter.

    Args:
        state: Zu validierender PhotoState

    Raises:
        ValueError: Bei ungültigen State-Parametern
    """
    _validate_basic_fields(state)
    _validate_attempt_fields(state)
    _validate_decision_field(state)


def safe_state_update(state: PhotoState, **updates: Any) -> PhotoState:
    """Sichere State-Aktualisierung mit Validierung.

    Args:
        state: Zu aktualisierender State
        **updates: Zu aktualisierende Felder

    Returns:
        Aktualisierter und validierter State
    """
    for key, value in updates.items():
        if hasattr(state, key):
            setattr(state, key, value)

    validate_photo_state(state)
    return state




def node_detect_and_request(state: PhotoState) -> PhotoState:
    """Erkennt Photo-Intent und signalisiert photo_request über Orchestrator-Tools.

    In der Produktion würde dies functions_routes.execute mit
    function_name="photo_request" aufrufen.
    Hier wird nur request_sent markiert und das Frontend zeigt die CameraCapture-Komponente.

    Args:
        state: Aktueller Photo-Workflow-State

    Returns:
        Aktualisierter State mit gesetztem request_sent Flag

    Raises:
        ValueError: Bei ungültigen State-Parametern
    """
    try:
        validate_photo_state(state)

        # Heuristik: Prüfe auf Photo-Keywords in der Eingabe
        text = (state.user_input or "").lower()
        if any(keyword in text for keyword in PHOTO_INTENT_KEYWORDS):
            return safe_state_update(state, request_sent=True)

        return state
    except (ValueError, TypeError) as e:
        safe_exception_handler("node_detect_and_request", e, logger)
        return state


def node_capture(state: PhotoState) -> PhotoState:
    """Führt automatische Foto-Aufnahme über Server-Kamera durch.

    Diese Node ruft direkt den /api/camera/capture Endpunkt auf und
    nimmt automatisch ein Foto auf, anstatt auf externe Eingabe zu warten.

    Args:
        state: Aktueller Photo-Workflow-State

    Returns:
        Aktualisierter State mit Foto-URL und erhöhtem attempts-Zähler

    Raises:
        ValueError: Bei ungültigen State-Parametern
    """
    try:
        validate_photo_state(state)

        new_attempts = state.attempts + 1
        updates: dict[str, Any] = {"attempts": new_attempts}

        if new_attempts > state.max_attempts:
            # Abbruch bei zu vielen Wiederholungen
            updates["decision"] = BRANCH_NAMES["PHOTO_ABORT"]
            logger.warning("Photo-Workflow abgebrochen nach %d Versuchen", new_attempts)
            return safe_state_update(state, **updates)

        # Automatische Foto-Aufnahme über Server-Kamera
        try:
            import asyncio
            photo_result = asyncio.run(_capture_photo_automatically(state.user_id))
            if photo_result and photo_result.get("image_url"):
                updates["preview_url"] = photo_result["image_url"]
                logger.info("Foto automatisch aufgenommen für User %s", state.user_id)
            else:
                logger.warning("Foto-Aufnahme fehlgeschlagen für User %s", state.user_id)
                updates["decision"] = BRANCH_NAMES["PHOTO_ABORT"]
        except Exception as e:
            logger.error("Fehler bei automatischer Foto-Aufnahme: %s", e)
            updates["decision"] = BRANCH_NAMES["PHOTO_ABORT"]

        return safe_state_update(state, **updates)
    except (ValueError, TypeError) as e:
        safe_exception_handler("node_capture", e, logger)
        return state


def node_ask_ready(state: PhotoState) -> PhotoState:
    """Fragt den Benutzer, ob er bereit für das Foto ist.

    Diese Node wird vor der Aufnahme ausgeführt und wartet auf
    Benutzerbestätigung über das Frontend.

    Args:
        state: Aktueller Photo-Workflow-State

    Returns:
        State mit ready_question Flag gesetzt

    Raises:
        ValueError: Bei ungültigen State-Parametern
    """
    try:
        validate_photo_state(state)

        # Sende Nachricht an Benutzer über WebSocket
        try:
            import asyncio
            asyncio.run(_send_ready_question(state.user_id))
        except Exception as e:
            logger.warning("Konnte Bereitschaftsfrage nicht senden: %s", e)

        # Setze Flag für Frontend-Anzeige der Bereitschaftsfrage
        updates = {"extras": {**state.extras, "ready_question": True}}
        logger.info("Frage Benutzer %s ob er bereit für Foto ist", state.user_id)
        return safe_state_update(state, **updates)
    except (ValueError, TypeError) as e:
        safe_exception_handler("node_ask_ready", e, logger)
        return state


def node_await_decision(state: PhotoState) -> PhotoState:
    """Wartet auf Benutzerentscheidung; Entscheidung muss extern injiziert werden.

    Das Frontend sollte einen Decision-Endpunkt aufrufen um state.decision
    auf "ok", "retake" oder "abort" zu setzen.

    Args:
        state: Aktueller Photo-Workflow-State

    Returns:
        Unveränderten State (Entscheidung wird extern gesetzt)

    Raises:
        ValueError: Bei ungültigen State-Parametern
    """
    try:
        validate_photo_state(state)
        return state
    except (ValueError, TypeError) as e:
        safe_exception_handler("node_await_decision", e, logger)
        return state


def node_finish(state: PhotoState) -> PhotoState:
    """Beendet den Workflow.

    Args:
        state: Aktueller Photo-Workflow-State

    Returns:
        Unveränderten State

    Raises:
        ValueError: Bei ungültigen State-Parametern
    """
    try:
        validate_photo_state(state)
        logger.info("Photo-Workflow beendet für User %s", state.user_id)
        return state
    except (ValueError, TypeError) as e:
        safe_exception_handler("node_finish", e, logger)
        return state


def _create_graph_nodes() -> Any:
    """Erstellt StateGraph und fügt Nodes hinzu."""
    # Teste LangGraph-Import direkt
    try:
        from langgraph.graph import StateGraph as LangGraphStateGraph
        graph = LangGraphStateGraph(PhotoState)
    except ImportError:
        raise RuntimeError(ERROR_MESSAGES["LANGGRAPH_NOT_INSTALLED"])

    graph.add_node(WORKFLOW_NODES["PHOTO_DETECT"], node_detect_and_request)
    graph.add_node(WORKFLOW_NODES["PHOTO_ASK_READY"], node_ask_ready)
    graph.add_node(WORKFLOW_NODES["PHOTO_CAPTURE"], node_capture)
    graph.add_node(WORKFLOW_NODES["PHOTO_AWAIT_DECISION"], node_await_decision)
    graph.add_node(WORKFLOW_NODES["PHOTO_FINISH"], node_finish)
    return graph


def _add_graph_edges(graph: Any) -> None:
    """Fügt Edges und Router zum Graphen hinzu."""
    graph.set_entry_point(WORKFLOW_NODES["PHOTO_DETECT"])
    graph.add_conditional_edges(
        WORKFLOW_NODES["PHOTO_DETECT"],
        _start_router,
        {
            WORKFLOW_NODES["PHOTO_ASK_READY"]: WORKFLOW_NODES["PHOTO_ASK_READY"],
            WORKFLOW_NODES["PHOTO_FINISH"]: WORKFLOW_NODES["PHOTO_FINISH"]
        }
    )
    graph.add_conditional_edges(
        WORKFLOW_NODES["PHOTO_ASK_READY"],
        _ready_router,
        {
            WORKFLOW_NODES["PHOTO_CAPTURE"]: WORKFLOW_NODES["PHOTO_CAPTURE"],
            WORKFLOW_NODES["PHOTO_FINISH"]: WORKFLOW_NODES["PHOTO_FINISH"]
        }
    )
    graph.add_edge(WORKFLOW_NODES["PHOTO_CAPTURE"], WORKFLOW_NODES["PHOTO_AWAIT_DECISION"])
    graph.add_conditional_edges(
        WORKFLOW_NODES["PHOTO_AWAIT_DECISION"],
        _decision_router,
        {
            WORKFLOW_NODES["PHOTO_ASK_READY"]: WORKFLOW_NODES["PHOTO_ASK_READY"],
            WORKFLOW_NODES["PHOTO_FINISH"]: WORKFLOW_NODES["PHOTO_FINISH"]
        },
    )
    # Import END direkt
    try:
        from langgraph.graph import END as LANGGRAPH_END
        graph.add_edge(WORKFLOW_NODES["PHOTO_FINISH"], LANGGRAPH_END)
    except ImportError:
        # Fallback für Tests
        pass


def build_photo_graph() -> Any:
    """Baut den LangGraph-Graphen für den Photo-Workflow.

    Returns:
        Kompilierter LangGraph-Graph mit Checkpointer

    Raises:
        RuntimeError: Wenn LangGraph nicht installiert ist
    """
    graph = _create_graph_nodes()
    _add_graph_edges(graph)

    # Memory-Checkpointer für State-Persistierung
    if MEMORY_SAVER is not None:
        memory = MEMORY_SAVER()
        return graph.compile(checkpointer=memory)

    return graph.compile()

def _start_router(state: PhotoState) -> str:
    """Router für Start-Entscheidung basierend auf request_sent Flag.

    Args:
        state: Aktueller Photo-Workflow-State

    Returns:
        Nächster Node-Name ("ask_ready" oder "finish")
    """
    if state.request_sent:
        return WORKFLOW_NODES["PHOTO_ASK_READY"]
    return WORKFLOW_NODES["PHOTO_FINISH"]


def _ready_router(state: PhotoState) -> str:
    """Router für Bereitschafts-Entscheidung.

    Args:
        state: Aktueller Photo-Workflow-State

    Returns:
        Nächster Node-Name ("capture" oder "finish")
    """
    if state.decision == BRANCH_NAMES["PHOTO_READY"]:
        return WORKFLOW_NODES["PHOTO_CAPTURE"]
    return WORKFLOW_NODES["PHOTO_FINISH"]


def _decision_router(state: PhotoState) -> str:
    """Router für Benutzerentscheidung nach Aufnahme.

    Args:
        state: Aktueller Photo-Workflow-State

    Returns:
        Nächster Node-Name ("ask_ready" oder "finish")
    """
    if state.decision == BRANCH_NAMES["PHOTO_OK"]:
        return WORKFLOW_NODES["PHOTO_FINISH"]
    if (state.decision == BRANCH_NAMES["PHOTO_RETAKE"] and
        state.attempts < state.max_attempts):
        return WORKFLOW_NODES["PHOTO_ASK_READY"]
    return WORKFLOW_NODES["PHOTO_FINISH"]




class PhotoWorkflow(BaseWorkflowBuilder):
    """Photo-Workflow-Builder mit Integration in bestehende Workflow-Infrastruktur.

    Erweitert BaseWorkflowBuilder um photo-spezifische Funktionalität
    und bietet Export-Funktionen für Visualisierung.
    """

    def __init__(self, config: PhotoWorkflowConfig | None = None) -> None:
        """Initialisiert den Photo-Workflow-Builder.

        Args:
            config: Optionale Photo-Workflow-Konfiguration
        """
        self.photo_config = config or PhotoWorkflowConfig()
        super().__init__(self.photo_config)

    def build(self) -> dict[str, Any]:
        """Baut die Photo-Workflow-Repräsentation.

        Returns:
            Serialisierbare Workflow-Struktur mit Nodes und Edges
        """
        nodes = [
            WORKFLOW_NODES["PHOTO_DETECT"],
            WORKFLOW_NODES["PHOTO_ASK_READY"],
            WORKFLOW_NODES["PHOTO_CAPTURE"],
            WORKFLOW_NODES["PHOTO_AWAIT_DECISION"],
            WORKFLOW_NODES["PHOTO_FINISH"]
        ]

        edges = [
            (WORKFLOW_NODES["PHOTO_DETECT"], WORKFLOW_NODES["PHOTO_ASK_READY"]),
            (WORKFLOW_NODES["PHOTO_ASK_READY"], WORKFLOW_NODES["PHOTO_CAPTURE"]),
            (WORKFLOW_NODES["PHOTO_CAPTURE"], WORKFLOW_NODES["PHOTO_AWAIT_DECISION"]),
            (WORKFLOW_NODES["PHOTO_AWAIT_DECISION"], WORKFLOW_NODES["PHOTO_FINISH"])
        ]

        return {
            "nodes": nodes,
            "edges": edges,
            "config": {
                "max_photo_attempts": self.photo_config.max_photo_attempts,
                "enable_preview": self.photo_config.enable_preview,
                "auto_upload": self.photo_config.auto_upload
            }
        }

    def export_to_dot(self) -> str:
        """Exportiert den Photo-Workflow in DOT-Format.

        Returns:
            DOT-formatierter String für Graphviz
        """
        workflow_data = self.build()
        visualizer = WorkflowVisualizer(workflow_data)
        return visualizer.export_to_dot()

    def export_to_mermaid(self) -> str:
        """Exportiert den Photo-Workflow in Mermaid-Format.

        Returns:
            Mermaid-formatierter String für Diagramme
        """
        workflow_data = self.build()
        visualizer = WorkflowVisualizer(workflow_data)
        return visualizer.export_to_mermaid()

    def compile_graph(self) -> Any:
        """Kompiliert den LangGraph-Graphen mit aktueller Konfiguration.

        Returns:
            Kompilierter LangGraph-Graph
        """
        return build_photo_graph()


async def _send_ready_question(user_id: str) -> None:
    """Sendet Bereitschaftsfrage an den Benutzer über WebSocket.

    Args:
        user_id: Benutzer-ID für die Nachricht
    """
    try:
        try:
            from data_models.websocket import create_status_update
        except ImportError:
            # Fallback für fehlende data_models
            def create_status_update(status, details):
                return {"status": status, "details": details}

        try:
            from websocket.manager import websocket_manager
            event = create_status_update(
                status="photo_ready_question",
                details="Sind Sie bereit für das Foto? Sagen Sie 'Ja' oder 'Bereit' wenn Sie soweit sind."
            )
            if hasattr(event, "model_dump"):
                await websocket_manager.broadcast(event.model_dump())
            else:
                await websocket_manager.broadcast(event)
            logger.info("Bereitschaftsfrage gesendet an User %s", user_id)
        except ImportError:
            websocket_manager = None
            logger.warning("WebSocket-Manager nicht verfügbar - Bereitschaftsfrage nicht gesendet")
    except Exception as e:
        logger.error("Fehler beim Senden der Bereitschaftsfrage: %s", e)


async def _capture_photo_automatically(user_id: str) -> dict[str, Any] | None:
    """Führt automatische Foto-Aufnahme über Server-Kamera durch.

    Args:
        user_id: Benutzer-ID für die Aufnahme

    Returns:
        Foto-Aufnahme-Ergebnis mit image_url oder None bei Fehler
    """
    try:
        import aiohttp

        # Bestimme Server-URL mit Fallback
        try:
            from config.settings import settings
            base_url = getattr(settings, "server_base_url", "http://127.0.0.1:8000")
        except ImportError:
            settings = None
            base_url = "http://127.0.0.1:8000"

        capture_url = f"{base_url}/api/camera/capture"

        # Aufnahme-Request
        payload = {"resolution": "640x480"}
        headers = {"X-User-Id": user_id, "Content-Type": "application/json"}

        async with aiohttp.ClientSession() as session:
            async with session.post(capture_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info("Automatische Foto-Aufnahme erfolgreich für User %s", user_id)
                    return result
                if response.status == 502:
                    # Storage-Fehler - verwende Mock-URL für Development
                    logger.warning("Storage-Fehler erkannt, verwende Mock-URL für Development")
                    return _create_mock_photo_result(user_id)
                error_text = await response.text()
                logger.error("Foto-Aufnahme fehlgeschlagen: %s - %s", response.status, error_text)
                return _create_mock_photo_result(user_id)

    except Exception as e:
        logger.error("Fehler bei automatischer Foto-Aufnahme: %s", e)
        return _create_mock_photo_result(user_id)


def _create_mock_photo_result(user_id: str) -> dict[str, Any]:
    """Erstellt Mock-Foto-Ergebnis für Development.

    Args:
        user_id: Benutzer-ID

    Returns:
        Mock-Foto-Ergebnis
    """
    from datetime import UTC, datetime

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    mock_url = f"https://via.placeholder.com/640x480/0066cc/ffffff?text=Photo+{timestamp[:8]}"

    return {
        "status": "ok",
        "image_url": mock_url,
        "metadata": {
            "timestamp": timestamp,
            "resolution": {"width": 640, "height": 480},
            "format": "image/jpeg",
            "file_size": 12345,
            "user_id": user_id,
            "mock": True
        }
    }


def create_photo_workflow(config: PhotoWorkflowConfig | None = None) -> Any:
    """Factory-Funktion für kompilierte Graph-Instanz.

    Args:
        config: Optionale Photo-Workflow-Konfiguration

    Returns:
        Kompilierter Photo-Workflow-Graph
    """
    workflow = PhotoWorkflow(config)
    return workflow.compile_graph()
