"""Orchestrator Tools Konfiguration.

Stellt die Tool-Definitionen für den Orchestrator-Agent bereit.
"""

from data_models import Function, FunctionParameter

from .agent_operations import (
    analyze_and_maybe_generate_image as _analyze_and_maybe_generate_image,
)
from .agent_operations import (
    analyze_and_maybe_take_photo as _analyze_and_maybe_take_photo,
)
from .agent_operations import (
    delegate_to_agent_implementation,
    delegate_to_best_agent_implementation,
    discover_agents_implementation,
    get_agent_details_implementation,
    monitor_execution_implementation,
)
from .agent_operations import (
    generate_image_implementation as _generate_image_implementation,
)
from .agent_operations import (
    perform_web_research_implementation as _perform_web_research_implementation,
)
from .agent_operations import (
    photo_request_implementation as _photo_request_implementation,
)


def get_orchestrator_tools() -> dict[str, Function]:
    """Stellt Orchestrator-Tools bereit."""
    return {
        "discover_available_agents": Function(
            name="discover_available_agents",
            description="Entdeckt verfügbare Agents basierend auf Aufgabenbeschreibung und Anforderungen",
            parameters=[
                FunctionParameter(
                    "task_description", "string", "Beschreibung der auszuführenden Aufgabe"
                ),
                FunctionParameter("required_capabilities", "array", "Benötigte Agent-Fähigkeiten"),
                FunctionParameter("preferred_category", "string", "Bevorzugte Agent-Kategorie"),
            ],
            func=discover_agents_implementation,
        ),
        "delegate_to_agent": Function(
            name="delegate_to_agent",
            description="Delegiert eine Aufgabe an einen spezifischen Agent",
            parameters=[
                FunctionParameter("agent_id", "string", "ID des Ziel-Agents"),
                FunctionParameter(
                    "task_specification",
                    "object",
                    "Task-Spezifikation mit instruction und priority",
                ),
                FunctionParameter("execution_mode", "string", "Ausführungsmodus: async oder sync"),
            ],
            func=delegate_to_agent_implementation,
        ),
        "delegate_to_best_agent": Function(
            name="delegate_to_best_agent",
            description="Wählt automatisch den besten Agent für eine Aufgabe aus und delegiert sie",
            parameters=[
                FunctionParameter(
                    "task_description", "string", "Beschreibung der auszuführenden Aufgabe"
                ),
                FunctionParameter(
                    "selection_criteria", "object", "Auswahlkriterien für Agent-Selektion"
                ),
            ],
            func=delegate_to_best_agent_implementation,
        ),
        "get_agent_details": Function(
            name="get_agent_details",
            description="Ruft detaillierte Informationen über einen spezifischen Agent ab",
            parameters=[FunctionParameter("agent_id", "string", "ID des abzufragenden Agents")],
            func=get_agent_details_implementation,
        ),
        "monitor_execution_status": Function(
            name="monitor_execution_status",
            description="Überwacht den Status einer laufenden Agent-Ausführung",
            parameters=[
                FunctionParameter("execution_id", "string", "ID der zu überwachenden Ausführung"),
                FunctionParameter("include_logs", "boolean", "Logs in Antwort einschließen"),
            ],
            func=monitor_execution_implementation,
        ),
        # Neues Tool: Web Research via Deep Research API
        "perform_web_research": Function(
            name="perform_web_research",
            description="Führt umfassende Web-Recherche zu einer gegebenen Frage durch",
            parameters=[
                FunctionParameter("query", "string", "Recherchefrage des Benutzers"),
                FunctionParameter(
                    "max_iterations", "integer", "Maximale Recherche-Iterationen", False, 3
                ),
                FunctionParameter(
                    "context", "object", "Zusätzlicher Kontext für die Recherche", False, {}
                ),
            ],
            func=_perform_web_research_implementation,  # type: ignore[name-defined]
        ),
        # Neues Tool: Bildgenerierung
        "generate_image": Function(
            name="generate_image",
            description="Generiert ein Bild basierend auf einer Textbeschreibung",
            parameters=[
                FunctionParameter("prompt", "string", "Beschreibung des zu generierenden Bildes"),
                FunctionParameter(
                    "size",
                    "string",
                    "Bildgröße (1024x1024|1024x1792|1792x1024)",
                    False,
                    "1024x1024",
                ),
                FunctionParameter("quality", "string", "Qualität (standard|hd)", False, "standard"),
                FunctionParameter(
                    "style",
                    "string",
                    "Stil (Realistic|Artistic|Cartoon|Photography|Digital Art)",
                    False,
                    "Realistic",
                ),
                FunctionParameter("user_id", "string", "Benutzer-ID", False, None),
                FunctionParameter("session_id", "string", "Sitzungs-ID", False, None),
            ],
            func=_generate_image_implementation,  # type: ignore[name-defined]
        ),
        # Neues Tool: Fotoanfrage (aktiviert Fotofunktion auf Nachfrage)
        "photo_request": Function(
            name="photo_request",
            description="Aktiviert die Fotofunktion für den Benutzer",
            parameters=[
                FunctionParameter("user_id", "string", "Benutzer-ID", False, None),
            ],
            func=_photo_request_implementation,  # type: ignore[name-defined]
        ),
        # Intent-Analyse und ggf. Rückfrage für Bildgenerierung
        "analyze_and_maybe_generate_image": Function(
            name="analyze_and_maybe_generate_image",
            description="Analysiert Benutzereingabe und generiert bei Bedarf ein Bild",
            parameters=[
                FunctionParameter("user_input", "string", "Nutzereingabe (Freitext)"),
                FunctionParameter("user_id", "string", "Benutzer-ID", False, None),
                FunctionParameter("session_id", "string", "Sitzungs-ID", False, None),
            ],
            func=_analyze_and_maybe_generate_image,  # type: ignore[name-defined]
        ),
        # Intent-Analyse und ggf. Foto-Aufnahme
        "analyze_and_maybe_take_photo": Function(
            name="analyze_and_maybe_take_photo",
            description="Analysiert Benutzereingabe und nimmt bei Bedarf ein Foto auf",
            parameters=[
                FunctionParameter("user_input", "string", "Nutzereingabe (Freitext)"),
                FunctionParameter("user_id", "string", "Benutzer-ID", False, None),
                FunctionParameter("session_id", "string", "Sitzungs-ID", False, None),
            ],
            func=_analyze_and_maybe_take_photo,  # type: ignore[name-defined]
        ),
    }
