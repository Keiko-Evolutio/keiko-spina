# backend/api/routes/common.py
"""Gemeinsame Funktionalität für alle Route-Module."""


# Standard Error-Responses für alle Router
STANDARD_RESPONSES = {
    400: {"description": "Ungültige Anfrage"},
    404: {"description": "Ressource nicht gefunden"},
    500: {"description": "Interner Server-Fehler"}
}

# Spezifische Error-Responses
AGENT_RESPONSES = {
    **STANDARD_RESPONSES,
    404: {"description": "Agent nicht gefunden"},
    500: {"description": "Agent-Ausführungsfehler"}
}

CHAT_RESPONSES = {
    **STANDARD_RESPONSES,
    404: {"description": "Chat-Sitzung nicht gefunden"}
}

FUNCTION_RESPONSES = {
    **STANDARD_RESPONSES,
    404: {"description": "Funktion nicht gefunden"},
    500: {"description": "Funktions-Ausführungsfehler"}
}

HEALTH_RESPONSES = {
    503: {"description": "Service nicht verfügbar"},
    500: {"description": "Interner Health Check Fehler"}
}
