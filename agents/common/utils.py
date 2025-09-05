"""Gemeinsame Utility-Funktionen für Agent-Operationen."""

from typing import Any


def extract_token_usage(result: Any) -> tuple[int, int]:
    """Extrahiert Token-Nutzung aus Ergebnisobjekt.

    Unterstützt verschiedene Token-Usage-Formate:
    - OpenAI-Format: {"usage": {"prompt_tokens": X, "completion_tokens": Y}}
    - Alternative Formate: {"token_usage": {...}}, {"input_tokens": X, "output_tokens": Y}

    Args:
        result: Ergebnisobjekt mit potentieller Token-Usage-Information

    Returns:
        Tuple mit (prompt_tokens, completion_tokens). Bei Fehlern (0, 0).
    """
    try:
        if isinstance(result, dict):
            usage = result.get("usage") or result.get("token_usage") or {}
            prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            return prompt, completion
    except (ValueError, TypeError, AttributeError):
        pass
    return 0, 0


def extract_framework_from_agent_id(agent_id: str) -> tuple[str | None, str]:
    """Extrahiert Framework aus Agent-ID im Format 'framework_agentid'.

    Args:
        agent_id: Agent-ID, möglicherweise mit Framework-Präfix

    Returns:
        Tuple mit (framework, clean_agent_id). Framework ist None wenn nicht erkennbar.
    """
    if "_" in agent_id:
        parts = agent_id.split("_", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    return None, agent_id


def create_agent_key(framework: str, agent_id: str) -> str:
    """Erstellt einheitlichen Agent-Key für Caching/Registry.

    Args:
        framework: Framework-Name
        agent_id: Agent-ID

    Returns:
        Einheitlicher Key im Format 'framework_agentid'
    """
    return f"{framework}_{agent_id}"


def extract_instruction_from_kwargs(kwargs: dict[str, Any]) -> str:
    """Ermittelt Instruktion/Query/Task aus kwargs für Monitoring.

    Args:
        kwargs: Keyword-Argumente mit potentieller Instruktion

    Returns:
        Instruktions-String oder leerer String
    """
    return kwargs.get("instruction") or kwargs.get("query") or kwargs.get("task") or ""


def is_error_result(result: Any) -> bool:
    """Prüft ob Ergebnis einen Fehler repräsentiert.

    Args:
        result: Zu prüfendes Ergebnis

    Returns:
        True wenn Ergebnis einen Fehler repräsentiert
    """
    if isinstance(result, dict):
        return result.get("error") is not None or result.get("success") is False
    return False


def create_error_result(error_message: str, **additional_fields: Any) -> dict[str, Any]:
    """Erstellt standardisiertes Error-Result-Dictionary.

    Args:
        error_message: Fehlermeldung
        **additional_fields: Zusätzliche Felder für das Error-Result

    Returns:
        Standardisiertes Error-Dictionary
    """
    result = {
        "error": error_message,
        "success": False,
        **additional_fields
    }
    return result


def create_success_result(data: Any = None, **additional_fields: Any) -> dict[str, Any]:
    """Erstellt standardisiertes Success-Result-Dictionary.

    Args:
        data: Erfolgs-Daten
        **additional_fields: Zusätzliche Felder für das Success-Result

    Returns:
        Standardisiertes Success-Dictionary
    """
    result = {
        "success": True,
        **additional_fields
    }
    if data is not None:
        result["data"] = data
    return result
