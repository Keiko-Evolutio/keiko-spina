"""KEI MCP Utility-Funktionen.

Gemeinsame Hilfsfunktionen für das KEI MCP System:
- ID-Generierung
- URL-Validierung
- String-Sanitization
- Error-Formatting
"""

import hashlib
import re
import uuid
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin, urlparse

from kei_logging import get_logger

from .constants import VALIDATION_PATTERNS

# Logger für dieses Modul
logger = get_logger(__name__)


def generate_correlation_id() -> str:
    """Generiert eindeutige Correlation-ID für Request-Tracking."""
    return str(uuid.uuid4())


def generate_request_id() -> str:
    """Generiert eindeutige Request-ID."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    random_part = str(uuid.uuid4())[:8]
    return f"req_{timestamp}_{random_part}"


def sanitize_server_name(server_name: str) -> str:
    """Sanitisiert Server-Namen für sichere Verwendung.

    Args:
        server_name: Ursprünglicher Server-Name

    Returns:
        Sanitisierter Server-Name

    Raises:
        ValueError: Wenn Server-Name ungültig ist
    """
    if not server_name:
        raise ValueError("Server-Name darf nicht leer sein")

    # Entferne führende/nachfolgende Whitespaces
    sanitized = server_name.strip()

    # Prüfe gegen Pattern
    if not re.match(VALIDATION_PATTERNS["SERVER_NAME"], sanitized):
        raise ValueError(
            f"Ungültiger Server-Name: {server_name}. "
            f"Erlaubt sind nur alphanumerische Zeichen, Unterstriche und Bindestriche."
        )

    return sanitized


def validate_url(url: str) -> bool:
    """Validiert URL-Format.

    Args:
        url: Zu validierende URL

    Returns:
        True wenn URL gültig ist
    """
    if not url:
        return False

    try:
        # Basic URL-Pattern-Check
        if not re.match(VALIDATION_PATTERNS["URL"], url):
            return False

        # Detaillierte URL-Parsing
        parsed = urlparse(url)

        # Prüfe Schema
        if parsed.scheme not in ("http", "https"):
            return False

        # Prüfe Hostname
        return parsed.netloc

    except Exception:
        return False


def normalize_url(base_url: str, endpoint: str) -> str:
    """Normalisiert URL durch Kombination von Base-URL und Endpoint.

    Args:
        base_url: Basis-URL
        endpoint: Endpoint-Pfad

    Returns:
        Vollständige normalisierte URL
    """
    # Entferne trailing slash von base_url
    base_url = base_url.rstrip("/")

    # Stelle sicher, dass endpoint mit / beginnt
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    return urljoin(base_url, endpoint)


def format_error_message(
    error: Exception,
    context: dict[str, Any] | None = None
) -> str:
    """Formatiert Error-Message mit Kontext-Informationen.

    Args:
        error: Exception-Objekt
        context: Zusätzliche Kontext-Informationen

    Returns:
        Formatierte Error-Message
    """
    parts = [str(error)]

    if context:
        context_parts = []

        if context.get("server_name"):
            context_parts.append(f"Server: {context['server_name']}")

        if context.get("tool_name"):
            context_parts.append(f"Tool: {context['tool_name']}")

        if context.get("request_id"):
            context_parts.append(f"Request: {context['request_id']}")

        if context_parts:
            parts.append(f"[{', '.join(context_parts)}]")

    return " ".join(parts)


def calculate_hash(data: str | bytes | dict[str, Any]) -> str:
    """Berechnet SHA-256 Hash für gegebene Daten.

    Args:
        data: Daten für Hash-Berechnung

    Returns:
        Hexadezimaler Hash-String
    """
    if isinstance(data, dict):
        # Sortiere Dictionary für konsistente Hashes
        import json
        data_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
        data_bytes = data_str.encode("utf-8")
    elif isinstance(data, str):
        data_bytes = data.encode("utf-8")
    else:
        data_bytes = data

    return hashlib.sha256(data_bytes).hexdigest()


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Kürzt String auf maximale Länge.

    Args:
        text: Zu kürzender Text
        max_length: Maximale Länge
        suffix: Suffix für gekürzte Strings

    Returns:
        Gekürzter String
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def deep_merge_dicts(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    """Führt zwei Dictionaries rekursiv zusammen.

    Args:
        dict1: Erstes Dictionary
        dict2: Zweites Dictionary (überschreibt dict1)

    Returns:
        Zusammengeführtes Dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def extract_domain_from_url(url: str) -> str | None:
    """Extrahiert Domain aus URL.

    Args:
        url: URL

    Returns:
        Domain oder None bei ungültiger URL
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except (ValueError, TypeError) as e:
        logger.debug(f"URL-Domain-Extraktion fehlgeschlagen - Format-/Typ-Fehler: {e}")
        return None
    except Exception as e:
        logger.warning(f"URL-Domain-Extraktion fehlgeschlagen - Unerwarteter Fehler: {e}")
        return None


def is_valid_api_key(api_key: str) -> bool:
    """Validiert API-Key-Format.

    Args:
        api_key: Zu validierender API-Key

    Returns:
        True wenn API-Key gültig ist
    """
    if not api_key:
        return False

    return bool(re.match(VALIDATION_PATTERNS["API_KEY"], api_key))


def format_duration_ms(duration_seconds: float) -> str:
    """Formatiert Dauer in Millisekunden für Logging.

    Args:
        duration_seconds: Dauer in Sekunden

    Returns:
        Formatierte Dauer-String
    """
    duration_ms = duration_seconds * 1000

    if duration_ms < 1:
        return f"{duration_ms:.2f}ms"
    if duration_ms < 1000:
        return f"{duration_ms:.0f}ms"
    return f"{duration_seconds:.2f}s"


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Sicheres JSON-Parsing mit Fallback.

    Args:
        json_str: JSON-String
        default: Fallback-Wert bei Parse-Fehler

    Returns:
        Geparste Daten oder Fallback-Wert
    """
    try:
        import json
        return json.loads(json_str)
    except (ImportError, AttributeError):
        # Import failed or json module not available
        return default
    except Exception:
        # JSON parsing failed or other errors
        return default


def create_cache_key(*parts: str) -> str:
    """Erstellt Cache-Key aus mehreren Teilen.

    Args:
        parts: Teile für Cache-Key

    Returns:
        Cache-Key
    """
    # Sanitisiere und kombiniere Teile
    sanitized_parts = []
    for part in parts:
        if part:
            # Entferne problematische Zeichen
            sanitized = re.sub(r"[^\w\-.]", "_", str(part))
            sanitized_parts.append(sanitized)

    return ":".join(sanitized_parts)


def batch_items(items: list[Any], batch_size: int) -> list[list[Any]]:
    """Teilt Liste in Batches auf.

    Args:
        items: Liste der Items
        batch_size: Größe pro Batch

    Returns:
        Liste von Batches
    """
    if batch_size <= 0:
        raise ValueError("Batch-Größe muss positiv sein")

    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])

    return batches


def mask_sensitive_data(data: dict[str, Any], sensitive_keys: list[str] | None = None) -> dict[str, Any]:
    """Maskiert sensitive Daten für Logging.

    Args:
        data: Dictionary mit Daten
        sensitive_keys: Liste sensitiver Schlüssel

    Returns:
        Dictionary mit maskierten Daten
    """
    if sensitive_keys is None:
        sensitive_keys = ["api_key", "password", "token", "secret", "authorization"]

    masked_data = data.copy()

    for key, value in masked_data.items():
        if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
            if isinstance(value, str) and len(value) > 4:
                masked_data[key] = value[:2] + "*" * (len(value) - 4) + value[-2:]
            else:
                masked_data[key] = "***"

    return masked_data
