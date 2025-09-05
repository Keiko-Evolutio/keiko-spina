"""Idempotenz- und Deduplikations-Store für KEI-Bus.

Konsolidierte Implementierung, die die robuste task_management Idempotency-Engine verwendet.
"""

from __future__ import annotations

from kei_logging import get_logger
from task_management.idempotency_manager import (
    DuplicateDetectionStrategy,
    idempotency_manager,
)

logger = get_logger(__name__)

# Konstanten für Backward-Compatibility
DEFAULT_TTL_SECONDS = 3600


async def is_duplicate(operation: str, key: str) -> bool:
    """Prüft, ob eine Operation mit Key bereits verarbeitet wurde.

    Args:
        operation: Operation-Typ (z.B. "publish", "kafka_publish")
        key: Eindeutiger Key für die Operation

    Returns:
        True wenn Duplikat, False sonst
    """
    try:
        # Verwende robuste task_management Implementierung
        request_data = {"operation": operation, "key": key}
        result = await idempotency_manager.check_duplicate(
            request_data=request_data,
            idempotency_key=f"{operation}:{key}",
            strategy=DuplicateDetectionStrategy.EXACT_MATCH
        )
        return result.is_duplicate
    except Exception as exc:  # pragma: no cover - defensiv
        logger.debug("Idempotency check failed: %s", exc)
        return False


async def remember(operation: str, key: str, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
    """Speichert Idempotenz-Key für TTL-Periode.

    Args:
        operation: Operation-Typ (z.B. "publish", "kafka_publish")
        key: Eindeutiger Key für die Operation
        ttl_seconds: Time-to-Live in Sekunden
    """
    try:
        # Verwende robuste task_management Implementierung
        request_data = {"operation": operation, "key": key}
        idempotency_key = f"{operation}:{key}"

        # Cache Request für Idempotenz
        await idempotency_manager.cache_request(
            request_id=idempotency_key,
            idempotency_key=idempotency_key,
            request_data=request_data,
            ttl_seconds=ttl_seconds
        )

        # Markiere als completed für zukünftige Duplicate-Checks
        await idempotency_manager.update_response(
            request_id=idempotency_key,
            response_data={"status": "processed"},
            response_status="success"
        )
    except Exception as exc:  # pragma: no cover - defensiv
        logger.debug("Idempotency remember failed: %s", exc)
