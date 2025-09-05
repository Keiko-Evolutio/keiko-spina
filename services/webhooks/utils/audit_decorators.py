"""Audit-Decorators für das Webhook-System.

Vereinfacht Audit-Logging durch Decorator-Pattern und eliminiert Duplicate Code.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from kei_logging import get_logger

from ..audit_logger import WebhookAuditEventType, WebhookAuditOperation, webhook_audit

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def audit_operation(
    event_type: WebhookAuditEventType,
    operation: WebhookAuditOperation,
    *,
    extract_correlation_id: Callable[[Any], str | None] | None = None,
    extract_delivery_id: Callable[[Any], str | None] | None = None,
    extract_target_id: Callable[[Any], str | None] | None = None,
    extract_tenant_id: Callable[[Any], str | None] | None = None,
    extract_event_type: Callable[[Any], str | None] | None = None,
    on_success: bool = True,
    on_failure: bool = True,
    suppress_errors: bool = True,
) -> Callable[[F], F]:
    """Decorator für automatisches Audit-Logging.

    Args:
        event_type: Typ des Audit-Events
        operation: Art der Operation
        extract_correlation_id: Funktion um correlation_id aus Argumenten zu extrahieren
        extract_delivery_id: Funktion um delivery_id aus Argumenten zu extrahieren
        extract_target_id: Funktion um target_id aus Argumenten zu extrahieren
        extract_tenant_id: Funktion um tenant_id aus Argumenten zu extrahieren
        extract_event_type: Funktion um event_type aus Argumenten zu extrahieren
        on_success: Audit-Log bei erfolgreichem Aufruf
        on_failure: Audit-Log bei Fehlern
        suppress_errors: Audit-Fehler unterdrücken (empfohlen)

    Returns:
        Decorator-Funktion
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Extrahiere Audit-Parameter
            audit_params = _extract_audit_params(
                args, kwargs,
                extract_correlation_id,
                extract_delivery_id,
                extract_target_id,
                extract_tenant_id,
                extract_event_type,
            )

            try:
                result = await func(*args, **kwargs)

                # Success Audit-Log
                if on_success:
                    await _log_audit_event(
                        event_type=event_type,
                        operation=operation,
                        success=True,
                        suppress_errors=suppress_errors,
                        **audit_params
                    )

                return result

            except Exception as exc:
                # Failure Audit-Log
                if on_failure:
                    await _log_audit_event(
                        event_type=event_type,
                        operation=operation,
                        success=False,
                        error_details={"error": str(exc), "type": type(exc).__name__},
                        suppress_errors=suppress_errors,
                        **audit_params
                    )
                raise

        return wrapper  # type: ignore[return-value]
    return decorator


def audit_delivery_operation(
    event_type: WebhookAuditEventType,
    operation: WebhookAuditOperation,
    **kwargs
) -> Callable[[F], F]:
    """Spezialisierter Decorator für Delivery-Operationen.

    Extrahiert automatisch Standard-Parameter aus DeliveryRecord, WebhookTarget, WebhookEvent.
    """
    return audit_operation(
        event_type=event_type,
        operation=operation,
        extract_correlation_id=lambda args, kwargs: _get_from_record(args, kwargs, "correlation_id"),
        extract_delivery_id=lambda args, kwargs: _get_from_record(args, kwargs, "delivery_id"),
        extract_target_id=lambda args, kwargs: _get_from_target(args, kwargs, "id"),
        extract_tenant_id=lambda args, kwargs: _get_from_event_meta(args, kwargs, "tenant"),
        extract_event_type=lambda args, kwargs: _get_from_event(args, kwargs, "event_type"),
        **kwargs
    )


def audit_target_operation(
    event_type: WebhookAuditEventType,
    operation: WebhookAuditOperation,
    **kwargs
) -> Callable[[F], F]:
    """Spezialisierter Decorator für Target-Operationen."""
    return audit_operation(
        event_type=event_type,
        operation=operation,
        extract_target_id=lambda args, kwargs: _get_from_target(args, kwargs, "id"),
        extract_tenant_id=lambda args, kwargs: _get_from_args(args, kwargs, "tenant_id"),
        **kwargs
    )


def _extract_audit_params(
    args: tuple,
    kwargs: dict,
    extract_correlation_id: Callable | None,
    extract_delivery_id: Callable | None,
    extract_target_id: Callable | None,
    extract_tenant_id: Callable | None,
    extract_event_type: Callable | None,
) -> dict[str, Any]:
    """Extrahiert Audit-Parameter aus Funktionsargumenten."""
    params = {}

    if extract_correlation_id:
        params["correlation_id"] = extract_correlation_id(args, kwargs)
    if extract_delivery_id:
        params["delivery_id"] = extract_delivery_id(args, kwargs)
    if extract_target_id:
        params["target_id"] = extract_target_id(args, kwargs)
    if extract_tenant_id:
        params["tenant_id"] = extract_tenant_id(args, kwargs)
    if extract_event_type:
        params["event_type"] = extract_event_type(args, kwargs)

    return {k: v for k, v in params.items() if v is not None}


async def _log_audit_event(
    event_type: WebhookAuditEventType,
    operation: WebhookAuditOperation,
    success: bool,
    suppress_errors: bool,
    error_details: dict[str, Any] | None = None,
    **params
) -> None:
    """Loggt Audit-Event mit Error-Handling."""
    if suppress_errors:
        with contextlib.suppress(Exception):
            await _do_audit_log(event_type, operation, success, error_details, **params)
    else:
        await _do_audit_log(event_type, operation, success, error_details, **params)


async def _do_audit_log(
    event_type: WebhookAuditEventType,
    operation: WebhookAuditOperation,
    success: bool,
    error_details: dict[str, Any] | None,
    **params
) -> None:
    """Führt das eigentliche Audit-Logging durch."""
    if success:
        # Success-spezifische Audit-Methoden
        if event_type == WebhookAuditEventType.OUTBOUND_DELIVERED:
            await webhook_audit.outbound_delivered(**params)
        elif event_type == WebhookAuditEventType.OUTBOUND_ENQUEUED:
            await webhook_audit.outbound_enqueued(**params)
        # Weitere Event-Types nach Bedarf
    # Failure-spezifische Audit-Methoden
    elif event_type == WebhookAuditEventType.OUTBOUND_FAILED:
        await webhook_audit.outbound_failed(
            error_details=error_details or {},
            will_retry=False,  # Default, kann überschrieben werden
            **params
        )


# Helper-Funktionen für Parameter-Extraktion
def _get_from_record(args: tuple, kwargs: dict, attr: str) -> str | None:
    """Extrahiert Attribut aus DeliveryRecord (erstes Argument)."""
    if args and hasattr(args[0], attr):
        return getattr(args[0], attr)
    if "record" in kwargs and hasattr(kwargs["record"], attr):
        return getattr(kwargs["record"], attr)
    return None


def _get_from_target(args: tuple, kwargs: dict, attr: str) -> str | None:
    """Extrahiert Attribut aus WebhookTarget (zweites Argument oder 'target')."""
    if len(args) > 1 and hasattr(args[1], attr):
        return getattr(args[1], attr)
    if "target" in kwargs and hasattr(kwargs["target"], attr):
        return getattr(kwargs["target"], attr)
    return None


def _get_from_event(args: tuple, kwargs: dict, attr: str) -> str | None:
    """Extrahiert Attribut aus WebhookEvent (drittes Argument oder 'event')."""
    if len(args) > 2 and hasattr(args[2], attr):
        return getattr(args[2], attr)
    if "event" in kwargs and hasattr(kwargs["event"], attr):
        return getattr(kwargs["event"], attr)
    return None


def _get_from_event_meta(args: tuple, kwargs: dict, attr: str) -> str | None:
    """Extrahiert Attribut aus WebhookEvent.meta."""
    event = None
    if len(args) > 2:
        event = args[2]
    elif "event" in kwargs:
        event = kwargs["event"]

    if event and hasattr(event, "meta") and event.meta and hasattr(event.meta, attr):
        return getattr(event.meta, attr)
    return None


def _get_from_args(args: tuple, kwargs: dict, key: str) -> str | None:
    """Extrahiert Wert direkt aus kwargs."""
    return kwargs.get(key)


__all__ = [
    "audit_delivery_operation",
    "audit_operation",
    "audit_target_operation",
]
