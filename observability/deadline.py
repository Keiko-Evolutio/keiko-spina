"""Deadline- und Budget-Propagation für REST und gRPC.

Dieses Modul stellt Hilfsfunktionen bereit, um eine Deadline (Zeitbudget)
pro eingehender Anfrage zu verwalten und an Downstream-Aufrufe zu
propagieren. Die Deadline wird als absoluter Zeitpunkt (monotonic) intern
gehalten; nach außen werden Headerwerte in Millisekunden verwendet.
"""

from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar
from typing import Any

from .base_metrics import MetricsConstants

# Kontextvariable für absolute Deadline (monotonic timestamp in Sekunden)
_DEADLINE_TS: ContextVar[float | None] = ContextVar("kei_deadline_ts", default=None)


def set_deadline_ms_from_now(budget_ms: int) -> None:
    """Setzt Deadline relativ zu jetzt in Millisekunden.

    Args:
        budget_ms: Restzeit ab jetzt in Millisekunden
    """
    now = time.monotonic()
    _DEADLINE_TS.set(now + max(0.0, float(budget_ms)) / 1000.0)


def set_deadline_abs_ts(deadline_ts_monotonic: float) -> None:
    """Setzt eine absolute Deadline (monotonic timestamp)."""
    _DEADLINE_TS.set(deadline_ts_monotonic)


def clear_deadline() -> None:
    """Löscht die gesetzte Deadline."""
    _DEADLINE_TS.set(None)


def get_remaining_budget_ms() -> int | None:
    """Gibt verbleibendes Zeitbudget in Millisekunden zurück, falls gesetzt."""
    deadline = _DEADLINE_TS.get()
    if deadline is None:
        return None
    rem = max(0.0, deadline - time.monotonic()) * 1000.0
    return int(rem)


def build_outgoing_headers(existing: dict[str, str] | None = None, *, safety_ms: int = MetricsConstants.DEFAULT_SAFETY_MARGIN_MS) -> dict[str, str]:
    """Erstellt Header mit Time-Budget-Propagation.

    Args:
        existing: Bestehende Header
        safety_ms: Sicherheitsabschlag in Millisekunden

    Returns:
        Header-Dict inkl. `X-Time-Budget-Ms`
    """
    hdrs: dict[str, str] = {}
    if existing:
        hdrs.update(existing)
    rem = get_remaining_budget_ms()
    if rem is not None:
        budget = max(0, rem - max(0, safety_ms))
        hdrs[MetricsConstants.TIME_BUDGET_HEADER] = str(budget)
    return hdrs


async def run_with_deadline(awaitable: Any) -> Any:
    """Führt Awaitable unter Berücksichtigung der Deadline aus.

    Bei gesetzter Deadline wird `asyncio.wait_for` verwendet, andernfalls
    direkt awaited.
    """
    rem = get_remaining_budget_ms()
    if rem is None:
        return await awaitable
    timeout_s = max(0.001, rem / 1000.0)
    return await asyncio.wait_for(awaitable, timeout=timeout_s)


__all__ = [
    "build_outgoing_headers",
    "clear_deadline",
    "get_remaining_budget_ms",
    "run_with_deadline",
    "set_deadline_abs_ts",
    "set_deadline_ms_from_now",
]
