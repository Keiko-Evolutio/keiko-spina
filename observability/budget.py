"""Budget- und Kosten-Propagation (Tokens/Costs) für REST/gRPC.

Stellt kontextbasierte Budgets bereit und Hilfen zur Weitergabe per Header:
- X-Token-Budget: verbleibende Token (Ganzzahl)
- X-Cost-Budget-Usd: verbleibendes Kostenbudget in USD (Float)

Funktionen erlauben Setzen, Abfragen und Verbrauchs-Tracking.
"""

from __future__ import annotations

from contextvars import ContextVar

from .base_metrics import build_headers_with_context

_TOKEN_BUDGET_REMAINING: ContextVar[int | None] = ContextVar("kei_token_budget", default=None)
_COST_BUDGET_REMAINING_USD: ContextVar[float | None] = ContextVar("kei_cost_budget_usd", default=None)


def set_token_budget(tokens: int | None) -> None:
    """Setzt verbleibendes Tokenbudget."""
    if tokens is not None and tokens < 0:
        tokens = 0
    _TOKEN_BUDGET_REMAINING.set(tokens)


def set_cost_budget_usd(cost_usd: float | None) -> None:
    """Setzt verbleibendes Kostenbudget (USD)."""
    if cost_usd is not None and cost_usd < 0.0:
        cost_usd = 0.0
    _COST_BUDGET_REMAINING_USD.set(cost_usd)


def get_token_budget_remaining() -> int | None:
    """Gibt verbleibende Tokens zurück (oder None)."""
    return _TOKEN_BUDGET_REMAINING.get()


def get_cost_budget_remaining_usd() -> float | None:
    """Gibt verbleibendes Kostenbudget (USD) zurück (oder None)."""
    return _COST_BUDGET_REMAINING_USD.get()


def try_consume_tokens(tokens: int) -> bool:
    """Versucht, Tokens zu verbrauchen. Gibt False zurück, wenn Budget nicht reicht."""
    if tokens <= 0:
        return True
    remaining = _TOKEN_BUDGET_REMAINING.get()
    if remaining is None:
        return True
    if remaining < tokens:
        return False
    _TOKEN_BUDGET_REMAINING.set(remaining - tokens)
    return True


def try_consume_cost_usd(amount_usd: float) -> bool:
    """Versucht, Kostenbudget zu verbrauchen. Gibt False zurück, wenn Budget nicht reicht."""
    if amount_usd <= 0.0:
        return True
    remaining = _COST_BUDGET_REMAINING_USD.get()
    if remaining is None:
        return True
    if remaining < amount_usd:
        return False
    _COST_BUDGET_REMAINING_USD.set(remaining - amount_usd)
    return True


def build_outgoing_budget_headers(existing: dict[str, str] | None = None) -> dict[str, str]:
    """Erstellt Header für Budget-Weitergabe (Tokens/Costs)."""
    return build_headers_with_context(
        existing_headers=existing,
        token_budget=get_token_budget_remaining(),
        cost_budget_usd=get_cost_budget_remaining_usd()
    )





__all__ = [
    "build_outgoing_budget_headers",
    "get_cost_budget_remaining_usd",
    "get_token_budget_remaining",
    "set_cost_budget_usd",
    "set_token_budget",
    "try_consume_cost_usd",
    "try_consume_tokens",
]
