"""Budget-Propagation Middleware.

Liest eingehende Header `X-Token-Budget` und `X-Cost-Budget-Usd` und setzt
kontextuelle Budgets. Spiegelt verbleibende Budgets in Response-Headers wider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

from kei_logging import get_logger
from observability.budget import (
    get_cost_budget_remaining_usd,
    get_token_budget_remaining,
    set_cost_budget_usd,
    set_token_budget,
)

if TYPE_CHECKING:
    from fastapi import Request, Response
    from starlette.types import ASGIApp

# add_cost_sample_ms nicht verfügbar - entfernt


logger = get_logger(__name__)


class BudgetMiddleware(BaseHTTPMiddleware):
    """Middleware zur Verwaltung von Token-/Kosten-Budgets pro Request."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # Eingehende Header lesen
        token_budget: int | None = None
        cost_budget: float | None = None
        try:
            if request.headers.get("X-Token-Budget"):
                token_budget = int(request.headers["X-Token-Budget"])  # type: ignore[index]
        except (ValueError, TypeError) as e:
            logger.warning(f"Ungültiger X-Token-Budget Header: {e}")
            token_budget = None
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Parsen des X-Token-Budget Headers: {e}")
            token_budget = None
        try:
            if request.headers.get("X-Cost-Budget-Usd"):
                cost_budget = float(request.headers["X-Cost-Budget-Usd"])  # type: ignore[index]
        except (ValueError, TypeError) as e:
            logger.warning(f"Ungültiger X-Cost-Budget-Usd Header: {e}")
            cost_budget = None
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Parsen des X-Cost-Budget-Usd Headers: {e}")
            cost_budget = None

        # Kontexte setzen
        set_token_budget(token_budget)
        set_cost_budget_usd(cost_budget)

        response: Response = await call_next(request)

        # Cache-Operationen in Budget aufnehmen (best effort) - deaktiviert
        # TODO: add_cost_sample_ms Funktion implementieren wenn benötigt - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/106
        try:
            lookup_ms = getattr(request.state, "cache_lookup_ms", None)
            if lookup_ms is not None:
                # add_cost_sample_ms("cache_lookup", float(lookup_ms))
                pass
        except AttributeError:
            # Request.state hat kein cache_lookup_ms Attribut - das ist normal
            pass
        except Exception as e:
            logger.debug(f"Fehler beim Abrufen der Cache-Lookup-Zeit: {e}")
        try:
            write_ms = getattr(request.state, "cache_write_ms", None)
            if write_ms is not None:
                # add_cost_sample_ms("cache_write", float(write_ms))
                pass
        except AttributeError:
            # Request.state hat kein cache_write_ms Attribut - das ist normal
            pass
        except Exception as e:
            logger.debug(f"Fehler beim Abrufen der Cache-Write-Zeit: {e}")

        # Verbleibende Budgets zurückgeben
        try:
            t = get_token_budget_remaining()
            if t is not None:
                response.headers["X-Token-Budget-Remaining"] = str(max(0, int(t)))
        except (ValueError, TypeError) as e:
            logger.warning(f"Fehler beim Setzen des X-Token-Budget-Remaining Headers: {e}")
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Token-Budget-Header: {e}")
        try:
            c = get_cost_budget_remaining_usd()
            if c is not None:
                response.headers["X-Cost-Budget-Usd-Remaining"] = f"{max(0.0, float(c)):.6f}"
        except (ValueError, TypeError) as e:
            logger.warning(f"Fehler beim Setzen des X-Cost-Budget-Usd-Remaining Headers: {e}")
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Cost-Budget-Header: {e}")
        return response


__all__ = ["BudgetMiddleware"]
