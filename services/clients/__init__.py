# backend/services/clients/__init__.py
"""Services Clients Paket."""

from __future__ import annotations

import contextlib


def _ensure_minimal_attrs(mod) -> None:
    """Stellt sicher, dass ein ersetztes/stub Modul erwartete Attribute hat.

    Einige Tests ersetzen `services.clients.clients` durch SimpleNamespace Stubs.
    Diese Funktion ergänzt fehlende Flags/Funktionen, damit andere Tests, die
    patchen, nicht mit AttributeError fehlschlagen.
    """
    # Feature-Flags mit Standardwerten versehen
    if not hasattr(mod, "_AIOHTTP_AVAILABLE"):
        mod._AIOHTTP_AVAILABLE = False
    if not hasattr(mod, "_AZURE_AVAILABLE"):
        mod._AZURE_AVAILABLE = False
    if not hasattr(mod, "_AZURE_AI_AVAILABLE"):
        mod._AZURE_AI_AVAILABLE = False

    # Cleanup/Session-Utilities als No-Op ergänzen
    if not hasattr(mod, "cleanup_all_sessions"):
        async def _noop_cleanup() -> None:
            return None
        mod.cleanup_all_sessions = _noop_cleanup

    if not hasattr(mod, "managed_session"):
        class _DummyCtx:
            async def __aenter__(self):
                return None
            async def __aexit__(self, *args):
                return None
        mod.managed_session = lambda **kwargs: _DummyCtx()


try:
    # Falls Tests bereits ein Stub in sys.modules registriert haben, ergänzen
    import sys as _sys
    _existing = _sys.modules.get("services.clients.clients")
    if _existing is not None:
        with contextlib.suppress(Exception):
            _ensure_minimal_attrs(_existing)

    from . import clients as _clients_mod
    _ensure_minimal_attrs(_clients_mod)
    from .clients import (
        Services,
        foundry_credential,
        http_client,
    )

    _CLIENTS_AVAILABLE = True

except ImportError:  # pragma: no cover - optional dependency
    # Fallback-Definitionen für fehlgeschlagene Imports
    _sys = None  # Fallback für sys import
    _clients_mod = None  # Fallback für clients module import

    Services = None
    def http_client():  # type: ignore[override]
        """Fallback HTTP-Client."""
        return

    def foundry_credential():  # type: ignore[override]
        """Fallback Azure Credential."""
        return

    _CLIENTS_AVAILABLE = False

__all__ = [
    "Services",
    "foundry_credential",
    "http_client",
]

# Package Metadaten
__version__ = "0.0.1"
__author__ = "Keiko Development Team"
