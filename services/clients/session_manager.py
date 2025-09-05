# backend/services/clients/session_manager.py
"""Session-Management für HTTP-Clients."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import aiohttp

from kei_logging import get_logger

from .common import (
    HTTPClientConfig,
    create_aiohttp_connector,
    create_aiohttp_session_config,
)

logger = get_logger(__name__)


class SessionManager:
    """Session-Manager."""

    def __init__(self) -> None:
        self._active_sessions: set[aiohttp.ClientSession] = set()

    def register(self, session: aiohttp.ClientSession) -> None:
        """Registriert Session für Tracking."""
        self._active_sessions.add(session)
        logger.debug(f"Session registriert: {id(session)}")

    def unregister(self, session: aiohttp.ClientSession) -> None:
        """Entfernt Session aus Tracking."""
        self._active_sessions.discard(session)
        logger.debug(f"Session entfernt: {id(session)}")

    async def close_all(self) -> int:
        """Schließt alle aktiven Sessions."""
        count = 0
        for session in list(self._active_sessions):
            try:
                if not session.closed:
                    await session.close()
                    count += 1
            except Exception as e:
                logger.debug(f"Session close Fehler: {e}")
            finally:
                self.unregister(session)

        logger.info(f"Sessions geschlossen: {count}")
        return count


# Globaler Session-Manager
_session_manager = SessionManager()


@asynccontextmanager
async def managed_session(
    config: HTTPClientConfig | None = None,
    **kwargs: Any
) -> AsyncIterator[aiohttp.ClientSession]:
    """Context Manager für HTTP-Sessions mit Standard-Konfiguration.

    Args:
        config: HTTP Client Konfiguration (optional)
        **kwargs: Überschreibungen für Session-Parameter
    """
    # Standard-Konfiguration mit optionalen Überschreibungen
    session_config = create_aiohttp_session_config(config, **kwargs)

    # Connector separat erstellen
    connector_config = session_config.pop("connector_config", {})
    connector = create_aiohttp_connector(connector_config)
    session_config["connector"] = connector

    session = aiohttp.ClientSession(**session_config)
    _session_manager.register(session)

    try:
        yield session
    finally:
        try:
            if not session.closed:
                await session.close()
        except Exception as e:
            logger.debug(f"Session close Fehler: {e}")
        finally:
            _session_manager.unregister(session)


async def cleanup_all_sessions() -> int:
    """Schließt alle Sessions."""
    return await _session_manager.close_all()
