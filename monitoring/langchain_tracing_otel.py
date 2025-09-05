"""LangChain Callback → OpenTelemetry Spans.

Dieses Modul stellt einen vereinfachten Callback-Handler für LangChain bereit,
der Events in OTel-Spans abbildet. Doppel-Instrumentierung wird vermieden,
indem nur ein Pfad (OTel-first) genutzt wird.
"""

from __future__ import annotations

from typing import Any

from kei_logging import get_logger
from observability import trace_function

logger = get_logger(__name__)


class LangChainOTelCallback:
    """Vereinfachter LangChain-Callback-Handler für OTel."""

    def __init__(self, component_name: str = "langchain") -> None:
        """Initialisiert den Handler.

        Args:
            component_name: Name für Komponenten-Attribut in Spans.
        """
        self.component_name = component_name

    @trace_function("langchain.on_start", {"component": "langchain"})
    async def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        """Behandelt Start eines Chains."""
        logger.debug("LC start: %s", serialized.get("name"))

    @trace_function("langchain.on_end", {"component": "langchain"})
    async def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Behandelt Ende eines Chains."""
        logger.debug("LC end")

    @trace_function("langchain.on_error", {"component": "langchain"})
    async def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Behandelt Fehler in Chains."""
        logger.error("LC error: %s", error)


__all__ = ["LangChainOTelCallback"]
