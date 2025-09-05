"""Unified KEI-RPC Client basierend auf UnifiedHTTPClient.

Migriert KEIRPCClient zur neuen UnifiedHTTPClient-Architektur
während die bestehende API beibehalten wird.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from services.core.client_factory import ClientFactory

# Optional imports für erweiterte Features
try:
    from observability import trace_function
except ImportError:
    def trace_function(_name):
        def decorator(func):
            return func
        return decorator


if TYPE_CHECKING:
    from services.core.unified_client import UnifiedHTTPClient

    from .kei_rpc_client import KEIRPCClientConfig

logger = get_logger(__name__)


class UnifiedKEIRPCClient:
    """Unified KEI-RPC Client basierend auf UnifiedHTTPClient.

    Drop-in Replacement für KEIRPCClient mit verbesserter Architektur:
    - Verwendet UnifiedHTTPClient für HTTP-Kommunikation
    - Behält vollständige API-Kompatibilität bei
    - Verbesserte Error-Handling und Circuit Breaker Integration
    - Konsolidierte Konfiguration und Monitoring
    """

    def __init__(self, config: KEIRPCClientConfig):
        """Initialisiert den Unified KEI-RPC Client.

        Args:
            config: Konfiguration für den KEI-RPC Client
        """
        self.config = config
        self._unified_client: UnifiedHTTPClient | None = None

    async def __aenter__(self):
        """Async Context Manager Entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async Context Manager Exit."""
        await self.close()

    async def _ensure_client(self) -> None:
        """Stellt sicher, dass der Unified HTTP Client initialisiert ist."""
        if self._unified_client is not None:
            return

        # Erstelle UnifiedHTTPClient über ClientFactory
        self._unified_client = ClientFactory.create_kei_rpc_client(
            base_url=self.config.base_url,
            api_token=self.config.api_token,
            tenant_id=self.config.tenant_id,
            timeout_seconds=self.config.timeout_seconds
        )

        # Initialisiere den Client
        await self._unified_client._ensure_client()

        logger.debug(f"Unified KEI-RPC Client für {self.config.base_url} initialisiert")

    @trace_function("kei_rpc_client_get_resource")
    async def get_resource(self, resource_id: str) -> dict[str, Any]:
        """Ruft eine Ressource über die KEI-RPC API ab.

        Args:
            resource_id: ID der abzurufenden Ressource

        Returns:
            Dictionary mit Ressourcen-Daten

        Raises:
            httpx.HTTPStatusError: Bei HTTP-Fehlern
            httpx.RequestError: Bei Verbindungsfehlern
        """
        await self._ensure_client()

        return await self._unified_client.get_json(f"/resources/{resource_id}")

    @trace_function("kei_rpc_client_list_resources")
    async def list_resources(
        self,
        limit: int | None = None,
        offset: int | None = None,
        filters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Listet Ressourcen über die KEI-RPC API auf.

        Args:
            limit: Maximale Anzahl der Ergebnisse
            offset: Offset für Paginierung
            filters: Zusätzliche Filter-Parameter

        Returns:
            Dictionary mit Ressourcen-Liste und Metadaten
        """
        await self._ensure_client()

        params = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if filters:
            params.update(filters)

        return await self._unified_client.get_json("/resources", params=params)

    @trace_function("kei_rpc_client_create_resource")
    async def create_resource(self, resource_data: dict[str, Any]) -> dict[str, Any]:
        """Erstellt eine neue Ressource über die KEI-RPC API.

        Args:
            resource_data: Daten für die neue Ressource

        Returns:
            Dictionary mit erstellter Ressource
        """
        await self._ensure_client()

        return await self._unified_client.post_json("/resources", json_data=resource_data)

    @trace_function("kei_rpc_client_update_resource")
    async def update_resource(
        self,
        resource_id: str,
        resource_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Aktualisiert eine bestehende Ressource über die KEI-RPC API.

        Args:
            resource_id: ID der zu aktualisierenden Ressource
            resource_data: Neue Daten für die Ressource

        Returns:
            Dictionary mit aktualisierter Ressource
        """
        await self._ensure_client()

        return await self._unified_client.put_json(
            f"/resources/{resource_id}",
            json_data=resource_data
        )

    @trace_function("kei_rpc_client_delete_resource")
    async def delete_resource(self, resource_id: str) -> None:
        """Löscht eine Ressource über die KEI-RPC API.

        Args:
            resource_id: ID der zu löschenden Ressource
        """
        await self._ensure_client()

        await self._unified_client.delete(f"/resources/{resource_id}")

    @trace_function("kei_rpc_client_execute_action")
    async def execute_action(
        self,
        action_name: str,
        action_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Führt eine Aktion über die KEI-RPC API aus.

        Args:
            action_name: Name der auszuführenden Aktion
            action_data: Optionale Daten für die Aktion

        Returns:
            Dictionary mit Aktions-Ergebnis
        """
        await self._ensure_client()

        payload = {
            "action": action_name,
            "data": action_data or {}
        }

        return await self._unified_client.post_json("/actions", json_data=payload)

    @trace_function("kei_rpc_client_get_status")
    async def get_status(self) -> dict[str, Any]:
        """Ruft den Status der KEI-RPC API ab.

        Returns:
            Dictionary mit Status-Informationen
        """
        await self._ensure_client()

        return await self._unified_client.get_json("/status")

    async def health_check(self) -> bool:
        """Führt Health-Check für die KEI-RPC API durch.

        Returns:
            True wenn API erreichbar und gesund
        """
        await self._ensure_client()

        try:
            health_result = await self._unified_client.health_check("/health")
            return health_result["status"] == "healthy"
        except Exception as exc:
            logger.debug(f"Health-Check fehlgeschlagen für KEI-RPC API: {exc}")
            return False

    async def close(self) -> None:
        """Schließt den KEI-RPC Client und gibt Ressourcen frei."""
        if self._unified_client:
            await self._unified_client.close()
            self._unified_client = None

        logger.debug(f"Unified KEI-RPC Client für {self.config.base_url} geschlossen")


# Backward-Compatibility Alias
KEIRPCClient = UnifiedKEIRPCClient
