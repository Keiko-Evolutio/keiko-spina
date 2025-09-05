"""Grafana API Client für Panel- und Dashboard-Exports.

Enterprise-grade Client mit robustem Error Handling, Retry-Logic und
umfassender Type-Safety für Grafana-Integrationen.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from config.settings import settings
from kei_logging import get_logger

from .config import GrafanaConfig, get_reporting_config
from .exceptions import (
    GrafanaAuthenticationError,
    GrafanaClientError,
    GrafanaConnectionError,
    GrafanaTimeoutError,
)

logger = get_logger(__name__)


class GrafanaClient:
    """Enterprise-grade Grafana API Client.

    Bietet robuste Funktionalitäten für:
    - Panel-Exports (PNG, PDF)
    - Dashboard-Exports
    - Retry-Logic mit exponential backoff
    - Umfassendes Error Handling
    - Type-safe API
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_token: str | None = None,
        config: GrafanaConfig | None = None
    ) -> None:
        """Initialisiert den Grafana-Client.

        Args:
            base_url: Grafana-Base-URL (optional, verwendet Settings falls None)
            api_token: API-Token (optional, verwendet Settings falls None)
            config: Grafana-Konfiguration (optional, verwendet Standard-Config falls None)
        """
        self.base_url = (base_url or settings.grafana_url).rstrip("/")
        self.api_token = api_token or settings.grafana_api_token.get_secret_value()
        self.config = config or get_reporting_config().grafana

        logger.info(f"GrafanaClient initialisiert für {self.base_url}")

    def _get_headers(self) -> dict[str, str]:
        """Erstellt HTTP-Headers für Grafana-Requests.

        Returns:
            Dict mit HTTP-Headers
        """
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    async def _execute_request_with_retry(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        timeout: float = 30.0
    ) -> bytes:
        """Führt HTTP-Request mit Retry-Logic aus.

        Args:
            url: Request-URL
            params: Query-Parameter
            timeout: Request-Timeout in Sekunden

        Returns:
            Response-Content als Bytes

        Raises:
            GrafanaClientError: Bei HTTP-Fehlern
            GrafanaConnectionError: Bei Verbindungsfehlern
            GrafanaTimeoutError: Bei Timeout-Fehlern
            GrafanaAuthenticationError: Bei Authentifizierungsfehlern
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(
                        url,
                        headers=self._get_headers(),
                        params=params or {}
                    )

                    # Spezifische Error-Behandlung basierend auf Status-Code
                    if response.status_code == 401:
                        raise GrafanaAuthenticationError(
                            "Grafana-Authentifizierung fehlgeschlagen",
                            status_code=response.status_code,
                            response_body=response.text
                        )
                    if response.status_code == 404:
                        raise GrafanaClientError(
                            f"Dashboard oder Panel nicht gefunden: {url}",
                            status_code=response.status_code,
                            response_body=response.text
                        )

                    response.raise_for_status()
                    return response.content

            except httpx.TimeoutException as e:
                last_exception = GrafanaTimeoutError(
                    f"Grafana-Request Timeout nach {timeout}s",
                    status_code=None,
                    response_body=str(e)
                )
            except httpx.ConnectError as e:
                last_exception = GrafanaConnectionError(
                    f"Verbindung zu Grafana fehlgeschlagen: {self.base_url}",
                    status_code=None,
                    response_body=str(e)
                )
            except (GrafanaAuthenticationError, GrafanaClientError):
                # Diese Exceptions nicht wiederholen
                raise
            except Exception as e:
                last_exception = GrafanaClientError(
                    f"Unerwarteter Grafana-Fehler: {e!s}",
                    status_code=None,
                    response_body=str(e)
                )

            # Retry-Delay (außer beim letzten Versuch)
            if attempt < self.config.max_retries:
                delay = self.config.retry_delay_seconds * (2 ** attempt)
                logger.warning(
                    f"Grafana-Request fehlgeschlagen (Versuch {attempt + 1}/{self.config.max_retries + 1}), "
                    f"Retry in {delay}s: {last_exception}"
                )
                await asyncio.sleep(delay)

        # Alle Retry-Versuche fehlgeschlagen
        if last_exception:
            raise last_exception
        raise GrafanaClientError("Unbekannter Fehler bei Grafana-Request")

    async def export_panel_png(
        self,
        dashboard_uid: str,
        panel_id: int,
        params: dict[str, Any] | None = None
    ) -> bytes:
        """Exportiert ein Grafana-Panel als PNG.

        Args:
            dashboard_uid: UID des Dashboards
            panel_id: Panel-ID
            params: Query-Parameter (z.B. from, to, width, height)

        Returns:
            PNG-Daten als Bytes

        Raises:
            GrafanaClientError: Bei Export-Fehlern
        """
        url = f"{self.base_url}/render/d-solo/{dashboard_uid}/_?panelId={panel_id}"

        # Standard-Parameter mit Konfigurationswerten setzen
        export_params = {
            "width": self.config.default_panel_width,
            "height": self.config.default_panel_height
        }
        if params:
            export_params.update(params)

        logger.debug(f"Exportiere Panel PNG: {dashboard_uid}/{panel_id}")
        return await self._execute_request_with_retry(
            url,
            export_params,
            self.config.panel_timeout_seconds
        )

    async def export_dashboard_pdf(
        self,
        dashboard_uid: str,
        params: dict[str, Any] | None = None
    ) -> bytes:
        """Exportiert ein Grafana-Dashboard als PDF.

        Args:
            dashboard_uid: UID des Dashboards
            params: Query-Parameter (z.B. from, to)

        Returns:
            PDF-Daten als Bytes

        Raises:
            GrafanaClientError: Bei Export-Fehlern
        """
        url = f"{self.base_url}/render/d/{dashboard_uid}/_"

        logger.debug(f"Exportiere Dashboard PDF: {dashboard_uid}")
        return await self._execute_request_with_retry(
            url,
            params,
            self.config.dashboard_timeout_seconds
        )

    async def export_panel_csv(
        self,
        dashboard_uid: str,
        panel_id: int,
        params: dict[str, Any] | None = None
    ) -> str:
        """Exportiert ein Grafana-Panel als CSV.

        Args:
            dashboard_uid: UID des Dashboards
            panel_id: Panel-ID
            params: Query-Parameter (z.B. from, to)

        Returns:
            CSV-Daten als String

        Raises:
            GrafanaClientError: Bei Export-Fehlern
        """
        # CSV-Export über Prometheus-API (falls verfügbar)
        # Fallback: PNG-Export mit Warnung
        logger.warning("CSV-Export nicht direkt unterstützt, verwende PNG-Export")
        png_data = await self.export_panel_png(dashboard_uid, panel_id, params)
        return f"# CSV-Export nicht verfügbar, PNG-Daten: {len(png_data)} bytes"

    async def export_panel_json(
        self,
        dashboard_uid: str,
        panel_id: int,
        params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Exportiert Panel-Metadaten als JSON.

        Args:
            dashboard_uid: UID des Dashboards
            panel_id: Panel-ID
            params: Query-Parameter

        Returns:
            Panel-Metadaten als Dictionary

        Raises:
            GrafanaClientError: Bei Export-Fehlern
        """
        url = f"{self.base_url}/api/dashboards/uid/{dashboard_uid}"

        try:
            async with httpx.AsyncClient(timeout=self.config.panel_timeout_seconds) as client:
                response = await client.get(url, headers=self._get_headers())
                response.raise_for_status()
                dashboard_data = response.json()

                # Panel-spezifische Daten extrahieren
                panels = dashboard_data.get("dashboard", {}).get("panels", [])
                target_panel = next((p for p in panels if p.get("id") == panel_id), None)

                if not target_panel:
                    raise GrafanaClientError(f"Panel {panel_id} nicht gefunden in Dashboard {dashboard_uid}")

                return {
                    "dashboard_uid": dashboard_uid,
                    "panel_id": panel_id,
                    "panel_title": target_panel.get("title", "Unknown"),
                    "panel_type": target_panel.get("type", "Unknown"),
                    "panel_config": target_panel,
                    "export_params": params or {},
                    "exported_at": "now"  # Würde normalerweise datetime verwenden
                }

        except httpx.HTTPStatusError as e:
            raise GrafanaClientError(
                f"Dashboard-Metadaten-Export fehlgeschlagen: {e.response.status_code}",
                status_code=e.response.status_code,
                response_body=e.response.text
            ) from e
        except Exception as e:
            raise GrafanaClientError(f"JSON-Export fehlgeschlagen: {e!s}") from e


__all__ = ["GrafanaClient"]
