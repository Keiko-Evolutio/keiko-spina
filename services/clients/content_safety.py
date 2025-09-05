# backend/services/clients/content_safety.py
"""Azure Content Safety Textanalyse (asynchron) mit Retry-Logik."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.settings import settings
from kei_logging import get_logger

from .common import (
    CONTENT_SAFETY_API_VERSION,
    CONTENT_SAFETY_CATEGORIES,
    CONTENT_SAFETY_FALLBACK_CATEGORY,
    CONTENT_SAFETY_REQUEST_EVENT,
    CONTENT_SAFETY_UNAVAILABLE_EVENT,
    CONTENT_SAFETY_UNAVAILABLE_REASON,
    MAX_SEVERITY_LEVEL,
    SAFE_SEVERITY_THRESHOLD,
    RetryableClient,
    StandardHTTPClientConfig,
    create_aiohttp_connector,
    create_aiohttp_session_config,
    create_azure_headers,
    create_content_safety_retry_config,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class ContentSafetyResult:
    """Ergebnis der Content-Safety-Analyse."""
    is_safe: bool
    score: float
    category: str
    raw: dict[str, Any]


class ContentSafetyClient(RetryableClient):
    """Azure Content Safety Client für Text-Analyse.

    Nutzt REST-API mit automatischer Retry-Logik.
    Optional, fällt bei fehlender Konfiguration sauber zurück.
    """

    def __init__(self, endpoint: str | None = None, api_key: str | None = None) -> None:
        # Retry-Konfiguration initialisieren
        super().__init__(create_content_safety_retry_config())

        # Service-Konfiguration laden
        self._endpoint = endpoint or settings.azure_content_safety_endpoint
        self._api_key = api_key or (
            settings.azure_content_safety_key.get_secret_value()
            if getattr(settings, "azure_content_safety_key", None)
            else ""
        )
        self._available = bool(self._endpoint and self._api_key)

        # HTTP Client Konfiguration
        self._http_config = StandardHTTPClientConfig.content_safety()

        logger.debug({
            "event": "content_safety_client_init",
            "available": self._available,
            "endpoint": bool(self._endpoint),
            "api_key": bool(self._api_key),
        })

    @property
    def is_available(self) -> bool:
        """Gibt zurück, ob der Client konfiguriert ist."""
        return self._available

    def _create_request_url(self) -> str:
        """Erstellt die Request-URL für die Content Safety API."""
        return (
            f"{self._endpoint.rstrip('/')}/contentsafety/text:analyze"
            f"?api-version={CONTENT_SAFETY_API_VERSION}"
        )

    def _create_request_payload(self, text: str) -> dict[str, Any]:
        """Erstellt das Request-Payload für die Content Safety API."""
        return {
            "text": text,
            "categories": CONTENT_SAFETY_CATEGORIES,
            "outputType": "FourSeverityLevels",
        }

    def _parse_response_data(self, data: dict[str, Any]) -> ContentSafetyResult:
        """Parst die API-Response und erstellt ContentSafetyResult.

        Args:
            data: Response-Daten von der API

        Returns:
            ContentSafetyResult mit analysierten Daten
        """
        severities = []

        # Azure Content Safety kann verschiedene Response-Formate haben
        categories_analysis = data.get("categoriesAnalysis", data.get("CategoriesAnalysis", {}))

        if isinstance(categories_analysis, list):
            # Neues Format: Liste von Kategorien
            for item in categories_analysis:
                if isinstance(item, dict):
                    sev = item.get("severity", 0)
                    severities.append(int(sev))
        else:
            # Altes Format: Dict mit Kategorien
            for category in CONTENT_SAFETY_CATEGORIES:
                sev = categories_analysis.get(category, {}).get("severity", 0)
                severities.append(int(sev))

        max_sev = max(severities) if severities else 0
        is_safe = max_sev <= SAFE_SEVERITY_THRESHOLD
        score = float(max_sev) / MAX_SEVERITY_LEVEL
        category = "safe" if is_safe else "unsafe"

        return ContentSafetyResult(
            is_safe=is_safe,
            score=score,
            category=category,
            raw=data
        )

    async def _perform_analysis_request(self, text: str) -> ContentSafetyResult:
        """Führt die eigentliche Content Safety Analyse durch.

        Args:
            text: Zu analysierender Text

        Returns:
            ContentSafetyResult mit Analyseergebnis
        """
        url = self._create_request_url()
        headers = create_azure_headers(self._api_key)
        payload = self._create_request_payload(text)

        logger.debug({
            "event": CONTENT_SAFETY_REQUEST_EVENT,
            "endpoint": self._endpoint,
            "text_length": len(text),
        })

        # HTTP Session mit Standard-Konfiguration erstellen
        session_config = create_aiohttp_session_config(self._http_config)

        # Connector separat erstellen
        connector_config = session_config.pop("connector_config", {})
        connector = create_aiohttp_connector(connector_config)
        session_config["connector"] = connector

        import aiohttp
        async with aiohttp.ClientSession(**session_config) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                data: dict[str, Any] = await resp.json(content_type=None)

                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}: {data}")

                return self._parse_response_data(data)

    async def analyze_text(self, text: str) -> ContentSafetyResult:
        """Analysiert Text auf problematische Inhalte.

        Args:
            text: Eingabetext (Prompt)

        Returns:
            ContentSafetyResult: Ergebnis mit Score und Kategorie
        """
        # Fallback bei fehlender Konfiguration
        if not self._available:
            logger.debug({"event": CONTENT_SAFETY_UNAVAILABLE_EVENT})
            return ContentSafetyResult(
                is_safe=True,
                score=0.0,
                category=CONTENT_SAFETY_FALLBACK_CATEGORY,
                raw={"reason": CONTENT_SAFETY_UNAVAILABLE_REASON}
            )

        # Analyse mit Retry-Logik durchführen
        try:
            return await self._execute_with_retry(self._perform_analysis_request, text)
        except Exception as e:
            logger.warning(f"Content Safety Analyse fehlgeschlagen: {e}")
            return ContentSafetyResult(
                is_safe=True,
                score=0.0,
                category=CONTENT_SAFETY_FALLBACK_CATEGORY,
                raw={"error": str(e)}
            )


def create_content_safety_client() -> ContentSafetyClient:
    """Factory für ContentSafetyClient aus Settings."""
    return ContentSafetyClient()
