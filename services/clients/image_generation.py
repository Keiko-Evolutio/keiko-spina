# backend/services/clients/image_generation.py
"""Azure OpenAI (DALL·E-3) Bildgenerierung (asynchron) mit Retry-Logik.

Kommentare auf Deutsch, Identifier Englisch, PEP 8/257/484 konform.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from openai import AsyncOpenAI

from config.settings import settings
from kei_logging import get_logger
from services.safety.prompt_safety_filter import prompt_safety_filter

from .common import (
    CONTENT_POLICY_VIOLATION_DETECTED_EVENT,
    DEFAULT_IMAGE_API_VERSION,
    DEFAULT_IMAGE_CONTENT_TYPE,
    DEFAULT_IMAGE_DEPLOYMENT,
    DEFAULT_IMAGE_QUALITY,
    DEFAULT_IMAGE_RESPONSE_FORMAT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_USER_ID,
    IMAGE_GENERATION_NO_CONTENT_ERROR,
    IMAGE_GENERATION_REQUEST_EVENT,
    IMAGE_SERVICE_CLIENT_READY_EVENT,
    OPENAI_IMAGES_GENERATE_CALL_EVENT,
    OPENAI_IMAGES_GENERATE_ERROR_EVENT,
    OPENAI_IMAGES_GENERATE_OK_EVENT,
    OPENAI_IMAGES_NO_CONTENT_EVENT,
    PROMPT_SAFETY_VIOLATION_EVENT,
    PROMPT_SANITIZED_EVENT,
    USING_FALLBACK_PROMPT_EVENT,
    RetryableClient,
    StandardHTTPClientConfig,
    create_httpx_client_config,
    create_image_generation_retry_config,
    is_content_policy_violation,
)
from .common.http_config import create_aiohttp_session_config

logger = get_logger(__name__)

ImageSize = Literal["1024x1024", "1024x1792", "1792x1024"]
ImageQuality = Literal["standard", "hd"]


@dataclass(slots=True)
class ImageGenerationRequest:
    """Anfrageparameter für die Bildgenerierung."""
    prompt: str
    size: ImageSize = DEFAULT_IMAGE_SIZE  # type: ignore
    quality: ImageQuality = DEFAULT_IMAGE_QUALITY  # type: ignore
    style: str = "Realistic"
    user_id: str | None = None
    session_id: str | None = None


@dataclass(slots=True)
class ImageGenerationResult:
    """Ergebnis der Bildgenerierung."""
    image_bytes: bytes
    content_type: str
    raw: dict[str, Any]


class ImageGenerationService(RetryableClient):
    """Service für DALL·E-3 über Azure OpenAI kompatible API.

    Nutzt openai.AsyncOpenAI mit automatischer Retry-Logik und
    integrierter Prompt Safety Filterung.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        deployment: str = DEFAULT_IMAGE_DEPLOYMENT,
        api_version: str = DEFAULT_IMAGE_API_VERSION
    ) -> None:
        # Retry-Konfiguration initialisieren
        super().__init__(create_image_generation_retry_config())

        # Service-Konfiguration laden
        self._endpoint = endpoint or settings.project_keiko_image_endpoint
        self._api_key = api_key or (
            settings.project_keiko_image_api_key.get_secret_value()
            if getattr(settings, "project_keiko_image_api_key", None)
            else ""
        )
        self._deployment = deployment
        self._api_version = api_version
        self._available = bool(self._endpoint and self._api_key)
        self._client: AsyncOpenAI | None = None

        # HTTP Client Konfiguration
        self._http_config = StandardHTTPClientConfig.image_generation()

    @property
    def is_available(self) -> bool:
        """Gibt zurück, ob der Service konfiguriert ist."""
        return self._available

    def _create_base_url(self) -> str:
        """Erstellt die Base-URL für Azure OpenAI Images API."""
        return f"{self._endpoint.rstrip('/')}/openai/deployments/{self._deployment}"

    def _ensure_client(self) -> AsyncOpenAI:
        """Initialisiert den OpenAI-Client lazily."""
        if self._client is None:
            base_url = self._create_base_url()

            # HTTP Client mit Standard-Konfiguration erstellen
            client_config = create_httpx_client_config(
                self._http_config,
                base_url=base_url
            )
            http_client = httpx.AsyncClient(**client_config)

            # OpenAI Client erstellen
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=base_url,
                http_client=http_client
            )

            logger.debug({
                "event": IMAGE_SERVICE_CLIENT_READY_EVENT,
                "endpoint": self._endpoint,
                "deployment": self._deployment,
                "api_version": self._api_version,
                "base_url": base_url,
                "has_key": bool(self._api_key)
            })
        return self._client

    def _apply_prompt_safety_filter(self, prompt: str) -> str:
        """Wendet Prompt Safety Filter an und gibt sanitisierten Prompt zurück.

        Args:
            prompt: Original-Prompt

        Returns:
            Sanitisierter Prompt
        """
        safety_result = prompt_safety_filter.analyze_prompt(prompt)

        if not safety_result.is_safe:
            logger.warning({
                "event": PROMPT_SAFETY_VIOLATION_EVENT,
                "original_prompt": prompt,
                "violations": safety_result.violations,
                "confidence_score": safety_result.confidence_score
            })

            # Verwende sanitisierten Prompt
            actual_prompt = safety_result.sanitized_prompt
            logger.info({
                "event": PROMPT_SANITIZED_EVENT,
                "original_prompt": prompt,
                "sanitized_prompt": actual_prompt
            })
            return actual_prompt

        return prompt

    async def _download_image_from_url(self, image_url: str) -> bytes:
        """Lädt ein Bild von einer URL herunter.

        Args:
            image_url: URL des Bildes

        Returns:
            Bild-Bytes
        """
        import aiohttp

        session_config = create_aiohttp_session_config(self._http_config)
        async with aiohttp.ClientSession(**session_config) as session:
            async with session.get(image_url) as resp:
                if resp.status != 200:
                    raise RuntimeError(
                        f"Fehler beim Herunterladen des Bildes: HTTP {resp.status}"
                    )
                return await resp.read()

    def _extract_image_bytes(self, response_data: Any) -> bytes:
        """Extrahiert Bild-Bytes aus der OpenAI API Response.

        Args:
            response_data: Response-Daten von der OpenAI API

        Returns:
            Bild-Bytes
        """
        data0 = response_data.data[0]

        # Azure OpenAI Images API kann base64 oder URL zurückgeben
        image_b64 = getattr(data0, "b64_json", None)
        image_url = getattr(data0, "url", None)

        if image_b64:
            # Base64-Format
            return base64.b64decode(image_b64)
        if image_url:
            # URL-Format - Bild herunterladen (async call needed)
            import asyncio
            return asyncio.create_task(self._download_image_from_url(image_url))
        logger.error({
            "event": OPENAI_IMAGES_NO_CONTENT_EVENT,
            "available_attrs": [attr for attr in dir(data0) if not attr.startswith("_")],
            "b64_json": image_b64,
            "url": image_url
        })
        raise RuntimeError(IMAGE_GENERATION_NO_CONTENT_ERROR)

    async def _perform_image_generation(
        self,
        req: ImageGenerationRequest,
        actual_prompt: str
    ) -> ImageGenerationResult:
        """Führt die eigentliche Bildgenerierung durch.

        Args:
            req: Image Generation Request
            actual_prompt: Sanitisierter Prompt

        Returns:
            ImageGenerationResult mit generiertem Bild
        """
        client = self._ensure_client()

        logger.debug({
            "event": OPENAI_IMAGES_GENERATE_CALL_EVENT,
            "size": req.size,
            "quality": req.quality,
            "prompt_length": len(actual_prompt),
        })

        response = await client.images.generate(
            model=self._deployment,
            prompt=actual_prompt,
            size=req.size,
            quality=req.quality,
            response_format=DEFAULT_IMAGE_RESPONSE_FORMAT,
            user=req.user_id or req.session_id or DEFAULT_USER_ID,
            extra_query={"api-version": self._api_version},
        )

        # Bild-Bytes extrahieren
        if hasattr(response.data[0], "url") and response.data[0].url:
            # URL-Format
            image_bytes = await self._download_image_from_url(response.data[0].url)
            format_type = "url"
        else:
            # Base64-Format
            image_bytes = base64.b64decode(response.data[0].b64_json)
            format_type = "base64"

        # Response-Daten für Raw-Output
        raw: dict[str, Any] = getattr(
            response,
            "model_dump",
            lambda: getattr(response, "dict", lambda: {"response": str(response)})()
        )()

        logger.debug({
            "event": OPENAI_IMAGES_GENERATE_OK_EVENT,
            "format": format_type,
            "image_size": len(image_bytes),
        })

        return ImageGenerationResult(
            image_bytes=image_bytes,
            content_type=DEFAULT_IMAGE_CONTENT_TYPE,
            raw=raw
        )

    async def generate(self, req: ImageGenerationRequest) -> ImageGenerationResult:
        """Generiert ein Bild mittels DALL·E-3 und gibt Bytes zurück.

        Args:
            req: Parameter der Generierung

        Returns:
            ImageGenerationResult mit generiertem Bild
        """
        if not self._available:
            from .common import ServiceNotConfiguredException
            raise ServiceNotConfiguredException("ImageGenerationService")

        # Safety Filter anwenden
        actual_prompt = self._apply_prompt_safety_filter(req.prompt)

        logger.debug({
            "event": IMAGE_GENERATION_REQUEST_EVENT,
            "size": req.size,
            "quality": req.quality,
            "original_prompt": req.prompt,
            "final_prompt": actual_prompt,
        })

        # Bildgenerierung mit Retry-Logik durchführen
        try:
            return await self._execute_with_retry(self._perform_image_generation, req, actual_prompt)
        except Exception as e:
            # Content Policy Violation spezielle Behandlung
            if is_content_policy_violation(str(e)):
                logger.warning({
                    "event": CONTENT_POLICY_VIOLATION_DETECTED_EVENT,
                    "original_prompt": req.prompt,
                    "sanitized_prompt": actual_prompt,
                    "error": str(e)
                })

                # Fallback-Prompt verwenden
                fallback_prompt = prompt_safety_filter.get_safe_fallback_prompt(req.prompt)
                logger.info({
                    "event": USING_FALLBACK_PROMPT_EVENT,
                    "fallback_prompt": fallback_prompt
                })

                try:
                    return await self._execute_with_retry(
                        self._perform_image_generation,
                        req,
                        fallback_prompt
                    )
                except Exception:
                    # Auch Fallback fehlgeschlagen - hilfreiche Fehlermeldung
                    suggestions = prompt_safety_filter.create_safe_prompt_suggestions(req.prompt)
                    raise RuntimeError(
                        f"Bildgenerierung aufgrund von Content Policy Violation fehlgeschlagen. "
                        f"Versuchen Sie einen dieser sicheren Prompts: {', '.join(suggestions)}"
                    )

            # Andere Fehler weiterleiten
            logger.exception({
                "event": OPENAI_IMAGES_GENERATE_ERROR_EVENT,
                "error": str(e),
                "error_type": type(e).__name__,
            })
            raise


def create_image_generation_service() -> ImageGenerationService:
    """Factory-Funktion für ImageGenerationService."""
    # Deployment/API-Version optional aus .env laden
    try:
        from os import getenv
        deployment = getenv("PROJECT_KEIKO_IMAGE_DEPLOYMENT", DEFAULT_IMAGE_DEPLOYMENT)
        api_version = getenv("PROJECT_KEIKO_IMAGE_API_VERSION", DEFAULT_IMAGE_API_VERSION)
    except Exception:
        deployment = DEFAULT_IMAGE_DEPLOYMENT
        api_version = DEFAULT_IMAGE_API_VERSION

    return ImageGenerationService(deployment=deployment, api_version=api_version)
