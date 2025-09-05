"""Bildgenerator-Agent für KI-basierte Bilderstellung.

Implementiert Bildgenerierung mit Content Safety, Storage-Upload und
strukturierter Fehlerbehandlung.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from agents.base_agent import AgentMetadata, AgentPerformanceMetrics, BaseAgent
from agents.constants import (
    ALLOWED_IMAGE_QUALITIES,
    ALLOWED_IMAGE_SIZES,
    ALLOWED_IMAGE_STYLES,
    DEFAULT_SAFETY_SCORE,
    IMAGE_GENERATION_ERROR_THRESHOLD_MS,
    IMAGE_GENERATION_WARNING_THRESHOLD_MS,
    METRICS_INCREMENT,
    MIN_PROMPT_LENGTH,
    SECONDS_TO_MILLISECONDS,
    AgentStatus,
    AgentType,
    ImageQuality,
    ImageSize,
    ImageStyle,
)
from agents.constants import (
    ErrorMessagesImageGenerator as ErrorMessages,
)
from agents.constants import (
    LogEventsImageGenerator as LogEvents,
)
from agents.constants import (
    MetricsNamesImageGenerator as MetricsNames,
)
from config.settings import settings
from core.exceptions import KeikoValidationError
from kei_logging import get_logger
from services.clients.content_safety import ContentSafetyClient, create_content_safety_client
from services.clients.image_generation import (
    ImageGenerationRequest,
    ImageGenerationService,
    create_image_generation_service,
)

if TYPE_CHECKING:
    from monitoring.custom_metrics import MetricsCollector

logger = get_logger(__name__)


# =============================================================================
# Datenklassen für Task und Ergebnisse
# =============================================================================

@dataclass(slots=True)
class ImageTask:
    """Eingabeparameter für den Agent."""

    prompt: str
    size: ImageSize = ImageSize.SQUARE
    quality: ImageQuality = ImageQuality.STANDARD
    style: ImageStyle = ImageStyle.REALISTIC
    user_id: str | None = None
    session_id: str | None = None


@dataclass(slots=True)
class PromptResult:
    """Ergebnis der Prompt-Vorbereitung."""
    original_prompt: str
    sanitized_prompt: str
    optimized_prompt: str
    sanitization_result: Any  # SanitizationResult


@dataclass(slots=True)
class SafetyResult:
    """Ergebnis des Content Safety Checks."""
    is_safe: bool
    score: float
    category: str
    duration_ms: float


@dataclass(slots=True)
class GenerationResult:
    """Ergebnis der Bildgenerierung."""
    image_bytes: bytes
    content_type: str
    duration_ms: float


@dataclass(slots=True)
class UploadResult:
    """Ergebnis des Storage-Uploads."""
    blob_url: str
    sas_url: str
    filename: str
    duration_ms: float


class ImageGeneratorAgent(BaseAgent[ImageTask, dict[str, Any]]):
    """Bildgenerator-Agent für KI-basierte Bilderstellung.

    Implementiert sichere Bildgenerierung mit Content Safety Checks,
    automatischem Storage-Upload und strukturierter Fehlerbehandlung.
    """

    def __init__(
        self,
        *,
        storage_container: str | None = None,
        content_safety: ContentSafetyClient | None = None,
        image_service: ImageGenerationService | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Initialisiert den Image Generator Agent.

        Args:
            storage_container: Optionaler Storage-Container Name
            content_safety: Optionaler Content Safety Client
            image_service: Optionaler Image Generation Service
            metrics: Optionaler Metrics Collector
        """
        # Externe Services initialisieren
        self._safety = content_safety or create_content_safety_client()
        self._images = image_service or create_image_generation_service()

        # Storage und Prompt Utilities
        from utils.prompt_sanitizer import PromptSanitizer, SanitizationStrategy
        from utils.storage_utils import StorageUtils

        self._storage = StorageUtils(
            default_container=storage_container or settings.keiko_storage_container_for_img
        )
        self._prompt_sanitizer = PromptSanitizer(strategy=SanitizationStrategy.COMPREHENSIVE)

        # Agent-Status bestimmen
        agent_status = (
            AgentStatus.AVAILABLE
            if self._images.is_available and self._storage.is_available
            else AgentStatus.OFFLINE
        )

        # BaseAgent initialisieren
        super().__init__(
            agent_id=settings.agent_image_generator_id or "agent_image_generator",
            name="Image Generator Agent",
            agent_type=AgentType.CUSTOM,  # Verwende Framework-Type, nicht Function-Type
            description="Generiert Bilder via DALL·E-3 und speichert in Azure Storage",
            capabilities=["image_generation", "content_safety", "storage_upload"],
            metrics_collector=metrics,
        )

        # Status aktualisieren
        self.update_status(agent_status)

        # Initialisiere Mixin-kompatible Attribute
        self._metadata = self.metadata
        self._performance_metrics = self.performance_metrics

    # Implementierung der abstrakten Properties aus den Mixins
    @property
    def metadata(self) -> AgentMetadata:
        """Agent-Metadaten für Mixin-Kompatibilität."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: AgentMetadata) -> None:
        """Setter für Agent-Metadaten."""
        self._metadata = value

    @property
    def performance_metrics(self) -> AgentPerformanceMetrics:
        """Performance-Metriken für Mixin-Kompatibilität."""
        return self._performance_metrics

    @performance_metrics.setter
    def performance_metrics(self, value: AgentPerformanceMetrics) -> None:
        """Setter für Performance-Metriken."""
        self._performance_metrics = value

    def _get_performance_thresholds(self) -> tuple[float, float]:
        """Gibt spezifische Performance-Schwellenwerte für Image Generation zurück.

        Returns:
            Tuple von (warning_threshold_ms, error_threshold_ms)
        """
        return IMAGE_GENERATION_WARNING_THRESHOLD_MS, IMAGE_GENERATION_ERROR_THRESHOLD_MS

    @staticmethod
    def _calculate_duration_ms(start_time: float) -> float:
        """Berechnet die Ausführungsdauer in Millisekunden.

        Args:
            start_time: Start-Zeit von time.perf_counter()

        Returns:
            Dauer in Millisekunden
        """
        return (time.perf_counter() - start_time) * SECONDS_TO_MILLISECONDS

    async def _execute_task(self, task: ImageTask) -> dict[str, Any]:
        """Führt die Bildgenerierung durch.

        Args:
            task: Bildgenerierungsauftrag

        Returns:
            Strukturierte Antwort mit Storage-URL und Metadaten.
        """
        # Validierung und Prompt-Vorbereitung
        prompt_result = await self._validate_and_prepare(task)

        # Content Safety Check und Bildgenerierung
        generation_result = await self._check_safety_and_generate(task, prompt_result)

        # Upload und Response-Building
        return await self._upload_and_respond(task, prompt_result, generation_result)

    async def _validate_and_prepare(self, task: ImageTask) -> PromptResult:
        """Validiert Task und bereitet Prompt vor.

        Args:
            task: Bildgenerierungsauftrag

        Returns:
            Vorbereitetes Prompt-Ergebnis
        """
        self._validate_task(task)
        return await self._prepare_prompt(task)

    async def _check_safety_and_generate(self, task: ImageTask, prompt_result: PromptResult) -> GenerationResult:
        """Führt Safety-Check durch und generiert Bild.

        Args:
            task: Bildgenerierungsauftrag
            prompt_result: Vorbereitetes Prompt-Ergebnis

        Returns:
            Generiertes Bild-Ergebnis

        Raises:
            Exception: Bei Content Safety Blockierung oder Generierungsfehlern
        """
        safety_result = await self._perform_safety_check(prompt_result.optimized_prompt)
        if not safety_result.is_safe:
            # Strukturierte Exception für Safety-Blockierung
            error_details = {
                "reason": "content_safety_blocked",
                "score": safety_result.score,
                "category": safety_result.category,
            }
            raise KeikoValidationError(
                ErrorMessages.CONTENT_SAFETY_BLOCKED, details=error_details
            )

        return await self._generate_image(task, prompt_result.optimized_prompt)

    async def _upload_and_respond(
        self,
        task: ImageTask,
        prompt_result: PromptResult,
        generation_result: GenerationResult
    ) -> dict[str, Any]:
        """Lädt Bild hoch und erstellt Response.

        Args:
            task: Bildgenerierungsauftrag
            prompt_result: Prompt-Ergebnis
            generation_result: Generierungs-Ergebnis

        Returns:
            Strukturierte Erfolgs-Antwort
        """
        upload_result = await self._upload_to_storage(
            generation_result.image_bytes,
            generation_result.content_type,
            task.user_id,
            task.session_id,
        )

        # Safety-Result für Response erstellen
        safety_result = SafetyResult(
            is_safe=True,
            score=1.0,
            category="safe",
            duration_ms=0.0,
        )

        return self._build_success_response(
            task=task,
            prompt_result=prompt_result,
            safety_result=safety_result,
            generation_result=generation_result,
            upload_result=upload_result,
        )

    def _validate_task(self, task: ImageTask) -> None:
        """Validiert Task-Parameter.

        Args:
            task: Zu validierender ImageTask

        Raises:
            KeikoValidationError: Bei ungültigen Parametern
        """
        if task.size not in ALLOWED_IMAGE_SIZES:
            raise KeikoValidationError(ErrorMessages.INVALID_SIZE, details={"size": task.size})

        if task.quality not in ALLOWED_IMAGE_QUALITIES:
            raise KeikoValidationError(ErrorMessages.INVALID_QUALITY, details={"quality": task.quality})

        if task.style not in ALLOWED_IMAGE_STYLES:
            raise KeikoValidationError(ErrorMessages.INVALID_STYLE, details={"style": task.style})

        if not task.prompt or len(task.prompt.strip()) < MIN_PROMPT_LENGTH:
            raise KeikoValidationError(ErrorMessages.PROMPT_REQUIRED)

    async def _prepare_prompt(self, task: ImageTask) -> PromptResult:
        """Bereitet und optimiert den Prompt vor.

        Args:
            task: ImageTask mit Prompt-Informationen

        Returns:
            PromptResult mit sanitisiertem und optimiertem Prompt
        """
        original_prompt = task.prompt.strip()

        # Prompt bereinigen
        sanitization_result = self._prompt_sanitizer.sanitize(original_prompt)

        # Prompt für gewählten Stil optimieren
        optimized_prompt = self._prompt_sanitizer.optimize_for_style(
            sanitization_result.sanitized_prompt,
            task.style
        )

        logger.debug(
            {
                "event": "prompt_prepared",
                "original_length": len(original_prompt),
                "sanitized_length": len(sanitization_result.sanitized_prompt),
                "optimized_length": len(optimized_prompt),
                "is_safe": sanitization_result.is_safe,
                "violations": sanitization_result.violations_detected,
            }
        )

        return PromptResult(
            original_prompt=original_prompt,
            sanitized_prompt=sanitization_result.sanitized_prompt,
            optimized_prompt=optimized_prompt,
            sanitization_result=sanitization_result,
        )

    async def _perform_safety_check(self, prompt: str) -> SafetyResult:
        """Führt Content Safety Check durch."""
        start_time = time.perf_counter()

        try:
            safety_result = await self._safety.analyze_text(prompt)
            duration_ms = self._calculate_duration_ms(start_time)

            # Performance-Metriken erfassen
            self._record_metric(MetricsNames.CONTENT_SAFETY_LATENCY, duration_ms)

            if not safety_result.is_safe:
                self._record_metric(MetricsNames.CONTENT_SAFETY_BLOCKED, METRICS_INCREMENT)

            logger.debug(
                {
                    "event": LogEvents.CONTENT_SAFETY_CHECK,
                    "is_safe": safety_result.is_safe,
                    "score": safety_result.score,
                    "category": safety_result.category,
                    "duration_ms": duration_ms,
                }
            )

            return SafetyResult(
                is_safe=safety_result.is_safe,
                score=safety_result.score,
                category=safety_result.category,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = self._calculate_duration_ms(start_time)
            logger.exception(
                {
                    "event": LogEvents.CONTENT_SAFETY_ERROR,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_ms": duration_ms,
                }
            )

            # Bei Fehler als unsicher behandeln
            return SafetyResult(
                is_safe=False,
                score=DEFAULT_SAFETY_SCORE,
                category="error",
                duration_ms=duration_ms,
            )

    async def _generate_image(self, task: ImageTask, prompt: str) -> GenerationResult:
        """Generiert das Bild via DALL·E-3."""
        start_time = time.perf_counter()

        try:
            image_req = ImageGenerationRequest(
                prompt=prompt,
                size=task.size.value,
                quality=task.quality.value,
                style=task.style.value,
                user_id=task.user_id,
                session_id=task.session_id,
            )

            logger.debug(
                {
                    "event": LogEvents.IMAGE_GENERATION_START,
                    "size": task.size,
                    "quality": task.quality,
                    "style": task.style,
                }
            )

            result = await self._images.generate(image_req)
            duration_ms = self._calculate_duration_ms(start_time)

            # Performance-Metriken erfassen
            self._record_metric(MetricsNames.IMAGE_GENERATION_LATENCY, duration_ms)

            logger.debug(
                {
                    "event": LogEvents.IMAGE_GENERATION_COMPLETE,
                    "duration_ms": duration_ms,
                    "content_type": result.content_type,
                    "size_bytes": len(result.image_bytes),
                }
            )

            return GenerationResult(
                image_bytes=result.image_bytes,
                content_type=result.content_type,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = self._calculate_duration_ms(start_time)
            logger.exception(
                {
                    "event": LogEvents.IMAGE_GENERATION_ERROR,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_ms": duration_ms,
                    "prompt_length": len(prompt),
                }
            )
            # Metrik für fehlgeschlagene Generierung erfassen
            self._record_metric(MetricsNames.IMAGE_GENERATION_LATENCY, duration_ms)
            raise

    async def _upload_to_storage(
        self,
        image_data: bytes,
        content_type: str,
        user_id: str | None,
        session_id: str | None,
    ) -> UploadResult:
        """Lädt das Bild in Azure Storage hoch."""
        start_time = time.perf_counter()

        try:
            upload_result = await self._storage.upload_image(
                image_data=image_data,
                content_type=content_type,
                user_id=user_id,
                session_id=session_id,
            )

            duration_ms = self._calculate_duration_ms(start_time)

            # Performance-Metriken erfassen
            self._record_metric(MetricsNames.STORAGE_UPLOAD_LATENCY, duration_ms)

            return UploadResult(
                blob_url=upload_result["blob_url"],
                sas_url=upload_result["sas_url"],
                filename=upload_result["filename"],
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = self._calculate_duration_ms(start_time)
            logger.warning(
                f"Storage-Upload fehlgeschlagen, verwende Base64-Fallback: {e}"
            )
            logger.exception(
                {
                    "event": LogEvents.STORAGE_UPLOAD_ERROR,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_ms": duration_ms,
                    "fallback": "base64_data_url",
                    "image_size_bytes": len(image_data),
                }
            )

            # Fallback auf Base64 Data URL
            import base64
            base64_data = base64.b64encode(image_data).decode("utf-8")
            data_url = f"data:{content_type};base64,{base64_data}"

            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            fallback_filename = f"fallback/{timestamp}.png"

            return UploadResult(
                blob_url=data_url,
                sas_url=data_url,
                filename=fallback_filename,
                duration_ms=duration_ms,
            )

    @staticmethod
    def _build_success_response(
        task: ImageTask,
        prompt_result: PromptResult,
        safety_result: SafetyResult,
        generation_result: GenerationResult,
        upload_result: UploadResult,
    ) -> dict[str, Any]:
        """Erstellt erfolgreiche Antwort.

        Args:
            task: Bildgenerierungsauftrag mit Parametern
            prompt_result: Ergebnis der Prompt-Vorbereitung
            safety_result: Ergebnis des Content Safety Checks
            generation_result: Ergebnis der Bildgenerierung
            upload_result: Ergebnis des Storage-Uploads

        Returns:
            Strukturierte Erfolgs-Antwort mit allen Metadaten
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

        return {
            "status": "success",
            "storage_url": upload_result.sas_url,
            "blob_url": upload_result.blob_url,
            "file_name": upload_result.filename,
            "content_type": generation_result.content_type,
            "metadata": {
                "original_prompt": prompt_result.original_prompt,
                "optimized_prompt": prompt_result.optimized_prompt,
                "timestamp": timestamp,
                "user_id": task.user_id,
                "session_id": task.session_id,
                "parameters": {
                    "size": task.size,
                    "style": task.style,
                    "quality": task.quality,
                },
                "safety": {
                    "score": safety_result.score,
                    "category": safety_result.category,
                },
                "performance": {
                    "safety_ms": safety_result.duration_ms,
                    "generation_ms": generation_result.duration_ms,
                    "upload_ms": upload_result.duration_ms,
                },
            },
        }

    @staticmethod
    def _build_error_response(
        reason: str, extra: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Erstellt Fehler-Antwort.

        Args:
            reason: Grund für den Fehler
            extra: Zusätzliche Fehler-Details

        Returns:
            Strukturierte Fehler-Antwort
        """
        response = {"status": "failed", "reason": reason}
        if extra:
            response.update(extra)
        return response
