"""KEI-Stream Rate-Limiting Error-Handler.

Implementiert umfassendes Error-Handling für Rate-Limiting:
- HTTP 429 Response-Management
- Retry-After-Header-Berechnung
- Graceful Degradation bei System-Fehlern
- Client-Guidance für Rate-Limit-Recovery

@version 1.0.0
"""

import time
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from middleware.rate_limiting_types import (
    KEIStreamEndpointType,
    KEIStreamRateLimitConfig,
    KEIStreamRateLimitResult,
    KEIStreamRateLimitStrategy,
    RateLimitErrorType,
)
from observability import get_logger, record_custom_metric

logger = get_logger(__name__)


@dataclass
class RateLimitErrorContext:
    """Kontext-Informationen für Rate-Limiting-Fehler."""
    error_type: RateLimitErrorType
    identifier: str
    strategy: KEIStreamRateLimitStrategy
    endpoint_type: KEIStreamEndpointType
    current_usage: dict[str, Any]
    limits: dict[str, Any]
    retry_after_seconds: int | None = None
    recovery_suggestions: list[str] = None

    def __post_init__(self):
        if self.recovery_suggestions is None:
            self.recovery_suggestions = []


class KEIStreamRateLimitErrorHandler:
    """Spezialisierter Error-Handler für KEI-Stream Rate-Limiting.

    Bietet intelligente Error-Responses mit:
    - Detaillierte Fehlerinformationen
    - Client-Guidance für Recovery
    - Adaptive Retry-After-Berechnung
    - Graceful Degradation-Strategien
    """

    def __init__(self):
        self.error_counts: dict[str, int] = {}
        self.last_error_times: dict[str, float] = {}

    def create_rate_limit_error_response(
        self,
        request: Request,
        result: KEIStreamRateLimitResult,
        config: KEIStreamRateLimitConfig,
        error_context: RateLimitErrorContext | None = None
    ) -> JSONResponse:
        """Erstellt detaillierte HTTP 429 Response für Rate-Limiting-Fehler.

        Args:
            request: FastAPI-Request-Objekt
            result: Rate-Limiting-Prüfungsergebnis
            config: Rate-Limiting-Konfiguration
            error_context: Zusätzliche Fehler-Kontext-Informationen

        Returns:
            JSONResponse mit detaillierten Fehlerinformationen
        """
        # Bestimme Error-Type basierend auf Kontext
        error_type = self._determine_error_type(result, config, error_context)

        # Berechne intelligente Retry-After-Zeit
        retry_after = self._calculate_adaptive_retry_after(
            request, result, config, error_type
        )

        # Erstelle Recovery-Suggestions
        recovery_suggestions = self._generate_recovery_suggestions(
            error_type, config, result
        )

        # Erstelle Response-Headers
        headers = self._create_response_headers(
            result, config, retry_after
        )

        # Erstelle detaillierte Error-Response
        error_response = self._create_error_response_body(
            request, result, config, error_type, retry_after, recovery_suggestions
        )

        # Aufzeichnung für Monitoring
        self._record_error_metrics(request, error_type, config)

        return JSONResponse(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            content=error_response,
            headers=headers
        )

    def _determine_error_type(
        self,
        result: KEIStreamRateLimitResult,
        config: KEIStreamRateLimitConfig,
        error_context: RateLimitErrorContext | None
    ) -> RateLimitErrorType:
        """Bestimmt spezifischen Error-Type basierend auf Kontext."""
        if error_context and error_context.error_type:
            return error_context.error_type

        # Analysiere Result für Error-Type
        if result.frame_rate_status and result.frame_rate_status.get("current_frame_tokens", 0) <= 0:
            return RateLimitErrorType.FRAME_RATE_EXCEEDED

        if result.stream_limits and result.stream_limits.get("stream_count", 0) >= config.max_concurrent_streams:
            return RateLimitErrorType.CONCURRENT_STREAMS_EXCEEDED

        if result.current_usage and result.current_usage.get("current_tokens", 0) <= 0:
            if config.burst_capacity and result.remaining_requests <= 0:
                return RateLimitErrorType.BURST_CAPACITY_EXCEEDED
            return RateLimitErrorType.REQUEST_RATE_EXCEEDED

        return RateLimitErrorType.REQUEST_RATE_EXCEEDED

    def _calculate_adaptive_retry_after(
        self,
        request: Request,
        result: KEIStreamRateLimitResult,
        config: KEIStreamRateLimitConfig,
        error_type: RateLimitErrorType
    ) -> int:
        """Berechnet adaptive Retry-After-Zeit basierend auf Error-Type und Historie."""
        base_retry_after = result.retry_after_seconds or 1

        # Identifier für Error-Tracking
        identifier = f"{request.client.host if request.client else 'unknown'}:{request.url.path}"

        # Erhöhe Retry-After bei wiederholten Fehlern
        current_time = time.time()
        if identifier in self.last_error_times:
            time_since_last_error = current_time - self.last_error_times[identifier]

            # Wenn Fehler innerhalb kurzer Zeit wiederholt auftreten
            if time_since_last_error < 60:  # 1 Minute
                self.error_counts[identifier] = self.error_counts.get(identifier, 0) + 1

                # Exponential Backoff für wiederholte Fehler
                backoff_multiplier = min(2 ** self.error_counts[identifier], 16)
                base_retry_after = int(base_retry_after * backoff_multiplier)
            else:
                # Reset bei längerer Pause
                self.error_counts[identifier] = 1
        else:
            self.error_counts[identifier] = 1

        self.last_error_times[identifier] = current_time

        # Error-Type-spezifische Anpassungen
        if error_type == RateLimitErrorType.FRAME_RATE_EXCEEDED:
            # Kürzere Retry-Zeit für Frame-Rate-Limits
            base_retry_after = max(1, base_retry_after // 2)
        elif error_type == RateLimitErrorType.CONCURRENT_STREAMS_EXCEEDED:
            # Längere Retry-Zeit für Stream-Limits
            base_retry_after = max(30, base_retry_after * 2)
        elif error_type == RateLimitErrorType.BURST_CAPACITY_EXCEEDED:
            # Mittlere Retry-Zeit für Burst-Limits
            base_retry_after = max(5, base_retry_after)

        # Begrenze maximale Retry-Zeit
        return min(base_retry_after, config.max_backoff_seconds)

    def _generate_recovery_suggestions(
        self,
        error_type: RateLimitErrorType,
        config: KEIStreamRateLimitConfig,
        result: KEIStreamRateLimitResult
    ) -> list[str]:
        """Generiert spezifische Recovery-Suggestions basierend auf Error-Type."""
        suggestions = []

        if error_type == RateLimitErrorType.REQUEST_RATE_EXCEEDED:
            suggestions.extend([
                f"Reduzieren Sie Ihre Request-Rate auf maximal {config.requests_per_second} Requests pro Sekunde",
                "Implementieren Sie exponential backoff in Ihrem Client",
                "Verwenden Sie Batch-Requests wo möglich"
            ])

        elif error_type == RateLimitErrorType.FRAME_RATE_EXCEEDED:
            if config.frames_per_second:
                suggestions.extend([
                    f"Reduzieren Sie Ihre Frame-Rate auf maximal {config.frames_per_second} Frames pro Sekunde",
                    "Implementieren Sie Frame-Batching für bessere Effizienz",
                    "Verwenden Sie Compression für große Frames"
                ])

        elif error_type == RateLimitErrorType.CONCURRENT_STREAMS_EXCEEDED:
            if config.max_concurrent_streams:
                suggestions.extend([
                    f"Begrenzen Sie gleichzeitige Streams auf maximal {config.max_concurrent_streams}",
                    "Schließen Sie ungenutzte Streams",
                    "Implementieren Sie Stream-Pooling"
                ])

        elif error_type == RateLimitErrorType.BURST_CAPACITY_EXCEEDED:
            suggestions.extend([
                f"Warten Sie bis Burst-Kapazität wieder verfügbar ist (max. {config.burst_capacity} Requests)",
                "Verteilen Sie Requests gleichmäßiger über die Zeit",
                "Implementieren Sie Request-Queuing"
            ])

        # Allgemeine Suggestions
        suggestions.extend([
            "Prüfen Sie Ihre Tenant-Limits und erwägen Sie ein Upgrade",
            "Implementieren Sie Client-seitiges Rate-Limiting",
            "Kontaktieren Sie den Support bei anhaltenden Problemen"
        ])

        return suggestions

    def _create_response_headers(
        self,
        result: KEIStreamRateLimitResult,
        config: KEIStreamRateLimitConfig,
        retry_after: int
    ) -> dict[str, str]:
        """Erstellt umfassende Response-Headers für Rate-Limiting."""
        headers = {
            # Standard Rate-Limiting-Headers
            "X-RateLimit-Limit": str(int(config.requests_per_second * config.window_size_seconds)),
            "X-RateLimit-Remaining": str(result.remaining_requests),
            "X-RateLimit-Reset": str(int(result.reset_time)),
            "X-RateLimit-Window": str(config.window_size_seconds),
            "Retry-After": str(retry_after),

            # KEI-Stream-spezifische Headers
            "X-RateLimit-Strategy": config.strategy.value,
            "X-RateLimit-Endpoint-Type": config.endpoint_type.value,
            "X-RateLimit-Requests-Per-Second": str(config.requests_per_second),
            "X-RateLimit-Burst-Capacity": str(config.burst_capacity),
        }

        # Optionale KEI-Stream-Headers
        if config.frames_per_second:
            headers["X-RateLimit-Frames-Per-Second"] = str(config.frames_per_second)

        if config.max_concurrent_streams:
            headers["X-RateLimit-Max-Concurrent-Streams"] = str(config.max_concurrent_streams)

        if config.max_stream_duration_seconds:
            headers["X-RateLimit-Max-Stream-Duration"] = str(config.max_stream_duration_seconds)

        # Frame-Rate-Status-Headers
        if result.frame_rate_status:
            headers["X-RateLimit-Frame-Tokens-Remaining"] = str(
                result.frame_rate_status.get("current_frame_tokens", 0)
            )

        # Stream-Limit-Headers
        if result.stream_limits:
            headers["X-RateLimit-Current-Streams"] = str(
                result.stream_limits.get("stream_count", 0)
            )

        return headers

    def _create_error_response_body(
        self,
        request: Request,
        result: KEIStreamRateLimitResult,
        config: KEIStreamRateLimitConfig,
        error_type: RateLimitErrorType,
        retry_after: int,
        recovery_suggestions: list[str]
    ) -> dict[str, Any]:
        """Erstellt detaillierte Error-Response-Body."""
        return {
            "error": "kei_stream_rate_limit_exceeded",
            "error_type": error_type.value,
            "message": self._get_error_message(error_type, config),
            "timestamp": time.time(),
            "request_id": request.headers.get("X-Request-ID", "unknown"),

            "rate_limit_details": {
                "strategy": config.strategy.value,
                "endpoint_type": config.endpoint_type.value,
                "current_limits": {
                    "requests_per_second": config.requests_per_second,
                    "burst_capacity": config.burst_capacity,
                    "frames_per_second": config.frames_per_second,
                    "max_concurrent_streams": config.max_concurrent_streams,
                    "window_seconds": config.window_size_seconds
                },
                "current_usage": result.current_usage or {},
                "remaining_requests": result.remaining_requests,
                "reset_time": result.reset_time,
                "retry_after_seconds": retry_after
            },

            "kei_stream_specific": {
                "frame_rate_status": result.frame_rate_status,
                "stream_limits": result.stream_limits,
                "limit_exceeded_by": result.limit_exceeded_by
            },

            "recovery_guidance": {
                "suggestions": recovery_suggestions,
                "retry_after_seconds": retry_after,
                "backoff_strategy": "exponential",
                "max_backoff_seconds": config.max_backoff_seconds
            },

            "support_information": {
                "documentation_url": "https://docs.services-streaming.com/rate-limiting",
                "contact_support": "support@services-streaming.com",
                "upgrade_options": "https://services-streaming.com/pricing"
            }
        }

    def _get_error_message(
        self,
        error_type: RateLimitErrorType,
        config: KEIStreamRateLimitConfig
    ) -> str:
        """Gibt benutzerfreundliche Error-Message zurück."""
        messages = {
            RateLimitErrorType.REQUEST_RATE_EXCEEDED:
                f"Request-Rate-Limit überschritten. Maximal {config.requests_per_second} Requests pro Sekunde erlaubt.",
            RateLimitErrorType.FRAME_RATE_EXCEEDED:
                f"Frame-Rate-Limit überschritten. Maximal {config.frames_per_second} Frames pro Sekunde erlaubt.",
            RateLimitErrorType.CONCURRENT_STREAMS_EXCEEDED:
                f"Concurrent-Stream-Limit überschritten. Maximal {config.max_concurrent_streams} gleichzeitige Streams erlaubt.",
            RateLimitErrorType.BURST_CAPACITY_EXCEEDED:
                f"Burst-Kapazität überschritten. Maximal {config.burst_capacity} Requests im Burst erlaubt.",
            RateLimitErrorType.STREAM_DURATION_EXCEEDED:
                f"Stream-Dauer-Limit überschritten. Maximal {config.max_stream_duration_seconds} Sekunden pro Stream erlaubt.",
            RateLimitErrorType.QUOTA_EXCEEDED:
                "Quota-Limit überschritten. Bitte erwägen Sie ein Upgrade Ihres Plans.",
            RateLimitErrorType.SYSTEM_ERROR:
                "Temporärer System-Fehler beim Rate-Limiting. Bitte versuchen Sie es später erneut."
        }

        return messages.get(error_type, "Rate-Limit überschritten. Bitte verlangsamen Sie Ihre Anfragen.")

    def _record_error_metrics(
        self,
        request: Request,
        error_type: RateLimitErrorType,
        config: KEIStreamRateLimitConfig
    ):
        """Zeichnet Error-Metriken für Monitoring auf."""
        try:
            record_custom_metric("kei_stream_rate_limiting_errors_total", 1, {
                "error_type": error_type.value,
                "strategy": config.strategy.value,
                "endpoint_type": config.endpoint_type.value,
                "method": request.method,
                "endpoint": request.url.path
            })
        except Exception as e:
            logger.warning(f"Fehler beim Aufzeichnen von Error-Metriken: {e}")


# Globale Error-Handler-Instanz
_error_handler: KEIStreamRateLimitErrorHandler | None = None


def get_rate_limit_error_handler() -> KEIStreamRateLimitErrorHandler:
    """Holt oder erstellt globalen Rate-Limit-Error-Handler."""
    global _error_handler
    if _error_handler is None:
        _error_handler = KEIStreamRateLimitErrorHandler()
    return _error_handler
