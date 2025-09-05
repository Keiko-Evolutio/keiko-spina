"""Gemeinsame Typen für Rate Limiting.

Dieses Modul enthält gemeinsame Typen und Enums, die von verschiedenen
Rate Limiting Modulen verwendet werden, um zirkuläre Imports zu vermeiden.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class RateLimitErrorType(str, Enum):
    """Typen von Rate Limiting Fehlern."""
    # Standard Rate-Limiting-Fehler
    QUOTA_EXCEEDED = "quota_exceeded"
    BURST_LIMIT_EXCEEDED = "burst_limit_exceeded"
    CONCURRENT_LIMIT_EXCEEDED = "concurrent_limit_exceeded"
    BANDWIDTH_LIMIT_EXCEEDED = "bandwidth_limit_exceeded"
    REQUEST_SIZE_EXCEEDED = "request_size_exceeded"
    CUSTOM_LIMIT_EXCEEDED = "custom_limit_exceeded"

    # KEI-Stream-spezifische Fehler
    REQUEST_RATE_EXCEEDED = "request_rate_exceeded"
    FRAME_RATE_EXCEEDED = "frame_rate_exceeded"
    CONCURRENT_STREAMS_EXCEEDED = "concurrent_streams_exceeded"
    BURST_CAPACITY_EXCEEDED = "burst_capacity_exceeded"
    STREAM_DURATION_EXCEEDED = "stream_duration_exceeded"
    SYSTEM_ERROR = "system_error"


class KEIStreamEndpointType(Enum):
    """Typen von KEI Stream Endpoints."""
    WEBSOCKET = "websocket"
    SSE = "sse"
    GRPC = "grpc"
    REST = "rest"
    REST_API = "rest_api"  # Alias für REST
    WEBHOOK = "webhook"
    STREAM_MANAGEMENT = "stream_management"
    TOOL_EXECUTION = "tool_execution"


class KEIStreamRateLimitStrategy(Enum):
    """Rate Limiting Strategien für KEI Stream."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class KEIStreamRateLimitResult(BaseModel):
    """Ergebnis einer Rate Limit Prüfung."""
    allowed: bool
    remaining: int | None = None
    reset_time: float | None = None
    retry_after: int | None = None
    error_type: RateLimitErrorType | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = {}


class KEIStreamRateLimitConfig(BaseModel):
    """Konfiguration für KEI Stream Rate Limiting."""
    enabled: bool = True
    strategy: KEIStreamRateLimitStrategy = KEIStreamRateLimitStrategy.TOKEN_BUCKET
    requests_per_minute: int = 60
    burst_size: int = 10
    concurrent_connections: int = 100
    bandwidth_limit_mbps: float | None = None
    max_request_size_mb: float | None = None
    custom_limits: dict[str, Any] = {}


class RateLimitErrorContext(BaseModel):
    """Kontext für Rate Limiting Fehler."""
    error_type: RateLimitErrorType
    endpoint_type: KEIStreamEndpointType
    user_id: str | None = None
    client_ip: str | None = None
    endpoint_path: str | None = None
    current_usage: int | None = None
    limit: int | None = None
    reset_time: float | None = None
    retry_after: int | None = None
    additional_info: dict[str, Any] = {}


__all__ = [
    "KEIStreamEndpointType",
    "KEIStreamRateLimitConfig",
    "KEIStreamRateLimitResult",
    "KEIStreamRateLimitStrategy",
    "RateLimitErrorContext",
    "RateLimitErrorType",
]
