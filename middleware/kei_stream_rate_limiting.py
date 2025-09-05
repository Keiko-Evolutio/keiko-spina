"""KEI-Stream Rate-Limiting-Middleware.

Implementiert umfassende Rate-Limiting-Funktionalität speziell für KEI-Stream:
- Token-Bucket-Algorithmus für gleichmäßige Rate-Limiting
- Verschiedene Rate-Limiting-Strategien (IP, User, API-Key, Tenant)
- Endpoint-spezifische Limits (WebSocket, SSE, REST)
- Redis-basierte verteilte Rate-Limiting für Skalierbarkeit
- Integration mit KEI-Stream Token-Bucket-System

@version 1.0.0
"""

import builtins
import contextlib
import json
import time
from dataclasses import dataclass
from typing import Any

import redis.asyncio as redis
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from observability import get_logger, record_custom_metric
from services.streaming.token_bucket import TokenBucket as KEITokenBucket

from .constants import RateLimitConstants
from .rate_limiting_types import (
    KEIStreamEndpointType,
    KEIStreamRateLimitResult,
    RateLimitErrorContext,
    RateLimitErrorType,
)
from .rate_limiting_types import KEIStreamRateLimitStrategy as AlgorithmStrategy
from .utils.client_identification import ClientIdentificationUtils, IdentificationStrategy


# Import error handler function (delayed to avoid circular import)
def _get_rate_limit_error_handler():
    """Lazy import für Error Handler um zirkuläre Imports zu vermeiden."""
    from .rate_limiting_error_handler import get_rate_limit_error_handler
    return get_rate_limit_error_handler()

logger = get_logger(__name__)


# Verwende die konsolidierte IdentificationStrategy aus utils
KEIStreamIdentificationStrategy = IdentificationStrategy


# KEIStreamEndpointType wird aus rate_limiting_types importiert


@dataclass
class KEIStreamRateLimitConfig:
    """Konfiguration für KEI-Stream Rate-Limiting."""
    requests_per_second: float
    burst_capacity: int
    window_size_seconds: int = 60
    identification_strategy: KEIStreamIdentificationStrategy = KEIStreamIdentificationStrategy.IP_ADDRESS
    endpoint_type: KEIStreamEndpointType = KEIStreamEndpointType.REST_API
    enabled: bool = True

    # Algorithmus-Strategie (für Kompatibilität mit Pydantic-Modell)
    algorithm_strategy: AlgorithmStrategy = AlgorithmStrategy.TOKEN_BUCKET

    # KEI-Stream-spezifische Konfiguration
    frames_per_second: float | None = None  # Für WebSocket/SSE
    max_concurrent_streams: int | None = None
    max_stream_duration_seconds: int | None = None

    # Erweiterte Konfiguration
    grace_period_seconds: int = 5
    backoff_multiplier: float = 1.5
    max_backoff_seconds: int = 300

    def __post_init__(self):
        """Validiert Konfigurationsparameter."""
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second muss positiv sein")
        if self.burst_capacity <= 0:
            raise ValueError("burst_capacity muss positiv sein")
        if self.window_size_seconds <= 0:
            raise ValueError("window_size_seconds muss positiv sein")


@dataclass
class KEIStreamRateLimitResult:
    """Ergebnis einer KEI-Stream Rate-Limiting-Prüfung."""
    allowed: bool
    remaining_requests: int
    reset_time: float
    retry_after_seconds: int | None = None
    limit_exceeded_by: str | None = None
    current_usage: dict[str, Any] | None = None

    # KEI-Stream-spezifische Informationen
    stream_limits: dict[str, Any] | None = None
    frame_rate_status: dict[str, Any] | None = None


class KEIStreamDistributedTokenBucket:
    """Redis-basierter verteilter Token-Bucket speziell für KEI-Stream.

    Erweitert den Standard-Token-Bucket um KEI-Stream-spezifische Features:
    - Frame-Rate-Limiting für WebSocket/SSE
    - Stream-basierte Limits
    - Session-Management-Integration
    """

    def __init__(self, redis_client: redis.Redis, key_prefix: str = RateLimitConstants.DEFAULT_KEY_PREFIX):
        self.redis = redis_client
        self.key_prefix = key_prefix

    async def consume_tokens(
        self,
        identifier: str,
        tokens_requested: int,
        config: KEIStreamRateLimitConfig,
        stream_id: str | None = None,
        frame_type: str | None = None
    ) -> KEIStreamRateLimitResult:
        """Versucht Tokens zu konsumieren mit KEI-Stream-spezifischer Logik.

        Args:
            identifier: Eindeutige Kennung für den Bucket
            tokens_requested: Anzahl der angeforderten Tokens
            config: KEI-Stream Rate-Limiting-Konfiguration
            stream_id: Optional Stream-ID für Stream-spezifische Limits
            frame_type: Optional Frame-Typ für Frame-spezifische Limits

        Returns:
            KEIStreamRateLimitResult mit detaillierten Informationen
        """
        bucket_key = f"{self.key_prefix}:{identifier}"
        current_time = time.time()

        # Erweiterte Lua-Script für KEI-Stream-spezifische Token-Bucket-Operation
        lua_script = """
        local bucket_key = KEYS[1]
        local stream_bucket_key = KEYS[2]
        local current_time = tonumber(ARGV[1])
        local tokens_requested = tonumber(ARGV[2])
        local capacity = tonumber(ARGV[3])
        local refill_rate = tonumber(ARGV[4])
        local window_size = tonumber(ARGV[5])
        local frames_per_second = tonumber(ARGV[6]) or 0
        local max_concurrent_streams = tonumber(ARGV[7]) or 0

        -- Hauptbucket-Logik
        local bucket_data = redis.call('HMGET', bucket_key, 'tokens', 'last_refill')
        local current_tokens = tonumber(bucket_data[1]) or capacity
        local last_refill = tonumber(bucket_data[2]) or current_time

        -- Token-Refill berechnen
        local time_passed = current_time - last_refill
        local tokens_to_add = time_passed * refill_rate
        current_tokens = math.min(capacity, current_tokens + tokens_to_add)

        -- Prüfe Hauptbucket
        local main_allowed = current_tokens >= tokens_requested
        local remaining_tokens = current_tokens

        if main_allowed then
            remaining_tokens = current_tokens - tokens_requested
        end

        -- Stream-spezifische Prüfungen (falls Stream-ID vorhanden)
        local stream_allowed = true
        local stream_info = {}

        if stream_bucket_key ~= "" then
            local stream_data = redis.call('HMGET', stream_bucket_key, 'frame_tokens', 'last_frame_time', 'stream_count')
            local frame_tokens = tonumber(stream_data[1]) or (frames_per_second > 0 and frames_per_second or capacity)
            local last_frame_time = tonumber(stream_data[2]) or current_time
            local stream_count = tonumber(stream_data[3]) or 0

            -- Frame-Rate-Limiting prüfen
            if frames_per_second > 0 then
                local frame_time_passed = current_time - last_frame_time
                local frame_tokens_to_add = frame_time_passed * frames_per_second
                frame_tokens = math.min(frames_per_second * 10, frame_tokens + frame_tokens_to_add)

                stream_allowed = frame_tokens >= 1
                if stream_allowed then
                    frame_tokens = frame_tokens - 1
                    redis.call('HMSET', stream_bucket_key, 'frame_tokens', frame_tokens, 'last_frame_time', current_time)
                end
            end

            -- Concurrent-Stream-Limit prüfen
            if max_concurrent_streams > 0 and stream_count >= max_concurrent_streams then
                stream_allowed = false
            end

            stream_info = {
                frame_tokens = frame_tokens,
                stream_count = stream_count,
                frames_per_second = frames_per_second
            }
        end

        local final_allowed = main_allowed and stream_allowed

        -- Bucket-Daten aktualisieren
        if final_allowed then
            redis.call('HMSET', bucket_key, 'tokens', remaining_tokens, 'last_refill', current_time)
            redis.call('EXPIRE', bucket_key, window_size * 2)
        end

        -- Reset-Zeit berechnen
        local reset_time = current_time + ((capacity - remaining_tokens) / refill_rate)

        return {
            final_allowed and 1 or 0,
            math.floor(remaining_tokens),
            reset_time,
            current_tokens,
            cjson.encode(stream_info)
        }
        """

        try:
            # Stream-spezifischer Bucket-Key
            stream_bucket_key = ""
            if stream_id:
                stream_bucket_key = f"{self.key_prefix}:stream:{stream_id}"

            result = await self.redis.eval(
                lua_script,
                2,  # Anzahl Keys
                bucket_key,
                stream_bucket_key,
                current_time,
                tokens_requested,
                config.burst_capacity,
                config.requests_per_second,
                config.window_size_seconds,
                config.frames_per_second or 0,
                config.max_concurrent_streams or 0
            )

            allowed, remaining, reset_time, current_tokens, stream_info_json = result

            # Stream-Informationen parsen
            stream_info = {}
            with contextlib.suppress(builtins.BaseException):
                stream_info = json.loads(stream_info_json) if stream_info_json else {}

            # Retry-After berechnen
            retry_after = None
            if not allowed:
                tokens_needed = tokens_requested - current_tokens
                retry_after = max(1, int(tokens_needed / config.requests_per_second))

            return KEIStreamRateLimitResult(
                allowed=bool(allowed),
                remaining_requests=int(remaining),
                reset_time=float(reset_time),
                retry_after_seconds=retry_after,
                current_usage={
                    "current_tokens": current_tokens,
                    "capacity": config.burst_capacity,
                    "refill_rate": config.requests_per_second
                },
                stream_limits=stream_info if stream_info else None,
                frame_rate_status={
                    "frames_per_second_limit": config.frames_per_second,
                    "current_frame_tokens": stream_info.get("frame_tokens"),
                } if config.frames_per_second and stream_info else None
            )

        except Exception as e:
            logger.exception(f"Redis KEI-Stream Token-Bucket-Fehler für {identifier}: {e}")
            # Fallback: Erlaube Request bei Redis-Fehlern
            return KEIStreamRateLimitResult(
                allowed=True,
                remaining_requests=config.burst_capacity,
                reset_time=current_time + config.window_size_seconds,
                current_usage={"error": str(e)}
            )

    async def register_stream(self, identifier: str, stream_id: str) -> bool:
        """Registriert einen neuen Stream für Concurrent-Stream-Limiting."""
        try:
            stream_bucket_key = f"{self.key_prefix}:stream:{stream_id}"
            user_streams_key = f"{self.key_prefix}:user_streams:{identifier}"

            # Füge Stream zur User-Stream-Liste hinzu
            await self.redis.sadd(user_streams_key, stream_id)
            await self.redis.expire(user_streams_key, RateLimitConstants.DEFAULT_TTL_SECONDS)

            # Initialisiere Stream-Bucket
            await self.redis.hmset(stream_bucket_key, {
                "created_at": time.time(),
                "frame_tokens": RateLimitConstants.INITIAL_FRAME_TOKENS,
                "last_frame_time": time.time()
            })
            await self.redis.expire(stream_bucket_key, RateLimitConstants.DEFAULT_TTL_SECONDS)

            return True

        except Exception as e:
            logger.exception(f"Fehler beim Registrieren von Stream {stream_id}: {e}")
            return False

    async def unregister_stream(self, identifier: str, stream_id: str) -> bool:
        """Entfernt einen Stream aus dem Concurrent-Stream-Limiting."""
        try:
            stream_bucket_key = f"{self.key_prefix}:stream:{stream_id}"
            user_streams_key = f"{self.key_prefix}:user_streams:{identifier}"

            # Entferne Stream aus User-Stream-Liste
            await self.redis.srem(user_streams_key, stream_id)

            # Lösche Stream-Bucket
            await self.redis.delete(stream_bucket_key)

            return True

        except Exception as e:
            logger.exception(f"Fehler beim Entfernen von Stream {stream_id}: {e}")
            return False


class KEIStreamRateLimitingMiddleware(BaseHTTPMiddleware):
    """FastAPI-Middleware für KEI-Stream-spezifisches Rate-Limiting.

    Implementiert verschiedene Rate-Limiting-Strategien mit Redis-Backend
    und vollständiger Integration in das KEI-Stream-System.
    """

    def __init__(
        self,
        app,
        redis_url: str = "redis://localhost:6379",
        default_config: KEIStreamRateLimitConfig | None = None,
        tenant_configs: dict[str, KEIStreamRateLimitConfig] | None = None,
        endpoint_configs: dict[str, KEIStreamRateLimitConfig] | None = None
    ):
        super().__init__(app)

        # Standard-Konfiguration für KEI-Stream
        self.default_config = default_config or KEIStreamRateLimitConfig(
            requests_per_second=50.0,
            burst_capacity=100,
            window_size_seconds=60,
            frames_per_second=20.0,  # 20 Frames pro Sekunde für WebSocket/SSE
            max_concurrent_streams=5
        )

        # Tenant-spezifische Konfigurationen
        self.tenant_configs = tenant_configs or self._get_default_tenant_configs()

        # Endpoint-spezifische Konfigurationen
        self.endpoint_configs = endpoint_configs or self._get_default_endpoint_configs()

        # Redis-Client für verteiltes Rate-Limiting
        self.redis_client = None
        self.redis_url = redis_url

        # Token-Bucket für lokales Fallback
        self.local_buckets: dict[str, KEITokenBucket] = {}

        # Metriken-Tracking
        self.metrics = {
            "requests_allowed": 0,
            "requests_blocked": 0,
            "frames_allowed": 0,
            "frames_blocked": 0,
            "streams_created": 0,
            "streams_rejected": 0,
            "redis_errors": 0,
            "fallback_used": 0
        }

    def _get_default_tenant_configs(self) -> dict[str, KEIStreamRateLimitConfig]:
        """Standard-Tenant-Konfigurationen für verschiedene Tier-Level."""
        return {
            "free": KEIStreamRateLimitConfig(
                requests_per_second=10.0,
                burst_capacity=20,
                frames_per_second=5.0,
                max_concurrent_streams=2,
                identification_strategy=KEIStreamIdentificationStrategy.USER_ID
            ),
            "premium": KEIStreamRateLimitConfig(
                requests_per_second=100.0,
                burst_capacity=200,
                frames_per_second=50.0,
                max_concurrent_streams=10,
                identification_strategy=KEIStreamIdentificationStrategy.API_KEY
            ),
            "enterprise": KEIStreamRateLimitConfig(
                requests_per_second=500.0,
                burst_capacity=1000,
                frames_per_second=200.0,
                max_concurrent_streams=50,
                identification_strategy=KEIStreamIdentificationStrategy.TENANT_ID
            )
        }

    def _get_default_endpoint_configs(self) -> dict[str, KEIStreamRateLimitConfig]:
        """Standard-Endpoint-Konfigurationen für verschiedene KEI-Stream-Endpunkte."""
        return {
            # WebSocket-Endpunkte
            "GET:/stream/ws/{session_id}": KEIStreamRateLimitConfig(
                requests_per_second=5.0,  # Weniger Verbindungen
                burst_capacity=10,
                frames_per_second=50.0,   # Aber mehr Frames
                max_concurrent_streams=10,
                endpoint_type=KEIStreamEndpointType.WEBSOCKET,
                identification_strategy=KEIStreamIdentificationStrategy.SESSION_ID
            ),

            # SSE-Endpunkte
            "GET:/stream/sse/{session_id}/{stream_id}": KEIStreamRateLimitConfig(
                requests_per_second=10.0,
                burst_capacity=20,
                frames_per_second=30.0,
                max_concurrent_streams=5,
                endpoint_type=KEIStreamEndpointType.SSE,
                identification_strategy=KEIStreamIdentificationStrategy.SESSION_ID
            ),

            # Stream-Management-Endpunkte
            "POST:/stream/create": KEIStreamRateLimitConfig(
                requests_per_second=2.0,
                burst_capacity=5,
                endpoint_type=KEIStreamEndpointType.STREAM_MANAGEMENT,
                identification_strategy=KEIStreamIdentificationStrategy.USER_ID
            ),

            # Tool-Execution-Endpunkte
            "POST:/tools/execute": KEIStreamRateLimitConfig(
                requests_per_second=20.0,
                burst_capacity=40,
                endpoint_type=KEIStreamEndpointType.TOOL_EXECUTION,
                identification_strategy=KEIStreamIdentificationStrategy.API_KEY
            )
        }

    async def startup(self):
        """Initialisiert Redis-Verbindung."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("✅ Redis-Verbindung für KEI-Stream Rate-Limiting erfolgreich")
        except Exception as e:
            logger.warning(f"⚠️ Redis-Verbindung fehlgeschlagen: {e}")
            logger.info("📦 Verwende lokale Token-Buckets als Fallback")

    async def shutdown(self):
        """Schließt Redis-Verbindung."""
        if self.redis_client:
            await self.redis_client.close()

    def get_rate_limit_config(
        self,
        request: Request,
        tenant_id: str | None = None,
        api_key: str | None = None
    ) -> KEIStreamRateLimitConfig:
        """Ermittelt die passende KEI-Stream Rate-Limiting-Konfiguration.

        Auflösungsreihenfolge:
        1. Endpoint-spezifische Konfiguration
        2. Tenant-spezifische Konfiguration
        3. API-Key-spezifische Konfiguration
        4. Standard-Konfiguration
        """
        # Endpoint-spezifische Konfiguration prüfen
        endpoint_key = f"{request.method}:{request.url.path}"
        if endpoint_key in self.endpoint_configs:
            return self.endpoint_configs[endpoint_key]

        # Pattern-basierte Endpoint-Matching
        for pattern, config in self.endpoint_configs.items():
            if self._matches_endpoint_pattern(endpoint_key, pattern):
                return config

        # Tenant-spezifische Konfiguration prüfen
        if tenant_id and tenant_id in self.tenant_configs:
            return self.tenant_configs[tenant_id]

        # Standard-Konfiguration verwenden
        return self.default_config

    def _matches_endpoint_pattern(self, endpoint: str, pattern: str) -> bool:
        """Prüft ob Endpoint einem Pattern entspricht (einfache Wildcard-Unterstützung)."""
        # Einfache Pattern-Matching-Logik
        if "{" in pattern and "}" in pattern:
            # Ersetze {param} durch Wildcard für Matching
            import re
            regex_pattern = re.sub(r"\{[^}]+\}", r"[^/]+", pattern)
            return bool(re.match(f"^{regex_pattern}$", endpoint))
        return endpoint == pattern

    def extract_rate_limit_identifier(
        self,
        request: Request,
        config: KEIStreamRateLimitConfig
    ) -> str:
        """Extrahiert Identifier für Rate-Limiting basierend auf KEI-Stream-Strategie."""
        return ClientIdentificationUtils.generate_client_id(request, config.identification_strategy)



    async def check_rate_limit(
        self,
        request: Request,
        tokens_requested: int = 1
    ) -> KEIStreamRateLimitResult:
        """Führt KEI-Stream-spezifische Rate-Limiting-Prüfung durch.

        Args:
            request: FastAPI-Request-Objekt
            tokens_requested: Anzahl der angeforderten Tokens

        Returns:
            KEIStreamRateLimitResult mit Prüfungsergebnis
        """
        # Extrahiere KEI-Stream-spezifische Informationen
        tenant_id = request.headers.get("X-Tenant-ID")
        api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
        stream_id = self._extract_stream_id_from_request(request)
        frame_type = request.headers.get("X-Frame-Type")

        # Ermittle Rate-Limiting-Konfiguration
        config = self.get_rate_limit_config(request, tenant_id, api_key)

        # Prüfe ob Rate-Limiting aktiviert ist
        if not config.enabled:
            return KEIStreamRateLimitResult(
                allowed=True,
                remaining_requests=config.burst_capacity,
                reset_time=time.time() + config.window_size_seconds
            )

        # Extrahiere Identifier für Rate-Limiting
        identifier = self.extract_rate_limit_identifier(request, config)

        # Verwende Redis-basiertes Rate-Limiting wenn verfügbar
        if self.redis_client:
            try:
                bucket = KEIStreamDistributedTokenBucket(self.redis_client)
                result = await bucket.consume_tokens(
                    identifier,
                    tokens_requested,
                    config,
                    stream_id,
                    frame_type
                )

                # Metriken aktualisieren
                if result.allowed:
                    self.metrics["requests_allowed"] += 1
                    if frame_type:
                        self.metrics["frames_allowed"] += 1
                else:
                    self.metrics["requests_blocked"] += 1
                    if frame_type:
                        self.metrics["frames_blocked"] += 1

                return result

            except Exception as e:
                logger.exception(f"Redis KEI-Stream Rate-Limiting-Fehler: {e}")
                self.metrics["redis_errors"] += 1
                # Fallback auf lokale Token-Buckets

        # Lokales Fallback-Rate-Limiting
        return await self._local_rate_limit_check(identifier, tokens_requested, config)

    def _extract_stream_id_from_request(self, request: Request) -> str | None:
        """Extrahiert Stream-ID aus Request für Stream-spezifische Limits."""
        # Aus URL-Path extrahieren
        path_parts = request.url.path.split("/")

        # SSE-Endpunkt: /stream/sse/{session_id}/{stream_id}
        if "sse" in path_parts:
            try:
                sse_index = path_parts.index("sse")
                if sse_index + 2 < len(path_parts):
                    return path_parts[sse_index + 2]
            except ValueError:
                pass

        # WebSocket-Endpunkt: Stream-ID aus Query-Parameter oder Header
        if "ws" in path_parts:
            stream_id = request.query_params.get("stream_id")
            if stream_id:
                return stream_id

        # Fallback auf Header
        return request.headers.get("X-Stream-ID")

    async def _local_rate_limit_check(
        self,
        identifier: str,
        tokens_requested: int,
        config: KEIStreamRateLimitConfig
    ) -> KEIStreamRateLimitResult:
        """Lokales Rate-Limiting als Fallback für KEI-Stream."""
        self.metrics["fallback_used"] += 1

        # Hole oder erstelle lokalen Token-Bucket
        if identifier not in self.local_buckets:
            self.local_buckets[identifier] = KEITokenBucket(
                capacity=config.burst_capacity,
                refill_rate=config.requests_per_second
            )

        bucket = self.local_buckets[identifier]

        # Versuche Tokens zu konsumieren
        if await bucket.consume(tokens_requested):
            available_tokens = await bucket.get_available_tokens()
            return KEIStreamRateLimitResult(
                allowed=True,
                remaining_requests=int(available_tokens),
                reset_time=time.time() + config.window_size_seconds,
                current_usage={
                    "current_tokens": available_tokens,
                    "capacity": config.burst_capacity,
                    "refill_rate": config.requests_per_second,
                    "backend": "local_fallback"
                }
            )
        # Berechne Retry-After
        available_tokens = await bucket.get_available_tokens()
        tokens_needed = tokens_requested - available_tokens
        retry_after = max(1, int(tokens_needed / config.requests_per_second))

        return KEIStreamRateLimitResult(
            allowed=False,
            remaining_requests=0,
            reset_time=time.time() + retry_after,
            retry_after_seconds=retry_after,
            limit_exceeded_by="local_bucket",
            current_usage={
                "current_tokens": available_tokens,
                "capacity": config.burst_capacity,
                "refill_rate": config.requests_per_second,
                "backend": "local_fallback"
            }
        )

    def create_rate_limit_response(
        self,
        result: KEIStreamRateLimitResult,
        config: KEIStreamRateLimitConfig,
        request: Request
    ) -> JSONResponse:
        """Erstellt HTTP 429 Response mit erweiterten KEI-Stream-spezifischen Informationen."""
        # Verwende erweiterten Error-Handler
        error_handler = _get_rate_limit_error_handler()

        # Erstelle Error-Context basierend auf Result
        error_context = self._create_error_context(result, config, request)

        return error_handler.create_rate_limit_error_response(
            request, result, config, error_context
        )

    def _create_error_context(
        self,
        result: KEIStreamRateLimitResult,
        config: KEIStreamRateLimitConfig,
        request: Request
    ) -> RateLimitErrorContext:
        """Erstellt Error-Context für detaillierte Fehlerbehandlung."""
        # Bestimme Error-Type basierend auf Result
        error_type = RateLimitErrorType.REQUEST_RATE_EXCEEDED

        if result.frame_rate_status and result.frame_rate_status.get("current_frame_tokens", 0) <= 0:
            error_type = RateLimitErrorType.FRAME_RATE_EXCEEDED
        elif result.stream_limits and result.stream_limits.get("stream_count", 0) >= config.max_concurrent_streams:
            error_type = RateLimitErrorType.CONCURRENT_STREAMS_EXCEEDED
        elif result.current_usage and result.current_usage.get("current_tokens", 0) <= 0:
            if config.burst_capacity and result.remaining_requests <= 0:
                error_type = RateLimitErrorType.BURST_CAPACITY_EXCEEDED

        # Extrahiere Identifier
        self.extract_rate_limit_identifier(request, config)

        return RateLimitErrorContext(
            error_type=error_type,
            endpoint_type=config.endpoint_type,
            endpoint_path=request.url.path,
            current_usage=result.remaining_requests,
            limit=int(config.requests_per_second * config.window_size_seconds),
            reset_time=result.reset_time,
            retry_after=result.retry_after_seconds,
            additional_info={
                "requests_per_second": config.requests_per_second,
                "burst_capacity": config.burst_capacity,
                "frames_per_second": config.frames_per_second,
                "max_concurrent_streams": config.max_concurrent_streams,
                "window_seconds": config.window_size_seconds
            }
        )

    async def dispatch(self, request: Request, call_next) -> Response:
        """Hauptmiddleware-Logik für KEI-Stream Rate-Limiting."""
        start_time = time.time()

        try:
            # Führe Rate-Limiting-Prüfung durch
            result = await self.check_rate_limit(request)

            if not result.allowed:
                # Rate-Limit überschritten - erstelle 429 Response
                config = self.get_rate_limit_config(request)
                response = self.create_rate_limit_response(result, config, request)

                # Metriken für Monitoring
                await self._record_rate_limit_metrics(request, result, blocked=True)

                return response

            # Request ist erlaubt - weiterleiten
            response = await call_next(request)

            # KEI-Stream Rate-Limiting-Headers zu Response hinzufügen
            config = self.get_rate_limit_config(request)
            response.headers["X-RateLimit-Limit"] = str(int(config.requests_per_second * config.window_size_seconds))
            response.headers["X-RateLimit-Remaining"] = str(result.remaining_requests)
            response.headers["X-RateLimit-Reset"] = str(int(result.reset_time))
            response.headers["X-RateLimit-Algorithm-Strategy"] = config.algorithm_strategy.value
            response.headers["X-RateLimit-Identification-Strategy"] = config.identification_strategy.value
            response.headers["X-RateLimit-Endpoint-Type"] = config.endpoint_type.value

            # KEI-Stream-spezifische Headers
            if config.frames_per_second:
                response.headers["X-RateLimit-Frames-Per-Second"] = str(config.frames_per_second)

            if config.max_concurrent_streams:
                response.headers["X-RateLimit-Max-Concurrent-Streams"] = str(config.max_concurrent_streams)

            # Metriken für Monitoring
            await self._record_rate_limit_metrics(request, result, blocked=False)

            return response

        except Exception as e:
            logger.exception(f"KEI-Stream Rate-Limiting-Middleware-Fehler: {e}")
            # Bei Fehlern Request durchlassen
            return await call_next(request)
        finally:
            # Performance-Metriken
            processing_time = (time.time() - start_time) * 1000
            record_custom_metric("kei_stream_rate_limiting_processing_time_ms", processing_time)

    async def _record_rate_limit_metrics(
        self,
        request: Request,
        result: KEIStreamRateLimitResult,
        blocked: bool
    ):
        """Zeichnet KEI-Stream-spezifische Metriken für Rate-Limiting auf."""
        try:
            # Basis-Metriken
            record_custom_metric("kei_stream_rate_limiting_requests_total", 1, {
                "method": request.method,
                "endpoint": request.url.path,
                "blocked": str(blocked).lower(),
                "identification_strategy": self.get_rate_limit_config(request).identification_strategy.value,
                "endpoint_type": self.get_rate_limit_config(request).endpoint_type.value
            })

            if blocked:
                record_custom_metric("kei_stream_rate_limiting_requests_blocked_total", 1, {
                    "method": request.method,
                    "endpoint": request.url.path,
                    "reason": result.limit_exceeded_by or "unknown",
                    "endpoint_type": self.get_rate_limit_config(request).endpoint_type.value
                })

            # Token-Bucket-Metriken
            if result.current_usage:
                record_custom_metric("kei_stream_rate_limiting_tokens_remaining", result.remaining_requests)

                if "current_tokens" in result.current_usage:
                    # Ensure numeric types for division operation
                    current_tokens = float(result.current_usage["current_tokens"])
                    capacity = float(result.current_usage["capacity"])
                    utilization = 1.0 - (current_tokens / capacity)
                    record_custom_metric("kei_stream_rate_limiting_bucket_utilization", utilization)

            # KEI-Stream-spezifische Metriken
            if result.frame_rate_status:
                record_custom_metric("kei_stream_frame_rate_tokens_remaining",
                    result.frame_rate_status.get("current_frame_tokens", 0))

            if result.stream_limits:
                record_custom_metric("kei_stream_concurrent_streams",
                    result.stream_limits.get("stream_count", 0))

        except Exception as e:
            logger.warning(f"Fehler beim Aufzeichnen von KEI-Stream Rate-Limiting-Metriken: {e}")

    def get_metrics(self) -> dict[str, Any]:
        """Gibt aktuelle KEI-Stream Rate-Limiting-Metriken zurück."""
        return {
            **self.metrics,
            "local_buckets_count": len(self.local_buckets),
            "redis_connected": self.redis_client is not None,
            "tenant_configs_count": len(self.tenant_configs),
            "endpoint_configs_count": len(self.endpoint_configs)
        }

    async def register_stream(self, identifier: str, stream_id: str) -> bool:
        """Registriert einen neuen Stream für Concurrent-Stream-Limiting."""
        if self.redis_client:
            try:
                bucket = KEIStreamDistributedTokenBucket(self.redis_client)
                success = await bucket.register_stream(identifier, stream_id)
                if success:
                    self.metrics["streams_created"] += 1
                else:
                    self.metrics["streams_rejected"] += 1
                return success
            except Exception as e:
                logger.exception(f"Fehler beim Registrieren von Stream {stream_id}: {e}")
                self.metrics["streams_rejected"] += 1
                return False

        # Lokales Fallback - einfache Zählung
        self.metrics["streams_created"] += 1
        return True

    async def unregister_stream(self, identifier: str, stream_id: str) -> bool:
        """Entfernt einen Stream aus dem Concurrent-Stream-Limiting."""
        if self.redis_client:
            try:
                bucket = KEIStreamDistributedTokenBucket(self.redis_client)
                return await bucket.unregister_stream(identifier, stream_id)
            except Exception as e:
                logger.exception(f"Fehler beim Entfernen von Stream {stream_id}: {e}")
                return False

        # Lokales Fallback - keine Aktion erforderlich
        return True


# Factory-Funktionen für einfache Integration

def create_kei_stream_rate_limiting_middleware(
    app,
    redis_url: str = "redis://localhost:6379",
    enable_redis: bool = True,
    tenant_tier_configs: dict[str, str] | None = None
) -> KEIStreamRateLimitingMiddleware:
    """Factory-Funktion für KEI-Stream Rate-Limiting-Middleware.

    Args:
        app: FastAPI-Application
        redis_url: Redis-Verbindungs-URL
        enable_redis: Ob Redis verwendet werden soll
        tenant_tier_configs: Mapping von Tenant-ID zu Tier-Level

    Returns:
        Konfigurierte KEIStreamRateLimitingMiddleware
    """
    # Standard-Tenant-Konfigurationen basierend auf Tier-Level
    tenant_configs = {}
    if tenant_tier_configs:
        default_tiers = {
            "free": KEIStreamRateLimitConfig(
                requests_per_second=10.0,
                burst_capacity=20,
                frames_per_second=5.0,
                max_concurrent_streams=2
            ),
            "premium": KEIStreamRateLimitConfig(
                requests_per_second=100.0,
                burst_capacity=200,
                frames_per_second=50.0,
                max_concurrent_streams=10
            ),
            "enterprise": KEIStreamRateLimitConfig(
                requests_per_second=500.0,
                burst_capacity=1000,
                frames_per_second=200.0,
                max_concurrent_streams=50
            )
        }

        for tenant_id, tier in tenant_tier_configs.items():
            if tier in default_tiers:
                tenant_configs[tenant_id] = default_tiers[tier]

    return KEIStreamRateLimitingMiddleware(
        app=app,
        redis_url=redis_url if enable_redis else None,
        tenant_configs=tenant_configs
    )


def get_rate_limit_status_for_request(
    middleware: KEIStreamRateLimitingMiddleware,
    request: Request
) -> dict[str, Any]:
    """Hilfsfunktion um Rate-Limit-Status für einen Request zu ermitteln.

    Args:
        middleware: KEIStreamRateLimitingMiddleware-Instanz
        request: FastAPI-Request-Objekt

    Returns:
        Dictionary mit Rate-Limit-Status-Informationen
    """
    config = middleware.get_rate_limit_config(request)
    identifier = middleware.extract_rate_limit_identifier(request, config)

    return {
        "identifier": identifier,
        "algorithm_strategy": config.algorithm_strategy.value,
        "identification_strategy": config.identification_strategy.value,
        "endpoint_type": config.endpoint_type.value,
        "limits": {
            "requests_per_second": config.requests_per_second,
            "burst_capacity": config.burst_capacity,
            "frames_per_second": config.frames_per_second,
            "max_concurrent_streams": config.max_concurrent_streams,
            "window_size_seconds": config.window_size_seconds
        },
        "enabled": config.enabled
    }


def get_kei_stream_rate_limiting_config():
    """Erstellt Standard-Konfiguration für KEI-Stream Rate Limiting."""
    # Einfache Konfiguration ohne komplexe Abhängigkeiten
    return {
        "requests_per_second": 50.0,
        "burst_capacity": 100,
        "window_size_seconds": 60,
        "enabled": True
    }
