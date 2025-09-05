# backend/services/llm/llm_client.py
"""LLM Client Integration für Orchestrator Service.

Implementiert OpenAI/Azure OpenAI Client mit Retry-Logic, Rate-Limiting,
Cost-Tracking, Prompt-Template-Management und Response-Caching.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from kei_logging import get_logger
from observability.budget import try_consume_cost_usd, try_consume_tokens
from policy_engine.enhanced_pii_redaction import EnhancedPIIRedactor
from services.clients.common.retry_utils import RetryableClient, RetryConfig
from storage.cache.redis_cache import get_cache_client

logger = get_logger(__name__)


@dataclass
class LLMRequest:
    """LLM Request Datenmodell."""

    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.1
    max_tokens: int | None = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    user_id: str | None = None
    session_id: str | None = None


@dataclass
class LLMResponse:
    """LLM Response Datenmodell."""

    content: str
    model: str
    usage: dict[str, int]
    cost_usd: float
    response_time_ms: float
    cached: bool = False
    request_id: str | None = None


@dataclass
class LLMClientConfig:
    """Konfiguration für LLM Client."""

    # OpenAI/Azure OpenAI Konfiguration
    api_key: str
    endpoint: str | None = None
    api_version: str = "2024-02-15-preview"
    deployment: str | None = None

    # Rate Limiting
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 150000

    # Cost Management
    max_cost_per_hour_usd: float = 10.0
    cost_alert_threshold_usd: float = 5.0

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 Stunde

    # Retry Konfiguration
    max_retries: int = 3
    initial_delay: float = 1.0
    backoff_multiplier: float = 2.0

    # Fallback
    enable_fallback: bool = True
    fallback_model: str = "gpt-3.5-turbo"

    # PII Redaction
    enable_pii_redaction: bool = True


@dataclass
class RateLimitState:
    """Rate Limiting Zustand."""

    requests_count: int = 0
    tokens_count: int = 0
    window_start: datetime = field(default_factory=datetime.utcnow)

    def reset_if_needed(self) -> None:
        """Setzt Zähler zurück wenn Zeitfenster abgelaufen."""
        now = datetime.utcnow()
        if now - self.window_start >= timedelta(minutes=1):
            self.requests_count = 0
            self.tokens_count = 0
            self.window_start = now

    def can_make_request(self, config: LLMClientConfig, estimated_tokens: int = 0) -> bool:
        """Prüft ob Request innerhalb der Rate Limits liegt."""
        self.reset_if_needed()

        if self.requests_count >= config.max_requests_per_minute:
            return False

        if self.tokens_count + estimated_tokens > config.max_tokens_per_minute:
            return False

        return True

    def record_request(self, tokens_used: int) -> None:
        """Registriert einen Request."""
        self.requests_count += 1
        self.tokens_count += tokens_used


class LLMClient(RetryableClient):
    """Enterprise LLM Client mit Rate-Limiting, Cost-Tracking und Caching."""

    def __init__(self, config: LLMClientConfig):
        """Initialisiert LLM Client.

        Args:
            config: Client-Konfiguration
        """
        # Retry-Konfiguration für Parent-Klasse
        retry_config = RetryConfig(
            max_retries=config.max_retries,
            initial_delay=config.initial_delay,
            backoff_multiplier=config.backoff_multiplier,
            exceptions=(Exception,)
        )
        super().__init__(retry_config)

        self.config = config
        self._client: AsyncOpenAI | None = None
        self._rate_limit_state = RateLimitState()
        self._pii_redactor: EnhancedPIIRedactor | None = None

        # Token-Kosten-Mapping (USD pro 1K Tokens)
        self._token_costs = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        }

        logger.info({
            "event": "llm_client_initialized",
            "model_costs_loaded": len(self._token_costs),
            "rate_limit_rpm": config.max_requests_per_minute,
            "caching_enabled": config.enable_caching
        })

    async def _ensure_client(self) -> AsyncOpenAI:
        """Initialisiert OpenAI Client lazily."""
        if self._client is None:
            if self.config.endpoint:
                # Azure OpenAI
                self._client = AsyncOpenAI(
                    api_key=self.config.api_key,
                    base_url=f"{self.config.endpoint.rstrip('/')}/openai/deployments/{self.config.deployment}",
                    api_version=self.config.api_version
                )
            else:
                # Standard OpenAI
                self._client = AsyncOpenAI(api_key=self.config.api_key)

            logger.debug({
                "event": "openai_client_created",
                "endpoint": self.config.endpoint,
                "deployment": self.config.deployment
            })

        return self._client

    async def _ensure_pii_redactor(self) -> EnhancedPIIRedactor:
        """Initialisiert PII Redactor lazily."""
        if self._pii_redactor is None:
            self._pii_redactor = EnhancedPIIRedactor()
        return self._pii_redactor

    def _estimate_tokens(self, messages: list[dict[str, str]]) -> int:
        """Schätzt Token-Anzahl für Messages (grobe Approximation)."""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        # Grobe Schätzung: 4 Zeichen = 1 Token
        return max(1, total_chars // 4)

    def _calculate_cost(self, model: str, usage: dict[str, int]) -> float:
        """Berechnet Kosten für LLM Request."""
        if model not in self._token_costs:
            # Fallback auf gpt-4 Preise für unbekannte Modelle
            model = "gpt-4"

        costs = self._token_costs[model]
        input_cost = (usage.get("prompt_tokens", 0) / 1000) * costs["input"]
        output_cost = (usage.get("completion_tokens", 0) / 1000) * costs["output"]

        return input_cost + output_cost

    def _create_cache_key(self, request: LLMRequest) -> str:
        """Erstellt Cache-Key für Request."""
        # Erstelle deterministischen Hash aus Request-Parametern
        request_data = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty
        }

        request_json = json.dumps(request_data, sort_keys=True)
        cache_hash = hashlib.sha256(request_json.encode()).hexdigest()[:16]

        return f"llm_response:{cache_hash}"

    async def _get_cached_response(self, cache_key: str) -> LLMResponse | None:
        """Lädt Response aus Cache."""
        if not self.config.enable_caching:
            return None

        try:
            redis = await get_cache_client()
            cached_data = await redis.get(cache_key)

            if cached_data:
                response_data = json.loads(cached_data)
                response = LLMResponse(**response_data)
                response.cached = True

                logger.debug({
                    "event": "llm_cache_hit",
                    "cache_key": cache_key,
                    "model": response.model
                })

                return response
        except Exception as e:
            logger.warning(f"Cache-Lookup fehlgeschlagen: {e}")

        return None

    async def _cache_response(self, cache_key: str, response: LLMResponse) -> None:
        """Speichert Response im Cache."""
        if not self.config.enable_caching:
            return

        try:
            redis = await get_cache_client()
            response_data = {
                "content": response.content,
                "model": response.model,
                "usage": response.usage,
                "cost_usd": response.cost_usd,
                "response_time_ms": response.response_time_ms,
                "request_id": response.request_id
            }

            await redis.setex(
                cache_key,
                self.config.cache_ttl_seconds,
                json.dumps(response_data)
            )

            logger.debug({
                "event": "llm_response_cached",
                "cache_key": cache_key,
                "ttl": self.config.cache_ttl_seconds
            })
        except Exception as e:
            logger.warning(f"Cache-Speicherung fehlgeschlagen: {e}")

    async def _redact_pii_if_enabled(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Redaktiert PII in Messages falls aktiviert."""
        if not self.config.enable_pii_redaction:
            return messages

        try:
            redactor = await self._ensure_pii_redactor()
            redacted_messages = []

            for message in messages:
                content = message.get("content", "")
                if content:
                    redaction_result = await redactor.redact_pii(content)
                    redacted_message = message.copy()
                    redacted_message["content"] = redaction_result.redacted_text
                    redacted_messages.append(redacted_message)
                else:
                    redacted_messages.append(message)

            return redacted_messages
        except Exception as e:
            logger.warning(f"PII-Redaction fehlgeschlagen: {e}")
            return messages

    async def _check_rate_limits(self, estimated_tokens: int) -> None:
        """Prüft Rate Limits und wartet falls nötig."""
        if not self._rate_limit_state.can_make_request(self.config, estimated_tokens):
            # Berechne Wartezeit bis zum nächsten Zeitfenster
            time_until_reset = 60 - (datetime.utcnow() - self._rate_limit_state.window_start).total_seconds()

            if time_until_reset > 0:
                logger.warning({
                    "event": "rate_limit_hit",
                    "wait_seconds": time_until_reset,
                    "requests_count": self._rate_limit_state.requests_count,
                    "tokens_count": self._rate_limit_state.tokens_count
                })

                await asyncio.sleep(time_until_reset)
                self._rate_limit_state.reset_if_needed()

    async def _check_cost_budget(self, estimated_cost: float) -> bool:
        """Prüft Cost Budget und gibt False zurück wenn überschritten."""
        if not try_consume_cost_usd(estimated_cost):
            logger.error({
                "event": "cost_budget_exceeded",
                "estimated_cost": estimated_cost,
                "budget_exhausted": True
            })
            return False

        return True

    async def _make_openai_request(self, request: LLMRequest) -> ChatCompletion:
        """Macht den eigentlichen OpenAI API Request."""
        client = await self._ensure_client()

        # Redaktiere PII falls aktiviert
        redacted_messages = await self._redact_pii_if_enabled(request.messages)

        # Erstelle OpenAI Request
        openai_kwargs = {
            "model": request.model,
            "messages": redacted_messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty
        }

        if request.max_tokens:
            openai_kwargs["max_tokens"] = request.max_tokens

        if request.user_id:
            openai_kwargs["user"] = request.user_id

        return await client.chat.completions.create(**openai_kwargs)

    async def chat_completion(self, request: LLMRequest) -> LLMResponse:
        """Führt Chat Completion Request aus mit allen Features.

        Args:
            request: LLM Request

        Returns:
            LLM Response

        Raises:
            Exception: Bei API-Fehlern oder Budget-Überschreitung
        """
        start_time = time.time()

        # Cache-Lookup
        cache_key = self._create_cache_key(request)
        cached_response = await self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        # Token-Schätzung für Rate Limiting
        estimated_tokens = self._estimate_tokens(request.messages)

        # Rate Limiting prüfen
        await self._check_rate_limits(estimated_tokens)

        # Grobe Kostenschätzung
        estimated_cost = (estimated_tokens / 1000) * self._token_costs.get(request.model, self._token_costs["gpt-4"])["input"]

        # Cost Budget prüfen
        if not await self._check_cost_budget(estimated_cost):
            raise Exception("Cost Budget überschritten")

        try:
            # OpenAI Request mit Retry-Logic
            completion = await self._execute_with_retry(
                self._make_openai_request,
                request
            )

            # Response verarbeiten
            response_time_ms = (time.time() - start_time) * 1000
            actual_cost = self._calculate_cost(request.model, completion.usage.model_dump())

            response = LLMResponse(
                content=completion.choices[0].message.content or "",
                model=completion.model,
                usage=completion.usage.model_dump(),
                cost_usd=actual_cost,
                response_time_ms=response_time_ms,
                request_id=completion.id
            )

            # Rate Limit State aktualisieren
            total_tokens = completion.usage.prompt_tokens + completion.usage.completion_tokens
            self._rate_limit_state.record_request(total_tokens)

            # Token Budget verbrauchen
            try_consume_tokens(total_tokens)

            # Response cachen
            await self._cache_response(cache_key, response)

            logger.info({
                "event": "llm_request_completed",
                "model": request.model,
                "tokens_used": total_tokens,
                "cost_usd": actual_cost,
                "response_time_ms": response_time_ms,
                "cached": False
            })

            return response

        except Exception as e:
            # Fallback auf lokales Modell falls konfiguriert
            if self.config.enable_fallback and request.model != self.config.fallback_model:
                logger.warning({
                    "event": "llm_request_failed_fallback",
                    "original_model": request.model,
                    "fallback_model": self.config.fallback_model,
                    "error": str(e)
                })

                fallback_request = LLMRequest(
                    model=self.config.fallback_model,
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    user_id=request.user_id,
                    session_id=request.session_id
                )

                return await self.chat_completion(fallback_request)

            logger.error({
                "event": "llm_request_failed",
                "model": request.model,
                "error": str(e),
                "error_type": type(e).__name__
            })

            raise
