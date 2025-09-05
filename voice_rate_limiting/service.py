"""Voice Rate Limiting Service Implementation.
Hauptservice für Voice-spezifisches Rate Limiting.
"""

from typing import Any

from kei_logging import get_logger

from .adaptive_limiter import AdaptiveRateLimiter
from .interfaces import (
    IAdaptiveRateLimiter,
    IRateLimitStore,
    IVoiceRateLimiter,
    IVoiceRateLimitService,
    UserTier,
    VoiceOperation,
    VoiceRateLimitContext,
    VoiceRateLimitSettings,
)
from .storage import create_rate_limit_store
from .voice_rate_limiter import VoiceRateLimiter

logger = get_logger(__name__)


class VoiceRateLimitService(IVoiceRateLimitService):
    """Voice Rate Limiting Service Implementation.
    Orchestriert alle Voice Rate Limiting Komponenten.
    """

    def __init__(self, settings: VoiceRateLimitSettings):
        self.settings = settings

        # Core Components
        self._store: IRateLimitStore | None = None
        self._rate_limiter: IVoiceRateLimiter | None = None
        self._adaptive_limiter: IAdaptiveRateLimiter | None = None

        # Service Status
        self._initialized = False
        self._running = False

        logger.info(f"Voice rate limiting service created with settings: {settings}")

    @property
    def rate_limiter(self) -> IVoiceRateLimiter:
        """Voice Rate Limiter."""
        if not self._rate_limiter:
            raise RuntimeError("Voice rate limiting service not initialized")
        return self._rate_limiter

    @property
    def adaptive_limiter(self) -> IAdaptiveRateLimiter:
        """Adaptive Rate Limiter."""
        if not self._adaptive_limiter:
            raise RuntimeError("Voice rate limiting service not initialized")
        return self._adaptive_limiter

    async def initialize(self) -> None:
        """Initialisiert Rate Limiting Service."""
        if self._initialized:
            return

        try:
            logger.info("Initializing voice rate limiting service...")

            # Storage initialisieren mit robustem Fallback
            redis_url = self.settings.redis_url if self.settings.redis_enabled else None
            force_in_memory = not self.settings.redis_enabled

            self._store = create_rate_limit_store(redis_url, force_in_memory)

            # Redis-Verbindung herstellen (falls verwendet und verfügbar)
            if hasattr(self._store, "connect"):
                try:
                    await self._store.connect()
                    logger.info("Redis connection established successfully")
                except Exception as e:
                    logger.warning(f"Redis connection failed ({e}), continuing with fallback store")
                    # Fallback zu In-Memory Store
                    self._store = create_rate_limit_store(None, force_in_memory=True)

            # In-Memory Store Cleanup starten (falls verwendet)
            if hasattr(self._store, "start_cleanup_task"):
                await self._store.start_cleanup_task()

            # Adaptive Limiter initialisieren
            self._adaptive_limiter = AdaptiveRateLimiter(self.settings)

            # User-Tier-Multiplier erstellen
            user_tier_multipliers = {
                UserTier.ANONYMOUS: self.settings.anonymous_multiplier,
                UserTier.STANDARD: self.settings.standard_multiplier,
                UserTier.PREMIUM: self.settings.premium_multiplier,
                UserTier.ENTERPRISE: self.settings.enterprise_multiplier
            }

            # Default-Konfigurationen für alle Operationen erstellen
            default_configs = {}
            for operation in VoiceOperation:
                # Verwende Adaptive Limiter für Standard-Tier als Basis
                config = await self._adaptive_limiter.get_current_limits(operation, UserTier.STANDARD)
                default_configs[operation] = config

            # Voice Rate Limiter mit Multi-Window-Unterstützung initialisieren
            self._rate_limiter = VoiceRateLimiter(
                store=self._store,
                default_configs=default_configs,
                user_tier_multipliers=user_tier_multipliers,
                settings=self.settings
            )

            # Adaptive Monitoring starten (falls aktiviert)
            if self.settings.adaptive_enabled:
                await self._adaptive_limiter.start_monitoring()

            self._initialized = True
            self._running = True

            logger.info("Voice rate limiting service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize voice rate limiting service: {e}")
            raise

    async def shutdown(self) -> None:
        """Fährt Rate Limiting Service herunter."""
        if not self._running:
            return

        try:
            logger.info("Shutting down voice rate limiting service...")

            # Adaptive Monitoring stoppen
            if self._adaptive_limiter:
                await self._adaptive_limiter.stop_monitoring()

            # Store Cleanup stoppen
            if self._store and hasattr(self._store, "stop_cleanup_task"):
                await self._store.stop_cleanup_task()

            # Redis-Verbindung schließen
            if self._store and hasattr(self._store, "disconnect"):
                await self._store.disconnect()

            self._running = False

            logger.info("Voice rate limiting service shut down successfully")

        except Exception as e:
            logger.error(f"Error during voice rate limiting service shutdown: {e}")

    async def get_rate_limit_status(self, context: VoiceRateLimitContext) -> dict[str, Any]:
        """Gibt Rate Limit Status für Kontext zurück."""
        if not self._initialized:
            return {"error": "Service not initialized"}

        status = {
            "service": {
                "initialized": self._initialized,
                "running": self._running,
                "settings": {
                    "enabled": self.settings.enabled,
                    "redis_enabled": self.settings.redis_enabled,
                    "adaptive_enabled": self.settings.adaptive_enabled
                }
            },
            "context": {
                "user_id": context.user_id,
                "user_tier": context.user_tier.value,
                "session_id": context.session_id,
                "ip_address": context.ip_address,
                "endpoint": context.endpoint
            },
            "operations": {}
        }

        # Status für alle Voice-Operationen sammeln
        for operation in VoiceOperation:
            try:
                operation_status = await self._rate_limiter.get_rate_limit_status(operation, context)
                status["operations"][operation.value] = operation_status
            except Exception as e:
                status["operations"][operation.value] = {"error": str(e)}

        # Adaptive Limiter Status hinzufügen
        if self._adaptive_limiter:
            try:
                status["adaptive"] = self._adaptive_limiter.get_adaptation_status()
            except Exception as e:
                status["adaptive"] = {"error": str(e)}

        return status

    async def reset_rate_limits(self, context: VoiceRateLimitContext) -> None:
        """Setzt Rate Limits für Kontext zurück (Admin-Funktion)."""
        if not self._initialized:
            raise RuntimeError("Service not initialized")

        logger.warning(f"Resetting rate limits for user {context.user_id}")

        # Generiere alle möglichen Keys für den Kontext
        operations = list(VoiceOperation)

        for operation in operations:
            # User-basierte Keys
            user_key = f"user:{context.user_id}:{operation.value}"
            await self._store.clear(user_key)
            await self._store.clear(f"concurrent:{user_key}")

            # Session-basierte Keys (falls verfügbar)
            if context.session_id:
                session_key = f"session:{context.session_id}:{operation.value}"
                await self._store.clear(session_key)
                await self._store.clear(f"concurrent:{session_key}")

            # IP-basierte Keys (falls verfügbar)
            if context.ip_address:
                ip_key = f"ip:{context.ip_address}:{operation.value}"
                await self._store.clear(ip_key)
                await self._store.clear(f"concurrent:{ip_key}")

        logger.info(f"Rate limits reset for user {context.user_id}")

    async def update_user_tier(self, user_id: str, new_tier: UserTier) -> None:
        """Aktualisiert User-Tier (Admin-Funktion)."""
        logger.info(f"User tier updated for {user_id}: {new_tier.value}")

        # In einer vollständigen Implementation würde man hier
        # die User-Tier-Information in einer Datenbank speichern
        # Für jetzt loggen wir nur die Änderung

    async def get_global_statistics(self) -> dict[str, Any]:
        """Gibt globale Rate Limiting Statistiken zurück."""
        if not self._initialized:
            return {"error": "Service not initialized"}

        stats = {
            "service_status": {
                "initialized": self._initialized,
                "running": self._running,
                "uptime_seconds": 0  # TODO: Implement uptime tracking - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/105
            },
            "settings": {
                "enabled": self.settings.enabled,
                "redis_enabled": self.settings.redis_enabled,
                "adaptive_enabled": self.settings.adaptive_enabled,
                "monitoring_enabled": self.settings.monitoring_enabled
            }
        }

        # Store-Statistiken (falls verfügbar)
        if hasattr(self._store, "get_stats"):
            try:
                stats["store"] = self._store.get_stats()
            except Exception as e:
                stats["store"] = {"error": str(e)}

        # Adaptive Limiter Statistiken
        if self._adaptive_limiter:
            try:
                stats["adaptive"] = self._adaptive_limiter.get_adaptation_status()
            except Exception as e:
                stats["adaptive"] = {"error": str(e)}

        return stats

    async def health_check(self) -> dict[str, Any]:
        """Führt Health Check für Rate Limiting Service durch."""
        health = {
            "healthy": True,
            "details": {}
        }

        # Service-Status prüfen
        if not self._initialized or not self._running:
            health["healthy"] = False
            health["details"]["service"] = "Service not running"
        else:
            health["details"]["service"] = "OK"

        # Store-Verbindung prüfen
        try:
            if self._store:
                # Einfacher Test: Count für Test-Key abrufen
                await self._store.get_count("health_check", 60)
                health["details"]["store"] = "OK"
            else:
                health["healthy"] = False
                health["details"]["store"] = "Store not initialized"
        except Exception as e:
            health["healthy"] = False
            health["details"]["store"] = f"Store error: {e!s}"

        # Adaptive Limiter prüfen
        try:
            if self._adaptive_limiter:
                # Test: Aktuelle Limits für Standard-Operation abrufen
                await self._adaptive_limiter.get_current_limits(
                    VoiceOperation.SPEECH_TO_TEXT,
                    UserTier.STANDARD
                )
                health["details"]["adaptive_limiter"] = "OK"
            else:
                health["details"]["adaptive_limiter"] = "Not initialized"
        except Exception as e:
            health["healthy"] = False
            health["details"]["adaptive_limiter"] = f"Error: {e!s}"

        return health


def create_voice_rate_limit_service(settings: VoiceRateLimitSettings = None) -> VoiceRateLimitService:
    """Factory-Funktion für Voice Rate Limiting Service.

    Args:
        settings: Rate Limiting Settings, falls None werden Defaults verwendet

    Returns:
        Voice Rate Limiting Service Instance
    """
    if settings is None:
        settings = VoiceRateLimitSettings()

    return VoiceRateLimitService(settings)
