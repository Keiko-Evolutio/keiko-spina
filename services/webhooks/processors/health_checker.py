"""Health-Checker für das KEI-Webhook System.

Sammelt und aggregiert Health-Status-Informationen von allen Webhook-Komponenten.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

if TYPE_CHECKING:
    from ..utils.redis_manager import RedisManager
    from .event_processor import WebhookEventProcessor
    from .worker_pool_manager import WebhookWorkerPoolManager

logger = get_logger(__name__)


class WebhookHealthChecker:
    """Prüft und aggregiert Health-Status des Webhook-Systems."""

    def __init__(
        self,
        redis_manager: RedisManager,
        worker_pool_manager: WebhookWorkerPoolManager,  # Forward reference
        event_processor: WebhookEventProcessor,  # Forward reference
    ) -> None:
        """Initialisiert den Health-Checker.

        Args:
            redis_manager: Redis-Manager für Connectivity-Checks
            worker_pool_manager: Worker-Pool-Manager für Worker-Status
            event_processor: Event-Processor für Queue-Depth
        """
        self.redis_manager = redis_manager
        self.worker_pool_manager = worker_pool_manager
        self.event_processor = event_processor

    async def get_health_status(self) -> dict[str, Any]:
        """Sammelt umfassenden Health-Status aller Komponenten.

        Returns:
            Dictionary mit Health-Status-Informationen
        """
        # Redis-Connectivity prüfen
        redis_healthy = await self._check_redis_health()

        # Worker-Pool-Status abrufen
        worker_status = self._get_worker_pool_health()

        # Queue-Depth ermitteln
        queue_depth = await self._get_queue_depth()

        # Gesamtstatus bestimmen
        overall_healthy = (
            redis_healthy and
            worker_status.get("healthy", False) and
            queue_depth < 10000  # Threshold für "gesunde" Queue-Tiefe
        )

        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": self._get_current_timestamp(),
            "components": {
                "redis": {
                    "status": "healthy" if redis_healthy else "unhealthy",
                    "available": redis_healthy,
                },
                "worker_pool": worker_status,
                "queues": {
                    "status": "healthy" if queue_depth < 10000 else "degraded",
                    "total_depth": queue_depth,
                },
            },
            "metrics": {
                "outbox_depth": queue_depth,
                "active_workers": worker_status.get("active_workers", 0),
                "configured_workers": worker_status.get("configured_workers", 0),
            },
        }

    async def get_simple_health(self) -> dict[str, Any]:
        """Gibt vereinfachten Health-Status zurück (für Load Balancer).

        Returns:
            Minimaler Health-Status
        """
        worker_status = self._get_worker_pool_health()
        healthy = worker_status.get("active_workers", 0) > 0

        return {
            "status": "healthy" if healthy else "unhealthy",
            "active_workers": worker_status.get("active_workers", 0),
        }

    async def _check_redis_health(self) -> bool:
        """Prüft Redis-Connectivity.

        Returns:
            True wenn Redis verfügbar, False sonst
        """
        try:
            return await self.redis_manager.is_available()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Redis health check failed: %s", exc)
            return False

    def _get_worker_pool_health(self) -> dict[str, Any]:
        """Sammelt Worker-Pool-Health-Informationen.

        Returns:
            Worker-Pool-Status-Dictionary
        """
        try:
            pool_status = self.worker_pool_manager.get_pool_status()
            active_workers = pool_status.get("active_workers", 0)
            configured_workers = pool_status.get("configured_workers", 0)

            # Worker-Pool ist gesund wenn mindestens 50% der Worker aktiv sind
            healthy = (
                pool_status.get("running", False) and
                active_workers > 0 and
                (active_workers / max(configured_workers, 1)) >= 0.5
            )

            return {
                "status": "healthy" if healthy else "unhealthy",
                "healthy": healthy,
                "active_workers": active_workers,
                "configured_workers": configured_workers,
                "running": pool_status.get("running", False),
                "details": pool_status,
            }

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Worker pool health check failed: %s", exc)
            return {
                "status": "unhealthy",
                "healthy": False,
                "active_workers": 0,
                "configured_workers": 0,
                "running": False,
                "error": str(exc),
            }

    async def _get_queue_depth(self) -> int:
        """Ermittelt Gesamttiefe aller Queues.

        Returns:
            Gesamtanzahl Events in allen Queues
        """
        try:
            shard_names = self.worker_pool_manager.config.shard_names
            return await self.event_processor.get_queue_depth(shard_names)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Queue depth check failed: %s", exc)
            return 0

    def _get_current_timestamp(self) -> str:
        """Gibt aktuellen Timestamp im ISO-Format zurück.

        Returns:
            ISO-formatierter Timestamp
        """
        return datetime.now(UTC).isoformat()

    async def check_component_health(self, component: str) -> dict[str, Any]:
        """Prüft Health-Status einer spezifischen Komponente.

        Args:
            component: Name der Komponente ("redis", "workers", "queues")

        Returns:
            Health-Status der spezifischen Komponente

        Raises:
            ValueError: Wenn unbekannte Komponente angefragt wird
        """
        if component == "redis":
            healthy = await self._check_redis_health()
            return {
                "component": "redis",
                "status": "healthy" if healthy else "unhealthy",
                "available": healthy,
            }

        if component == "workers":
            return {
                "component": "workers",
                **self._get_worker_pool_health(),
            }

        if component == "queues":
            depth = await self._get_queue_depth()
            healthy = depth < 10000
            return {
                "component": "queues",
                "status": "healthy" if healthy else "degraded",
                "total_depth": depth,
            }

        raise ValueError(f"Unknown component: {component}")

    async def get_metrics_summary(self) -> dict[str, Any]:
        """Gibt Metriken-Zusammenfassung für Monitoring zurück.

        Returns:
            Dictionary mit Key-Metriken
        """
        worker_status = self._get_worker_pool_health()
        queue_depth = await self._get_queue_depth()
        redis_available = await self._check_redis_health()

        return {
            "webhook_active_workers": worker_status.get("active_workers", 0),
            "webhook_configured_workers": worker_status.get("configured_workers", 0),
            "webhook_queue_depth_total": queue_depth,
            "webhook_redis_available": 1 if redis_available else 0,
            "webhook_system_healthy": 1 if (
                redis_available and
                worker_status.get("healthy", False) and
                queue_depth < 10000
            ) else 0,
        }


__all__ = ["WebhookHealthChecker"]
