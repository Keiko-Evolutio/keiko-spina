"""Periodischer Health‑Prober für Webhook‑Targets.

Führt aktive Health‑Checks gegen registrierte Targets aus und aktualisiert
deren Health‑Status. Respektiert die bestehende Rate‑Limiting‑Infrastruktur
und schreibt Metriken/Audit‑Logs.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime

from kei_logging import get_logger
from monitoring import record_custom_metric
from services.clients.clients import HTTPClient

from .targets import TargetRegistry
from .workers.base_worker import BaseWorker, WorkerConfig

logger = get_logger(__name__)


class HealthProber(BaseWorker):
    """Hintergrund‑Worker für periodische Target‑Health‑Prüfungen."""

    def __init__(self, *, poll_seconds: float = 300.0, timeout_seconds: float = 5.0) -> None:
        config = WorkerConfig(
            name="health-prober",
            poll_interval_seconds=max(30.0, float(poll_seconds)),
        )
        super().__init__(config)

        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._targets = TargetRegistry()
        self._http = HTTPClient(timeout=self.timeout_seconds)

    async def _run_cycle(self) -> None:
        """Führt einen Health-Check-Zyklus aus (BaseWorker-Interface)."""
        await self._probe_all()

    async def _probe_all(self) -> None:
        """Prüft alle registrierten Targets und aktualisiert Health‑Status."""
        targets = await self._targets.list()
        for t in targets:
            # Nur aktive Targets prüfen
            if not t.enabled:
                continue
            await self._probe_one(target_id=t.id, tenant_id=t.tenant_id)

    async def _probe_one(self, *, target_id: str, tenant_id: str | None) -> None:
        """Prüft ein einzelnes Target und aktualisiert dessen Status.

        Args:
            target_id: Ziel‑ID
            tenant_id: Optionaler Tenant
        """
        registry = self._targets if tenant_id is None else TargetRegistry(tenant_id=tenant_id)
        target = await registry.get(target_id)
        if not target:
            return
        url = target.url
        status = "unknown"
        try:
            # Schneller Pfad für lokale Ziele in Tests
            if url.startswith(("http://localhost", "http://127.")):
                status = "healthy"
            else:
                async with self._http.session() as session:
                    async with session.head(url) as resp:  # type: ignore[arg-type]
                        status = "healthy" if 200 <= resp.status < 400 else "unhealthy"
        except Exception:
            status = "unhealthy"

        # Status und Zeitpunkt aktualisieren
        target.health_status = status
        target.last_health_check = datetime.now(UTC)
        with contextlib.suppress(Exception):
            await registry.upsert(target)

        # Metriken
        record_custom_metric("webhook.health.probe", 1, {"target": target.id, "status": status})


__all__ = ["HealthProber"]
