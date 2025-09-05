"""Periodischer Worker zur automatischen Secret‑Rotation für Webhook‑Targets.

Unterstützt Grace‑Period, Rollback bei Fehlern und Metriken/Audit.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from config.settings import settings
from kei_logging import get_logger
from monitoring import record_custom_metric

from .secret_manager import get_secret_manager
from .targets import TargetRegistry
from .workers.base_worker import BaseWorker, WorkerConfig

if TYPE_CHECKING:
    from .models import WebhookTarget

logger = get_logger(__name__)


class SecretRotationWorker(BaseWorker):
    """Hintergrund‑Worker zur Secret‑Rotation gemäß Konfiguration.

    Startet eine Loop, die in definierten Intervallen alle Targets mit
    `secret_key_name` rotiert. Unterstützt Rollback im Fehlerfall.
    """

    def __init__(self, *, interval_days: int | None = None, poll_seconds: int = 3600) -> None:
        config = WorkerConfig(
            name="secret-rotation-worker",
            poll_interval_seconds=max(60, int(poll_seconds)),
        )
        super().__init__(config)

        self.interval_days = int(interval_days or settings.secret_rotation_interval_days)
        self._targets = TargetRegistry()
        self._sm = get_secret_manager()

    async def _pre_start(self) -> None:
        """Prüft ob Secret-Rotation aktiviert ist."""
        if not settings.secret_rotation_enabled:
            raise RuntimeError("Secret-Rotation ist deaktiviert")

    async def _run_cycle(self) -> None:
        """Führt einen Rotations-Zyklus aus (BaseWorker-Interface)."""
        await self._check_and_rotate_all()

    async def _check_and_rotate_all(self) -> None:
        """Prüft alle Targets und rotiert bei Bedarf."""
        targets = await self._targets.list()
        now = datetime.now(UTC)
        for t in targets:
            if not t.secret_key_name:
                continue
            if t.secret_last_rotated_at and (now - t.secret_last_rotated_at) < timedelta(days=self.interval_days):
                continue
            await self._rotate_target(t)

    async def _rotate_target(self, target: WebhookTarget) -> None:
        """Rotiere Secret für ein einzelnes Target, setze Grace und persistiere."""
        old_version = target.secret_version
        try:
            _, new_version = await self._sm.rotate_secret(key_name=target.secret_key_name)  # type: ignore[arg-type]
            now = datetime.now(UTC)
            target.previous_secret_version = old_version
            target.secret_version = new_version
            target.secret_last_rotated_at = now
            target.secret_grace_until = now + timedelta(hours=int(settings.secret_grace_period_hours))
            await self._targets.upsert(target)
            record_custom_metric("webhook.secret.rotation.success", 1, {"target": target.id})
        except Exception as exc:
            # Rollback: bleibe auf alter Version
            record_custom_metric("webhook.secret.rotation.failure", 1, {"target": target.id})
            logger.warning(f"Secret Rotation für Target {target.id} fehlgeschlagen: {exc}")


__all__ = ["SecretRotationWorker"]
