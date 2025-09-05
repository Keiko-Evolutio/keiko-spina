"""Verwaltung von Webhook‑Targets und Target‑Health.

Target-Registry mit RedisManager-Integration und verbesserter
Architektur für bessere Performance und Wartbarkeit.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from kei_logging import get_logger

from .audit_logger import WebhookAuditOperation, webhook_audit
from .keys import targets_hash_key
from .models import Subscription, WebhookTarget
from .utils.redis_operations import redis_hash_delete, redis_hash_get_all, redis_hash_set

if TYPE_CHECKING:
    import builtins

logger = get_logger(__name__)


class TargetRegistry:
    """Verwaltet Webhook‑Targets in Redis mit Memory‑Fallback."""

    def __init__(
        self,
        registry_name: str = "default",
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Initialisiert die Target-Registry.

        Args:
            registry_name: Name der Registry
            tenant_id: Optional Tenant-ID
        """
        self.registry_name = registry_name
        self.tenant_id = tenant_id
        self._memory: dict[str, WebhookTarget] = {}

    def _parse_target_json(self, target_json: str) -> WebhookTarget | None:
        """Parst Target-JSON zu WebhookTarget-Objekt.

        Args:
            target_json: JSON-String des Targets

        Returns:
            WebhookTarget-Objekt oder None bei Fehlern
        """
        try:
            target_data = json.loads(target_json)
            return WebhookTarget.model_validate(target_data)
        except Exception as exc:
            logger.debug(f"Fehler beim Parsen von Target-JSON: {exc}")
            return None



    async def list(self) -> builtins.list[WebhookTarget]:
        """Listet alle Targets."""
        # Versuche aus Redis zu laden
        targets = await self._load_from_redis()
        if targets is not None:
            # Memory-Cache aktualisieren
            self._memory = {t.id: t for t in targets}
            return targets

        # Fallback auf Memory-Cache
        return list(self._memory.values())

    async def _load_from_redis(self) -> builtins.list[WebhookTarget] | None:
        """Lädt alle Targets aus Redis.

        Returns:
            Liste der Targets oder None bei Fehler
        """
        hash_key = targets_hash_key(self.tenant_id, self.registry_name)
        targets_data = await redis_hash_get_all(hash_key, WebhookTarget)

        if not targets_data:
            return None

        return list(targets_data.values())



    async def get(self, target_id: str) -> WebhookTarget | None:
        """Lädt Target per ID."""
        # Prüfe Memory-Cache zuerst
        if target_id in self._memory:
            return self._memory[target_id]

        # Versuche aus Redis zu laden
        target = await self._load_target_from_redis(target_id)
        if target:
            # Memory-Cache aktualisieren
            self._memory[target_id] = target
            return target

        return None

    async def _load_target_from_redis(self, target_id: str) -> WebhookTarget | None:
        """Lädt ein einzelnes Target aus Redis.

        Args:
            target_id: Target-ID

        Returns:
            WebhookTarget oder None wenn nicht gefunden
        """
        try:
            hash_key = targets_hash_key(self.tenant_id, self.registry_name)
            # Verwende redis_hash_get_all und extrahiere das spezifische Target
            targets_data = await redis_hash_get_all(hash_key, WebhookTarget)

            if not targets_data or target_id not in targets_data:
                return None

            return targets_data[target_id]

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Failed to load target %s from Redis: %s", target_id, exc)
            return None

    async def upsert(self, target: WebhookTarget) -> None:
        """Erstellt oder aktualisiert ein Target."""
        is_create = target.id not in self._memory

        # Memory-Cache aktualisieren
        self._memory[target.id] = target

        # In Redis speichern
        await self._save_target_to_redis(target)

        # Audit-Logging
        operation = WebhookAuditOperation.CREATE if is_create else WebhookAuditOperation.UPDATE
        await self._record_target_audit(target, operation)

    async def _save_target_to_redis(self, target: WebhookTarget) -> None:
        """Speichert Target in Redis.

        Args:
            target: Zu speicherndes Target
        """
        hash_key = targets_hash_key(self.tenant_id, self.registry_name)
        # Secrets niemals persistieren: legacy_secret ausschließen
        target_data = target.model_copy(update={}, exclude={"legacy_secret"})
        success = await redis_hash_set(hash_key, target.id, target_data)
        if not success:
            logger.warning("Failed to save target %s to Redis", target.id)

    async def _record_target_audit(
        self,
        target: WebhookTarget,
        operation: WebhookAuditOperation
    ) -> None:
        """Zeichnet Audit-Log für Target-Operation auf.

        Args:
            target: Betroffenes Target
            operation: Art der Operation
        """
        try:
            await webhook_audit.target_changed(
                operation=operation,
                target_id=target.id,
                user_id=None,
                tenant_id=self.tenant_id,
                correlation_id=None,
                details={"url": target.url, "enabled": target.enabled},
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Failed to record target audit for %s: %s", target.id, exc)

    async def delete(self, target_id: str) -> None:
        """Löscht ein Target."""
        # Aus Memory-Cache entfernen
        self._memory.pop(target_id, None)

        # Aus Redis entfernen
        await self._delete_target_from_redis(target_id)

        # Audit-Logging
        await self._record_target_deletion_audit(target_id)

    async def _delete_target_from_redis(self, target_id: str) -> None:
        """Löscht Target aus Redis.

        Args:
            target_id: ID des zu löschenden Targets
        """
        try:
            hash_key = targets_hash_key(self.tenant_id, self.registry_name)
            success = await redis_hash_delete(hash_key, target_id)

            if not success:
                logger.debug("Target %s was not found in Redis", target_id)

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Failed to delete target %s from Redis: %s", target_id, exc)

    async def _record_target_deletion_audit(self, target_id: str) -> None:
        """Zeichnet Audit-Log für Target-Löschung auf.

        Args:
            target_id: ID des gelöschten Targets
        """
        try:
            await webhook_audit.target_changed(
                operation=WebhookAuditOperation.DELETE,
                target_id=target_id,
                user_id=None,
                tenant_id=self.tenant_id,
                correlation_id=None,
                details={},
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Failed to record target deletion audit for %s: %s", target_id, exc)

    # =====================================================================
    # Subscription Management (einfach, innerhalb des Zielobjekts)
    # =====================================================================
    async def add_subscription(self, target_id: str, subscription: Subscription) -> bool:
        """Fügt eine Subscription zu einem Target hinzu und persistiert sie."""
        target = await self.get(target_id)
        if not target:
            return False
        subs = list(target.subscriptions)
        subs = [s for s in subs if s.id != subscription.id]
        subs.append(subscription)
        target.subscriptions = subs
        await self.upsert(target)
        return True

    async def remove_subscription(self, target_id: str, subscription_id: str) -> bool:
        """Entfernt eine Subscription eines Targets."""
        target = await self.get(target_id)
        if not target:
            return False
        orig = len(target.subscriptions)
        target.subscriptions = [s for s in target.subscriptions if s.id != subscription_id]
        if len(target.subscriptions) == orig:
            return False
        await self.upsert(target)
        return True

    async def list_subscriptions(self, target_id: str) -> builtins.list[Subscription]:
        """Listet Subscriptions eines Targets auf."""
        target = await self.get(target_id)
        return list(target.subscriptions) if target else []


__all__ = ["TargetRegistry"]
