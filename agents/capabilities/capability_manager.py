# backend/agents/capabilities/capability_manager.py
"""Capability Manager für das Agent System.

Verwaltet Capabilities mit automatischen Health/Readiness-Checks,
Versionierung und kategorie-spezifischer Validierung.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from kei_logging import get_logger

from ..metadata.agent_metadata import (
    CapabilityCategory,
    CapabilityStatus,
)
from .enhanced_capabilities import (
    CapabilityMetrics,
    DefaultHealthChecker,
    DefaultReadinessChecker,
    EnhancedCapability,
    HealthChecker,
    HealthCheckResult,
    ReadinessChecker,
    ReadinessCheckResult,
)

logger = get_logger(__name__)


class CapabilityValidationError(Exception):
    """Fehler bei Capability-Validierung."""


class CapabilityManager:
    """Manager für Enhanced Capabilities mit automatischen Health/Readiness-Checks."""

    def __init__(
        self,
        health_check_interval: int = None,
        readiness_check_interval: int = None
    ) -> None:
        """Initialisiert Capability Manager.

        Args:
            health_check_interval: Intervall für Health-Checks in Sekunden (Standard: 30)
            readiness_check_interval: Intervall für Readiness-Checks in Sekunden (Standard: 10)
        """
        from .capability_utils import CapabilityConstants
        if health_check_interval is None:
            health_check_interval = CapabilityConstants.HEALTH_CHECK_INTERVAL
        if readiness_check_interval is None:
            readiness_check_interval = CapabilityConstants.READINESS_CHECK_INTERVAL
        self.capabilities: dict[str, EnhancedCapability] = {}
        self.health_checkers: dict[str, HealthChecker] = {}
        self.readiness_checkers: dict[str, ReadinessChecker] = {}
        self.metrics: dict[str, CapabilityMetrics] = defaultdict(CapabilityMetrics)
        # Agent-Capabilities-Mapping für Heartbeat-Service
        self._agent_capabilities: dict[str, Any] = {}



        self.health_check_interval = health_check_interval
        self.readiness_check_interval = readiness_check_interval

        self._health_check_task: asyncio.Task | None = None
        self._readiness_check_task: asyncio.Task | None = None
        self._running = False


        self._category_validators: dict[
            CapabilityCategory, Callable[[EnhancedCapability], bool]
        ] = {
            CapabilityCategory.TOOLS: self._validate_tools_capability,
            CapabilityCategory.SKILLS: self._validate_skills_capability,
            CapabilityCategory.DOMAINS: self._validate_domains_capability,
            CapabilityCategory.POLICIES: self._validate_policies_capability,
        }

    async def start(self) -> None:
        """Startet automatische Health/Readiness-Checks."""
        if self._running:
            return

        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._readiness_check_task = asyncio.create_task(self._readiness_check_loop())

        logger.info("Capability Manager gestartet")

    async def stop(self) -> None:
        """Stoppt automatische Checks."""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._readiness_check_task:
            self._readiness_check_task.cancel()
            try:
                await self._readiness_check_task
            except asyncio.CancelledError:
                pass

        logger.info("Capability Manager gestoppt")

    def register_capability(
        self,
        capability: EnhancedCapability,
        health_checker: HealthChecker | None = None,
        readiness_checker: ReadinessChecker | None = None,
    ) -> None:
        """Registriert neue Capability mit optionalen Checkern.

        Args:
            capability: Capability-Definition
            health_checker: Optionaler Health-Checker
            readiness_checker: Optionaler Readiness-Checker

        Raises:
            CapabilityValidationError: Bei Validierungsfehlern
        """
        self._validate_capability(capability)

        self.capabilities[capability.id] = capability

        if health_checker is None:
            health_checker = DefaultHealthChecker(capability.name)
        if readiness_checker is None:
            readiness_checker = DefaultReadinessChecker(capability.name, capability.dependencies)

        self.health_checkers[capability.id] = health_checker
        self.readiness_checkers[capability.id] = readiness_checker

        if capability.id not in self.metrics:
            self.metrics[capability.id] = CapabilityMetrics()

        logger.info(f"Capability registriert: {capability.id} ({capability.category.value})")

    def unregister_capability(self, capability_id: str) -> bool:
        """Entfernt Capability aus Registry.

        Args:
            capability_id: ID der zu entfernenden Capability

        Returns:
            True wenn erfolgreich entfernt, False wenn nicht gefunden
        """
        if capability_id not in self.capabilities:
            return False

        del self.capabilities[capability_id]
        self.health_checkers.pop(capability_id, None)
        self.readiness_checkers.pop(capability_id, None)
        self.metrics.pop(capability_id, None)

        logger.info(f"Capability entfernt: {capability_id}")
        return True

    def get_capability(self, capability_id: str) -> EnhancedCapability | None:
        """Gibt Capability zurück."""
        return self.capabilities.get(capability_id)

    def list_capabilities(
        self,
        category: CapabilityCategory | None = None,
        status: CapabilityStatus | None = None,
        available_only: bool = False,
    ) -> list[EnhancedCapability]:
        """Listet Capabilities mit optionalen Filtern.

        Args:
            category: Optionale Kategorie-Filterung
            status: Optionaler Status-Filter
            available_only: Nur verfügbare Capabilities

        Returns:
            Liste der gefilterten Capabilities
        """
        capabilities = list(self.capabilities.values())

        if category:
            capabilities = [c for c in capabilities if c.category == category]

        if status:
            capabilities = [c for c in capabilities if c.status == status]

        if available_only:
            capabilities = [c for c in capabilities if c.is_available()]

        return capabilities

    def get_capability_metrics(self, capability_id: str) -> CapabilityMetrics | None:
        """Gibt Metriken für Capability zurück."""
        return self.metrics.get(capability_id)

    async def check_capability_health(self, capability_id: str) -> HealthCheckResult | None:
        """Führt manuellen Health-Check für Capability durch."""
        if capability_id not in self.capabilities:
            return None

        checker = self.health_checkers.get(capability_id)
        if not checker:
            return None

        try:
            result = await checker.check_health()

            capability = self.capabilities[capability_id]
            capability.health_status = result.status
            capability.last_health_check = result.timestamp
            capability.updated_at = datetime.now(UTC)

            return result
        except Exception as e:
            logger.error(f"Health-Check für {capability_id} fehlgeschlagen: {e}")
            return None

    async def check_capability_readiness(
        self, capability_id: str
    ) -> ReadinessCheckResult | None:
        """Führt manuellen Readiness-Check für Capability durch."""
        if capability_id not in self.capabilities:
            return None

        checker = self.readiness_checkers.get(capability_id)
        if not checker:
            return None

        try:
            result = await checker.check_readiness()

            capability = self.capabilities[capability_id]
            capability.readiness_status = result.status
            capability.last_readiness_check = result.timestamp
            capability.updated_at = datetime.now(UTC)

            return result
        except Exception as e:
            logger.error(f"Readiness-Check für {capability_id} fehlgeschlagen: {e}")
            return None

    def record_capability_invocation(
        self, capability_id: str, success: bool, response_time_ms: float
    ) -> None:
        """Zeichnet Capability-Aufruf für Metriken auf.

        Args:
            capability_id: ID der aufgerufenen Capability
            success: Ob Aufruf erfolgreich war
            response_time_ms: Antwortzeit in Millisekunden
        """
        if capability_id not in self.metrics:
            return

        metrics = self.metrics[capability_id]
        metrics.total_invocations += 1

        if success:
            metrics.successful_invocations += 1
        else:
            metrics.failed_invocations += 1

        if metrics.total_invocations == 1:
            metrics.average_response_time_ms = response_time_ms
        else:
            alpha = 0.1
            metrics.average_response_time_ms = (
                alpha * response_time_ms + (1 - alpha) * metrics.average_response_time_ms
            )

        metrics.last_invocation_at = datetime.now(UTC)

    def _validate_capability(self, capability: EnhancedCapability) -> None:
        """Validiert Capability basierend auf Kategorie.

        Args:
            capability: Zu validierende Capability

        Raises:
            CapabilityValidationError: Bei Validierungsfehlern
        """
        if not capability.id or not capability.name:
            raise CapabilityValidationError("ID und Name sind erforderlich")

        if capability.id in self.capabilities:
            raise CapabilityValidationError(f"Capability {capability.id} bereits registriert")


        validator = self._category_validators.get(capability.category)
        if validator and not validator(capability):
            raise CapabilityValidationError(
                f"Validierung für Kategorie {capability.category.value} fehlgeschlagen"
            )

    @staticmethod
    def _validate_tools_capability(capability: EnhancedCapability) -> bool:
        """Validiert Tools-Capability.

        Args:
            capability: Zu validierende Tools-Capability

        Returns:
            True wenn Capability gültig ist
        """
        return bool(capability.endpoints or capability.parameters)

    @staticmethod
    def _validate_skills_capability(capability: EnhancedCapability) -> bool:
        """Validiert Skills-Capability.

        Args:
            capability: Zu validierende Skills-Capability

        Returns:
            True wenn Capability gültig ist
        """
        return bool(capability.description and capability.metadata)

    @staticmethod
    def _validate_domains_capability(capability: EnhancedCapability) -> bool:
        """Validiert Domains-Capability.

        Args:
            capability: Zu validierende Domains-Capability

        Returns:
            True wenn Capability gültig ist
        """
        return bool(capability.tags)

    @staticmethod
    def _validate_policies_capability(capability: EnhancedCapability) -> bool:
        """Validiert Policies-Capability.

        Args:
            capability: Zu validierende Policies-Capability

        Returns:
            True wenn Capability gültig ist
        """
        return bool(capability.parameters)

    async def _health_check_loop(self) -> None:
        """Automatische Health-Check-Schleife."""
        while self._running:
            try:
                for capability_id in list(self.capabilities.keys()):
                    await self.check_capability_health(capability_id)

                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler in Health-Check-Loop: {e}")
                await asyncio.sleep(5)

    async def _readiness_check_loop(self) -> None:
        """Automatische Readiness-Check-Schleife."""
        while self._running:
            try:
                for capability_id in list(self.capabilities.keys()):
                    await self.check_capability_readiness(capability_id)

                await asyncio.sleep(self.readiness_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler in Readiness-Check-Loop: {e}")
                await asyncio.sleep(5)
