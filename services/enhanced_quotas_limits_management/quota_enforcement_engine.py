# backend/services/enhanced_quotas_limits_management/quota_enforcement_engine.py
"""Quota Enforcement Engine für Real-time Monitoring und Enforcement.

Implementiert Real-time Quota-Enforcement mit Alerting,
Auto-Scaling und Integration mit Security-Systemen.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

from kei_logging import get_logger
from services.enhanced_security_integration import SecurityContext

from .data_models import (
    EnforcementAction,
    QuotaCheckResult,
    QuotaScope,
    QuotaViolation,
    ResourceType,
)

logger = get_logger(__name__)


class QuotaEnforcementEngine:
    """Quota Enforcement Engine für Real-time Monitoring und Enforcement."""

    def __init__(self):
        """Initialisiert Quota Enforcement Engine."""
        # Enforcement-Konfiguration
        self.enable_real_time_enforcement = True
        self.enable_auto_scaling = False  # Placeholder für Auto-Scaling
        self.enable_alerting = True
        self.enable_security_integration = True

        # Enforcement-Aktionen
        self._enforcement_handlers = {
            EnforcementAction.ALLOW: self._handle_allow,
            EnforcementAction.DENY: self._handle_deny,
            EnforcementAction.THROTTLE: self._handle_throttle,
            EnforcementAction.QUEUE: self._handle_queue,
            EnforcementAction.ALERT: self._handle_alert,
            EnforcementAction.SCALE: self._handle_scale
        }

        # Violation-Tracking
        self._active_violations: dict[str, QuotaViolation] = {}
        self._violation_history: list[QuotaViolation] = []
        self._max_violation_history = 10000

        # Throttling-State
        self._throttled_requests: dict[str, dict[str, Any]] = {}
        self._throttle_queues: dict[str, asyncio.Queue] = {}

        # Performance-Tracking
        self._enforcement_count = 0
        self._total_enforcement_time_ms = 0.0
        self._violation_count = 0
        self._alert_count = 0

        # Background-Tasks
        self._monitoring_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._is_running = False

        logger.info("Quota Enforcement Engine initialisiert")

    async def start(self) -> None:
        """Startet Quota Enforcement Engine."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("Quota Enforcement Engine gestartet")

    async def stop(self) -> None:
        """Stoppt Quota Enforcement Engine."""
        self._is_running = False

        # Stoppe Background-Tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        await asyncio.gather(
            self._monitoring_task,
            self._cleanup_task,
            return_exceptions=True
        )

        logger.info("Quota Enforcement Engine gestoppt")

    async def enforce_quota_violation(
        self,
        quota_check_result: QuotaCheckResult,
        security_context: SecurityContext | None = None
    ) -> dict[str, Any]:
        """Führt Quota-Violation-Enforcement durch.

        Args:
            quota_check_result: Quota-Check-Result
            security_context: Security-Context

        Returns:
            Enforcement-Result
        """
        start_time = time.time()

        try:
            logger.debug({
                "event": "quota_enforcement_started",
                "quota_id": quota_check_result.quota_id,
                "enforcement_action": quota_check_result.enforcement_action.value,
                "current_usage": quota_check_result.current_usage,
                "limit": quota_check_result.limit
            })

            # Erstelle Quota-Violation
            violation = await self._create_quota_violation(quota_check_result, security_context)

            # Führe Enforcement-Action durch
            enforcement_handler = self._enforcement_handlers.get(
                quota_check_result.enforcement_action,
                self._handle_deny
            )

            enforcement_result = await enforcement_handler(violation, quota_check_result, security_context)

            # Tracke Violation
            self._active_violations[violation.violation_id] = violation
            self._violation_history.append(violation)

            # Memory-Limit prüfen
            if len(self._violation_history) > self._max_violation_history:
                self._violation_history = self._violation_history[-self._max_violation_history:]

            # Performance-Tracking
            enforcement_time_ms = (time.time() - start_time) * 1000
            self._update_enforcement_performance_stats(enforcement_time_ms)

            # Security-Event (falls Security-Integration aktiviert)
            if self.enable_security_integration and security_context:
                await self._create_security_event(violation, security_context)

            logger.info({
                "event": "quota_enforcement_completed",
                "violation_id": violation.violation_id,
                "enforcement_action": quota_check_result.enforcement_action.value,
                "action_taken": enforcement_result.get("action_taken", False),
                "enforcement_time_ms": enforcement_time_ms
            })

            return enforcement_result

        except Exception as e:
            logger.error(f"Quota enforcement fehlgeschlagen: {e}")

            return {
                "action_taken": False,
                "error": str(e),
                "enforcement_time_ms": (time.time() - start_time) * 1000
            }

    async def _create_quota_violation(
        self,
        quota_check_result: QuotaCheckResult,
        security_context: SecurityContext | None
    ) -> QuotaViolation:
        """Erstellt Quota-Violation."""
        try:
            import uuid

            violation = QuotaViolation(
                violation_id=str(uuid.uuid4()),
                quota_id=quota_check_result.quota_id,
                resource_type=ResourceType.API_CALL,  # TODO: Aus Context ableiten - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/114
                scope=QuotaScope.GLOBAL,  # TODO: Aus Context ableiten - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/114
                scope_id="global",  # TODO: Aus Context ableiten - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/114
                limit_value=quota_check_result.limit,
                actual_value=quota_check_result.current_usage,
                excess_amount=max(0, quota_check_result.current_usage - quota_check_result.limit),
                enforcement_action=quota_check_result.enforcement_action,
                request_id=security_context.request_id if security_context else None,
                user_id=security_context.user_id if security_context else None,
                details={
                    "remaining": quota_check_result.remaining,
                    "rate_limited": quota_check_result.rate_limited,
                    "retry_after_seconds": quota_check_result.retry_after_seconds
                }
            )

            self._violation_count += 1
            return violation

        except Exception as e:
            logger.error(f"Quota violation creation fehlgeschlagen: {e}")
            raise

    async def _handle_allow(
        self,
        _violation: QuotaViolation,
        _quota_check_result: QuotaCheckResult,
        _security_context: SecurityContext | None
    ) -> dict[str, Any]:
        """Behandelt ALLOW-Enforcement-Action."""
        return {
            "action_taken": True,
            "action": "allow",
            "message": "Request allowed despite quota violation"
        }

    async def _handle_deny(
        self,
        violation: QuotaViolation,
        quota_check_result: QuotaCheckResult,
        _security_context: SecurityContext | None
    ) -> dict[str, Any]:
        """Behandelt DENY-Enforcement-Action."""
        try:
            logger.warning({
                "event": "quota_violation_denied",
                "violation_id": violation.violation_id,
                "quota_id": violation.quota_id,
                "user_id": violation.user_id,
                "excess_amount": violation.excess_amount
            })

            return {
                "action_taken": True,
                "action": "deny",
                "message": f"Request denied due to quota violation. Limit: {violation.limit_value}, Usage: {violation.actual_value}",
                "retry_after_seconds": quota_check_result.retry_after_seconds
            }

        except Exception as e:
            logger.error(f"Deny enforcement fehlgeschlagen: {e}")
            return {"action_taken": False, "error": str(e)}

    async def _handle_throttle(
        self,
        violation: QuotaViolation,
        quota_check_result: QuotaCheckResult,
        _security_context: SecurityContext | None
    ) -> dict[str, Any]:
        """Behandelt THROTTLE-Enforcement-Action."""
        try:
            throttle_key = f"{violation.quota_id}_{violation.user_id or 'anonymous'}"

            # Erstelle Throttle-Entry
            self._throttled_requests[throttle_key] = {
                "violation_id": violation.violation_id,
                "throttle_start": time.time(),
                "retry_after_seconds": quota_check_result.retry_after_seconds or 60,
                "request_count": self._throttled_requests.get(throttle_key, {}).get("request_count", 0) + 1
            }

            logger.warning({
                "event": "quota_violation_throttled",
                "violation_id": violation.violation_id,
                "throttle_key": throttle_key,
                "retry_after_seconds": quota_check_result.retry_after_seconds
            })

            return {
                "action_taken": True,
                "action": "throttle",
                "message": f"Request throttled due to quota violation. Retry after {quota_check_result.retry_after_seconds} seconds",
                "retry_after_seconds": quota_check_result.retry_after_seconds,
                "throttle_key": throttle_key
            }

        except Exception as e:
            logger.error(f"Throttle enforcement fehlgeschlagen: {e}")
            return {"action_taken": False, "error": str(e)}

    async def _handle_queue(
        self,
        violation: QuotaViolation,
        _quota_check_result: QuotaCheckResult,
        _security_context: SecurityContext | None
    ) -> dict[str, Any]:
        """Behandelt QUEUE-Enforcement-Action."""
        try:
            queue_key = f"{violation.quota_id}_queue"

            # Erstelle Queue falls nicht vorhanden
            if queue_key not in self._throttle_queues:
                self._throttle_queues[queue_key] = asyncio.Queue(maxsize=1000)

            queue = self._throttle_queues[queue_key]

            # Füge Request zur Queue hinzu
            if not queue.full():
                await queue.put({
                    "violation_id": violation.violation_id,
                    "request_id": violation.request_id,
                    "user_id": violation.user_id,
                    "timestamp": time.time()
                })

                queue_position = queue.qsize()

                logger.info({
                    "event": "quota_violation_queued",
                    "violation_id": violation.violation_id,
                    "queue_key": queue_key,
                    "queue_position": queue_position
                })

                return {
                    "action_taken": True,
                    "action": "queue",
                    "message": f"Request queued due to quota violation. Position: {queue_position}",
                    "queue_position": queue_position,
                    "estimated_wait_seconds": queue_position * 5  # 5 Sekunden pro Request
                }
            # Queue voll - verweigern
            return {
                "action_taken": True,
                "action": "deny",
                "message": "Request denied - quota violation queue is full",
                "queue_full": True
            }

        except Exception as e:
            logger.error(f"Queue enforcement fehlgeschlagen: {e}")
            return {"action_taken": False, "error": str(e)}

    async def _handle_alert(
        self,
        violation: QuotaViolation,
        _quota_check_result: QuotaCheckResult,
        security_context: SecurityContext | None
    ) -> dict[str, Any]:
        """Behandelt ALERT-Enforcement-Action."""
        try:
            # Sende Alert
            await self._send_quota_alert(violation, security_context)

            self._alert_count += 1

            logger.warning({
                "event": "quota_violation_alert",
                "violation_id": violation.violation_id,
                "quota_id": violation.quota_id,
                "user_id": violation.user_id,
                "excess_amount": violation.excess_amount
            })

            return {
                "action_taken": True,
                "action": "alert",
                "message": "Quota violation alert sent - request allowed",
                "alert_sent": True
            }

        except Exception as e:
            logger.error(f"Alert enforcement fehlgeschlagen: {e}")
            return {"action_taken": False, "error": str(e)}

    async def _handle_scale(
        self,
        violation: QuotaViolation,
        quota_check_result: QuotaCheckResult,
        security_context: SecurityContext | None
    ) -> dict[str, Any]:
        """Behandelt SCALE-Enforcement-Action."""
        try:
            if not self.enable_auto_scaling:
                logger.warning("Auto-scaling nicht aktiviert - fallback zu ALERT")
                return await self._handle_alert(violation, quota_check_result, security_context)

            # TODO: Implementiere Auto-Scaling-Logic - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/114
            logger.info({
                "event": "quota_violation_auto_scale_triggered",
                "violation_id": violation.violation_id,
                "quota_id": violation.quota_id
            })

            return {
                "action_taken": True,
                "action": "scale",
                "message": "Auto-scaling triggered due to quota violation",
                "scaling_initiated": True
            }

        except Exception as e:
            logger.error(f"Scale enforcement fehlgeschlagen: {e}")
            return {"action_taken": False, "error": str(e)}

    async def _send_quota_alert(
        self,
        violation: QuotaViolation,
        _security_context: SecurityContext | None
    ) -> None:
        """Sendet Quota-Alert."""
        try:
            # TODO: Implementiere echtes Alerting-System - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/114
            alert_message = {
                "type": "quota_violation",
                "violation_id": violation.violation_id,
                "quota_id": violation.quota_id,
                "resource_type": violation.resource_type.value,
                "scope": violation.scope.value,
                "limit": violation.limit_value,
                "usage": violation.actual_value,
                "excess": violation.excess_amount,
                "user_id": violation.user_id,
                "timestamp": violation.violation_timestamp.isoformat()
            }

            logger.warning({
                "event": "quota_alert_sent",
                "alert": alert_message
            })

        except Exception as e:
            logger.error(f"Quota alert sending fehlgeschlagen: {e}")

    async def _create_security_event(
        self,
        violation: QuotaViolation,
        security_context: SecurityContext
    ) -> None:
        """Erstellt Security-Event für Quota-Violation."""
        try:
            # TODO: Integration mit Enhanced Security Integration Engine - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/114
            logger.info({
                "event": "quota_violation_security_event",
                "violation_id": violation.violation_id,
                "user_id": security_context.user_id,
                "tenant_id": security_context.tenant_id,
                "threat_level": "medium"
            })

        except Exception as e:
            logger.error(f"Security event creation fehlgeschlagen: {e}")

    async def _monitoring_loop(self) -> None:
        """Background-Loop für Real-time Monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(30)  # Monitor alle 30 Sekunden

                if self._is_running:
                    await self._monitor_active_violations()
                    await self._process_throttle_queues()

            except Exception as e:
                logger.error(f"Monitoring loop fehlgeschlagen: {e}")
                await asyncio.sleep(30)

    async def _cleanup_loop(self) -> None:
        """Background-Loop für Cleanup."""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # Cleanup alle 5 Minuten

                if self._is_running:
                    await self._cleanup_expired_violations()
                    await self._cleanup_expired_throttles()

            except Exception as e:
                logger.error(f"Cleanup loop fehlgeschlagen: {e}")
                await asyncio.sleep(300)

    async def _monitor_active_violations(self) -> None:
        """Monitort aktive Violations."""
        try:
            resolved_violations = []

            for violation_id, violation in self._active_violations.items():
                # Prüfe ob Violation resolved werden kann
                age_seconds = (datetime.utcnow() - violation.violation_timestamp).total_seconds()

                if age_seconds > 3600:  # 1 Stunde
                    violation.resolved_timestamp = datetime.utcnow()
                    resolved_violations.append(violation_id)

            # Entferne resolved Violations
            for violation_id in resolved_violations:
                del self._active_violations[violation_id]

            if resolved_violations:
                logger.debug(f"Monitoring: {len(resolved_violations)} violations resolved")

        except Exception as e:
            logger.error(f"Active violations monitoring fehlgeschlagen: {e}")

    async def _process_throttle_queues(self) -> None:
        """Verarbeitet Throttle-Queues."""
        try:
            for queue_key, queue in self._throttle_queues.items():
                if not queue.empty():
                    # Verarbeite nächsten Request in Queue
                    try:
                        queued_request = await asyncio.wait_for(queue.get(), timeout=0.1)

                        logger.debug({
                            "event": "throttle_queue_processed",
                            "queue_key": queue_key,
                            "request_id": queued_request.get("request_id"),
                            "queue_size": queue.qsize()
                        })

                    except TimeoutError:
                        pass

        except Exception as e:
            logger.error(f"Throttle queue processing fehlgeschlagen: {e}")

    async def _cleanup_expired_violations(self) -> None:
        """Bereinigt abgelaufene Violations."""
        try:
            from datetime import datetime, timedelta

            cutoff_time = datetime.utcnow() - timedelta(hours=24)

            original_count = len(self._violation_history)
            self._violation_history = [
                v for v in self._violation_history
                if v.violation_timestamp > cutoff_time
            ]

            cleaned_count = original_count - len(self._violation_history)
            if cleaned_count > 0:
                logger.debug(f"Violation cleanup: {cleaned_count} expired violations entfernt")

        except Exception as e:
            logger.error(f"Violation cleanup fehlgeschlagen: {e}")

    async def _cleanup_expired_throttles(self) -> None:
        """Bereinigt abgelaufene Throttles."""
        try:
            current_time = time.time()
            expired_throttles = []

            for throttle_key, throttle_data in self._throttled_requests.items():
                throttle_age = current_time - throttle_data["throttle_start"]
                retry_after = throttle_data["retry_after_seconds"]

                if throttle_age > retry_after:
                    expired_throttles.append(throttle_key)

            for throttle_key in expired_throttles:
                del self._throttled_requests[throttle_key]

            if expired_throttles:
                logger.debug(f"Throttle cleanup: {len(expired_throttles)} expired throttles entfernt")

        except Exception as e:
            logger.error(f"Throttle cleanup fehlgeschlagen: {e}")

    def _update_enforcement_performance_stats(self, enforcement_time_ms: float) -> None:
        """Aktualisiert Enforcement-Performance-Statistiken."""
        self._enforcement_count += 1
        self._total_enforcement_time_ms += enforcement_time_ms

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_enforcement_time = (
            self._total_enforcement_time_ms / self._enforcement_count
            if self._enforcement_count > 0 else 0.0
        )

        return {
            "total_enforcements": self._enforcement_count,
            "avg_enforcement_time_ms": avg_enforcement_time,
            "total_violations": self._violation_count,
            "active_violations": len(self._active_violations),
            "violation_history_size": len(self._violation_history),
            "throttled_requests": len(self._throttled_requests),
            "throttle_queues": len(self._throttle_queues),
            "alerts_sent": self._alert_count,
            "real_time_enforcement_enabled": self.enable_real_time_enforcement,
            "auto_scaling_enabled": self.enable_auto_scaling
        }
