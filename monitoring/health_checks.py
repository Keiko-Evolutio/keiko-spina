# backend/monitoring/health_checks.py
"""Einfaches Health-Check-System zur Prüfung interner Komponenten.

Unterstützt synchrone und asynchrone Prüfungen mit Zeitlimit und
liefert strukturierte Ergebnisse zur Aggregation.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from prometheus_client import Gauge

from config.settings import settings
from kei_logging import get_logger
from monitoring import record_custom_metric
from storage.cache.redis_cache import NoOpCache, get_cache_client

# Import der gemeinsamen Error-Handling-Patterns
from .error_handling import (
    log_and_ignore_error,
)

logger = get_logger(__name__)

# Prometheus Gauges für SLA - sichere Registrierung
try:
    SLA_AVAILABILITY = Gauge(
        "keiko_sla_availability_percentage",
        "SLA Verfügbarkeit in Prozent",
        labelnames=("service", "component", "environment"),
    )
except ValueError:
    # Metrik bereits registriert, verwende existierende
    from prometheus_client import REGISTRY
    SLA_AVAILABILITY = REGISTRY._names_to_collectors.get("keiko_sla_availability_percentage")

try:
    SLA_P95_LATENCY = Gauge(
        "keiko_sla_p95_latency_ms",
        "SLA p95 Latenz in Millisekunden",
        labelnames=("service", "component", "environment"),
    )
except ValueError:
    # Metrik bereits registriert, verwende existierende
    from prometheus_client import REGISTRY
    SLA_P95_LATENCY = REGISTRY._names_to_collectors.get("keiko_sla_p95_latency_ms")
try:
    SLA_ERROR_RATE = Gauge(
        "keiko_sla_error_rate_percentage",
        "SLA Fehlerrate in Prozent",
        labelnames=("service", "component", "environment"),
    )
except ValueError:
    # Metrik bereits registriert, verwende existierende
    from prometheus_client import REGISTRY
    SLA_ERROR_RATE = REGISTRY._names_to_collectors.get("keiko_sla_error_rate_percentage")

try:
    SLA_COMPLIANCE = Gauge(
        "keiko_sla_compliance_status",
        "SLA Compliance Status (1=konform, 0=abweichend)",
        labelnames=("service", "component", "environment"),
    )
except ValueError:
    # Metrik bereits registriert, verwende existierende
    from prometheus_client import REGISTRY
    SLA_COMPLIANCE = REGISTRY._names_to_collectors.get("keiko_sla_compliance_status")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Import der gemeinsamen Konfiguration
from .config_base import (
    RING_BUFFER_CAPACITY,
    HealthCheckConfig,
)

# ============================================================================
# HEALTH CHECK KOMPONENTE
# ============================================================================

class ComponentHealthCheck:
    """Einzelner Health Check für eine Komponente."""

    def __init__(self, name: str, check_function: Callable[[], Any | Awaitable[Any]]):
        self.name = name
        self.check_function = check_function

    async def execute(self, timeout: float) -> dict[str, Any]:
        """Führt Health Check aus."""
        start_time = datetime.now()

        try:
            # Timeout-Handler
            result = await asyncio.wait_for(
                self._run_check(),
                timeout=timeout
            )

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            return {
                "name": self.name,
                "healthy": True,
                "duration_ms": duration_ms,
                "details": result if isinstance(result, dict) else {"status": "ok"},
                "timestamp": start_time.isoformat()
            }

        except TimeoutError:
            return {
                "name": self.name,
                "healthy": False,
                "error": "timeout",
                "duration_ms": timeout * 1000,
                "timestamp": start_time.isoformat()
            }
        except Exception as e:
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            return {
                "name": self.name,
                "healthy": False,
                "error": str(e),
                "duration_ms": duration_ms,
                "timestamp": start_time.isoformat()
            }

    async def _run_check(self) -> Any:
        """Führt eigentlichen Check aus."""
        if asyncio.iscoroutinefunction(self.check_function):
            return await self.check_function()
        return self.check_function()


# ============================================================================
# HEALTH CHECK MANAGER - VEREINFACHT
# ============================================================================

class HealthCheckManager:
    """Health Check Manager."""

    def __init__(self, config: HealthCheckConfig | None = None):
        self.config = config or HealthCheckConfig()
        self._checks: list[ComponentHealthCheck] = []
        self._running = False
        # SLA Tracking (Rolling 24h, Minutenauflösung)
        self._total_runs: int = 0
        self._successful_runs: int = 0
        self._error_count: int = 0
        self._latency_samples_ms: list[float] = []
        self._ring_capacity: int = RING_BUFFER_CAPACITY
        self._availability_ring: list[float] = []  # 0..1 pro Minute
        self._p95_latency_ring: list[float] = []  # ms pro Minute
        self._error_rate_ring: list[float] = []  # 0..1 pro Minute
        self._ring_last_minute: int | None = None

    async def start(self) -> None:
        """Startet Health Check Manager."""
        self._running = True

        # Standard-Checks registrieren
        self.add_check("basic", self._basic_health_check)

        logger.info("Health Check Manager gestartet")

    async def stop(self) -> None:
        """Stoppt Health Check Manager."""
        self._running = False
        self._checks.clear()
        logger.info("Health Check Manager gestoppt")

    def add_check(self, name: str, check_function: Callable[[], Any | Awaitable[Any]]) -> None:
        """Fügt Health Check hinzu."""
        check = ComponentHealthCheck(name, check_function)
        self._checks.append(check)
        logger.debug(f"Health Check '{name}' hinzugefügt")

    async def check_health(self) -> dict[str, Any]:
        """Führt alle Health Checks aus."""
        # Frühe Rückgabe bei deaktiviertem oder nicht laufendem Manager
        early_return = self._check_early_return_conditions()
        if early_return:
            return early_return

        # Führe alle Health Checks parallel aus
        results = await self._execute_health_checks()
        if not results:
            return self._create_no_checks_response()

        # Analysiere Ergebnisse und berechne Metriken
        analysis = self._analyze_health_check_results(results)

        # Aktualisiere SLA-Tracking
        self._update_sla_tracking(analysis)

        # Persistiere Metriken
        await self._persist_sla_metrics(analysis)

        # Aktualisiere externe Metriken-Systeme
        self._update_external_metrics(analysis)

        # Erstelle finale Antwort
        return self._create_health_response(analysis)

    def _check_early_return_conditions(self) -> dict[str, Any] | None:
        """Prüft Bedingungen für frühe Rückgabe."""
        if not self.config.enabled:
            return {
                "healthy": True,
                "status": "disabled",
                "components": {}
            }

        if not self._running:
            return {
                "healthy": False,
                "status": "not_running",
                "components": {}
            }

        return None

    async def _execute_health_checks(self) -> list[Any]:
        """Führt alle Health Checks parallel aus."""
        tasks = [
            check.execute(self.config.timeout_seconds)
            for check in self._checks
        ]

        if not tasks:
            return []

        return await asyncio.gather(*tasks, return_exceptions=True)

    def _create_no_checks_response(self) -> dict[str, Any]:
        """Erstellt Antwort wenn keine Checks registriert sind."""
        return {
            "healthy": True,
            "status": "no_checks",
            "components": {}
        }

    def _analyze_health_check_results(self, results: list[Any]) -> dict[str, Any]:
        """Analysiert Health Check Ergebnisse und berechnet Metriken."""
        components: dict[str, Any] = {}
        overall_healthy = True
        warning_count = 0
        sla_latency_violations = 0

        # Verarbeite einzelne Ergebnisse
        for result in results:
            if isinstance(result, Exception):
                overall_healthy = False
                continue

            components[result["name"]] = result

            if not result["healthy"]:
                overall_healthy = False
            elif result["duration_ms"] > self.config.warning_threshold_ms:
                warning_count += 1

            # SLA Latenz prüfen
            if result.get("duration_ms", 0) > self.config.sla_latency_ms:
                sla_latency_violations += 1

        # Status bestimmen
        status = self._determine_overall_status(overall_healthy, warning_count)

        # Berechne SLA-Metriken
        sla_metrics = self._calculate_sla_metrics(components)

        return {
            "components": components,
            "overall_healthy": overall_healthy,
            "warning_count": warning_count,
            "sla_latency_violations": sla_latency_violations,
            "status": status,
            "sla_metrics": sla_metrics
        }

    def _determine_overall_status(self, overall_healthy: bool, warning_count: int) -> str:
        """Bestimmt den Gesamtstatus basierend auf Health Check Ergebnissen."""
        if overall_healthy and warning_count == 0:
            return "healthy"
        if overall_healthy and warning_count > 0:
            return "degraded"
        return "unhealthy"

    def _calculate_sla_metrics(self, components: dict[str, Any]) -> dict[str, Any]:
        """Berechnet SLA-Metriken basierend auf Health Check Ergebnissen."""
        # SLA Availability Tracking aktualisieren
        self._total_runs += max(1, len(self._checks))
        self._successful_runs += sum(1 for c in components.values() if c["healthy"]) or 0
        availability = (self._successful_runs / max(1, self._total_runs))

        # Fehlerquote berechnen
        total_components = max(1, len(components))
        error_rate = sum(1 for c in components.values() if not c["healthy"]) / total_components

        # P95 Latenz berechnen
        p95_latency = self._calculate_p95_latency(components)

        return {
            "availability": availability,
            "error_rate": error_rate,
            "p95_latency_ms": p95_latency
        }

    def _calculate_p95_latency(self, components: dict[str, Any]) -> float:
        """Berechnet P95 Latenz aus Health Check Ergebnissen."""
        try:
            latencies = [float(c.get("duration_ms", 0.0)) for c in components.values()]
            latencies = sorted(latencies)
            if latencies:
                idx = max(0, int(0.95 * (len(latencies) - 1)))
                return latencies[idx]
        except Exception:
            pass
        return 0.0

    def _update_sla_tracking(self, analysis: dict[str, Any]) -> None:
        """Aktualisiert SLA-Tracking mit Ringbuffer."""
        sla_metrics = analysis["sla_metrics"]
        availability = sla_metrics["availability"]
        p95_latency = sla_metrics["p95_latency_ms"]
        error_rate = sla_metrics["error_rate"]

        # Ringbuffer aktualisieren (Minutenauflösung)
        now_minute = int(datetime.now().timestamp() // 60)
        if self._ring_last_minute != now_minute:
            self._availability_ring.append(availability)
            self._p95_latency_ring.append(p95_latency)
            self._error_rate_ring.append(error_rate)
            self._availability_ring = self._availability_ring[-self._ring_capacity:]
            self._p95_latency_ring = self._p95_latency_ring[-self._ring_capacity:]
            self._error_rate_ring = self._error_rate_ring[-self._ring_capacity:]
            self._ring_last_minute = now_minute

    async def _persist_sla_metrics(self, analysis: dict[str, Any]) -> None:
        """Persistiert SLA-Metriken in Redis."""
        if not settings.sla_trend_enable_persistence:
            return

        sla_metrics = analysis["sla_metrics"]

        def persist_to_redis():
            client = get_cache_client()
            if client and not isinstance(client, NoOpCache):
                ts = int(datetime.now().timestamp())
                prefix = settings.sla_trend_redis_prefix

                # Keys erzeugen
                k_avail_24 = f"{prefix}:availability:24h"
                k_lat_24 = f"{prefix}:latency_p95_ms:24h"
                k_err_24 = f"{prefix}:error_rate_pct:24h"

                # Add als Sorted Set (score=timestamp)
                client.zadd(k_avail_24, {str(sla_metrics["availability"]): ts})  # type: ignore[attr-defined]
                client.zadd(k_lat_24, {str(sla_metrics["p95_latency_ms"]): ts})  # type: ignore[attr-defined]
                client.zadd(k_err_24, {str(sla_metrics["error_rate"] * 100.0): ts})  # type: ignore[attr-defined]

                # Trim nach Retention
                cutoff_24 = ts - settings.sla_trend_retention_seconds_24h
                client.zremrangebyscore(k_avail_24, 0, cutoff_24)  # type: ignore[attr-defined]
                client.zremrangebyscore(k_lat_24, 0, cutoff_24)  # type: ignore[attr-defined]
                client.zremrangebyscore(k_err_24, 0, cutoff_24)  # type: ignore[attr-defined]

        log_and_ignore_error(persist_to_redis, "SLA Metrics Redis Persistierung")

    def _update_external_metrics(self, analysis: dict[str, Any]) -> None:
        """Aktualisiert externe Metriken-Systeme (Prometheus, Custom Metrics)."""
        sla_metrics = analysis["sla_metrics"]
        components = analysis["components"]

        # Prometheus Gauges aktualisieren
        def update_prometheus_gauges():
            env = getattr(settings, "environment", "development")
            availability_pct = sla_metrics["availability"] * 100.0
            error_rate_pct = sla_metrics["error_rate"] * 100.0
            p95_latency = sla_metrics["p95_latency_ms"]

            SLA_AVAILABILITY.labels(service="keiko-api", component="health", environment=env).set(availability_pct)
            SLA_P95_LATENCY.labels(service="keiko-api", component="health", environment=env).set(p95_latency)
            SLA_ERROR_RATE.labels(service="keiko-api", component="health", environment=env).set(error_rate_pct)

            compliant = 1.0 if (
                availability_pct >= settings.sla_availability_target_pct and
                p95_latency <= settings.sla_latency_target_ms and
                error_rate_pct <= settings.sla_error_rate_target_pct
            ) else 0.0
            SLA_COMPLIANCE.labels(service="keiko-api", component="health", environment=env).set(compliant)

        log_and_ignore_error(update_prometheus_gauges, "Prometheus Gauges Update")

        # Custom Metrics aktualisieren
        def record_custom_metrics():
            base_tags = {"service_name": "keiko-api", "check_type": "health", "sla_tier": "gold", "tenant_id": "-"}
            record_custom_metric("sla.availability_percentage", sla_metrics["availability"] * 100.0, base_tags)
            record_custom_metric("sla.response_time_p95_ms", sla_metrics["p95_latency_ms"], base_tags)
            record_custom_metric("sla.error_rate_percentage", sla_metrics["error_rate"] * 100.0, base_tags)

            success_rate = (sum(1 for c in components.values() if c["healthy"]) / max(1, len(components))) * 100.0
            record_custom_metric("sla.health_check_success_rate", success_rate, base_tags)

        log_and_ignore_error(record_custom_metrics, "Custom Metrics Recording")

    def _create_health_response(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Erstellt die finale Health Check Antwort."""
        components = analysis["components"]
        sla_metrics = analysis["sla_metrics"]

        return {
            "healthy": analysis["overall_healthy"],
            "status": analysis["status"],
            "components": components,
            "summary": {
                "total_checks": len(self._checks),
                "healthy_checks": sum(1 for c in components.values() if c["healthy"]),
                "warning_checks": analysis["warning_count"],
                "sla": {
                    "latency_ms_target": self.config.sla_latency_ms,
                    "latency_violations": analysis["sla_latency_violations"],
                    "availability_target": self.config.sla_availability_target,
                    "availability": sla_metrics["availability"],
                    "availability_ok": sla_metrics["availability"] >= self.config.sla_availability_target,
                    "error_rate_pct": sla_metrics["error_rate"] * 100.0,
                    "p95_latency_ms": sla_metrics["p95_latency_ms"],
                    "availability_trend": self._availability_ring[-24:],
                    "latency_trend": self._p95_latency_ring[-24:],
                    "error_trend": [x * 100.0 for x in self._error_rate_ring[-24:]],
                },
            },
            "timestamp": datetime.now().isoformat(),
            "sla": {
                "availability": sla_metrics["availability"],
                "p95_latency_ms": sla_metrics["p95_latency_ms"],
                "error_rate_pct": sla_metrics["error_rate"] * 100.0,
                "availability_trend": self._get_trend_data(self._availability_ring),
                "latency_trend": self._get_trend_data(self._p95_latency_ring),
                "error_trend": [x * 100.0 for x in self._get_trend_data(self._error_rate_ring)],
            }
        }

    def _get_trend_data(self, ring_buffer: list[float]) -> list[float]:
        """Extrahiert Trend-Daten aus Ringbuffer."""
        if not ring_buffer:
            return []
        return ring_buffer[-24:][::-1][:24][::-1]

    async def _basic_health_check(self) -> dict[str, Any]:
        """Basis Health Check."""
        return {
            "status": "ok",
            "manager_running": self._running,
            "checks_registered": len(self._checks)
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_database_health_check(connection_string: str) -> Callable[[], Awaitable[dict[str, Any]]]:
    """Erstellt Database Health Check."""

    async def check():
        # DB-Check
        return {"database": "connected"}

    return check


def create_redis_health_check(redis_url: str) -> Callable[[], Awaitable[dict[str, Any]]]:
    """Erstellt Redis Health Check."""

    async def check():
        # Redis-Check
        return {"redis": "connected"}

    return check


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ComponentHealthCheck",
    "HealthCheckConfig",
    "HealthCheckManager",
    "create_database_health_check",
    "create_redis_health_check",
]
