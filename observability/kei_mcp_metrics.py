"""Erweiterte Observability-Metriken für externe MCP Server Integration.

Implementiert detaillierte Metriken für Circuit Breaker Status, Connection Pool
Statistiken und strukturierte Error Categorization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

logger = get_logger(__name__)


class ErrorCategory(Enum):
    """Kategorien für Fehler-Klassifizierung."""

    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    VALIDATION_ERROR = "validation_error"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    SERVER_ERROR = "server_error"
    CIRCUIT_BREAKER_ERROR = "circuit_breaker_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ConnectionPoolStats:
    """Statistiken für HTTP Connection Pool."""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    max_connections: int = 0
    connection_reuse_count: int = 0
    connection_create_count: int = 0
    connection_close_count: int = 0
    avg_connection_lifetime_seconds: float = 0.0


@dataclass
class ErrorMetrics:
    """Metriken für Fehler-Tracking."""

    total_errors: int = 0
    errors_by_category: dict[ErrorCategory, int] = field(default_factory=dict)
    errors_by_server: dict[str, int] = field(default_factory=dict)
    last_error_time: float | None = None
    error_rate_per_minute: float = 0.0

    def record_error(self, category: ErrorCategory, server_name: str):
        """Verzeichnet einen Fehler."""
        self.total_errors += 1
        self.errors_by_category[category] = self.errors_by_category.get(category, 0) + 1
        self.errors_by_server[server_name] = self.errors_by_server.get(server_name, 0) + 1
        self.last_error_time = time.time()


@dataclass
class PerformanceMetrics:
    """Performance-Metriken für MCP Operations."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    p50_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    response_times: list[float] = field(default_factory=list)

    def record_request(self, response_time_ms: float, success: bool):
        """Verzeichnet einen Request."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.response_times.append(response_time_ms)

        # Behalte nur die letzten 1000 Response Times für Percentile-Berechnung
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]

        self._update_percentiles()

    def _update_percentiles(self):
        """Aktualisiert Percentile-Werte."""
        if not self.response_times:
            return

        sorted_times = sorted(self.response_times)
        count = len(sorted_times)

        self.avg_response_time_ms = sum(sorted_times) / count
        self.p50_response_time_ms = sorted_times[int(count * 0.5)]
        self.p95_response_time_ms = sorted_times[int(count * 0.95)]
        self.p99_response_time_ms = sorted_times[int(count * 0.99)]


class KEIMCPMetricsCollector:
    """Sammelt und verwaltet Metriken für externe MCP Server Integration."""

    def __init__(self):
        """Initialisiert Metrics Collector."""
        self.start_time = time.time()
        self.error_metrics = ErrorMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.connection_pool_stats: dict[str, ConnectionPoolStats] = {}
        self.circuit_breaker_stats: dict[str, dict[str, Any]] = {}
        self.server_metrics: dict[str, dict[str, Any]] = {}

        # Rate Limiting Metriken
        self.rate_limit_checks = {}
        self.rate_limit_rejections = {}
        self.rate_limit_current_usage = {}
        self.rate_limit_soft_limit_warnings = {}
        self.rate_limit_backend_health = {}
        self.rate_limit_cleanup_operations = {}
        self.rate_limit_cleanup_deleted_entries = {}

    @trace_function("metrics.record_request")
    def record_request(
        self,
        server_name: str,
        operation: str,
        response_time_ms: float,
        success: bool,
        error_category: ErrorCategory | None = None
    ):
        """Verzeichnet einen Request mit allen relevanten Metriken.

        Args:
            server_name: Name des MCP Servers
            operation: Art der Operation (invoke, discovery, health)
            response_time_ms: Antwortzeit in Millisekunden
            success: Ob der Request erfolgreich war
            error_category: Kategorie des Fehlers (falls nicht erfolgreich)
        """
        # Performance-Metriken aktualisieren
        self.performance_metrics.record_request(response_time_ms, success)

        # Server-spezifische Metriken
        if server_name not in self.server_metrics:
            self.server_metrics[server_name] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "avg_response_time_ms": 0.0,
                "operations": {}
            }

        server_stats = self.server_metrics[server_name]
        server_stats["total_requests"] += 1

        if success:
            server_stats["successful_requests"] += 1
        else:
            server_stats["failed_requests"] += 1
            if error_category:
                self.error_metrics.record_error(error_category, server_name)

        # Operation-spezifische Metriken
        if operation not in server_stats["operations"]:
            server_stats["operations"][operation] = {
                "count": 0,
                "avg_response_time_ms": 0.0,
                "success_rate": 0.0
            }

        op_stats = server_stats["operations"][operation]
        op_stats["count"] += 1

        # Gleitender Durchschnitt für Response Time
        alpha = 0.1
        if op_stats["avg_response_time_ms"] == 0:
            op_stats["avg_response_time_ms"] = response_time_ms
        else:
            op_stats["avg_response_time_ms"] = (
                alpha * response_time_ms +
                (1 - alpha) * op_stats["avg_response_time_ms"]
            )

        # Success Rate aktualisieren
        if server_stats["total_requests"] > 0:
            op_stats["success_rate"] = (
                server_stats["successful_requests"] / server_stats["total_requests"]
            )

    def update_connection_pool_stats(self, server_name: str, stats: ConnectionPoolStats):
        """Aktualisiert Connection Pool Statistiken.

        Args:
            server_name: Name des MCP Servers
            stats: Connection Pool Statistiken
        """
        self.connection_pool_stats[server_name] = stats

    def update_circuit_breaker_stats(self, server_name: str, stats: dict[str, Any]):
        """Aktualisiert Circuit Breaker Statistiken.

        Args:
            server_name: Name des MCP Servers
            stats: Circuit Breaker Statistiken
        """
        self.circuit_breaker_stats[server_name] = stats

    def get_comprehensive_metrics(self) -> dict[str, Any]:
        """Gibt umfassende Metriken zurück.

        Returns:
            Dictionary mit allen gesammelten Metriken
        """
        current_time = time.time()
        uptime_seconds = current_time - self.start_time

        return {
            "timestamp": current_time,
            "uptime_seconds": uptime_seconds,

            # Allgemeine Performance-Metriken
            "performance": {
                "total_requests": self.performance_metrics.total_requests,
                "successful_requests": self.performance_metrics.successful_requests,
                "failed_requests": self.performance_metrics.failed_requests,
                "success_rate": (
                    self.performance_metrics.successful_requests /
                    max(1, self.performance_metrics.total_requests)
                ),
                "avg_response_time_ms": self.performance_metrics.avg_response_time_ms,
                "p50_response_time_ms": self.performance_metrics.p50_response_time_ms,
                "p95_response_time_ms": self.performance_metrics.p95_response_time_ms,
                "p99_response_time_ms": self.performance_metrics.p99_response_time_ms,
                "requests_per_second": (
                    self.performance_metrics.total_requests / max(1, uptime_seconds)
                )
            },

            # Fehler-Metriken
            "errors": {
                "total_errors": self.error_metrics.total_errors,
                "error_rate": (
                    self.error_metrics.total_errors /
                    max(1, self.performance_metrics.total_requests)
                ),
                "errors_by_category": {
                    category.value: count
                    for category, count in self.error_metrics.errors_by_category.items()
                },
                "errors_by_server": self.error_metrics.errors_by_server,
                "last_error_time": self.error_metrics.last_error_time
            },

            # Server-spezifische Metriken
            "servers": self.server_metrics,

            # Connection Pool Statistiken
            "connection_pools": {
                server: {
                    "total_connections": stats.total_connections,
                    "active_connections": stats.active_connections,
                    "idle_connections": stats.idle_connections,
                    "max_connections": stats.max_connections,
                    "connection_reuse_rate": (
                        stats.connection_reuse_count /
                        max(1, stats.connection_create_count)
                    ),
                    "avg_connection_lifetime_seconds": stats.avg_connection_lifetime_seconds
                }
                for server, stats in self.connection_pool_stats.items()
            },

            # Circuit Breaker Statistiken
            "circuit_breakers": self.circuit_breaker_stats
        }

    def get_health_summary(self) -> dict[str, Any]:
        """Gibt Gesundheits-Zusammenfassung zurück.

        Returns:
            Dictionary mit Gesundheitsstatus
        """
        metrics = self.get_comprehensive_metrics()

        # Bestimme Gesamtstatus
        error_rate = metrics["errors"]["error_rate"]
        avg_response_time = metrics["performance"]["avg_response_time_ms"]

        if error_rate > 0.1:  # > 10% Fehlerrate
            status = "unhealthy"
        elif error_rate > 0.05 or avg_response_time > 5000:  # > 5% Fehlerrate oder > 5s Response Time
            status = "degraded"
        else:
            status = "healthy"

        # Circuit Breaker Status prüfen
        open_circuits = sum(
            1 for stats in self.circuit_breaker_stats.values()
            if stats.get("state") == "open"
        )

        return {
            "status": status,
            "error_rate": error_rate,
            "avg_response_time_ms": avg_response_time,
            "total_servers": len(self.server_metrics),
            "open_circuit_breakers": open_circuits,
            "total_requests": metrics["performance"]["total_requests"],
            "uptime_seconds": metrics["uptime_seconds"]
        }

    def reset_metrics(self):
        """Setzt alle Metriken zurück."""
        self.start_time = time.time()
        self.error_metrics = ErrorMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.connection_pool_stats.clear()
        self.circuit_breaker_stats.clear()
        self.server_metrics.clear()

        logger.info("Externe MCP Metriken zurückgesetzt")


# Globale Metrics Collector Instanz
kei_mcp_metrics = KEIMCPMetricsCollector()

# Alias für Kompatibilität
mcp_metrics = kei_mcp_metrics


def categorize_error(exception: Exception) -> ErrorCategory:
    """Kategorisiert Exception in ErrorCategory.

    Args:
        exception: Aufgetretene Exception

    Returns:
        Passende ErrorCategory
    """
    import httpx
    from fastapi import HTTPException

    from agents.tools.mcp.kei_mcp_circuit_breaker import CircuitBreakerException

    if isinstance(exception, CircuitBreakerException):
        return ErrorCategory.CIRCUIT_BREAKER_ERROR

    if isinstance(exception, HTTPException):
        if exception.status_code == 401:
            return ErrorCategory.AUTHENTICATION_ERROR
        if exception.status_code == 403:
            return ErrorCategory.AUTHORIZATION_ERROR
        if exception.status_code == 422:
            return ErrorCategory.VALIDATION_ERROR
        if exception.status_code == 429:
            return ErrorCategory.RATE_LIMIT_ERROR
        if 500 <= exception.status_code < 600:
            return ErrorCategory.SERVER_ERROR

    elif isinstance(exception, httpx.ConnectError | httpx.NetworkError):
        return ErrorCategory.NETWORK_ERROR

    elif isinstance(exception, httpx.TimeoutException | TimeoutError):
        return ErrorCategory.TIMEOUT_ERROR

    elif "tool" in str(exception).lower():
        return ErrorCategory.TOOL_EXECUTION_ERROR

    return ErrorCategory.UNKNOWN_ERROR


__all__ = [
    "ConnectionPoolStats",
    "ErrorCategory",
    "ErrorMetrics",
    "KEIMCPMetricsCollector",
    "PerformanceMetrics",
    "categorize_error",
    "kei_mcp_metrics",
    "mcp_metrics"  # Alias für Kompatibilität
]
