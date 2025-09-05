# backend/services/llm/llm_monitoring.py
"""LLM Monitoring und Alerting System.

Implementiert umfassendes Monitoring für LLM-Requests mit
Token-Usage-Tracking, Cost-Monitoring und Alerting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger
from monitoring.metrics_collector import MetricsCollector

logger = get_logger(__name__)


@dataclass
class LLMMetrics:
    """LLM Metriken Datenmodell."""

    # Request Metriken
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cached_requests: int = 0

    # Token Metriken
    total_tokens_used: int = 0
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0

    # Cost Metriken
    total_cost_usd: float = 0.0
    cost_per_model: dict[str, float] = field(default_factory=dict)

    # Performance Metriken
    avg_response_time_ms: float = 0.0
    min_response_time_ms: float = float("inf")
    max_response_time_ms: float = 0.0

    # Rate Limiting Metriken
    rate_limit_hits: int = 0
    budget_exhaustions: int = 0

    # Zeitfenster
    window_start: datetime = field(default_factory=datetime.utcnow)

    def update_request_metrics(self, success: bool, cached: bool, response_time_ms: float) -> None:
        """Aktualisiert Request-Metriken."""
        self.total_requests += 1

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        if cached:
            self.cached_requests += 1

        # Response Time Metriken
        if response_time_ms > 0:
            self.min_response_time_ms = min(self.min_response_time_ms, response_time_ms)
            self.max_response_time_ms = max(self.max_response_time_ms, response_time_ms)

            # Einfache gleitende Durchschnittsberechnung
            if self.avg_response_time_ms == 0:
                self.avg_response_time_ms = response_time_ms
            else:
                self.avg_response_time_ms = (self.avg_response_time_ms + response_time_ms) / 2

    def update_token_metrics(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Aktualisiert Token-Metriken."""
        self.prompt_tokens_used += prompt_tokens
        self.completion_tokens_used += completion_tokens
        self.total_tokens_used += prompt_tokens + completion_tokens

    def update_cost_metrics(self, model: str, cost_usd: float) -> None:
        """Aktualisiert Cost-Metriken."""
        self.total_cost_usd += cost_usd

        if model not in self.cost_per_model:
            self.cost_per_model[model] = 0.0
        self.cost_per_model[model] += cost_usd

    def get_success_rate(self) -> float:
        """Berechnet Erfolgsrate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    def get_cache_hit_rate(self) -> float:
        """Berechnet Cache-Hit-Rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cached_requests / self.total_requests


@dataclass
class AlertConfig:
    """Konfiguration für Alerts."""

    # Cost Alerts
    cost_threshold_usd: float = 5.0
    cost_alert_interval_minutes: int = 15

    # Performance Alerts
    response_time_threshold_ms: float = 5000.0
    error_rate_threshold: float = 0.1  # 10%

    # Rate Limiting Alerts
    rate_limit_alert_threshold: int = 5

    # Token Usage Alerts
    token_usage_threshold_per_hour: int = 100000


@dataclass
class Alert:
    """Alert Datenmodell."""

    alert_type: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    metrics: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class LLMMonitor:
    """LLM Monitoring und Alerting System."""

    def __init__(self, alert_config: AlertConfig):
        """Initialisiert LLM Monitor.

        Args:
            alert_config: Alert-Konfiguration
        """
        self.alert_config = alert_config
        self.metrics = LLMMetrics()
        self.hourly_metrics: list[LLMMetrics] = []
        self.alerts: list[Alert] = []

        # Letzte Alert-Zeitstempel für Throttling
        self._last_cost_alert = datetime.min
        self._last_performance_alert = datetime.min
        self._last_rate_limit_alert = datetime.min

        # Metrics Collector Integration
        self.metrics_collector = MetricsCollector()

        logger.info({
            "event": "llm_monitor_initialized",
            "cost_threshold": alert_config.cost_threshold_usd,
            "response_time_threshold": alert_config.response_time_threshold_ms
        })

    def record_request(
        self,
        model: str,
        success: bool,
        cached: bool,
        response_time_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0
    ) -> None:
        """Registriert LLM Request für Monitoring.

        Args:
            model: Verwendetes Modell
            success: Ob Request erfolgreich war
            cached: Ob Response aus Cache kam
            response_time_ms: Response-Zeit in Millisekunden
            prompt_tokens: Anzahl Prompt-Tokens
            completion_tokens: Anzahl Completion-Tokens
            cost_usd: Kosten in USD
        """
        # Metriken aktualisieren
        self.metrics.update_request_metrics(success, cached, response_time_ms)

        if success and not cached:
            self.metrics.update_token_metrics(prompt_tokens, completion_tokens)
            self.metrics.update_cost_metrics(model, cost_usd)

        # Prometheus Metriken exportieren
        self._export_prometheus_metrics(model, success, cached, response_time_ms, cost_usd)

        # Alerts prüfen
        self._check_alerts()

        logger.debug({
            "event": "llm_request_recorded",
            "model": model,
            "success": success,
            "cached": cached,
            "response_time_ms": response_time_ms,
            "cost_usd": cost_usd
        })

    def record_rate_limit_hit(self) -> None:
        """Registriert Rate Limit Hit."""
        self.metrics.rate_limit_hits += 1
        self._check_rate_limit_alerts()

    def record_budget_exhaustion(self) -> None:
        """Registriert Budget-Erschöpfung."""
        self.metrics.budget_exhaustions += 1
        self._create_alert(
            "budget_exhaustion",
            "critical",
            "LLM Budget erschöpft - weitere Requests blockiert",
            {"budget_exhaustions": self.metrics.budget_exhaustions}
        )

    def _export_prometheus_metrics(
        self,
        model: str,
        success: bool,
        cached: bool,
        response_time_ms: float,
        cost_usd: float
    ) -> None:
        """Exportiert Metriken zu Prometheus."""
        try:
            # Request Counter
            self.metrics_collector.increment_counter(
                "llm_requests_total",
                labels={"model": model, "success": str(success), "cached": str(cached)}
            )

            # Response Time Histogram
            self.metrics_collector.observe_histogram(
                "llm_response_time_ms",
                response_time_ms,
                labels={"model": model}
            )

            # Cost Gauge
            if cost_usd > 0:
                self.metrics_collector.set_gauge(
                    "llm_cost_usd_total",
                    self.metrics.total_cost_usd
                )

            # Token Usage Gauge
            self.metrics_collector.set_gauge(
                "llm_tokens_used_total",
                self.metrics.total_tokens_used
            )

        except Exception as e:
            logger.warning(f"Prometheus Metrics Export fehlgeschlagen: {e}")

    def _check_alerts(self) -> None:
        """Prüft alle Alert-Bedingungen."""
        self._check_cost_alerts()
        self._check_performance_alerts()
        self._check_error_rate_alerts()

    def _check_cost_alerts(self) -> None:
        """Prüft Cost-basierte Alerts."""
        now = datetime.utcnow()

        # Throttling: Nur alle X Minuten Cost-Alerts
        if (now - self._last_cost_alert).total_seconds() < self.alert_config.cost_alert_interval_minutes * 60:
            return

        if self.metrics.total_cost_usd >= self.alert_config.cost_threshold_usd:
            severity = "high" if self.metrics.total_cost_usd < self.alert_config.cost_threshold_usd * 2 else "critical"

            self._create_alert(
                "high_cost",
                severity,
                f"LLM Kosten überschreiten Threshold: ${self.metrics.total_cost_usd:.2f}",
                {
                    "total_cost_usd": self.metrics.total_cost_usd,
                    "threshold_usd": self.alert_config.cost_threshold_usd,
                    "cost_per_model": self.metrics.cost_per_model
                }
            )

            self._last_cost_alert = now

    def _check_performance_alerts(self) -> None:
        """Prüft Performance-basierte Alerts."""
        if self.metrics.avg_response_time_ms > self.alert_config.response_time_threshold_ms:
            now = datetime.utcnow()

            # Throttling: Nur alle 5 Minuten Performance-Alerts
            if (now - self._last_performance_alert).total_seconds() < 300:
                return

            self._create_alert(
                "slow_response",
                "medium",
                f"LLM Response-Zeit überschreitet Threshold: {self.metrics.avg_response_time_ms:.0f}ms",
                {
                    "avg_response_time_ms": self.metrics.avg_response_time_ms,
                    "threshold_ms": self.alert_config.response_time_threshold_ms,
                    "max_response_time_ms": self.metrics.max_response_time_ms
                }
            )

            self._last_performance_alert = now

    def _check_error_rate_alerts(self) -> None:
        """Prüft Error-Rate-basierte Alerts."""
        if self.metrics.total_requests < 10:  # Mindestens 10 Requests für aussagekräftige Rate
            return

        error_rate = 1.0 - self.metrics.get_success_rate()

        if error_rate > self.alert_config.error_rate_threshold:
            self._create_alert(
                "high_error_rate",
                "high",
                f"LLM Error-Rate überschreitet Threshold: {error_rate:.1%}",
                {
                    "error_rate": error_rate,
                    "threshold": self.alert_config.error_rate_threshold,
                    "failed_requests": self.metrics.failed_requests,
                    "total_requests": self.metrics.total_requests
                }
            )

    def _check_rate_limit_alerts(self) -> None:
        """Prüft Rate-Limit-basierte Alerts."""
        if self.metrics.rate_limit_hits >= self.alert_config.rate_limit_alert_threshold:
            now = datetime.utcnow()

            # Throttling: Nur alle 10 Minuten Rate-Limit-Alerts
            if (now - self._last_rate_limit_alert).total_seconds() < 600:
                return

            self._create_alert(
                "frequent_rate_limits",
                "medium",
                f"Häufige Rate-Limit-Hits: {self.metrics.rate_limit_hits}",
                {
                    "rate_limit_hits": self.metrics.rate_limit_hits,
                    "threshold": self.alert_config.rate_limit_alert_threshold
                }
            )

            self._last_rate_limit_alert = now

    def _create_alert(self, alert_type: str, severity: str, message: str, metrics: dict[str, Any]) -> None:
        """Erstellt neuen Alert."""
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            metrics=metrics
        )

        self.alerts.append(alert)

        # Alert loggen
        logger.warning({
            "event": "llm_alert_created",
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "metrics": metrics
        })

        # Hier könnte Integration mit externen Alerting-Systemen erfolgen
        # z.B. Slack, PagerDuty, etc.

    def get_current_metrics(self) -> dict[str, Any]:
        """Gibt aktuelle Metriken zurück."""
        return {
            "requests": {
                "total": self.metrics.total_requests,
                "successful": self.metrics.successful_requests,
                "failed": self.metrics.failed_requests,
                "cached": self.metrics.cached_requests,
                "success_rate": self.metrics.get_success_rate(),
                "cache_hit_rate": self.metrics.get_cache_hit_rate()
            },
            "tokens": {
                "total": self.metrics.total_tokens_used,
                "prompt": self.metrics.prompt_tokens_used,
                "completion": self.metrics.completion_tokens_used
            },
            "cost": {
                "total_usd": self.metrics.total_cost_usd,
                "per_model": self.metrics.cost_per_model
            },
            "performance": {
                "avg_response_time_ms": self.metrics.avg_response_time_ms,
                "min_response_time_ms": self.metrics.min_response_time_ms,
                "max_response_time_ms": self.metrics.max_response_time_ms
            },
            "rate_limiting": {
                "rate_limit_hits": self.metrics.rate_limit_hits,
                "budget_exhaustions": self.metrics.budget_exhaustions
            },
            "window_start": self.metrics.window_start.isoformat()
        }

    def get_recent_alerts(self, hours: int = 24) -> list[dict[str, Any]]:
        """Gibt aktuelle Alerts zurück."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        recent_alerts = [
            {
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "metrics": alert.metrics,
                "timestamp": alert.timestamp.isoformat()
            }
            for alert in self.alerts
            if alert.timestamp >= cutoff
        ]

        return recent_alerts

    def reset_metrics(self) -> None:
        """Setzt Metriken zurück (z.B. für neue Zeitfenster)."""
        # Aktuelle Metriken in Historie speichern
        self.hourly_metrics.append(self.metrics)

        # Nur letzte 24 Stunden behalten
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.hourly_metrics = [
            m for m in self.hourly_metrics
            if m.window_start >= cutoff
        ]

        # Neue Metriken initialisieren
        self.metrics = LLMMetrics()

        logger.info({
            "event": "llm_metrics_reset",
            "hourly_metrics_count": len(self.hourly_metrics)
        })
