"""Zentrale Prometheus-Metriken-Definitionen für das Monitoring-System.

Konsolidiert alle Prometheus-Metriken aus dem gesamten Monitoring-Modul
in einer einzigen, gut organisierten Datei. Eliminiert Code-Duplikation
und stellt konsistente Namenskonventionen sicher.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ============================================================================
# ANOMALIE-ERKENNUNG METRIKEN
# ============================================================================

ANOMALY_DETECTIONS_TOTAL = Counter(
    "anomaly_detections_total",
    "Gesamtzahl erkannter Anomalien",
    labelnames=("tenant", "metric", "severity"),
)

ANOMALY_SCORE_GAUGE = Gauge(
    "anomaly_last_score",
    "Letzter Anomalie-Score je Metrik/Tenant",
    labelnames=("tenant", "metric"),
)

# ============================================================================
# ATTACK/SECURITY METRIKEN
# ============================================================================

ATTACK_REQUESTS_SUSPECT = Counter(
    "attack_requests_suspect_total",
    "Anzahl verdächtiger Requests (Anomalie-Erkennung)",
    labelnames=("ip",),
)

ATTACK_IP_BLOCKED = Counter(
    "attack_ip_blocked_total",
    "Anzahl blockierter IPs (temporär)",
    labelnames=("ip",),
)

SECURITY_EVENTS_TOTAL = Counter(
    "security_events_total",
    "Gesamtzahl sicherheitsrelevanter Events",
    labelnames=("type",),
)

# ============================================================================
# CACHE METRIKEN
# ============================================================================

KEIKO_CACHE_OPERATIONS_TOTAL = Counter(
    "keiko_cache_operations_total",
    "Cache Operationen (hit/miss)",
    labelnames=("cache_type", "operation", "tenant_id"),
)

KEIKO_CACHE_HIT_RATIO = Gauge(
    "keiko_cache_hit_ratio",
    "Cache Hit Ratio (0..1)",
    labelnames=("cache_type", "tenant_id"),
)

KEIKO_REQUEST_DEDUP_TOTAL = Counter(
    "keiko_request_dedup_total",
    "Request Deduplication Ereignisse",
    labelnames=("tenant_id", "dedup_type"),
)

KEIKO_CACHE_SIZE_BYTES = Gauge(
    "keiko_cache_size_bytes",
    "Cache Größe in Bytes",
    labelnames=("cache_type", "tenant_id"),
)

KEIKO_CACHE_EVICTION_TOTAL = Counter(
    "keiko_cache_eviction_total",
    "Cache Eviction Events",
    labelnames=("cache_type", "tenant_id", "reason"),
)

KEIKO_CACHE_OP_LATENCY = Histogram(
    "keiko_cache_operation_latency_seconds",
    "Cache Operation Latenz",
    labelnames=("cache_type", "operation", "tenant_id"),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

KEIKO_CACHE_MEMORY_USAGE = Gauge(
    "keiko_cache_memory_usage_bytes",
    "Cache Memory Usage in Bytes",
    labelnames=("cache_type", "tenant_id"),
)

# ============================================================================
# HTTP RESPONSE CACHE METRIKEN
# ============================================================================

KEIKO_HTTP_CACHE_304_TOTAL = Counter(
    "keiko_http_cache_304_total",
    "HTTP 304 (Not Modified) Antworten aus Response Cache",
    labelnames=("endpoint", "type"),
)

KEIKO_HTTP_CACHE_ETAG_GENERATED_TOTAL = Counter(
    "keiko_http_cache_etag_generated_total",
    "ETag Generierung pro Endpoint",
    labelnames=("endpoint",),
)

KEIKO_HTTP_CACHE_TTL_RULE_TOTAL = Counter(
    "keiko_http_cache_ttl_rule_total",
    "TTL-Regel Anwendung pro Endpoint und Pattern",
    labelnames=("endpoint", "pattern", "ttl"),
)

# ============================================================================
# SLA/HEALTH METRIKEN
# ============================================================================

SLA_AVAILABILITY = Gauge(
    "keiko_sla_availability_percentage",
    "SLA Verfügbarkeit in Prozent",
    labelnames=("service", "component", "environment"),
)

SLA_P95_LATENCY = Gauge(
    "keiko_sla_p95_latency_ms",
    "SLA p95 Latenz in Millisekunden",
    labelnames=("service", "component", "environment"),
)

SLA_ERROR_RATE = Gauge(
    "keiko_sla_error_rate_percentage",
    "SLA Fehlerrate in Prozent",
    labelnames=("service", "component", "environment"),
)

SLA_COMPLIANCE = Gauge(
    "keiko_sla_compliance_status",
    "SLA Compliance Status (1=konform, 0=abweichend)",
    labelnames=("service", "component", "environment"),
)

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def record_cache_hit(cache_type: str, tenant_id: str = "default") -> None:
    """Zeichnet Cache-Hit auf."""
    KEIKO_CACHE_OPERATIONS_TOTAL.labels(
        cache_type=cache_type,
        operation="hit",
        tenant_id=tenant_id
    ).inc()

def record_cache_miss(cache_type: str, tenant_id: str = "default") -> None:
    """Zeichnet Cache-Miss auf."""
    KEIKO_CACHE_OPERATIONS_TOTAL.labels(
        cache_type=cache_type,
        operation="miss",
        tenant_id=tenant_id
    ).inc()

def observe_cache_size(cache_type: str, size_bytes: float, tenant_id: str = "default") -> None:
    """Aktualisiert Cache-Größe."""
    KEIKO_CACHE_SIZE_BYTES.labels(
        cache_type=cache_type,
        tenant_id=tenant_id
    ).set(size_bytes)

def record_eviction(cache_type: str, reason: str, tenant_id: str = "default") -> None:
    """Zeichnet Cache-Eviction auf."""
    KEIKO_CACHE_EVICTION_TOTAL.labels(
        cache_type=cache_type,
        tenant_id=tenant_id,
        reason=reason
    ).inc()

def record_dedup(tenant_id: str, dedup_type: str = "request") -> None:
    """Zeichnet Request-Deduplication auf."""
    KEIKO_REQUEST_DEDUP_TOTAL.labels(
        tenant_id=tenant_id,
        dedup_type=dedup_type
    ).inc()

def record_http_cache_304(endpoint: str, cache_type: str = "etag") -> None:
    """Zeichnet HTTP 304 Response auf."""
    KEIKO_HTTP_CACHE_304_TOTAL.labels(
        endpoint=endpoint,
        type=cache_type
    ).inc()

def record_etag_generation(endpoint: str) -> None:
    """Zeichnet ETag-Generierung auf."""
    KEIKO_HTTP_CACHE_ETAG_GENERATED_TOTAL.labels(endpoint=endpoint).inc()

def record_ttl_rule_application(endpoint: str, pattern: str, ttl: str) -> None:
    """Zeichnet TTL-Regel-Anwendung auf."""
    KEIKO_HTTP_CACHE_TTL_RULE_TOTAL.labels(
        endpoint=endpoint,
        pattern=pattern,
        ttl=ttl
    ).inc()

def record_anomaly_detection(tenant: str, metric: str, severity: str) -> None:
    """Zeichnet Anomalie-Erkennung auf."""
    ANOMALY_DETECTIONS_TOTAL.labels(
        tenant=tenant,
        metric=metric,
        severity=severity
    ).inc()

def update_anomaly_score(tenant: str, metric: str, score: float) -> None:
    """Aktualisiert Anomalie-Score."""
    ANOMALY_SCORE_GAUGE.labels(tenant=tenant, metric=metric).set(score)

def record_suspect_request(ip: str) -> None:
    """Zeichnet verdächtigen Request auf."""
    ATTACK_REQUESTS_SUSPECT.labels(ip=ip).inc()

def record_ip_blocked(ip: str) -> None:
    """Zeichnet IP-Blockierung auf."""
    ATTACK_IP_BLOCKED.labels(ip=ip).inc()

def record_security_event(event_type: str) -> None:
    """Zeichnet Security-Event auf."""
    SECURITY_EVENTS_TOTAL.labels(type=event_type).inc()

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Anomalie-Metriken
    "ANOMALY_DETECTIONS_TOTAL",
    "ANOMALY_SCORE_GAUGE",
    "ATTACK_IP_BLOCKED",
    # Attack/Security-Metriken
    "ATTACK_REQUESTS_SUSPECT",
    "KEIKO_CACHE_EVICTION_TOTAL",
    "KEIKO_CACHE_HIT_RATIO",
    "KEIKO_CACHE_MEMORY_USAGE",
    # Cache-Metriken
    "KEIKO_CACHE_OPERATIONS_TOTAL",
    "KEIKO_CACHE_OP_LATENCY",
    "KEIKO_CACHE_SIZE_BYTES",
    # HTTP Cache-Metriken
    "KEIKO_HTTP_CACHE_304_TOTAL",
    "KEIKO_HTTP_CACHE_ETAG_GENERATED_TOTAL",
    "KEIKO_HTTP_CACHE_TTL_RULE_TOTAL",
    "KEIKO_REQUEST_DEDUP_TOTAL",
    "SECURITY_EVENTS_TOTAL",
    # SLA/Health-Metriken
    "SLA_AVAILABILITY",
    "SLA_COMPLIANCE",
    "SLA_ERROR_RATE",
    "SLA_P95_LATENCY",
    "observe_cache_size",
    "record_anomaly_detection",
    "record_cache_hit",
    "record_cache_miss",
    "record_dedup",
    "record_etag_generation",
    "record_eviction",
    "record_http_cache_304",
    "record_ip_blocked",
    "record_security_event",
    "record_suspect_request",
    "record_ttl_rule_application",
    "update_anomaly_score",
]
