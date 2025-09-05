"""Zentrale Konstanten für KEI-Bus.

Konsolidiert alle Magic Numbers und Hard-coded Strings an einem Ort.
"""

from __future__ import annotations

# Cache-Keys und Prefixes
CHAOS_PROFILE_KEY = "bus:chaos:profile"
CHAOS_DELAY_MS_KEY = "bus:chaos:delay_ms"
REPLAY_LIST_KEY_TEMPLATE = "bus:replay:{subject}"
IDEMPOTENCY_KEY_PREFIX = "bus:idem"
OUTBOX_KEY_PREFIX = "bus:outbox"
INBOX_KEY_PREFIX = "bus:inbox"

# Chaos-Engineering-Konstanten
MILLISECONDS_TO_SECONDS = 1000.0
DROP_PROBABILITY_THRESHOLD = 0.5
MIXED_DROP_PROBABILITY_THRESHOLD = 0.7
DEFAULT_CHAOS_DELAY_MS = 100

# Topic-Management-Konstanten
DEFAULT_MAX_DELIVERY = 5
DEFAULT_RETENTION_POLICY = "limits"
DEFAULT_REPLAY_STORE_LIMIT = 10

# TTL-Konstanten (in Sekunden)
DEFAULT_IDEMPOTENCY_TTL_SECONDS = 3600  # 1 Stunde
DEFAULT_INBOX_TTL_SECONDS = 3600  # 1 Stunde
DEFAULT_CHAOS_NONCE_TTL_SECONDS = 300  # 5 Minuten

# RPC-Konstanten
RPC_REQUEST_TYPE = "rpc_request"
RPC_REPLY_SUBJECT_PREFIX = "kei.rpc.reply"
RPC_TIMEOUT_ERROR_CODE = "RPC_TIMEOUT"

# DLQ-Konstanten
DLQ_STREAM_NAME = "DLQ"
DLQ_SUBJECT_PREFIX = "kei.dlq.>"
PARKING_SUBJECT_PREFIX = "kei.parking"

# Provider-Konstanten
DEFAULT_OUTBOX_NAME = "default"
PUBLISH_OPERATION = "publish"
CONSUME_OPERATION = "consume"
KAFKA_PUBLISH_OPERATION = "kafka_publish"

# Subject-Pattern für Namenskonventionen
EVENT_SUBJECT_PATTERN = r"^kei\.[a-z0-9-]+\.[a-z0-9-]+\.[a-z0-9-]+\.[a-z0-9-]+\.v\d+$"
RPC_SUBJECT_PATTERN = r"^kei\.rpc\.[a-z0-9-]+\.[a-z0-9-]+\.v\d+$"

# Error-Messages
ERROR_INVALID_EVENT_SUBJECT = "Ungültiges Event-Subject-Format"
ERROR_INVALID_RPC_SUBJECT = "Ungültiges RPC-Subject-Format"
ERROR_JWT_CLAIMS_MISSING = "JWT Claims fehlen in Message-Headers"
ERROR_BUS_BASE_SCOPE_MISSING = "Fehlender Bus-Basis-Scope"
ERROR_TENANT_SCOPE_MISSING = "Fehlender Tenant-Scope"
ERROR_SUBJECT_SCOPES_MISSING = "Fehlende Scopes für Subject"

# Standard-Werte
DEFAULT_VERSION = 1
DEFAULT_QUEUE_NAME = "default"
DEFAULT_DURABLE_NAME = "default"

# Metrics-Namen
METRIC_PUBLISH_COUNT = "bus.messages.published"
METRIC_CONSUME_COUNT = "bus.messages.consumed"
METRIC_REDELIVERIES_COUNT = "bus.messages.redeliveries"
METRIC_DLQ_COUNT = "bus.messages.dlq"
METRIC_LATENCY_MS = "bus.end_to_end.latency_ms"
METRIC_ERRORS_COUNT = "bus.messages.errors"
METRIC_TIMEOUTS_COUNT = "bus.pull.timeouts"

# Limits und Thresholds
MAX_LATENCY_SAMPLES = 1000
DEFAULT_MAX_ITEMS = 100
DEFAULT_CORRELATION_DEPTH = 10

# Stream-Namen für NATS
NATS_AGENTS_STREAM = "AGENTS"
NATS_TASKS_STREAM = "TASKS"
NATS_EVENTS_STREAM = "EVENTS"
NATS_A2A_STREAM = "A2A"

# Subject-Patterns für NATS-Streams
NATS_AGENTS_SUBJECTS = ["kei.agents.>", "kei.agents.>.key.>"]
NATS_TASKS_SUBJECTS = ["kei.tasks.>", "kei.tasks.>.key.>"]
NATS_EVENTS_SUBJECTS = ["kei.events.>", "kei.events.>.key.>"]
NATS_A2A_SUBJECTS = ["kei.a2a.>", "kei.a2a.>.key.>"]

# Kafka-Konstanten
KAFKA_ACKS_ALL = "all"
KAFKA_AUTO_OFFSET_RESET_EARLIEST = "earliest"

# Privacy/Security-Konstanten
AUTHORIZATION_HEADER = "authorization"
BEARER_PREFIX = "bearer "
TRACEPARENT_HEADER = "traceparent"

# Envelope-Felder
ENVELOPE_FIELD_ID = "id"
ENVELOPE_FIELD_TYPE = "type"
ENVELOPE_FIELD_SUBJECT = "subject"
ENVELOPE_FIELD_PAYLOAD = "payload"
ENVELOPE_FIELD_HEADERS = "headers"
ENVELOPE_FIELD_TENANT = "tenant"
ENVELOPE_FIELD_CORRELATION_ID = "corr_id"
ENVELOPE_FIELD_CAUSATION_ID = "causation_id"
ENVELOPE_FIELD_MESSAGE_TYPE = "type"
ENVELOPE_FIELD_MESSAGE_ID = "id"

# Logging-Konstanten
LOG_FIELD_CORRELATION_ID = "correlation_id"
LOG_FIELD_CAUSATION_ID = "causation_id"
LOG_FIELD_TENANT = "tenant"
LOG_FIELD_SUBJECT = "subject"
LOG_FIELD_MESSAGE_TYPE = "type"
LOG_FIELD_MESSAGE_ID = "message_id"


__all__ = [
    # Privacy/Security
    "AUTHORIZATION_HEADER",
    "BEARER_PREFIX",
    "CHAOS_DELAY_MS_KEY",
    # Cache-Keys
    "CHAOS_PROFILE_KEY",
    "CONSUME_OPERATION",
    "DEFAULT_CHAOS_DELAY_MS",
    "DEFAULT_CHAOS_NONCE_TTL_SECONDS",
    "DEFAULT_CORRELATION_DEPTH",
    "DEFAULT_DURABLE_NAME",
    # TTL-Konstanten
    "DEFAULT_IDEMPOTENCY_TTL_SECONDS",
    "DEFAULT_INBOX_TTL_SECONDS",
    # Topic-Management
    "DEFAULT_MAX_DELIVERY",
    "DEFAULT_MAX_ITEMS",
    # Provider
    "DEFAULT_OUTBOX_NAME",
    "DEFAULT_QUEUE_NAME",
    "DEFAULT_REPLAY_STORE_LIMIT",
    "DEFAULT_RETENTION_POLICY",
    # Standard-Werte
    "DEFAULT_VERSION",
    # DLQ
    "DLQ_STREAM_NAME",
    "DLQ_SUBJECT_PREFIX",
    "DROP_PROBABILITY_THRESHOLD",
    "ENVELOPE_FIELD_CAUSATION_ID",
    "ENVELOPE_FIELD_CORRELATION_ID",
    "ENVELOPE_FIELD_HEADERS",
    # Envelope
    "ENVELOPE_FIELD_ID",
    "ENVELOPE_FIELD_PAYLOAD",
    "ENVELOPE_FIELD_SUBJECT",
    "ENVELOPE_FIELD_TENANT",
    "ENVELOPE_FIELD_TYPE",
    "ERROR_BUS_BASE_SCOPE_MISSING",
    # Error-Messages
    "ERROR_INVALID_EVENT_SUBJECT",
    "ERROR_INVALID_RPC_SUBJECT",
    "ERROR_JWT_CLAIMS_MISSING",
    "ERROR_SUBJECT_SCOPES_MISSING",
    "ERROR_TENANT_SCOPE_MISSING",
    # Subject-Pattern
    "EVENT_SUBJECT_PATTERN",
    "IDEMPOTENCY_KEY_PREFIX",
    "INBOX_KEY_PREFIX",
    # Kafka
    "KAFKA_ACKS_ALL",
    "KAFKA_AUTO_OFFSET_RESET_EARLIEST",
    "KAFKA_PUBLISH_OPERATION",
    "LOG_FIELD_CAUSATION_ID",
    # Logging
    "LOG_FIELD_CORRELATION_ID",
    "LOG_FIELD_MESSAGE_ID",
    "LOG_FIELD_MESSAGE_TYPE",
    "LOG_FIELD_SUBJECT",
    "LOG_FIELD_TENANT",
    # Limits
    "MAX_LATENCY_SAMPLES",
    "METRIC_CONSUME_COUNT",
    "METRIC_DLQ_COUNT",
    "METRIC_ERRORS_COUNT",
    "METRIC_LATENCY_MS",
    # Metrics
    "METRIC_PUBLISH_COUNT",
    "METRIC_REDELIVERIES_COUNT",
    "METRIC_TIMEOUTS_COUNT",
    # Chaos-Engineering
    "MILLISECONDS_TO_SECONDS",
    "MIXED_DROP_PROBABILITY_THRESHOLD",
    "NATS_A2A_STREAM",
    "NATS_A2A_SUBJECTS",
    # NATS
    "NATS_AGENTS_STREAM",
    "NATS_AGENTS_SUBJECTS",
    "NATS_EVENTS_STREAM",
    "NATS_EVENTS_SUBJECTS",
    "NATS_TASKS_STREAM",
    "NATS_TASKS_SUBJECTS",
    "OUTBOX_KEY_PREFIX",
    "PARKING_SUBJECT_PREFIX",
    "PUBLISH_OPERATION",
    "REPLAY_LIST_KEY_TEMPLATE",
    "RPC_REPLY_SUBJECT_PREFIX",
    # RPC
    "RPC_REQUEST_TYPE",
    "RPC_SUBJECT_PATTERN",
    "RPC_TIMEOUT_ERROR_CODE",
    "TRACEPARENT_HEADER",
]
