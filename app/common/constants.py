"""Konstanten für das App-Modul.

Zentrale Definition aller Magic Numbers, Strings und Konfigurationswerte
um Code-Duplikation zu vermeiden und Wartbarkeit zu verbessern.
"""

from __future__ import annotations

# Version Information
APP_VERSION = "2.0.0"
APP_TITLE = "Keiko Personal Assistant"
APP_DESCRIPTION = "Multi-Agent System mit Azure AI Foundry Integration"

# Network Configuration
DEFAULT_GRPC_PORT = 50051
DEFAULT_GRPC_BIND = "0.0.0.0:50051"
WEBSOCKET_CLOSE_CODE_SHUTDOWN = 1001
WEBSOCKET_CLOSE_REASON_SHUTDOWN = "Server Shutdown"

# Security Configuration
JWT_SECRET_LENGTH = 32
DEFAULT_CORS_ORIGINS: list[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
]

# Health Check Configuration
HEALTH_STATUS_HEALTHY = "healthy"
HEALTH_STATUS_DEGRADED = "degraded"
HEALTH_STATUS_WARNING = "warning"

# Service Names
SERVICE_KEI_BUS = "kei_bus"
SERVICE_AGENTS = "agents"
SERVICE_VOICE = "voice"
SERVICE_REDIS = "redis"
SERVICE_WEBHOOK = "webhook"
SERVICE_RATE_LIMITING = "rate_limiting"
SERVICE_DOMAIN_REVALIDATION = "domain_revalidation"
SERVICE_GRPC = "grpc"
SERVICE_WORK_STEALER = "work_stealer"

# Environment Variables
ENV_JWT_SECRET_KEY = "JWT_SECRET_KEY"
ENV_KEI_RPC_ENABLE_GRPC = "KEI_RPC_ENABLE_GRPC"
ENV_KEI_RPC_GRPC_BIND = "KEI_RPC_GRPC_BIND"

# Timeouts and Limits
STARTUP_TIMEOUT_SECONDS = 30
SHUTDOWN_TIMEOUT_SECONDS = 30
HEALTH_CHECK_TIMEOUT_SECONDS = 5

# WebSocket Event Types
WS_EVENT_AGENT_INPUT = "agent_input"
WS_EVENT_VOICE_INPUT = "voice_input"
WS_EVENT_SYSTEM_QUERY = "system_query"
WS_EVENT_PING = "ping"
WS_EVENT_PONG = "pong"
WS_EVENT_TASK_REQUEST = "task_request"
WS_EVENT_CAPABILITY_ADVERTISEMENT = "capability_advertisement"

# WebSocket Message Types
WS_MSG_ERROR = "error"
WS_MSG_AGENT_RESPONSE = "agent_response"
WS_MSG_VOICE_RESPONSE = "voice_response"
WS_MSG_SYSTEM_STATUS = "system_status"
WS_MSG_TASK_DELEGATION = "task_delegation"
WS_MSG_AGENT_REGISTERED = "agent_registered"

# Logging Configuration
LOG_FORMAT_TIMESTAMP = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"
LOG_LEVEL_CRITICAL = "CRITICAL"

# Error Messages
ERROR_SERVICE_UNAVAILABLE = "Service temporarily unavailable"
ERROR_INVALID_REQUEST = "Invalid request format"
ERROR_AUTHENTICATION_FAILED = "Authentication failed"
ERROR_RATE_LIMIT_EXCEEDED = "Rate limit exceeded"
ERROR_INTERNAL_SERVER_ERROR = "Internal server error"

# Success Messages
SUCCESS_SERVICE_STARTED = "Service started successfully"
SUCCESS_SERVICE_STOPPED = "Service stopped successfully"
SUCCESS_HEALTH_CHECK_PASSED = "Health check passed"

__all__ = [
    "APP_DESCRIPTION",
    "APP_TITLE",
    # Version
    "APP_VERSION",
    "DEFAULT_CORS_ORIGINS",
    "DEFAULT_GRPC_BIND",
    # Network
    "DEFAULT_GRPC_PORT",
    # Environment
    "ENV_JWT_SECRET_KEY",
    "ENV_KEI_RPC_ENABLE_GRPC",
    "ENV_KEI_RPC_GRPC_BIND",
    "ERROR_AUTHENTICATION_FAILED",
    "ERROR_INTERNAL_SERVER_ERROR",
    "ERROR_INVALID_REQUEST",
    "ERROR_RATE_LIMIT_EXCEEDED",
    # Messages
    "ERROR_SERVICE_UNAVAILABLE",
    "HEALTH_CHECK_TIMEOUT_SECONDS",
    "HEALTH_STATUS_DEGRADED",
    # Health
    "HEALTH_STATUS_HEALTHY",
    "HEALTH_STATUS_WARNING",
    # Security
    "JWT_SECRET_LENGTH",
    # Logging
    "LOG_FORMAT_TIMESTAMP",
    "LOG_LEVEL_CRITICAL",
    "LOG_LEVEL_DEBUG",
    "LOG_LEVEL_ERROR",
    "LOG_LEVEL_INFO",
    "LOG_LEVEL_WARNING",
    "SERVICE_AGENTS",
    "SERVICE_DOMAIN_REVALIDATION",
    "SERVICE_GRPC",
    # Services
    "SERVICE_KEI_BUS",
    "SERVICE_RATE_LIMITING",
    "SERVICE_REDIS",
    "SERVICE_VOICE",
    "SERVICE_WEBHOOK",
    "SERVICE_WORK_STEALER",
    "SHUTDOWN_TIMEOUT_SECONDS",
    # Timeouts
    "STARTUP_TIMEOUT_SECONDS",
    "SUCCESS_HEALTH_CHECK_PASSED",
    "SUCCESS_SERVICE_STARTED",
    "SUCCESS_SERVICE_STOPPED",
    "WEBSOCKET_CLOSE_CODE_SHUTDOWN",
    "WEBSOCKET_CLOSE_REASON_SHUTDOWN",
    # WebSocket Events
    "WS_EVENT_AGENT_INPUT",
    "WS_EVENT_CAPABILITY_ADVERTISEMENT",
    "WS_EVENT_PING",
    "WS_EVENT_PONG",
    "WS_EVENT_SYSTEM_QUERY",
    "WS_EVENT_TASK_REQUEST",
    "WS_EVENT_VOICE_INPUT",
    "WS_MSG_AGENT_REGISTERED",
    "WS_MSG_AGENT_RESPONSE",
    # WebSocket Messages
    "WS_MSG_ERROR",
    "WS_MSG_SYSTEM_STATUS",
    "WS_MSG_TASK_DELEGATION",
    "WS_MSG_VOICE_RESPONSE",
]
