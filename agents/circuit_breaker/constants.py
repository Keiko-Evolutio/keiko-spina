"""Konstanten für Agent Circuit Breaker.

⚠️  DEPRECATED: Diese Datei ist veraltet!
Verwende stattdessen: from agents.constants import *

Alle Konstanten wurden in die zentrale constants.py konsolidiert.
"""

from __future__ import annotations

import warnings

# Diese Datei ist deprecated - alle Konstanten wurden in die zentrale constants.py konsolidiert
# Der Import-Block wurde entfernt, da die Konstanten nicht verwendet werden

# Deprecation Warning
warnings.warn(
    "backend.agents.circuit_breaker.constants ist deprecated. "
    "Verwende 'from agents.constants import *'",
    DeprecationWarning,
    stacklevel=2
)

# Circuit Breaker-spezifische Konstanten die noch nicht konsolidiert sind
# Cache Konfiguration
DEFAULT_CACHE_MAX_SIZE: int = 1000
DEFAULT_CACHE_TTL_SECONDS: int = 300
DEFAULT_CACHE_CLEANUP_INTERVAL_SECONDS: int = 60

# Circuit Breaker-spezifische Error Messages
ERROR_CIRCUIT_BREAKER_OPEN: str = "Circuit breaker {name} is {state}"
ERROR_SERVICE_NOT_INITIALIZED: str = "Service not initialized"
ERROR_INVALID_STATE: str = "Invalid state: {state}. Use 'open', 'closed', or 'reset'"

# Circuit Breaker-spezifische Logging Messages
LOG_CIRCUIT_BREAKER_OPENED: str = "Circuit breaker {name} opened (failures: {failures}, next attempt: {next_attempt})"
LOG_CIRCUIT_BREAKER_CLOSED: str = "Circuit breaker {name} closed (successes: {successes})"
LOG_CIRCUIT_BREAKER_HALF_OPEN: str = "Circuit breaker {name} transitioned to half-open"
LOG_FAILURE_RECORDED: str = "Circuit breaker {name}: Failure recorded (type: {type}, consecutive: {consecutive}, error: {error})"
LOG_SUCCESS_RECORDED: str = "Circuit breaker {name}: Success recorded (time: {time}ms, consecutive: {consecutive})"
