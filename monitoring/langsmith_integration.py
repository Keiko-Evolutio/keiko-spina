"""LangSmith OpenTelemetry Bridge und Integrations-Hilfen.

Diese Integration stellt eine robuste Brücke zwischen OpenTelemetry (OTel)
und LangSmith bereit. Sie korreliert Trace-IDs, leitet AI-spezifische
Metriken weiter und bietet bequeme Kontextmanager für Agent-Ausführungen.
"""

from __future__ import annotations

import os
import random
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger(__name__)


# ==========================================================================
# OPTIONALE LANGSMITH SDK IMPORTS MIT ROBUSTEN FALLBACKS
# ==========================================================================

try:
    # LangSmith Python SDK (optional)
    from langsmith import Client as LangSmithClient  # type: ignore
    langsmith_trace = None  # type: ignore[assignment]
    LANGSMITH_AVAILABLE = True
except Exception:
    # Fallback-Symbole, damit Tests/Monkeypatch funktionieren
    LangSmithClient = None  # type: ignore[assignment]
    langsmith_trace = None  # type: ignore[assignment]
    LANGSMITH_AVAILABLE = False


try:
    # Zugriff auf aktuelle OTel Trace-Informationen
    from observability.tracing import get_current_trace_id
except Exception:
    def get_current_trace_id() -> str | None:  # type: ignore
        """Fallback, wenn OTel nicht verfügbar ist."""
        return None


# ==========================================================================
# KONFIGURATION
# ==========================================================================


@dataclass
class LangSmithConfig:
    """Konfiguration für LangSmith-Integration.

    Attributes:
        enabled: Aktiviert/deaktiviert die Integration global.
        api_key: API-Schlüssel für LangSmith.
        project_name: Projektname für Zuordnung in LangSmith.
        sampling_rate: Sampling-Rate (0.0–1.0) für Runs.
        enable_otel_bridge: Aktiviert Korrelation mit OTel Trace-IDs.
    """

    enabled: bool = True
    api_key: str | None = None
    project_name: str | None = None
    sampling_rate: float = 1.0
    enable_otel_bridge: bool = True

    @classmethod
    def from_environment(cls) -> LangSmithConfig:
        """Erzeugt Konfiguration aus Umgebungsvariablen."""
        return cls(
            enabled=os.getenv("LANGSMITH_ENABLED", "true").lower() == "true",
            api_key=os.getenv("LANGSMITH_API_KEY"),
            project_name=os.getenv("LANGSMITH_PROJECT_NAME") or os.getenv("LANGSMITH_PROJECT"),
            sampling_rate=float(os.getenv("LANGSMITH_SAMPLING_RATE", "1.0")),
            enable_otel_bridge=os.getenv("LANGSMITH_OTEL_BRIDGE", "true").lower() == "true",
        )


# ==========================================================================
# PII-REDAKTION
# ==========================================================================


_PII_PATTERNS = [
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # E-Mail
    re.compile(r"\b\+?\d{1,3}?[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b"),  # Telefon
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),  # Kreditkarte (rudimentär)
]


def redact_pii(text: str) -> str:
    """Maskiert bekannte PII-Muster in Texten.

    Args:
        text: Eingabetext.

    Returns:
        Bereinigter Text mit maskierter PII.
    """
    if not text:
        return text
    redacted = text
    for pattern in _PII_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


# ==========================================================================
# KERNINTEGRATION
# ==========================================================================


class LangSmithIntegration:
    """Zentrale LangSmith-Integration mit OTel-Korrelation.

    Stellt sichere No-Op-Fallbacks bereit, falls Abhängigkeiten fehlen oder
    die Integration deaktiviert ist.
    """

    def __init__(self, config: LangSmithConfig | None = None) -> None:
        self.config = config or LangSmithConfig.from_environment()
        self._is_active: bool = False
        self._client: Any | None = None

    def initialize(self) -> bool:
        """Initialisiert die LangSmith-Integration.

        Returns:
            True, wenn aktiv; False bei Deaktivierung oder Fehler.
        """
        if not self.config.enabled:
            logger.info("LangSmith deaktiviert (ENV)")
            return False

        if not LANGSMITH_AVAILABLE or LangSmithClient is None:
            logger.warning("LangSmith SDK nicht verfügbar – Integration im No-Op-Modus")
            return False

        if not self.config.api_key:
            logger.warning("LANGSMITH_API_KEY fehlt – Integration deaktiviert")
            return False

        try:
            self._client = LangSmithClient(api_key=self.config.api_key)
            self._is_active = True
            logger.info("LangSmith Integration initialisiert (Projekt: %s)", self.config.project_name or "default")
            return True
        except (ConnectionError, TimeoutError) as exc:
            logger.error("LangSmith Initialisierung fehlgeschlagen - Verbindungsproblem: %s", exc)
            self._is_active = False
            self._client = None
            return False
        except (ValueError, TypeError) as exc:
            logger.error("LangSmith Initialisierung fehlgeschlagen - Konfigurationsfehler: %s", exc)
            self._is_active = False
            self._client = None
            return False
        except Exception as exc:
            logger.exception("LangSmith Initialisierung fehlgeschlagen - Unerwarteter Fehler: %s", exc)
            self._is_active = False
            self._client = None
            return False

    @property
    def is_active(self) -> bool:
        """Gibt zurück, ob die Integration aktiv ist."""
        return self._is_active

    def _should_sample(self) -> bool:
        """Prüft Sampling-Entscheidung."""
        rate = max(0.0, min(1.0, self.config.sampling_rate))
        if rate >= 1.0:
            return True
        return random.random() < rate

    def _current_trace_context(self) -> dict[str, Any]:
        """Liest aktuelle OTel-Trace-Informationen aus."""
        if not self.config.enable_otel_bridge:
            return {}
        trace_id = get_current_trace_id()
        return {"otel.trace_id": trace_id} if trace_id else {}

    def record_token_usage(self, run_id: str | None, prompt_tokens: int, completion_tokens: int) -> None:
        """Übermittelt Token-Nutzung als Run-Metadaten.

        Args:
            run_id: Ziel-Run-ID (optional; ohne Effekt im No-Op-Modus).
            prompt_tokens: Anzahl Eingabetoken.
            completion_tokens: Anzahl Ausgabetoken.
        """
        if not self._is_active or not self._client or not run_id:
            return
        try:
            self._client.update_run(
                run_id=run_id,
                extra={
                    "metrics": {
                        "token.prompt": prompt_tokens,
                        "token.completion": completion_tokens,
                        "token.total": prompt_tokens + completion_tokens,
                    }
                },
            )
        except (ConnectionError, TimeoutError) as exc:
            logger.debug("LangSmith Token-Update ignoriert - Verbindungsproblem: %s", exc)
        except Exception as exc:
            logger.debug("LangSmith Token-Update ignoriert - Unerwarteter Fehler: %s", exc)

    @asynccontextmanager
    async def agent_run(self, name: str, inputs: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Asynchroner Kontext für Agent-Ausführungen mit Korrelation.

        Args:
            name: Logischer Name der Operation (z. B. "agent.execute").
            inputs: Eingabedaten; werden PII-bereinigt übertragen.

        Yields:
            Laufzeitdaten mit optionaler `run_id`.
        """
        start_time = time.time()
        run_info: dict[str, Any] = {"run_id": None}

        if not self._is_active or not self._client or not self._should_sample():
            yield run_info
            return

        try:
            metadata = {"component": "keiko-agent", **self._current_trace_context()}
            sanitized_inputs = {k: redact_pii(str(v)) for k, v in (inputs or {}).items()}

            run = self._client.create_run(
                name=name,
                inputs=sanitized_inputs,
                project_name=self.config.project_name,
                metadata=metadata,
            )
            run_info["run_id"] = getattr(run, "id", None)
            yield run_info

        except (ConnectionError, TimeoutError) as exc:
            logger.debug("LangSmith Run-Erstellung übersprungen - Verbindungsproblem: %s", exc)
            yield run_info
        except Exception as exc:
            logger.debug("LangSmith Run-Erstellung übersprungen - Unerwarteter Fehler: %s", exc)
            yield run_info
        finally:
            # Abschluss/Update mit Dauer
            if self._is_active and self._client and run_info.get("run_id"):
                try:
                    duration = time.time() - start_time
                    self._client.update_run(
                        run_id=run_info["run_id"],
                        extra={"metrics": {"latency.seconds": duration}},
                    )
                except Exception as exc:
                    logger.debug("LangSmith Run-Abschluss ignoriert: %s", exc)


# Globale bequeme Factory (Lazy)
_global_integration: LangSmithIntegration | None = None


def get_langsmith_integration() -> LangSmithIntegration:
    """Gibt globale Instanz der LangSmith-Integration zurück."""
    global _global_integration
    if _global_integration is None:
        _global_integration = LangSmithIntegration(LangSmithConfig.from_environment())
        _global_integration.initialize()
    return _global_integration


__all__ = [
    "LangSmithClient",
    "LangSmithConfig",
    "LangSmithIntegration",
    "get_langsmith_integration",
    "redact_pii",
]
