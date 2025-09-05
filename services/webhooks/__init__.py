"""Webhooks Service Package.

Vollständig migriert nach services/webhooks/ als Teil der kei_*-Module-Konsolidierung.

Verwendet Lazy‑Loader für schwergewichtige Symbole, um Import‑Nebenwirkungen
zu vermeiden (z. B. Prometheus‑Registrierung während Unit‑Tests).

Hauptkomponenten:
- Webhook-Manager und Delivery-Pipeline
- Security-Komponenten (Verification, Secret-Management)
- Circuit-Breaker und Health-Prober
- Schema-Registry für Webhook-Validierung
- Prometheus-Metrics und Alerting
- Delivery-Worker und Background-Processing
- Template-System für Webhook-Payloads
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Explizite Imports für PyCharm/IDE-Support (nur zur Typprüfung)
    from .manager import WebhookManager, get_webhook_manager, set_webhook_manager

__all__ = ["WebhookManager", "get_webhook_manager", "set_webhook_manager"]


def __getattr__(name: str) -> Any:
    """Lädt Symbole aus `manager` erst bei Zugriff nach.

    Unterstützte Namen: get_webhook_manager, WebhookManager, set_webhook_manager
    """
    if name in ("get_webhook_manager", "WebhookManager", "set_webhook_manager"):
        # Relative Auflösung erlaubt beide Paketpfade:
        # - "services.webhook" (wenn 'backend' bereits im sys.path liegt)
        # - "backend.services.webhook" (bei absolutem Import)
        module = importlib.import_module(".manager", package=__name__)
        return getattr(module, name)
    raise AttributeError(name)
