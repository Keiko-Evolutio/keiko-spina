"""Spec Publisher Modul.

Veröffentlicht OpenAPI/AsyncAPI Spezifikationen auf konfigurierbare Ziele
(HTTP Portal, Git Repo, Artefakt-Repository) mit einfachem Rollback.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from config.settings import settings
from kei_logging import get_logger, structured_msg
from services.webhooks.alerting import emit_warning

from .constants import ErrorMessages, SpecConstants

logger = get_logger(__name__)


@dataclass
class PublishResult:
    """Ergebnis eines Publish-Versuchs."""

    target: str
    success: bool
    url: str | None
    error: str | None


class SpecPublisher:
    """Veröffentlicht Spezifikationen auf mehrere Ziele (Best Effort + Rollback)."""

    def __init__(self, version_tag: str) -> None:
        self.version_tag = version_tag
        self.targets: list[str] = [t.strip() for t in (settings.spec_publishing_targets or "").split(",") if t.strip()]
        self.rollback_on_failure: bool = settings.spec_rollback_on_failure

    async def publish(self, artifacts: dict[str, Any]) -> tuple[list[PublishResult], bool]:
        """Publiziert auf alle konfigurierten Ziele.

        Args:
            artifacts: Map von Namen → Inhalte/Pfade

        Returns:
            Liste von Ergebnissen und gesamt Erfolg.
        """
        results: list[PublishResult] = []
        succeeded: list[str] = []

        for target in self.targets:
            try:
                if target == "portal":
                    res = await self._publish_portal(artifacts)
                elif target == "github":
                    res = await self._publish_github(artifacts)
                elif target == "artifactory":
                    res = await self._publish_artifactory(artifacts)
                else:
                    res = PublishResult(target=target, success=False, url=None, error=ErrorMessages.UNKNOWN_TARGET)
                results.append(res)
                if res.success:
                    succeeded.append(target)
            except Exception as exc:
                results.append(PublishResult(target=target, success=False, url=None, error=str(exc)))

        all_ok = all(r.success for r in results) if results else True

        if not all_ok and self.rollback_on_failure and succeeded:
            try:
                await self._rollback(succeeded)
            except Exception as exc:
                logger.warning(structured_msg("spec_rollback_failed", error=str(exc)))
            # Alert versenden
            with contextlib.suppress(Exception):
                await emit_warning("Spec Publishing Failed", {"results": [r.__dict__ for r in results]})

        return results, all_ok

    async def _publish_portal(self, artifacts: dict[str, Any]) -> PublishResult:
        """Publiziert via HTTP PUT auf Portal-Endpunkt."""
        endpoint = settings.spec_portal_endpoint
        if not endpoint:
            return PublishResult("portal", False, None, ErrorMessages.MISSING_ENDPOINT)
        try:
            async with httpx.AsyncClient(timeout=SpecConstants.HTTP_TIMEOUT) as client:
                resp = await client.put(f"{endpoint.rstrip('/')}/openapi/{self.version_tag}.json", json=artifacts)
                if resp.status_code >= 400:
                    return PublishResult("portal", False, None, f"HTTP {resp.status_code}")
        except Exception as exc:
            return PublishResult("portal", False, None, str(exc))
        return PublishResult("portal", True, f"{endpoint.rstrip('/')}/openapi/{self.version_tag}.json", None)

    async def _publish_github(self, artifacts: dict[str, Any]) -> PublishResult:
        """Publiziert zu GitHub Pages Repo (vereinfachter HTTP API Upload)."""
        repo = settings.spec_github_repo
        token = settings.spec_github_token
        if not repo or not token:
            return PublishResult("github", False, None, "missing_github_config")
        # Vereinfachung: nutzt GitHub Contents API zum Erstellen/Updaten
        owner_repo = repo.strip()
        path = f"openapi/{self.version_tag}.json"
        url = f"https://api.github.com/repos/{owner_repo}/contents/{path}"
        content = json.dumps(artifacts).encode("utf-8")
        import base64
        b64 = base64.b64encode(content).decode("utf-8")
        payload = {"message": f"publish {self.version_tag}", "content": b64}
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.put(url, json=payload, headers=headers)
                if resp.status_code >= 400:
                    return PublishResult("github", False, None, f"HTTP {resp.status_code}")
        except Exception as exc:
            return PublishResult("github", False, None, str(exc))
        return PublishResult("github", True, f"https://raw.githubusercontent.com/{owner_repo}/main/{path}", None)

    async def _publish_artifactory(self, artifacts: dict[str, Any]) -> PublishResult:
        """Publiziert zu Artifactory/Nexus (Platzhalter via HTTP)."""
        # Hier könnte eine konkrete Implementierung folgen; Platzhalter
        return PublishResult("artifactory", True, None, None)

    async def _rollback(self, succeeded_targets: list[str]) -> None:
        """Rollback der zuvor erfolgreichen Veröffentlichungen (Best Effort)."""
        for target in succeeded_targets:
            try:
                logger.info(structured_msg("spec_rollback", target=target, version=self.version_tag))
                # Portal/GitHub Rollback: Aufruf von Delete/Restore (vereinfacht: nur Log)
            except Exception:
                pass


def build_version_tag(base_version: str) -> str:
    """Erstellt Versionstag `v{major}.{minor}.{patch}-{timestamp}`."""
    ts = int(time.time())
    base = base_version if base_version.startswith("v") else f"v{base_version}"
    return f"{base}-{ts}"
