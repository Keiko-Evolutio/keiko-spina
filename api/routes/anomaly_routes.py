"""Anomaly Detection API.

Ermöglicht Konfiguration, Baseline-Lernen und ad-hoc Detection pro Tenant/Metrik.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.anomaly import AnomalyDetectionService

router = APIRouter(prefix="/api/v1/anomaly", tags=["anomaly"])


class TimeSeries(BaseModel):
    """Zeitreihendaten für Anomalieanalyse."""

    tenant: str = Field(..., description="Tenant-ID")
    metric: str = Field(..., description="Metrikname")
    values: list[float] = Field(..., description="Zeitreihenwerte (gleichmäßig gesampelt)")


_service = AnomalyDetectionService()


@router.post("/detect")
async def detect(ts: TimeSeries) -> dict[str, Any]:
    """Führt eine ad-hoc Anomalieerkennung aus."""
    return await _service.detect(tenant=ts.tenant, metric_name=ts.metric, values=ts.values)


@router.post("/baseline")
async def baseline(ts: TimeSeries) -> dict[str, Any]:
    """Lernt und persistiert Baseline-Statistiken."""
    ok = await _service.learn_baseline(tenant=ts.tenant, metric_name=ts.metric, values=ts.values)
    return {"success": ok}


@router.get("/threshold/{tenant}/{metric}")
async def threshold(tenant: str, metric: str) -> dict[str, Any]:
    """Liefert adaptiven Schwellwert (wenn vorhanden)."""
    bounds = await _service.adaptive_threshold(tenant=tenant, metric_name=metric)
    return {"bounds": bounds}
