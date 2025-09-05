"""Template-Management-API für n8n-Workflow-Templates."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from kei_logging import get_logger
from observability import trace_function
from services.n8n.workflow_sync_manager import workflow_sync_manager
from services.n8n.workflow_templates import workflow_template_registry

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/n8n/templates", tags=["n8n-templates"])


class TemplateInfo(BaseModel):
    """Metadaten eines Templates."""

    template_id: str
    display_name: str
    category: str
    description: str
    parameters: list[dict[str, Any]]


class RenderRequest(BaseModel):
    """Render-Request für Templates."""

    template_id: str = Field(..., description="Template Identifier")
    values: dict[str, Any] = Field(default_factory=dict, description="Parameterwerte")


class InstantiateRequest(BaseModel):
    """Erstellt und startet eine Workflow-Instanz aus einem Template."""

    template_id: str
    values: dict[str, Any] = Field(default_factory=dict)
    # Optionalen WebSocket connection_id für Live-Status mitgeben
    connection_id: str | None = Field(default=None)
    agent_id: str | None = Field(default=None)


@router.get("/", response_model=list[TemplateInfo])
@trace_function("api.n8n.templates.list")
async def list_templates() -> list[TemplateInfo]:
    """Listet alle verfügbaren Templates."""
    infos: list[TemplateInfo] = []
    for tpl in workflow_template_registry.list():
        infos.append(
            TemplateInfo(
                template_id=tpl.template_id,
                display_name=tpl.display_name,
                category=tpl.category,
                description=tpl.description,
                parameters=[
                    {
                        "name": p.name,
                        "type": p.type,
                        "required": p.required,
                        "description": p.description,
                        "default": p.default,
                    }
                    for p in tpl.parameters
                ],
            )
        )
    return infos


@router.post("/render")
@trace_function("api.n8n.templates.render")
async def render_template(req: RenderRequest) -> dict[str, Any]:
    """Rendert ein Template zu einem n8n-Workflow-Spezifikationsobjekt."""
    try:
        return workflow_template_registry.render(req.template_id, req.values)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:  # fehlender Parameter
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/instantiate")
@trace_function("api.n8n.templates.instantiate")
async def instantiate_template(req: InstantiateRequest) -> dict[str, Any]:
    """Erzeugt aus einem Template einen Workflow und startet ihn via SyncManager."""
    try:
        spec = workflow_template_registry.render(req.template_id, req.values)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Erwartung: Spezifikation enthält ein Feld "workflow_id" oder Name; hier vereinfacht
    workflow_id = spec.get("id") or spec.get("name") or req.template_id

    # Start über SyncManager – für REST-Trigger wird eine echte workflow_id benötigt
    result = await workflow_sync_manager.start_workflow(
        workflow_id=str(workflow_id),
        payload={"template": req.template_id, "values": req.values},
        connection_id=req.connection_id,
        agent_id=req.agent_id,
        trigger_mode="rest",
    )

    return {"template_id": req.template_id, **result}


__all__ = ["router"]
