"""A2A API – Senden strukturierter Agent-zu-Agent Nachrichten über KEI-Bus."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from data_models.a2a import A2AAttachment, A2AMessage, A2ARole, A2AToolCall
from kei_logging import get_logger
from security.kei_mcp_auth import require_auth, require_rate_limit
from services.a2a_service import A2AService

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/a2a", tags=["a2a"])


class ToolCallModel(BaseModel):
    """Pydantic-Repräsentation eines Tool-Calls."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None


class AttachmentModel(BaseModel):
    """Pydantic-Repräsentation eines Attachments."""

    id: str
    type: str
    uri: str
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SendA2ARequest(BaseModel):
    """Request für den Versand einer A2A-Nachricht."""

    from_agent_id: str
    to_agent_id: str
    protocol_version: str = Field(default="v1")
    role: A2ARole = Field(default=A2ARole.ASSISTANT)
    content: str
    tool_calls: list[ToolCallModel] = Field(default_factory=list)
    attachments: list[AttachmentModel] = Field(default_factory=list)
    corr_id: str | None = None
    causation_id: str | None = None
    traceparent: str | None = None
    headers: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tenant: str | None = None


class SendA2AResponse(BaseModel):
    """Antwort mit Message-ID nach Versand."""

    message_id: str


@router.post(
    "/send",
    response_model=SendA2AResponse,
    dependencies=[Depends(require_auth), Depends(lambda req: require_rate_limit(req, operation="default"))],
)
async def send_a2a(request: SendA2ARequest) -> SendA2AResponse:
    """Sendet eine A2A-Nachricht über den Bus."""
    svc = A2AService()
    msg = A2AMessage(
        protocol_version=request.protocol_version,
        role=request.role,
        content=request.content,
        tool_calls=[A2AToolCall(name=t.name, arguments=t.arguments, call_id=t.call_id) for t in request.tool_calls],
        attachments=[
            A2AAttachment(id=a.id, type=a.type, uri=a.uri, description=a.description, metadata=a.metadata)
            for a in request.attachments
        ],
        corr_id=request.corr_id,
        causation_id=request.causation_id,
        traceparent=request.traceparent,
        headers=request.headers,
        metadata=request.metadata,
    )
    # Schema-Validierung (HTTP 422 bei Fehler)
    try:
        msg.validate()
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    message_id = await svc.send_message(
        from_agent_id=request.from_agent_id,
        to_agent_id=request.to_agent_id,
        message=msg,
        tenant=request.tenant,
    )
    return SendA2AResponse(message_id=message_id)


__all__ = ["router"]
