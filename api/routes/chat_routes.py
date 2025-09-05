# backend/api/routes/chat_routes.py
"""Chat Management API Routes für Azure AI Foundry."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Path, Query
from pydantic import BaseModel, Field

from kei_logging import get_logger

from .base import (
    check_agents_integration,
    create_health_response,
    create_router,
    get_agent_system_status,
)
from .common import CHAT_RESPONSES

logger = get_logger(__name__)

# Router-Konfiguration
router = create_router("/chat", ["chat"])
router.responses.update(CHAT_RESPONSES)


# Datenmodelle
class ChatMessage(BaseModel):
    """Einzelne Chat-Nachricht."""
    id: str = Field(default_factory=lambda: f"msg_{uuid4().hex[:8]}")
    role: str = Field(..., description="Rolle: user, assistant, system")
    content: str = Field(..., description="Nachrichteninhalt")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] | None = Field(None, description="Zusätzliche Metadaten")


class ChatSessionRequest(BaseModel):
    """Request für neue Chat-Sitzung."""
    user_id: str = Field(..., description="Benutzer-ID")
    session_name: str | None = Field(None, description="Sitzungsname")
    agent_id: str | None = Field(None, description="Spezifische Agent-ID")
    system_prompt: str | None = Field(None, description="System-Prompt")
    context: dict[str, Any] | None = Field(None, description="Sitzungskontext")


class ChatMessageRequest(BaseModel):
    """Request für neue Chat-Nachricht."""
    content: str = Field(..., description="Nachrichteninhalt")
    context: dict[str, Any] | None = Field(None, description="Nachrichtenkontext")
    stream: bool = Field(False, description="Streaming-Response aktiviert")


class ChatSession(BaseModel):
    """Chat-Sitzung Modell."""
    id: str = Field(default_factory=lambda: f"chat_{uuid4().hex[:8]}")
    user_id: str = Field(..., description="Benutzer-ID")
    session_name: str = Field(..., description="Sitzungsname")
    agent_id: str | None = Field(None, description="Zugewiesene Agent-ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    messages: list[ChatMessage] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    status: str = Field("active", description="Sitzungsstatus")


class ChatResponse(BaseModel):
    """Chat-Response Modell."""
    session_id: str
    message: ChatMessage
    agent_info: dict[str, Any] | None = None
    processing_time_ms: int | None = None
    token_usage: dict[str, int] | None = None


# In-Memory Storage
chat_sessions: dict[str, ChatSession] = {}
user_sessions: dict[str, list[str]] = {}


# Helper Funktionen
def get_user_sessions(user_id: str) -> list[str]:
    """Holt oder erstellt Benutzer-Sitzungen."""
    if user_id not in user_sessions:
        user_sessions[user_id] = []
    return user_sessions[user_id]


async def process_chat_message(
    session: ChatSession,
    message_content: str,
    context: dict[str, Any] | None = None
) -> ChatMessage:
    """Verarbeitet Chat-Nachricht mit Agent System."""
    # Fallback für deaktiviertes Agent System
    if not check_agents_integration():
        return ChatMessage(
            role="assistant",
            content=f"Echo: {message_content} (Agent System nicht verfügbar)",
            metadata={"fallback": True}
        )

    try:
        from agents import execute_agent_task, find_best_agent_for_task

        # Agent bestimmen
        agent_id = session.agent_id
        if not agent_id:
            agent_id, _ = await find_best_agent_for_task(message_content)

        # Agent-Task ausführen
        result = await execute_agent_task(
            agent_id=agent_id,
            task=message_content,
            context=context or {},
            timeout=30
        )

        return ChatMessage(
            role="assistant",
            content=str(result),
            metadata={"agent_id": agent_id, "context": context}
        )

    except Exception as e:
        logger.exception(f"❌ Chat-Nachricht Verarbeitung fehlgeschlagen: {e}")
        return ChatMessage(
            role="assistant",
            content=f"Entschuldigung, ein Fehler ist aufgetreten: {e!s}",
            metadata={"error": True, "error_message": str(e)}
        )


# API Endpunkte
@router.post("/sessions", response_model=ChatSession)
async def create_chat_session(request: ChatSessionRequest):
    """Erstellt eine neue Chat-Sitzung."""
    session = ChatSession(
        user_id=request.user_id,
        session_name=request.session_name or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        agent_id=request.agent_id,
        context=request.context or {}
    )

    # System-Prompt als erste Nachricht hinzufügen
    if request.system_prompt:
        system_message = ChatMessage(
            role="system",
            content=request.system_prompt
        )
        session.messages.append(system_message)

    # Sitzung speichern
    chat_sessions[session.id] = session
    get_user_sessions(request.user_id).append(session.id)

    logger.info(f"💬 Neue Chat-Sitzung erstellt: {session.id}")
    return session


@router.get("/sessions", response_model=list[ChatSession])
async def list_user_sessions(user_id: str = Query(..., description="Benutzer-ID")):
    """Listet Chat-Sitzungen eines Benutzers."""
    session_ids = get_user_sessions(user_id)
    return [chat_sessions[session_id] for session_id in session_ids if session_id in chat_sessions]


@router.get("/sessions/{session_id}", response_model=ChatSession)
async def get_chat_session(session_id: str = Path(..., description="Chat-Sitzungs-ID")):
    """Holt eine spezifische Chat-Sitzung."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat-Sitzung nicht gefunden")
    return chat_sessions[session_id]


@router.post("/sessions/{session_id}/messages", response_model=ChatResponse)
async def send_chat_message(
    request: ChatMessageRequest,
    session_id: str = Path(..., description="Chat-Sitzungs-ID")
):
    """Sendet eine Nachricht an eine Chat-Sitzung."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat-Sitzung nicht gefunden")

    session = chat_sessions[session_id]
    start_time = datetime.now()

    # Benutzer-Nachricht hinzufügen
    user_message = ChatMessage(
        role="user",
        content=request.content,
        metadata=request.context or {}
    )
    session.messages.append(user_message)

    # Agent-Antwort verarbeiten
    assistant_message = await process_chat_message(
        session=session,
        message_content=request.content,
        context=request.context
    )
    session.messages.append(assistant_message)

    # Sitzung aktualisieren
    session.updated_at = datetime.now(UTC)

    processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

    response = ChatResponse(
        session_id=session_id,
        message=assistant_message,
        agent_info={"agent_id": session.agent_id} if session.agent_id else None,
        processing_time_ms=processing_time
    )

    logger.info(
        f"💬 Nachricht in Sitzung {session_id} verarbeitet ({processing_time}ms)")
    return response


@router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str = Path(..., description="Chat-Sitzungs-ID")):
    """Löscht eine Chat-Sitzung."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat-Sitzung nicht gefunden")

    session = chat_sessions[session_id]
    user_id = session.user_id

    # Aus Benutzer-Liste entfernen
    if user_id in user_sessions and session_id in user_sessions[user_id]:
        user_sessions[user_id].remove(session_id)

    # Sitzung löschen
    del chat_sessions[session_id]

    logger.info(f"🗑️ Chat-Sitzung {session_id} gelöscht")
    return {"message": "Chat-Sitzung erfolgreich gelöscht", "session_id": session_id}


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessage])
async def get_chat_messages(
    session_id: str = Path(..., description="Chat-Sitzungs-ID"),
    limit: int = Query(100, ge=1, le=500, description="Anzahl Nachrichten"),
    offset: int = Query(0, ge=0, description="Nachrichtennummer für Start")
):
    """Holt Nachrichten einer Chat-Sitzung."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat-Sitzung nicht gefunden")

    messages = chat_sessions[session_id].messages
    return messages[offset:offset + limit]


@router.get("/health")
async def chat_health_check():
    """Health Check für Chat System."""
    additional_data = {
        "chat_sessions": {
            "total_sessions": len(chat_sessions),
            "active_users": len(user_sessions),
            "total_messages": sum(len(s.messages) for s in chat_sessions.values())
        }
    }

    agent_status = get_agent_system_status()
    if agent_status:
        additional_data["agent_system_status"] = agent_status

    return create_health_response(additional_data)
