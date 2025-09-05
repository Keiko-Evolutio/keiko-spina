"""
Backend-eigener Agent Orchestrator Service
Ersetzt direkte SDK-Dependencies durch API-basierte Kommunikation
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
import json

try:
    from kei_logging import get_logger
except ImportError:
    # Fallback during migration
    import logging
    def get_logger(name: str):
        return logging.getLogger(name)

logger = get_logger(__name__)


class AgentStatus(str, Enum):
    """Agent Status Enumeration"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"


class AgentCapability(str, Enum):
    """Agent Capability Types"""
    CODE_GENERATION = "code_generation"
    WEB_RESEARCH = "web_research"
    IMAGE_GENERATION = "image_generation"
    DATA_ANALYSIS = "data_analysis"
    VOICE_PROCESSING = "voice_processing"


@dataclass
class AgentConfig:
    """Agent Configuration"""
    agent_id: str
    agent_type: str
    capabilities: List[str]
    metadata: Dict[str, str] = field(default_factory=dict)
    max_concurrent_requests: int = 5
    timeout_seconds: int = 30


@dataclass
class AgentRegistration:
    """Agent Registration Result"""
    agent_id: str
    registered: bool
    registered_at: str
    platform_agent_id: str
    status: AgentStatus = AgentStatus.ONLINE
    capabilities: List[str] = field(default_factory=list)


@dataclass
class AgentRequest:
    """Agent Request Structure"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    function_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timeout: int = 30
    priority: int = 0


@dataclass
class AgentResponse:
    """Agent Response Structure"""
    request_id: str
    agent_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class AgentOrchestratorService:
    """Backend-eigener Orchestrator für Agent-Management"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.registered_agents: Dict[str, AgentRegistration] = {}
        self.pending_requests: Dict[str, AgentRequest] = {}
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Configuration
        self.agent_service_url = self.config.get("agent_service_url", "http://localhost:8080")
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        
        logger.info("Agent Orchestrator Service initialisiert")

    async def register_agent(self, agent_config: AgentConfig) -> AgentRegistration:
        """Registriert Agent über Backend-interne Logik"""
        logger.info(f"Registriere Agent: {agent_config.agent_id}")

        try:
            # Backend-interne Agent-Registrierung
            registration = AgentRegistration(
                agent_id=agent_config.agent_id,
                registered=True,
                registered_at=datetime.now(UTC).isoformat(),
                platform_agent_id=f"platform_{agent_config.agent_id}",
                status=AgentStatus.ONLINE,
                capabilities=agent_config.capabilities
            )

            # Agent in Registry speichern
            self.registered_agents[agent_config.agent_id] = registration

            # Event publizieren
            await self._publish_agent_event("agent.registered", registration)

            logger.info(f"Agent erfolgreich registriert: {agent_config.agent_id}")
            return registration

        except Exception as e:
            logger.error(f"Agent-Registrierung fehlgeschlagen: {e}")
            return AgentRegistration(
                agent_id=agent_config.agent_id,
                registered=False,
                registered_at=datetime.now(UTC).isoformat(),
                platform_agent_id="",
                status=AgentStatus.ERROR
            )

    async def call_agent_via_api(self, request: AgentRequest) -> AgentResponse:
        """
        Ruft Agent über HTTP API auf (statt direkter SDK-Kommunikation)
        Implementiert die API-basierte Agent-Kommunikation aus dem Migrationsplan
        """
        start_time = time.time()
        
        try:
            logger.info(f"Rufe Agent auf: {request.agent_id}, Function: {request.function_name}")
            
            # Check if agent is registered
            if request.agent_id not in self.registered_agents:
                return AgentResponse(
                    request_id=request.request_id,
                    agent_id=request.agent_id,
                    success=False,
                    error=f"Agent {request.agent_id} nicht registriert",
                    execution_time_ms=int((time.time() - start_time) * 1000)
                )

            # Prepare API call
            api_payload = {
                "request_id": request.request_id,
                "function_name": request.function_name,
                "parameters": request.parameters,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "timeout": request.timeout
            }

            # Execute HTTP API call
            response = await self._execute_agent_api_call(request.agent_id, api_payload)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            if response:
                agent_response = AgentResponse(
                    request_id=request.request_id,
                    agent_id=request.agent_id,
                    success=True,
                    result=response,
                    execution_time_ms=execution_time
                )
                
                logger.info(f"Agent-Aufruf erfolgreich: {request.agent_id}")
                return agent_response
            else:
                return AgentResponse(
                    request_id=request.request_id,
                    agent_id=request.agent_id,
                    success=False,
                    error="Agent-API-Aufruf fehlgeschlagen",
                    execution_time_ms=execution_time
                )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Fehler bei Agent-API-Aufruf: {e}")
            
            return AgentResponse(
                request_id=request.request_id,
                agent_id=request.agent_id,
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )

    async def _execute_agent_api_call(
        self, 
        agent_id: str, 
        payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Führt HTTP API-Aufruf zu Agent-Service aus"""
        
        retries = 0
        while retries < self.max_retries:
            try:
                # API-Endpunkt für Agent-Aufruf
                url = f"{self.agent_service_url}/api/v1/agents/{agent_id}/call"
                
                response = await self.http_client.post(
                    url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-Request-ID": payload.get("request_id", ""),
                    }
                )
                
                response.raise_for_status()
                return response.json()
                
            except httpx.RequestError as e:
                retries += 1
                logger.warning(f"API-Aufruf fehlgeschlagen (Versuch {retries}/{self.max_retries}): {e}")
                
                if retries < self.max_retries:
                    await asyncio.sleep(self.retry_delay * retries)
                else:
                    logger.error(f"API-Aufruf nach {self.max_retries} Versuchen fehlgeschlagen")
                    raise
                    
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP-Fehler bei Agent-API-Aufruf: {e.response.status_code} - {e.response.text}")
                break
        
        return None

    async def get_agent_status(self, agent_id: str) -> Optional[AgentRegistration]:
        """Holt Agent-Status"""
        return self.registered_agents.get(agent_id)

    async def list_agents(self) -> List[AgentRegistration]:
        """Listet alle registrierten Agents auf"""
        return list(self.registered_agents.values())

    async def unregister_agent(self, agent_id: str) -> bool:
        """Entfernt Agent aus Registry"""
        if agent_id in self.registered_agents:
            registration = self.registered_agents[agent_id]
            registration.status = AgentStatus.OFFLINE
            
            # Event publizieren
            await self._publish_agent_event("agent.unregistered", registration)
            
            del self.registered_agents[agent_id]
            logger.info(f"Agent entfernt: {agent_id}")
            return True
        
        return False

    async def _publish_agent_event(self, event_type: str, data: AgentRegistration):
        """Publiziert Agent-Events über Backend-Messaging"""
        try:
            event_payload = {
                "event_type": event_type,
                "agent_id": data.agent_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "data": {
                    "agent_id": data.agent_id,
                    "platform_agent_id": data.platform_agent_id,
                    "status": data.status.value,
                    "capabilities": data.capabilities,
                    "registered_at": data.registered_at
                }
            }
            
            # TODO: Hier würde NATS/Kafka Event Publishing implementiert
            logger.debug(f"Agent-Event publiziert: {event_type} für {data.agent_id}")
            
        except Exception as e:
            logger.error(f"Fehler beim Publizieren von Agent-Event: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Health Check für Orchestrator Service"""
        return {
            "service": "agent_orchestrator",
            "status": "healthy",
            "registered_agents": len(self.registered_agents),
            "pending_requests": len(self.pending_requests),
            "timestamp": datetime.now(UTC).isoformat()
        }

    async def cleanup(self):
        """Cleanup-Methode für graceful shutdown"""
        logger.info("Orchestrator Service wird heruntergefahren...")
        
        # HTTP Client schließen
        await self.http_client.aclose()
        
        # Alle Agents als offline markieren
        for agent_id in list(self.registered_agents.keys()):
            await self.unregister_agent(agent_id)
        
        logger.info("Orchestrator Service heruntergefahren")


# Backward Compatibility Functions
async def registry_orchestrator(agent_id: str, **kwargs) -> Dict[str, Any]:
    """
    Backward compatibility function for existing code
    Ersetzt: from kei_agents.orchestrator.registry_orchestrator import registry_orchestrator
    """
    orchestrator = AgentOrchestratorService()
    
    # Legacy function call umwandeln
    request = AgentRequest(
        agent_id=agent_id,
        function_name=kwargs.get("function_name", "default"),
        parameters=kwargs.get("parameters", {}),
        user_id=kwargs.get("user_id"),
        session_id=kwargs.get("session_id")
    )
    
    response = await orchestrator.call_agent_via_api(request)
    
    if response.success:
        return response.result or {}
    else:
        raise Exception(f"Agent call failed: {response.error}")


# Factory Function
def create_agent_orchestrator(config: Optional[Dict[str, Any]] = None) -> AgentOrchestratorService:
    """Factory function für Agent Orchestrator Service"""
    return AgentOrchestratorService(config)


# Global instance for singleton pattern
_orchestrator_instance: Optional[AgentOrchestratorService] = None


async def get_orchestrator() -> AgentOrchestratorService:
    """Get global orchestrator instance"""
    global _orchestrator_instance
    
    if _orchestrator_instance is None:
        _orchestrator_instance = AgentOrchestratorService()
    
    return _orchestrator_instance