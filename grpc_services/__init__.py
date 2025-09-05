# backend/grpc_services/__init__.py
"""gRPC Services für Issue #56 Messaging-first Architecture
Implementiert gRPC Services basierend auf Protocol Buffers Contracts
"""

from .grpc_server import GRPCServer
from .sdk_platform_service import SDKPlatformCommunicationServiceImpl

__all__ = [
    "GRPCServer",
    "SDKPlatformCommunicationServiceImpl"
]
