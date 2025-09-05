"""Base Interceptor für gRPC Server mit Template Method Pattern.

Eliminiert Code-Duplikation zwischen verschiedenen Interceptor-Implementierungen
durch gemeinsame Wrapper-Logik und standardisierte Handler-Patterns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

import grpc

from kei_logging import get_logger

logger = get_logger(__name__)

# Type Aliases für bessere Lesbarkeit
HandlerCallDetails = grpc.HandlerCallDetails
ServicerContext = grpc.ServicerContext
GRPCHandler = grpc.RpcMethodHandler
UnaryUnaryHandler = Callable[[Any, ServicerContext], Awaitable[Any]]
UnaryStreamHandler = Callable[[Any, ServicerContext], Any]
StreamUnaryHandler = Callable[[Any, ServicerContext], Awaitable[Any]]
StreamStreamHandler = Callable[[Any, ServicerContext], Any]


class BaseInterceptor(grpc.aio.ServerInterceptor, ABC):
    """Basis-Interceptor mit Template Method Pattern.

    Eliminiert Code-Duplikation durch gemeinsame Wrapper-Logik.
    Subklassen implementieren nur die spezifische Interceptor-Logik.
    """

    def __init__(self, name: str) -> None:
        """Initialisiert Base Interceptor.

        Args:
            name: Name des Interceptors für Logging
        """
        self.name = name
        self.logger = get_logger(f"{__name__}.{name}")

    async def intercept_service(
        self,
        continuation: Callable[[HandlerCallDetails], Awaitable[GRPCHandler]],
        handler_call_details: HandlerCallDetails,
    ) -> GRPCHandler | None:
        """Template Method für Service-Interception.

        Args:
            continuation: Nächster Handler in der Chain
            handler_call_details: Details des gRPC-Calls

        Returns:
            Modifizierter gRPC Handler
        """
        handler = await continuation(handler_call_details)
        if handler is None:
            return None

        method_name = getattr(handler_call_details, "method", "unknown") or "unknown"

        # Template Method Pattern: Delegiere an spezifische Handler-Wrapper
        # Verwende grpc.unary_unary_rpc_method_handler statt direkter Instanziierung
        if hasattr(handler, "unary_unary") and handler.unary_unary:
            return grpc.unary_unary_rpc_method_handler(
                self._wrap_unary_unary(handler.unary_unary, method_name)
            )
        if hasattr(handler, "unary_stream") and handler.unary_stream:
            return grpc.unary_stream_rpc_method_handler(
                self._wrap_unary_stream(handler.unary_stream, method_name)
            )
        if hasattr(handler, "stream_unary") and handler.stream_unary:
            return grpc.stream_unary_rpc_method_handler(
                self._wrap_stream_unary(handler.stream_unary, method_name)
            )
        if hasattr(handler, "stream_stream") and handler.stream_stream:
            return grpc.stream_stream_rpc_method_handler(
                self._wrap_stream_stream(handler.stream_stream, method_name)
            )

        return handler

    def _wrap_unary_unary(self, behavior: UnaryUnaryHandler, method_name: str) -> UnaryUnaryHandler:
        """Wrapper für Unary-Unary Handler.

        Args:
            behavior: Original Handler-Funktion
            method_name: Name der gRPC-Methode

        Returns:
            Gewrappte Handler-Funktion
        """

        async def _wrapped(request: Any, context: ServicerContext) -> Any:
            # Pre-Processing Hook
            await self._before_call(request, context, method_name)

            try:
                # Hauptlogik des Interceptors
                result = await self._process_unary_unary(request, context, behavior, method_name)

                # Post-Processing Hook
                await self._after_call(request, result, context, method_name)
                return result

            except Exception as e:
                # Error-Processing Hook
                await self._on_error(request, e, context, method_name)
                raise

        return _wrapped

    def _wrap_unary_stream(
        self, behavior: UnaryStreamHandler, method_name: str
    ) -> UnaryStreamHandler:
        """Wrapper für Unary-Stream Handler."""

        async def _wrapped(request: Any, context: ServicerContext) -> Any:
            await self._before_call(request, context, method_name)
            try:
                result = BaseInterceptor._process_unary_stream(request, context, behavior, method_name)
                await self._after_call(request, result, context, method_name)
                return result
            except Exception as e:
                await self._on_error(request, e, context, method_name)
                raise

        return _wrapped

    def _wrap_stream_unary(
        self, behavior: StreamUnaryHandler, method_name: str
    ) -> StreamUnaryHandler:
        """Wrapper für Stream-Unary Handler."""

        async def _wrapped(request: Any, context: ServicerContext) -> Any:
            await self._before_call(request, context, method_name)
            try:
                result = await self._process_stream_unary(request, context, behavior, method_name)
                await self._after_call(request, result, context, method_name)
                return result
            except Exception as e:
                await self._on_error(request, e, context, method_name)
                raise

        return _wrapped

    def _wrap_stream_stream(
        self, behavior: StreamStreamHandler, method_name: str
    ) -> StreamStreamHandler:
        """Wrapper für Stream-Stream Handler."""

        async def _wrapped(request: Any, context: ServicerContext) -> Any:
            await self._before_call(request, context, method_name)
            try:
                result = BaseInterceptor._process_stream_stream(request, context, behavior, method_name)
                await self._after_call(request, result, context, method_name)
                return result
            except Exception as e:
                await self._on_error(request, e, context, method_name)
                raise

        return _wrapped

    # Template Methods - Subklassen implementieren spezifische Logik

    async def _before_call(self, request: Any, context: ServicerContext, method_name: str) -> None:
        """Hook vor dem Handler-Aufruf. Subklassen können überschreiben."""

    async def _after_call(
        self, request: Any, response: Any, context: ServicerContext, method_name: str
    ) -> None:
        """Hook nach dem Handler-Aufruf. Subklassen können überschreiben."""

    async def _on_error(
        self, request: Any, error: Exception, context: ServicerContext, method_name: str
    ) -> None:
        """Hook bei Fehlern. Subklassen können überschreiben."""
        self.logger.error(f"Fehler in {self.name} für {method_name}: {error}")

    # Abstract Methods - Subklassen MÜSSEN implementieren

    @abstractmethod
    async def _process_unary_unary(
        self, request: Any, context: ServicerContext, behavior: UnaryUnaryHandler, method_name: str
    ) -> Any:
        """Verarbeite Unary-Unary Request. Muss von Subklassen implementiert werden."""

    @staticmethod
    def _process_unary_stream(
        request: Any, context: ServicerContext, behavior: UnaryStreamHandler, _method_name: str
    ) -> Any:
        """Verarbeite Unary-Stream Request. Standard-Implementierung."""
        return behavior(request, context)

    async def _process_stream_unary(
        self, request: Any, context: ServicerContext, behavior: StreamUnaryHandler, _: str
    ) -> Any:
        """Verarbeite Stream-Unary Request. Standard-Implementierung."""
        return await behavior(request, context)

    @staticmethod
    def _process_stream_stream(
        request: Any,
        context: ServicerContext,
        behavior: StreamStreamHandler,
        _method_name: str,
    ) -> Any:
        """Verarbeite Stream-Stream Request. Standard-Implementierung."""
        return behavior(request, context)


class NoOpInterceptor(BaseInterceptor):
    """No-Operation Interceptor für Tests und als Beispiel.

    Führt keine Modifikationen durch, sondern ruft nur den Original-Handler auf.
    """

    def __init__(self) -> None:
        """Initialisiert NoOp Interceptor."""
        super().__init__("NoOp")

    async def _process_unary_unary(
        self, request: Any, context: ServicerContext, behavior: UnaryUnaryHandler, method_name: str
    ) -> Any:
        """Führt Original-Handler ohne Modifikationen aus."""
        return await behavior(request, context)


__all__ = [
    "BaseInterceptor",
    "GRPCHandler",
    "HandlerCallDetails",
    "NoOpInterceptor",
    "ServicerContext",
    "StreamStreamHandler",
    "StreamUnaryHandler",
    "UnaryStreamHandler",
    "UnaryUnaryHandler",
]
