"""gRPC Error Mapping Interceptor für KeikoExceptions.

Fängt KeikoExceptions im Servicer ab, mappt sie auf gRPC-Statuscodes,
setzt strukturierte Trailer-Metadaten und schreibt konsistente Logs.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import grpc

from core.exceptions import KeikoException
from kei_logging import get_logger, structured_msg

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)

HandlerCallDetails = grpc.HandlerCallDetails
ServicerContext = grpc.ServicerContext


def _classify(exc: KeikoException) -> grpc.StatusCode:
    """Mappt KeikoExceptions auf gRPC StatusCodes."""
    code = exc.error_code
    if code in {"VALIDATION_ERROR", "BAD_REQUEST"}:
        return grpc.StatusCode.INVALID_ARGUMENT
    if code in {"NOT_FOUND"}:
        return grpc.StatusCode.NOT_FOUND
    if code in {"AUTH_ERROR"}:
        return grpc.StatusCode.UNAUTHENTICATED
    if code in {"RATE_LIMIT_EXCEEDED"}:
        return grpc.StatusCode.RESOURCE_EXHAUSTED
    if code in {"TIMEOUT", "DEADLINE_EXCEEDED"}:
        return grpc.StatusCode.DEADLINE_EXCEEDED
    if code in {"CONFLICT"}:
        return grpc.StatusCode.FAILED_PRECONDITION
    if code in {"SERVICE_UNAVAILABLE", "DEPENDENCY_ERROR", "AZURE_ERROR", "NETWORK_ERROR"}:
        return grpc.StatusCode.UNAVAILABLE
    return grpc.StatusCode.INTERNAL


def _set_trailers(context: ServicerContext, exc: KeikoException) -> None:
    try:
        trailers = (
            ("kei-error-code", exc.error_code),
            ("kei-error-severity", str(exc.severity)),
        )
        context.set_trailing_metadata(trailers)
    except (AttributeError, TypeError) as e:
        logger.debug(f"Fehler beim Setzen der gRPC-Trailer-Metadaten: {e}")
    except Exception as e:
        logger.warning(f"Unerwarteter Fehler beim Setzen der gRPC-Trailer: {e}")


class ErrorMappingInterceptor(grpc.aio.ServerInterceptor):
    """Interceptor: Vereinheitlicht Fehlerbehandlung für gRPC via KeikoExceptions."""

    async def intercept_service(
        self,
        continuation: Callable[[HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]],
        handler_call_details: HandlerCallDetails,
    ) -> grpc.RpcMethodHandler | None:
        handler = await continuation(handler_call_details)
        if handler is None:
            return None

        method = getattr(handler_call_details, "method", "unknown") or "unknown"

        def _handle_keiko_exception(exc: KeikoException, context: ServicerContext) -> None:
            """Gemeinsame Behandlung von KeikoExceptions."""
            _set_trailers(context, exc)
            with contextlib.suppress(Exception):
                logger.exception(
                    structured_msg(
                        "gRPC KeikoException",
                        method=method,
                        code=exc.error_code,
                        severity=str(exc.severity),
                    )
                )
            context.abort(_classify(exc), exc.message)

        def _wrap_unary_unary(behavior):
            async def _wrapped(request, context: ServicerContext):
                try:
                    return await behavior(request, context)
                except KeikoException as exc:
                    _handle_keiko_exception(exc, context)

            return _wrapped

        def _wrap_unary_stream(behavior):
            async def _wrapped(request, context: ServicerContext):
                try:
                    async for resp in behavior(request, context):
                        yield resp
                except KeikoException as exc:
                    _handle_keiko_exception(exc, context)

            return _wrapped

        def _wrap_stream_unary(behavior):
            async def _wrapped(request_iter, context: ServicerContext):
                try:
                    return await behavior(request_iter, context)
                except KeikoException as exc:
                    _handle_keiko_exception(exc, context)

            return _wrapped

        def _wrap_stream_stream(behavior):
            async def _wrapped(request_iter, context: ServicerContext):
                try:
                    async for resp in behavior(request_iter, context):
                        yield resp
                except KeikoException as exc:
                    _handle_keiko_exception(exc, context)

            return _wrapped

        if hasattr(handler, "unary_unary") and handler.unary_unary:
            return grpc.unary_unary_rpc_method_handler(
                _wrap_unary_unary(handler.unary_unary),
                request_deserializer=getattr(handler, "request_deserializer", None),
                response_serializer=getattr(handler, "response_serializer", None),
            )
        if hasattr(handler, "unary_stream") and handler.unary_stream:
            return grpc.unary_stream_rpc_method_handler(
                _wrap_unary_stream(handler.unary_stream),
                request_deserializer=getattr(handler, "request_deserializer", None),
                response_serializer=getattr(handler, "response_serializer", None),
            )
        if hasattr(handler, "stream_unary") and handler.stream_unary:
            return grpc.stream_unary_rpc_method_handler(
                _wrap_stream_unary(handler.stream_unary),
                request_deserializer=getattr(handler, "request_deserializer", None),
                response_serializer=getattr(handler, "response_serializer", None),
            )
        if hasattr(handler, "stream_stream") and handler.stream_stream:
            return grpc.stream_stream_rpc_method_handler(
                _wrap_stream_stream(handler.stream_stream),
                request_deserializer=getattr(handler, "request_deserializer", None),
                response_serializer=getattr(handler, "response_serializer", None),
            )
        return handler


__all__ = ["ErrorMappingInterceptor"]
