"""V2 Functions API – striktere Schemas und Validierung.

Stellt Funktion-Listing, Detail und Ausführung unter `/api/v2/functions`
bereit. Nutzt Pydantic v2 Modelle mit engeren Constraints und liefert
einheitliche Fehlerrückgaben.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Path
from pydantic import ValidationError

from kei_logging import get_logger, structured_msg

from .constants import (
    DEFAULT_CATEGORY,
    DEFAULT_FUNCTION_NAME,
    DEFAULT_PARAM_NAME,
    ERROR_CODE_EXEC_ERROR,
    ERROR_CODE_HTTP_ERROR,
    ERROR_CODE_NOT_FOUND,
    ERROR_CODE_UNEXPECTED_ERROR,
    ERROR_CODE_VALIDATION_ERROR,
    HTTP_STATUS_ACCEPTED,
    HTTP_STATUS_NOT_FOUND,
    PARAM_TYPE_ANY,
    URL_PREFIX_V2_FUNCTIONS,
)
from .models import V2ExecuteRequest, V2ExecuteResult, V2FunctionSchema
from .utils import (
    calculate_duration_ms,
    create_execution_id,
    log_and_handle_exception,
)

logger = get_logger(__name__)


router = APIRouter(prefix=URL_PREFIX_V2_FUNCTIONS, tags=["v2-functions"])


def _map_v1_schema_to_v2(item: dict[str, Any]) -> V2FunctionSchema:
    """Transformiert v1 Schema zu v2 Schema.

    Args:
        item: V1 Schema-Dictionary

    Returns:
        V2FunctionSchema-Objekt
    """
    from .models import ParameterSpec

    params = [
        ParameterSpec(
            name=p.get("name", DEFAULT_PARAM_NAME),
            type=str(p.get("type", PARAM_TYPE_ANY)).lower(),
            description=p.get("description", ""),
            required=bool(p.get("required", True)),
            default=p.get("default"),
        )
        for p in item.get("parameters", [])
    ]
    return V2FunctionSchema(
        name=item.get("name", DEFAULT_FUNCTION_NAME),
        description=item.get("description", ""),
        parameters=params,
        return_type=str(item.get("return_type", PARAM_TYPE_ANY)).lower(),
        category=item.get("category", DEFAULT_CATEGORY),
        tags=item.get("tags", []),
    )


@router.get("/", response_model=list[V2FunctionSchema])
async def list_functions_v2() -> list[V2FunctionSchema]:
    """Listet Funktionen (v2) mit strikteren Schemas."""
    try:
        from api.routes.functions_routes import list_functions as v1_list  # type: ignore
        v1 = await v1_list()  # type: ignore[func-returns-value]
        out: list[V2FunctionSchema] = []
        for item in v1:
            try:
                out.append(_map_v1_schema_to_v2(item if isinstance(item, dict) else item.dict()))
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                # Logge spezifische Mapping-Fehler für Debugging
                logger.debug(structured_msg("v2.functions.mapping_error",
                                           item=str(item)[:100], error=str(exc)))
                continue
        return out
    except Exception as exc:  # pragma: no cover
        logger.warning(structured_msg("v2.functions.list error", error=str(exc)))
        return []


@router.get("/{function_name}", response_model=V2FunctionSchema)
async def get_function_v2(
    function_name: str = Path(..., description="Funktionsname")
) -> V2FunctionSchema:
    """Details einer Funktion (v2)."""
    try:
        from api.routes.functions_routes import get_function_details as v1_get  # type: ignore
        v1 = await v1_get(function_name)  # type: ignore[func-returns-value]
        data = v1 if isinstance(v1, dict) else v1.dict()
        return _map_v1_schema_to_v2(data)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=HTTP_STATUS_NOT_FOUND,
            detail={"code": ERROR_CODE_NOT_FOUND, "message": str(exc)}
        )


@router.post("/execute", response_model=V2ExecuteResult, status_code=HTTP_STATUS_ACCEPTED)
async def execute_function_v2(payload: V2ExecuteRequest) -> V2ExecuteResult:
    """Führt eine Funktion in v2 aus mit strenger Validierung."""
    exec_id = create_execution_id()
    started = datetime.now(UTC)

    try:
        result = await _execute_v1_function(payload)
        return _map_v1_result_to_v2(result, exec_id, payload.function, started)
    except ValidationError as ve:
        return _create_validation_error_result(exec_id, payload.function, started, ve)
    except HTTPException as he:
        return _create_http_error_result(exec_id, payload.function, started, he)
    except Exception as exc:
        log_and_handle_exception("v2.functions.execute", exc)
        return _create_unexpected_error_result(exec_id, payload.function, started, exc)


async def _execute_v1_function(payload: V2ExecuteRequest) -> Any:
    """Führt V1-Funktion aus.

    Args:
        payload: V2 Execution Request

    Returns:
        V1 Execution Result
    """
    from api.routes.functions_routes import execute_function as v1_exec  # type: ignore
    v1_req = {
        "function_name": payload.function,
        "parameters": payload.parameters,
        "context": None,
    }
    return await v1_exec(v1_req)  # type: ignore[misc]


def _map_v1_result_to_v2(
    v1_result: Any,
    exec_id: str,
    function_name: str,
    started: datetime
) -> V2ExecuteResult:
    """Mapp V1-Ergebnis zu V2-Format.

    Args:
        v1_result: V1 Execution Result
        exec_id: Execution ID
        function_name: Funktionsname
        started: Startzeitpunkt

    Returns:
        V2ExecuteResult
    """
    from .models import ExecutionError

    data = v1_result if isinstance(v1_result, dict) else v1_result.dict()
    err = data.get("error")

    return V2ExecuteResult(
        execution_id=data.get("execution_id", exec_id),
        function=data.get("function_name", function_name),
        status=data.get("status", "success"),
        result=data.get("result"),
        executed_at=data.get("executed_at", started),
        duration_ms=int(data.get("duration_ms", 0)),
        error=(
            ExecutionError(code=ERROR_CODE_EXEC_ERROR, message=str(err))
            if err else None
        ),
    )


def _create_validation_error_result(
    exec_id: str,
    function_name: str,
    started: datetime,
    validation_error: ValidationError
) -> V2ExecuteResult:
    """Erstellt Validation-Error-Result.

    Args:
        exec_id: Execution ID
        function_name: Funktionsname
        started: Startzeitpunkt
        validation_error: Validation Error

    Returns:
        V2ExecuteResult mit Validation Error
    """
    from .models import ExecutionError

    return V2ExecuteResult(
        execution_id=exec_id,
        function=function_name,
        status="error",
        executed_at=started,
        duration_ms=calculate_duration_ms(started),
        error=ExecutionError(
            code=ERROR_CODE_VALIDATION_ERROR,
            message="Parameter ungültig",
            details={"errors": validation_error.errors()}
        ),
    )


def _create_http_error_result(
    exec_id: str,
    function_name: str,
    started: datetime,
    http_exception: HTTPException
) -> V2ExecuteResult:
    """Erstellt HTTP-Error-Result.

    Args:
        exec_id: Execution ID
        function_name: Funktionsname
        started: Startzeitpunkt
        http_exception: HTTP Exception

    Returns:
        V2ExecuteResult mit HTTP Error
    """
    from .models import ExecutionError

    return V2ExecuteResult(
        execution_id=exec_id,
        function=function_name,
        status="error",
        executed_at=started,
        duration_ms=calculate_duration_ms(started),
        error=ExecutionError(
            code=ERROR_CODE_HTTP_ERROR,
            message=str(http_exception.detail)
        ),
    )


def _create_unexpected_error_result(
    exec_id: str,
    function_name: str,
    started: datetime,
    exception: Exception
) -> V2ExecuteResult:
    """Erstellt Unexpected-Error-Result.

    Args:
        exec_id: Execution ID
        function_name: Funktionsname
        started: Startzeitpunkt
        exception: Exception

    Returns:
        V2ExecuteResult mit Unexpected Error
    """
    from .models import ExecutionError

    return V2ExecuteResult(
        execution_id=exec_id,
        function=function_name,
        status="error",
        executed_at=started,
        duration_ms=calculate_duration_ms(started),
        error=ExecutionError(
            code=ERROR_CODE_UNEXPECTED_ERROR,
            message=str(exception)
        ),
    )


__all__ = ["router"]
