# backend/api/routes/functions_routes.py
"""Function Calling API Routes für Azure AI Foundry."""

import contextlib
import inspect
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Path
from pydantic import BaseModel, Field

from kei_logging import get_logger

from .base import check_agents_integration, create_health_response, create_router
from .common import FUNCTION_RESPONSES

logger = get_logger(__name__)

# Router-Konfiguration
router = create_router("/functions", ["functions"])
router.responses.update(FUNCTION_RESPONSES)


# Datenmodelle
class FunctionParameter(BaseModel):
    """Function Parameter Definition."""
    name: str = Field(..., description="Parameter-Name")
    type: str = Field(..., description="Parameter-Typ")
    description: str = Field(..., description="Parameter-Beschreibung")
    required: bool = Field(True, description="Ist Parameter erforderlich")
    default: Any | None = Field(None, description="Standard-Wert")


class FunctionSchema(BaseModel):
    """Function Schema Definition."""
    name: str = Field(..., description="Funktions-Name")
    description: str = Field(..., description="Funktions-Beschreibung")
    parameters: list[FunctionParameter] = Field(..., description="Funktions-Parameter")
    return_type: str = Field(..., description="Rückgabe-Typ")
    category: str = Field("general", description="Funktions-Kategorie")
    tags: list[str] = Field(default_factory=list, description="Funktions-Tags")


class FunctionExecutionRequest(BaseModel):
    """Request für Funktions-Ausführung."""
    function_name: str = Field(..., description="Name der auszuführenden Funktion")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Funktions-Parameter")
    context: dict[str, Any] | None = Field(None, description="Ausführungskontext")


class FunctionExecutionResult(BaseModel):
    """Result einer Funktions-Ausführung."""
    execution_id: str = Field(..., description="Eindeutige Ausführungs-ID")
    function_name: str = Field(..., description="Ausgeführte Funktion")
    status: str = Field(..., description="Ausführungsstatus")
    result: Any | None = Field(None, description="Funktionsergebnis")
    executed_at: datetime = Field(..., description="Ausführungszeit")
    duration_ms: int = Field(..., description="Ausführungsdauer")
    error: str | None = Field(None, description="Fehlermeldung")


# Built-in Funktionen
async def _builtin_get_current_time() -> str:
    """Gibt die aktuelle Zeit im ISO-Format zurück."""
    return datetime.now(UTC).isoformat()


async def _builtin_echo(text: str) -> str:
    """Echo-Funktion (Text zurückgeben)."""
    return f"Echo: {text}"


async def _builtin_add_numbers(a: float, b: float) -> float:
    """Addiert zwei Zahlen."""
    return a + b


async def _builtin_multiply_numbers(a: float, b: float) -> float:
    """Multipliziert zwei Zahlen."""
    return a * b


BUILT_IN_FUNCTIONS: dict[str, Callable] = {
    "get_current_time": _builtin_get_current_time,
    "echo": _builtin_echo,
    "add_numbers": _builtin_add_numbers,
    "multiply_numbers": _builtin_multiply_numbers,
}

# Registrierte Tools und Execution History
registered_tools: dict[str, FunctionSchema] = {}
execution_history: dict[str, FunctionExecutionResult] = {}


# Helper Funktionen
def get_function_schema(func: Callable, name: str) -> FunctionSchema:
    """Erstellt Schema aus Funktion."""
    sig = inspect.signature(func)
    parameters = []

    for param_name, param in sig.parameters.items():
        param_info = FunctionParameter(
            name=param_name,
            type=str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
            description=f"Parameter {param_name}",
            required=param.default == inspect.Parameter.empty,
            default=param.default if param.default != inspect.Parameter.empty else None
        )
        parameters.append(param_info)

    return FunctionSchema(
        name=name,
        description=func.__doc__ or f"Function {name}",
        parameters=parameters,
        return_type=str(
            sig.return_annotation) if sig.return_annotation != inspect.Parameter.empty else "Any"
    )


async def execute_function_safely(request: FunctionExecutionRequest) -> FunctionExecutionResult:
    """Führt Funktion sicher aus."""
    execution_id = f"exec_{uuid4().hex[:8]}"
    start_time = datetime.now()

    try:
        result: Any
        # Built-in Funktion prüfen
        if request.function_name in BUILT_IN_FUNCTIONS:
            func = BUILT_IN_FUNCTIONS[request.function_name]
            # Unterstützt synchrone und asynchrone Built-ins
            if inspect.iscoroutinefunction(func):
                result = await func(**request.parameters)
            else:
                result = func(**request.parameters)
        else:
            raise ValueError(f"Funktion '{request.function_name}' nicht gefunden")

        end_time = datetime.now()
        duration = int((end_time - start_time).total_seconds() * 1000)

        execution_result = FunctionExecutionResult(
            execution_id=execution_id,
            function_name=request.function_name,
            status="success",
            result=result,
            executed_at=start_time,
            duration_ms=duration
        )

        # In History speichern
        execution_history[execution_id] = execution_result
        return execution_result

    except Exception as e:
        end_time = datetime.now()
        duration = int((end_time - start_time).total_seconds() * 1000)

        execution_result = FunctionExecutionResult(
            execution_id=execution_id,
            function_name=request.function_name,
            status="error",
            executed_at=start_time,
            duration_ms=duration,
            error=str(e)
        )

        # In History speichern
        execution_history[execution_id] = execution_result
        return execution_result


def get_orchestrator_tools_safely() -> list[dict[str, Any]]:
    """Holt Orchestrator Tools mit Fehlerbehandlung."""
    if not check_agents_integration():
        return []

    try:
        from agents import get_orchestrator_tools
        tools_map = get_orchestrator_tools()  # Dict[str, data_models.Function]
        adapted: list[dict[str, Any]] = []
        for name, tool in tools_map.items():
            # Tool-Objekte in einfache Dicts konvertieren, inkl. callable
            params = []
            for p in getattr(tool, "parameters", []) or []:
                params.append({
                    "name": getattr(p, "name", "param"),
                    "type": getattr(p, "type", "Any"),
                    "description": getattr(p, "description", ""),
                    "required": getattr(p, "required", True),
                    "default": getattr(p, "default_value", None),
                })
            adapted.append({
                "name": getattr(tool, "name", name),
                "description": getattr(tool, "description", name),
                "parameters": params,
                "return_type": "Any",
                "category": "orchestrator",
                "func": getattr(tool, "func", None),
            })
        # Optionale Hilfstools für Kamera-Workflow ergänzen
        with contextlib.suppress(Exception):
            adapted.append({
                "name": "capture_photo",
                "description": "Capture a photo on demand and return SAS URL",
                "parameters": [
                    {"name": "resolution", "type": "string", "description": "640x480|1280x720|1920x1080", "required": False, "default": "640x480"},
                ],
                "return_type": "object",
                "category": "camera",
                "func": None,
            })

        return adapted
    except Exception as e:
        logger.warning(f"⚠️ Orchestrator Tools nicht verfügbar: {e}")
        return []


# API Endpunkte
@router.get("/", response_model=list[FunctionSchema])
async def list_functions():
    """Listet alle verfügbaren Funktionen."""
    functions = []

    # Built-in Funktionen
    for name, func in BUILT_IN_FUNCTIONS.items():
        schema = get_function_schema(func, name)
        schema.category = "built_in"
        functions.append(schema)

    # Registrierte Tools (inkl. Orchestrator-Tools wie perform_web_research)
    if not registered_tools:
        # Orchestrator-Tools dynamisch registrieren, wenn verfügbar
        orchestrator_tools = get_orchestrator_tools_safely()
        for tool in orchestrator_tools:
            try:
                schema = FunctionSchema(
                    name=tool["name"],
                    description=tool.get("description", tool["name"]),
                    parameters=[
                        FunctionParameter(
                            name=p["name"],
                            type=p.get("type", "Any"),
                            description=p.get("description", p["name"]),
                            required=p.get("required", True),
                            default=p.get("default"),
                        )
                        for p in tool.get("parameters", [])
                    ],
                    return_type=tool.get("return_type", "Any"),
                    category="orchestrator",
                )
                registered_tools[schema.name] = schema
            except Exception as e:
                logger.debug(f"Tool-Registrierung übersprungen: {e}")

    functions.extend(registered_tools.values())

    return functions


@router.get("/{function_name}", response_model=FunctionSchema)
async def get_function_details(function_name: str = Path(..., description="Funktions-Name")):
    """Holt Details einer spezifischen Funktion."""
    # Built-in prüfen
    if function_name in BUILT_IN_FUNCTIONS:
        func = BUILT_IN_FUNCTIONS[function_name]
        schema = get_function_schema(func, function_name)
        schema.category = "built_in"
        return schema

    # Registrierte Tools prüfen
    if function_name in registered_tools:
        return registered_tools[function_name]

    raise HTTPException(status_code=404, detail="Funktion nicht gefunden")


@router.post("/execute", response_model=FunctionExecutionResult)
async def execute_function(request: FunctionExecutionRequest):
    """Führt eine Funktion aus (inkl. orchestrator tools)."""
    # Built-ins/Sync-Wrapper
    if request.function_name in BUILT_IN_FUNCTIONS:
        return await execute_function_safely(request)

    # Orchestrator-Tools zur Laufzeit ausführen
    try:
        orchestrator_tools = get_orchestrator_tools_safely()
        tool_map: dict[str, Any] = {t["name"]: t for t in orchestrator_tools}
        if request.function_name not in tool_map:
            return await execute_function_safely(request)

        # Ausführen
        tool = tool_map[request.function_name]
        func = tool.get("func")
        if not callable(func):
            # Kein direkter Funktionszeiger verfügbar -> Fallback
            return await execute_function_safely(request)

        # Tool kann async oder sync sein
        if inspect.iscoroutinefunction(func):
            result = await func(**(request.parameters or {}))
        else:
            result = func(**(request.parameters or {}))

        # Falls Tool strukturierte Header zurückgibt, in Response übernehmen
        try:
            from fastapi import Response as response_class
            response = response_class
        except Exception:
            response = None  # type: ignore
        if isinstance(result, dict) and result.get("headers") and response is not None:
            # Hinweis: Hier keine Response-Instanz – Wrapper via FastAPI Dependency wäre ideal.
            # Minimal: Header im Result zurückgeben, Frontends/Proxies können diese übernehmen.
            pass


        execution_id = f"exec_{uuid4().hex[:8]}"
        execution_result = FunctionExecutionResult(
            execution_id=execution_id,
            function_name=request.function_name,
            status="success",
            result=result,
            executed_at=datetime.now(UTC),
            duration_ms=0,
        )
        execution_history[execution_id] = execution_result
        return execution_result
    except Exception as e:
        execution_id = f"exec_{uuid4().hex[:8]}"
        execution_result = FunctionExecutionResult(
            execution_id=execution_id,
            function_name=request.function_name,
            status="error",
            executed_at=datetime.now(UTC),
            duration_ms=0,
            error=str(e),
        )
        execution_history[execution_id] = execution_result
        return execution_result


@router.get("/executions/{execution_id}", response_model=FunctionExecutionResult)
async def get_execution_result(execution_id: str = Path(..., description="Ausführungs-ID")):
    """Holt Ergebnis einer Funktions-Ausführung."""
    if execution_id not in execution_history:
        raise HTTPException(status_code=404, detail="Ausführung nicht gefunden")

    return execution_history[execution_id]


@router.get("/categories/")
async def list_function_categories():
    """Listet Funktions-Kategorien."""
    categories = {"built_in": len(BUILT_IN_FUNCTIONS)}

    for tool in registered_tools.values():
        categories[tool.category] = categories.get(tool.category, 0) + 1

    return [
        {"name": cat, "count": count, "type": "built_in" if cat == "built_in" else "external"}
        for cat, count in categories.items()
    ]


@router.get("/health")
async def functions_health_check():
    """Health Check für Function System."""
    orchestrator_tools = get_orchestrator_tools_safely()

    additional_data = {
        "functions": {
            "built_in_functions": len(BUILT_IN_FUNCTIONS),
            "registered_tools": len(registered_tools),
            "total_executions": len(execution_history),
            "orchestrator_tools": len(orchestrator_tools)
        },
        "recent_executions": {
            "total": len(execution_history),
            "success": len([e for e in execution_history.values() if e.status == "success"]),
            "errors": len([e for e in execution_history.values() if e.status == "error"])
        }
    }

    return create_health_response(additional_data)
