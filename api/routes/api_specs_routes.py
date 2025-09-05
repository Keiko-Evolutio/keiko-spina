# backend/api/routes/api_specs_routes.py
"""API-Endpoints für Spezifikations-Management und -Generierung.

Stellt Endpoints für OpenAPI, AsyncAPI und MCP-Spezifikationen bereit
mit automatischer Generierung und Validierung.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import yaml
from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from kei_logging import (
    BusinessLogicError,
    LogLinkedError,
    ValidationError,
    get_logger,
    with_log_links,
)
from specs.api_spec_generator import (
    APISpecificationGenerator,
    get_api_spec_generator,
)

logger = get_logger(__name__)

# Router erstellen
router = APIRouter(prefix="/api/v1/specs", tags=["API Specifications"])


# Request/Response Models

class SpecificationGenerationRequest(BaseModel):
    """Request-Model für Spezifikations-Generierung."""
    spec_types: list[str] = Field(
        default=["openapi", "asyncapi", "mcp"],
        description="Zu generierende Spezifikations-Typen"
    )
    include_enhanced: bool = Field(
        default=True,
        description="Enhanced Management APIs einschließen"
    )
    include_examples: bool = Field(
        default=True,
        description="Beispiele einschließen"
    )
    include_security: bool = Field(
        default=True,
        description="Security-Schemas einschließen"
    )
    output_format: str = Field(
        default="yaml",
        description="Ausgabeformat (yaml, json)"
    )


class SpecificationMetadata(BaseModel):
    """Metadaten für API-Spezifikationen."""
    title: str = Field(..., description="Spezifikations-Titel")
    version: str = Field(..., description="Spezifikations-Version")
    description: str = Field(..., description="Spezifikations-Beschreibung")
    spec_type: str = Field(..., description="Spezifikations-Typ")
    generated_at: datetime = Field(..., description="Generierungszeitpunkt")
    generator_version: str = Field(..., description="Generator-Version")
    paths_count: int | None = Field(None, description="Anzahl API-Pfade")
    schemas_count: int | None = Field(None, description="Anzahl Schemas")
    channels_count: int | None = Field(None, description="Anzahl AsyncAPI-Channels")
    capabilities_count: int | None = Field(None, description="Anzahl MCP-Capabilities")


class SpecificationValidationResult(BaseModel):
    """Ergebnis der Spezifikations-Validierung."""
    valid: bool = Field(..., description="Validierungsstatus")
    spec_type: str = Field(..., description="Spezifikations-Typ")
    errors: list[str] = Field(default_factory=list, description="Validierungsfehler")
    warnings: list[str] = Field(default_factory=list, description="Validierungswarnungen")
    metadata: SpecificationMetadata | None = Field(None, description="Spezifikations-Metadaten")


class SpecificationListResponse(BaseModel):
    """Response für Spezifikations-Liste."""
    specifications: list[SpecificationMetadata] = Field(..., description="Verfügbare Spezifikationen")
    total_count: int = Field(..., description="Gesamtanzahl Spezifikationen")
    last_generated: datetime | None = Field(None, description="Letzte Generierung")


# Dependency-Funktionen

async def get_spec_generator() -> APISpecificationGenerator:
    """Dependency für API-Spezifikations-Generator."""
    from main import app  # Import hier um zirkuläre Imports zu vermeiden
    return get_api_spec_generator(app)


# API-Endpoints

@router.get("/", response_model=SpecificationListResponse)
@with_log_links(component="api_specs", operation="list_specifications")
async def list_specifications(
    spec_type: str | None = Query(None, description="Filter nach Spezifikations-Typ"),
    generator: APISpecificationGenerator = Depends(get_spec_generator)
) -> SpecificationListResponse:
    """Listet alle verfügbaren API-Spezifikationen auf.

    Args:
        spec_type: Filter nach Spezifikations-Typ (openapi, asyncapi, mcp)
        generator: API-Spezifikations-Generator

    Returns:
        Liste verfügbarer Spezifikationen
    """
    try:
        specifications = []

        # OpenAPI-Spezifikationen
        if not spec_type or spec_type == "openapi":
            try:
                openapi_spec = generator.generate_openapi_spec()

                metadata = SpecificationMetadata(
                    title=openapi_spec.get("info", {}).get("title", "KEI API"),
                    version=openapi_spec.get("info", {}).get("version", "1.0.0"),
                    description=openapi_spec.get("info", {}).get("description", ""),
                    spec_type="openapi",
                    generated_at=datetime.now(UTC),
                    generator_version="1.0.0",
                    paths_count=len(openapi_spec.get("paths", {})),
                    schemas_count=len(openapi_spec.get("components", {}).get("schemas", {}))
                )

                specifications.append(metadata)

            except Exception as e:
                logger.warning(f"Konnte OpenAPI-Spezifikation nicht laden: {e}")

        # AsyncAPI-Spezifikationen
        if not spec_type or spec_type == "asyncapi":
            try:
                asyncapi_spec = generator.generate_asyncapi_spec()

                metadata = SpecificationMetadata(
                    title=asyncapi_spec.get("info", {}).get("title", "KEI Events API"),
                    version=asyncapi_spec.get("info", {}).get("version", "1.0.0"),
                    description=asyncapi_spec.get("info", {}).get("description", ""),
                    spec_type="asyncapi",
                    generated_at=datetime.now(UTC),
                    generator_version="1.0.0",
                    channels_count=len(asyncapi_spec.get("channels", {}))
                )

                specifications.append(metadata)

            except Exception as e:
                logger.warning(f"Konnte AsyncAPI-Spezifikation nicht laden: {e}")

        # MCP-Profile
        if not spec_type or spec_type == "mcp":
            try:
                mcp_profiles = generator.generate_mcp_profiles()

                metadata = SpecificationMetadata(
                    title="KEI MCP Capabilities",
                    version="1.0.0",
                    description="Model Context Protocol Capability Profiles",
                    spec_type="mcp",
                    generated_at=datetime.now(UTC),
                    generator_version="1.0.0",
                    capabilities_count=len(mcp_profiles)
                )

                specifications.append(metadata)

            except Exception as e:
                logger.warning(f"Konnte MCP-Profile nicht laden: {e}")

        return SpecificationListResponse(
            specifications=specifications,
            total_count=len(specifications),
            last_generated=datetime.now(UTC) if specifications else None
        )

    except Exception as e:
        raise BusinessLogicError(
            message=f"Spezifikations-Auflistung fehlgeschlagen: {e}",
            component="api_specs",
            operation="list_specifications",
            cause=e
        )


@router.post("/generate", response_model=dict[str, Any])
@with_log_links(component="api_specs", operation="generate_specifications")
async def generate_specifications(
    request: SpecificationGenerationRequest,
    generator: APISpecificationGenerator = Depends(get_spec_generator)
) -> dict[str, Any]:
    """Generiert API-Spezifikationen basierend auf aktueller Codebase.

    Args:
        request: Generierungs-Request
        generator: API-Spezifikations-Generator

    Returns:
        Generierte Spezifikationen
    """
    try:
        generated_specs = {}
        generation_metadata = {
            "generated_at": datetime.now(UTC).isoformat(),
            "generator_version": "1.0.0",
            "request_config": request.dict()
        }

        # OpenAPI-Spezifikation
        if "openapi" in request.spec_types:
            try:
                openapi_spec = generator.generate_openapi_spec(
                    include_enhanced=request.include_enhanced,
                    include_examples=request.include_examples,
                    include_security=request.include_security
                )

                generated_specs["openapi"] = openapi_spec
                generation_metadata["openapi_paths_count"] = len(openapi_spec.get("paths", {}))
                generation_metadata["openapi_schemas_count"] = len(
                    openapi_spec.get("components", {}).get("schemas", {})
                )

            except Exception as e:
                logger.exception(f"OpenAPI-Generierung fehlgeschlagen: {e}")
                generated_specs["openapi"] = {"error": str(e)}

        # AsyncAPI-Spezifikation
        if "asyncapi" in request.spec_types:
            try:
                asyncapi_spec = generator.generate_asyncapi_spec()
                generated_specs["asyncapi"] = asyncapi_spec
                generation_metadata["asyncapi_channels_count"] = len(asyncapi_spec.get("channels", {}))

            except Exception as e:
                logger.exception(f"AsyncAPI-Generierung fehlgeschlagen: {e}")
                generated_specs["asyncapi"] = {"error": str(e)}

        # MCP-Profile
        if "mcp" in request.spec_types:
            try:
                mcp_profiles = generator.generate_mcp_profiles()
                generated_specs["mcp"] = mcp_profiles
                generation_metadata["mcp_capabilities_count"] = len(mcp_profiles)

            except Exception as e:
                logger.exception(f"MCP-Generierung fehlgeschlagen: {e}")
                generated_specs["mcp"] = {"error": str(e)}

        result = {
            "specifications": generated_specs,
            "metadata": generation_metadata,
            "success": True
        }

        logger.info(
            "API-Spezifikationen generiert",
            extra={
                "spec_types": request.spec_types,
                "include_enhanced": request.include_enhanced,
                "specs_generated": len(generated_specs),
                **{k: v for k, v in generation_metadata.items() if k.endswith("_count")}
            }
        )

        return result

    except Exception as e:
        raise BusinessLogicError(
            message=f"Spezifikations-Generierung fehlgeschlagen: {e}",
            component="api_specs",
            operation="generate_specifications",
            spec_types=request.spec_types,
            cause=e
        )


@router.get("/openapi", response_class=Response)
@with_log_links(component="api_specs", operation="get_openapi")
async def get_openapi_specification(
    spec_format: str = Query(default="yaml", description="Ausgabeformat (yaml, json)"),
    include_enhanced: bool = Query(default=True, description="Enhanced APIs einschließen"),
    include_examples: bool = Query(default=True, description="Beispiele einschließen"),
    generator: APISpecificationGenerator = Depends(get_spec_generator)
) -> Response:
    """Holt OpenAPI-Spezifikation.

    Args:
        spec_format: Ausgabeformat (yaml, json)
        include_enhanced: Enhanced APIs einschließen
        include_examples: Beispiele einschließen
        generator: API-Spezifikations-Generator

    Returns:
        OpenAPI-Spezifikation
    """
    try:
        openapi_spec = generator.generate_openapi_spec(
            include_enhanced=include_enhanced,
            include_examples=include_examples
        )

        if spec_format.lower() == "json":
            return JSONResponse(
                content=openapi_spec,
                headers={"Content-Type": "application/json"}
            )
        yaml_content = yaml.dump(openapi_spec, default_flow_style=False, allow_unicode=True)
        return Response(
            content=yaml_content,
            media_type="application/x-yaml",
            headers={"Content-Disposition": "attachment; filename=openapi.yaml"}
        )

    except Exception as e:
        raise BusinessLogicError(
            message=f"OpenAPI-Spezifikation konnte nicht abgerufen werden: {e}",
            component="api_specs",
            operation="get_openapi",
            format=spec_format,
            cause=e
        )


@router.get("/asyncapi", response_class=Response)
@with_log_links(component="api_specs", operation="get_asyncapi")
async def get_asyncapi_specification(
    spec_format: str = Query(default="yaml", description="Ausgabeformat (yaml, json)"),
    generator: APISpecificationGenerator = Depends(get_spec_generator)
) -> Response:
    """Holt AsyncAPI-Spezifikation.

    Args:
        spec_format: Ausgabeformat (yaml, json)
        generator: API-Spezifikations-Generator

    Returns:
        AsyncAPI-Spezifikation
    """
    try:
        asyncapi_spec = generator.generate_asyncapi_spec()

        if spec_format.lower() == "json":
            return JSONResponse(
                content=asyncapi_spec,
                headers={"Content-Type": "application/json"}
            )
        yaml_content = yaml.dump(asyncapi_spec, default_flow_style=False, allow_unicode=True)
        return Response(
            content=yaml_content,
            media_type="application/x-yaml",
            headers={"Content-Disposition": "attachment; filename=asyncapi.yaml"}
        )

    except Exception as e:
        raise BusinessLogicError(
            message=f"AsyncAPI-Spezifikation konnte nicht abgerufen werden: {e}",
            component="api_specs",
            operation="get_asyncapi",
            format=spec_format,
            cause=e
        )


@router.get("/mcp", response_model=dict[str, Any])
@with_log_links(component="api_specs", operation="get_mcp_profiles")
async def get_mcp_profiles(
    capability: str | None = Query(None, description="Spezifische Capability"),
    generator: APISpecificationGenerator = Depends(get_spec_generator)
) -> dict[str, Any]:
    """Holt MCP-Capability-Profile.

    Args:
        capability: Spezifische Capability (optional)
        generator: API-Spezifikations-Generator

    Returns:
        MCP-Profile
    """
    try:
        mcp_profiles = generator.generate_mcp_profiles()

        if capability:
            if capability not in mcp_profiles:
                raise ValidationError(
                    message=f"MCP-Capability '{capability}' nicht gefunden",
                    field="capability",
                    value=capability,
                    available_capabilities=list(mcp_profiles.keys())
                )

            return {capability: mcp_profiles[capability]}

        return mcp_profiles

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"MCP-Profile konnten nicht abgerufen werden: {e}",
            component="api_specs",
            operation="get_mcp_profiles",
            capability=capability,
            cause=e
        )


@router.post("/validate", response_model=list[SpecificationValidationResult])
@with_log_links(component="api_specs", operation="validate_specifications")
async def validate_specifications(
    spec_types: list[str] = Query(default=["openapi", "asyncapi", "mcp"], description="Zu validierende Spezifikations-Typen"),
    generator: APISpecificationGenerator = Depends(get_spec_generator)
) -> list[SpecificationValidationResult]:
    """Validiert API-Spezifikationen.

    Args:
        spec_types: Zu validierende Spezifikations-Typen
        generator: API-Spezifikations-Generator

    Returns:
        Validierungsergebnisse
    """
    try:
        validation_results = []

        for spec_type in spec_types:
            result = SpecificationValidationResult(
                valid=True,
                spec_type=spec_type,
                errors=[],
                warnings=[]
            )

            try:
                if spec_type == "openapi":
                    spec = generator.generate_openapi_spec()
                    # Validiere OpenAPI-Spezifikation
                    generator._validate_openapi_spec(spec)

                    result.metadata = SpecificationMetadata(
                        title=spec.get("info", {}).get("title", ""),
                        version=spec.get("info", {}).get("version", ""),
                        description=spec.get("info", {}).get("description", ""),
                        spec_type="openapi",
                        generated_at=datetime.now(UTC),
                        generator_version="1.0.0",
                        paths_count=len(spec.get("paths", {})),
                        schemas_count=len(spec.get("components", {}).get("schemas", {}))
                    )

                elif spec_type == "asyncapi":
                    spec = generator.generate_asyncapi_spec()
                    # TODO: Implementiere AsyncAPI-Validierung - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/109

                    result.metadata = SpecificationMetadata(
                        title=spec.get("info", {}).get("title", ""),
                        version=spec.get("info", {}).get("version", ""),
                        description=spec.get("info", {}).get("description", ""),
                        spec_type="asyncapi",
                        generated_at=datetime.now(UTC),
                        generator_version="1.0.0",
                        channels_count=len(spec.get("channels", {}))
                    )

                elif spec_type == "mcp":
                    profiles = generator.generate_mcp_profiles()
                    # TODO: Implementiere MCP-Validierung - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/109

                    result.metadata = SpecificationMetadata(
                        title="MCP Capabilities",
                        version="1.0.0",
                        description="Model Context Protocol Profiles",
                        spec_type="mcp",
                        generated_at=datetime.now(UTC),
                        generator_version="1.0.0",
                        capabilities_count=len(profiles)
                    )

                else:
                    result.valid = False
                    result.errors.append(f"Unbekannter Spezifikations-Typ: {spec_type}")

            except ValidationError as e:
                result.valid = False
                result.errors.append(str(e))

            except Exception as e:
                result.valid = False
                result.errors.append(f"Validierungsfehler: {e}")

            validation_results.append(result)

        logger.info(
            "Spezifikations-Validierung abgeschlossen",
            extra={
                "spec_types": spec_types,
                "valid_specs": sum(1 for r in validation_results if r.valid),
                "invalid_specs": sum(1 for r in validation_results if not r.valid)
            }
        )

        return validation_results

    except Exception as e:
        raise BusinessLogicError(
            message=f"Spezifikations-Validierung fehlgeschlagen: {e}",
            component="api_specs",
            operation="validate_specifications",
            spec_types=spec_types,
            cause=e
        )
