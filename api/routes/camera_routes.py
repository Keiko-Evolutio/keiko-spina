"""REST endpoints for camera capture and client photo upload to Azure Blob Storage.

POST /api/camera/capture
 - Captures a photo using the server camera (OpenCV), applies rate limiting,
   uploads the image to Azure Blob Storage and returns a SAS URL with metadata.

POST /api/camera/upload
 - Accepts a client-captured photo and stores it in Azure. Useful when the server
   cannot access a camera device.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from config.settings import settings
from kei_logging import get_logger
from services.camera.camera_service import (
    SUPPORTED_RESOLUTIONS,
    CameraAccessDeniedError,
    CameraInitializationError,
    CameraService,
)
from services.limits.rate_limiter import check_camera_limits_per_minute
from storage.azure_blob_storage.azure_blob_storage import generate_sas_url, upload_image_bytes

logger = get_logger(__name__)

router = APIRouter(prefix="/api/camera", tags=["camera"])


class CaptureRequest(BaseModel):
    """Input payload for server-side camera capture."""

    resolution: str = Field(default="640x480", description="640x480|1280x720|1920x1080")


class CaptureResponse(BaseModel):
    """Response payload for photo capture and upload APIs."""

    status: str
    image_url: str
    metadata: dict[str, Any]


def _select_resolution(res: str) -> tuple[int, int]:
    """Selects a whitelisted resolution or falls back to default."""
    if res not in SUPPORTED_RESOLUTIONS:
        return SUPPORTED_RESOLUTIONS["640x480"]
    return SUPPORTED_RESOLUTIONS[res]


def _sanitize_blob_name(name: str) -> str:
    """Sanitizes file names for safe blob paths."""
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_/."
    return "".join(c for c in name if c in allowed)


def _resolve_user_id(request: Request, header_user_id: str | None) -> str | None:
    """Resolves user id from JWT payload, header fallback, or development default."""
    # 1. Try JWT payload
    user_payload: dict[str, Any] | None = getattr(request.state, "user", None)
    if user_payload and "User_ID" in user_payload:
        uid = str(user_payload["User_ID"]).strip()
        if uid:
            return uid

    # 2. Try X-User-Id header
    if header_user_id:
        uid = header_user_id.strip()
        if uid:
            return uid

    # 3. Development fallback: Use default user ID
    from config.settings import settings
    if settings.environment == "development":
        return "dev_user_default"

    return None


@router.post("/capture", response_model=CaptureResponse)
async def capture_photo_endpoint(
    request: Request,
    body: CaptureRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> CaptureResponse:
    """Captures a photo using the server's camera and returns a SAS URL."""
    user_id = _resolve_user_id(request, x_user_id)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    allowed, _headers, retry_after = await check_camera_limits_per_minute(user_id)
    if not allowed:
        detail = {"message": "Rate limit exceeded", "retry_after": retry_after}
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=detail)

    resolution = _select_resolution(body.resolution)

    try:
        with CameraService(resolution=resolution, init_timeout_s=10.0) as cam:
            data, meta = cam.capture_frame()
    except CameraAccessDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except CameraInitializationError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    content_type = meta.get("format", "image/jpeg")
    ext = ".jpg" if content_type == "image/jpeg" else ".png"

    now = datetime.now(UTC)
    date_path = now.strftime("%Y/%m/%d")
    ts = now.strftime("%Y%m%dT%H%M%S%fZ")
    filename = _sanitize_blob_name(f"{ts}{ext}")
    blob_name = f"camera/{user_id}/{date_path}/{filename}"

    container = settings.keiko_storage_container_for_img

    try:
        url = await upload_image_bytes(
            container_name=container,
            blob_name=blob_name,
            data=data,
            content_type=content_type,
            overwrite=True,
            timeout_seconds=30,
        )
    except Exception as e:
        logger.exception(f"Upload failed: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Storage upload failed")

    try:
        sas_url = await generate_sas_url(container_name=container, blob_name=blob_name, expiry_minutes=30)
    except Exception as e:
        logger.warning(f"SAS generation failed, using public URL: {e}")
        sas_url = url

    return CaptureResponse(status="ok", image_url=sas_url, metadata=meta)


@router.post("/upload", response_model=CaptureResponse)
async def upload_photo_endpoint(
    request: Request,
    file: UploadFile = File(...),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> CaptureResponse:
    """Accepts a client-captured photo and stores it in Azure; returns a SAS URL."""
    user_id = _resolve_user_id(request, x_user_id)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    allowed, _headers, retry_after = await check_camera_limits_per_minute(user_id)
    if not allowed:
        detail = {"message": "Rate limit exceeded", "retry_after": retry_after}
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=detail)

    try:
        data = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid file: {e}")

    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 5MB limit")

    content_type = file.content_type or "image/jpeg"
    ext = ".jpg" if content_type == "image/jpeg" else ".png"

    now = datetime.now(UTC)
    date_path = now.strftime("%Y/%m/%d")
    ts = now.strftime("%Y%m%dT%H%M%S%fZ")
    filename = _sanitize_blob_name(f"{ts}{ext}")
    blob_name = f"camera/{user_id}/{date_path}/{filename}"
    container = settings.keiko_storage_container_for_img

    try:
        url = await upload_image_bytes(
            container_name=container,
            blob_name=blob_name,
            data=data,
            content_type=content_type,
            overwrite=True,
            timeout_seconds=30,
        )
    except Exception as e:
        logger.exception(f"Upload failed: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Storage upload failed")

    try:
        sas_url = await generate_sas_url(container_name=container, blob_name=blob_name, expiry_minutes=30)
    except Exception as e:
        logger.warning(f"SAS generation failed, using public URL: {e}")
        sas_url = url

    metadata = {
        "timestamp": now.isoformat(),
        "resolution": None,
        "format": content_type,
        "file_size": len(data),
    }
    return CaptureResponse(status="ok", image_url=sas_url, metadata=metadata)
