"""
v1 OCR endpoint.

Extracted from the legacy flat router so the route logic lives in a
single place and is referenced by both the /v1 and the legacy /ai mounts.
"""

import base64
import io
import time
import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

import tasks
from config import settings
from schemas.ocr import OCRResponse
from services.ocr_job import run_ocr_from_bytes

router = APIRouter(tags=["ocr"])
limiter = Limiter(key_func=get_remote_address)

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/jpg",
    "image/bmp",
    "image/tiff",
    "image/webp",
}


class QueuedOCRResponse(BaseModel):
    success: bool
    task_id: str
    status: str
    message: str
    status_url: str


@router.post("/ai/ocr")
@limiter.limit(settings.request_rate_limit)
async def process_ocr(
    request: Request,
    image: Annotated[UploadFile, File(description="Image file to process")],
    anchor_metadata: Annotated[
        Optional[str],
        Form(description="JSON encoded AnchorMetadata"),
    ] = None,
) -> OCRResponse:
    """Extract text fields from an uploaded document image."""
    start_time = time.time()
    trace_id = str(uuid.uuid4())

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_content_type",
                "message": (
                    f"Invalid content type: {image.content_type}. "
                    f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}"
                ),
            },
        )

    try:
        contents = await image.read()

        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "empty_image",
                    "message": "Uploaded image is empty",
                },
            )

        _validate_image_bytes(contents)

        # run_ocr_from_bytes returns a dict including anchor_metadata key.
        result_dict = run_ocr_from_bytes(contents, anchor_metadata)

        # Derive top-level confidence from per-field confidences if available.
        data = result_dict.get("data") or {}
        fields = data.get("fields") if isinstance(data, dict) else {}
        field_confidences = (
            [f["confidence"] for f in fields.values() if isinstance(f, dict)]
            if isinstance(fields, dict)
            else []
        )
        avg_confidence = (
            sum(field_confidences) / len(field_confidences)
            if field_confidences
            else None
        )

        reasons = ["OCR extraction completed successfully"]
        if avg_confidence is not None:
            reasons.append(f"Average field confidence: {avg_confidence:.2f}")

        # Strip envelope keys before spreading so our explicit values win.
        _ENVELOPE_KEYS = {
            "result", "confidence", "reasons", "anchor_metadata", "trace_id"
        }
        safe_dict = {k: v for k, v in result_dict.items() if k not in _ENVELOPE_KEYS}

        return OCRResponse(
            **safe_dict,
            # Standard result envelope fields (Issue #609)
            result="ocr_complete",
            confidence=avg_confidence,
            reasons=reasons,
            anchor_metadata=result_dict.get("anchor_metadata"),
            trace_id=trace_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        return OCRResponse(
            success=False,
            error={"code": "processing_error", "message": str(e)},
            processing_time_ms=processing_time_ms,
            result="ocr_error",
            reasons=[str(e)],
            anchor_metadata=None,
            trace_id=trace_id,
        )


@router.post(
    "/ai/ocr/jobs",
    response_model=QueuedOCRResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
@limiter.limit(settings.request_rate_limit)
async def queue_ocr_job(
    request: Request,
    image: Annotated[UploadFile, File(description="Image file to process")],
    anchor_metadata: Annotated[
        Optional[str],
        Form(description="JSON encoded AnchorMetadata"),
    ] = None,
) -> QueuedOCRResponse:
    """Queue OCR processing and return immediately with a pollable job URL."""
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_content_type",
                "message": (
                    f"Invalid content type: {image.content_type}. "
                    f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}"
                ),
            },
        )

    contents = await image.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "empty_image",
                "message": "Uploaded image is empty",
            },
        )

    _validate_image_bytes(contents)

    task_id = tasks.create_task(
        task_type="ocr",
        payload={
            "image_base64": base64.b64encode(contents).decode("ascii"),
            "content_type": image.content_type,
            "filename": image.filename,
            "anchor_metadata": anchor_metadata,
        },
    )

    return QueuedOCRResponse(
        success=True,
        task_id=task_id,
        status="pending",
        message="OCR job queued for processing",
        status_url=f"/v1/ai/jobs/{task_id}",
    )


def _validate_image_bytes(contents: bytes) -> None:
    from PIL import Image

    try:
        Image.open(io.BytesIO(contents)).verify()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_image",
                "message": f"Could not decode image: {str(e)}",
            },
        )
