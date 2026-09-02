"""
FinSight Voice Router — Speech-to-Text (STT) Layer
==================================================

Architectural Boundary & Invariants:
------------------------------------
1. The STT system has exactly ONE responsibility: AUDIO -> TRANSCRIPT TEXT.
2. It NEVER imports or invokes backend.engine.financial_engine.
3. It NEVER calls financial engine functions, selects financial intents, or calculates balances.
4. It NEVER automatically calls the existing /ask pipeline.
5. All uploaded audio is processed strictly in memory and released immediately.
6. Audio is never written to disk or permanently stored.
"""

import os
from typing import Optional
from fastapi import APIRouter, File, Form, UploadFile, status
from fastapi.responses import JSONResponse

from backend.schemas import VoiceTranscribeResponse, VoiceTranscribeErrorResponse
from ai.speech_to_text import transcribe_audio, resolve_mime_type

router = APIRouter(tags=["Voice Speech-to-Text"])

MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB max limit

ALLOWED_EXTENSIONS = {
    ".webm",
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".flac",
    ".aac",
}

# Error type to HTTP status code mapping
ERROR_TYPE_TO_HTTP_STATUS = {
    "empty_audio": status.HTTP_400_BAD_REQUEST,
    "corrupt_audio": status.HTTP_400_BAD_REQUEST,
    "unintelligible_speech": status.HTTP_400_BAD_REQUEST,
    "unsupported_media_type": status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    "rate_limit": status.HTTP_429_TOO_MANY_REQUESTS,
    "service_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
    "timeout": status.HTTP_504_GATEWAY_TIMEOUT,
    "authentication_error": status.HTTP_502_BAD_GATEWAY,
    "network_error": status.HTTP_502_BAD_GATEWAY,
    "provider_error": status.HTTP_502_BAD_GATEWAY,
    "model_not_found": status.HTTP_502_BAD_GATEWAY,
    "malformed_response": status.HTTP_502_BAD_GATEWAY,
    "unexpected_error": status.HTTP_502_BAD_GATEWAY,
}


@router.post(
    "/voice/transcribe",
    response_model=VoiceTranscribeResponse,
    responses={
        400: {"model": VoiceTranscribeErrorResponse, "description": "Invalid, empty, or unintelligible audio"},
        415: {"model": VoiceTranscribeErrorResponse, "description": "Unsupported audio format"},
        429: {"model": VoiceTranscribeErrorResponse, "description": "Rate limit exceeded"},
        502: {"model": VoiceTranscribeErrorResponse, "description": "Provider or network failure"},
        503: {"model": VoiceTranscribeErrorResponse, "description": "Provider temporarily unavailable"},
        504: {"model": VoiceTranscribeErrorResponse, "description": "Provider timeout"},
    },
    summary="Transcribe User Spoken Audio to Text",
    description=(
        "Converts browser-recorded audio into verbatim text. "
        "Preserves original spoken English, Hindi, and Hinglish without translation. "
        "Audio is processed in memory and released immediately after transcription."
    ),
)
@router.post(
    "/api/v1/voice/transcribe",
    response_model=VoiceTranscribeResponse,
    responses={
        400: {"model": VoiceTranscribeErrorResponse},
        415: {"model": VoiceTranscribeErrorResponse},
        429: {"model": VoiceTranscribeErrorResponse},
        502: {"model": VoiceTranscribeErrorResponse},
        503: {"model": VoiceTranscribeErrorResponse},
        504: {"model": VoiceTranscribeErrorResponse},
    },
    include_in_schema=False,
)
async def transcribe_voice(
    audio: UploadFile = File(..., description="Recorded audio file (webm, wav, mp3, m4a, ogg, flac, aac)"),
    language: Optional[str] = Form(None, description="Optional language hint (e.g. 'en', 'hi', 'hi-IN')"),
) -> JSONResponse:
    """
    Speech-to-Text endpoint.
    Accepts multipart/form-data audio file and returns verbatim transcript.
    """
    filename = audio.filename or ""
    content_type = audio.content_type or ""

    # 1. Validate file extension if filename is provided
    if filename:
        ext = os.path.splitext(filename.lower())[1]
        if ext and ext not in ALLOWED_EXTENSIONS:
            return JSONResponse(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                content={
                    "status": "error",
                    "message": (
                        f"Unsupported audio file extension '{ext}'. "
                        f"Supported extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
                    ),
                    "detail": f"Unsupported audio file extension '{ext}'.",
                },
            )

    # 2. Validate content type if provided and extension wasn't decisive
    resolved_mime = resolve_mime_type(content_type=content_type, filename=filename)
    if content_type and not resolved_mime:
        # Check if content_type is completely unrelated (e.g. text/plain, application/json)
        if not content_type.startswith("audio/") and not content_type == "video/webm":
            return JSONResponse(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                content={
                    "status": "error",
                    "message": f"Unsupported audio format '{content_type}'.",
                    "detail": f"Unsupported audio format '{content_type}'.",
                },
            )

    # 3. Read audio bytes in-memory
    try:
        audio_bytes = await audio.read()
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "message": f"Failed to read uploaded audio data: {str(e)}",
                "detail": f"Failed to read uploaded audio data: {str(e)}",
            },
        )
    finally:
        await audio.close()

    # 4. Validate audio size and non-emptiness
    if not audio_bytes or len(audio_bytes) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "message": "The uploaded audio file is empty. Please try speaking again.",
                "detail": "The uploaded audio file is empty.",
            },
        )

    if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
        del audio_bytes
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "message": (
                    f"Audio file size ({len(audio_bytes)} bytes) exceeds the maximum "
                    f"allowed limit of {MAX_AUDIO_SIZE_BYTES // (1024 * 1024)} MB."
                ),
                "detail": "Audio file size exceeds maximum allowed limit.",
            },
        )

    # 5. Execute STT transcription in memory
    try:
        result = transcribe_audio(
            audio_bytes=audio_bytes,
            filename=filename,
            content_type=content_type,
            language=language,
        )
    finally:
        # Immediately release memory buffer
        del audio_bytes

    # 6. Map and return response
    if result.get("status") == "success":
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "transcript": result.get("transcript", ""),
                "language": result.get("language") or language or "en",
            },
        )

    # Error handling mapping
    error_type = result.get("error_type", "unexpected_error")
    http_status = ERROR_TYPE_TO_HTTP_STATUS.get(
        error_type, status.HTTP_502_BAD_GATEWAY
    )
    error_message = result.get(
        "message", "An error occurred during audio transcription."
    )

    return JSONResponse(
        status_code=http_status,
        content={
            "status": "error",
            "message": error_message,
            "detail": error_message,
        },
    )
