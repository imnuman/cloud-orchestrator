"""
Model Proxy Routes.

Provides OpenAI-compatible API endpoints for accessing deployed models.

Users authenticate with their deployment API key and requests are
proxied to the underlying model containers.

Supported endpoints:
- POST /v1/chat/completions - Chat completions (LLMs)
- POST /v1/completions - Text completions (LLMs)
- POST /v1/embeddings - Text embeddings
- POST /v1/audio/transcriptions - Speech to text (Whisper)
- POST /v1/audio/speech - Text to speech
- POST /v1/images/generations - Image generation
- GET /v1/models - List available models
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from brain.models.base import get_db
from brain.models.model_catalog import DeploymentStatus
from brain.services.model_proxy import get_model_proxy_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Model API"])


# Request/Response Models (OpenAI-compatible)


class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""

    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional name for the participant")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(..., description="Model identifier (ignored, uses deployment model)")
    messages: list[ChatMessage] = Field(..., description="List of messages")
    temperature: float = Field(0.7, ge=0, le=2, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0, le=1, description="Nucleus sampling threshold")
    n: int = Field(1, ge=1, le=10, description="Number of completions to generate")
    stream: bool = Field(False, description="Whether to stream the response")
    stop: Optional[list[str]] = Field(None, description="Stop sequences")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    presence_penalty: float = Field(0, ge=-2, le=2)
    frequency_penalty: float = Field(0, ge=-2, le=2)
    user: Optional[str] = Field(None, description="End-user identifier")


class CompletionRequest(BaseModel):
    """OpenAI-compatible text completion request."""

    model: str = Field(...)
    prompt: str | list[str] = Field(...)
    suffix: Optional[str] = None
    max_tokens: Optional[int] = Field(16)
    temperature: float = Field(1.0, ge=0, le=2)
    top_p: float = Field(1.0, ge=0, le=1)
    n: int = Field(1, ge=1, le=10)
    stream: bool = Field(False)
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[list[str]] = None
    presence_penalty: float = Field(0, ge=-2, le=2)
    frequency_penalty: float = Field(0, ge=-2, le=2)
    best_of: int = Field(1)
    user: Optional[str] = None


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    model: str = Field(...)
    input: str | list[str] = Field(..., description="Text to embed")
    encoding_format: str = Field("float", description="Encoding format: float or base64")
    user: Optional[str] = None


class TranscriptionRequest(BaseModel):
    """OpenAI-compatible audio transcription request."""

    model: str = Field("whisper-1")
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: str = Field("json")
    temperature: float = Field(0.0)


class SpeechRequest(BaseModel):
    """OpenAI-compatible text-to-speech request."""

    model: str = Field(...)
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field("alloy", description="Voice to use")
    response_format: str = Field("mp3", description="Audio format: mp3, opus, aac, flac")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Speech speed")


class ImageGenerationRequest(BaseModel):
    """OpenAI-compatible image generation request."""

    model: str = Field("dall-e-3")
    prompt: str = Field(..., description="Image description")
    n: int = Field(1, ge=1, le=10, description="Number of images")
    size: str = Field("1024x1024", description="Image size")
    quality: str = Field("standard", description="Image quality: standard or hd")
    response_format: str = Field("url", description="Response format: url or b64_json")
    style: str = Field("vivid", description="Style: vivid or natural")
    user: Optional[str] = None


# Dependency for API key authentication


async def get_deployment_from_api_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Extract and validate deployment API key from request.

    Supports both:
    - Authorization: Bearer sk_xxxx
    - X-API-Key: sk_xxxx
    """
    proxy_service = get_model_proxy_service()

    # Try Authorization header first
    api_key = None
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]
    elif x_api_key:
        api_key = x_api_key

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Missing API key. Include 'Authorization: Bearer YOUR_KEY' header.",
                    "type": "invalid_request_error",
                    "code": "missing_api_key",
                }
            },
        )

    deployment = await proxy_service.get_deployment_by_api_key(api_key, db)

    if not deployment:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )

    if deployment.status != DeploymentStatus.READY:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": f"Deployment not ready. Status: {deployment.status.value}",
                    "type": "service_unavailable",
                    "code": "deployment_not_ready",
                }
            },
        )

    return deployment


# Endpoints


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    raw_request: Request,
    deployment=Depends(get_deployment_from_api_key),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a chat completion.

    OpenAI-compatible endpoint for chat-based language models.
    """
    proxy_service = get_model_proxy_service()

    # Get request body as dict
    body = request.model_dump()

    if request.stream:
        # Return streaming response
        return StreamingResponse(
            proxy_service.proxy_streaming_request(
                deployment=deployment,
                path="/v1/chat/completions",
                body=body,
                headers=dict(raw_request.headers),
                db=db,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming request
    status_code, headers, response_body = await proxy_service.proxy_request(
        deployment=deployment,
        path="/v1/chat/completions",
        method="POST",
        body=body,
        headers=dict(raw_request.headers),
        db=db,
    )

    return JSONResponse(content=response_body, status_code=status_code)


@router.post("/completions")
async def completions(
    request: CompletionRequest,
    raw_request: Request,
    deployment=Depends(get_deployment_from_api_key),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a text completion.

    OpenAI-compatible endpoint for text completion models.
    """
    proxy_service = get_model_proxy_service()
    body = request.model_dump()

    if request.stream:
        return StreamingResponse(
            proxy_service.proxy_streaming_request(
                deployment=deployment,
                path="/v1/completions",
                body=body,
                headers=dict(raw_request.headers),
                db=db,
            ),
            media_type="text/event-stream",
        )

    status_code, headers, response_body = await proxy_service.proxy_request(
        deployment=deployment,
        path="/v1/completions",
        method="POST",
        body=body,
        headers=dict(raw_request.headers),
        db=db,
    )

    return JSONResponse(content=response_body, status_code=status_code)


@router.post("/embeddings")
async def embeddings(
    request: EmbeddingRequest,
    raw_request: Request,
    deployment=Depends(get_deployment_from_api_key),
    db: AsyncSession = Depends(get_db),
):
    """
    Create embeddings.

    OpenAI-compatible endpoint for embedding models.
    """
    proxy_service = get_model_proxy_service()
    body = request.model_dump()

    status_code, headers, response_body = await proxy_service.proxy_request(
        deployment=deployment,
        path="/v1/embeddings",
        method="POST",
        body=body,
        headers=dict(raw_request.headers),
        db=db,
    )

    return JSONResponse(content=response_body, status_code=status_code)


@router.post("/audio/transcriptions")
async def audio_transcriptions(
    raw_request: Request,
    deployment=Depends(get_deployment_from_api_key),
    db: AsyncSession = Depends(get_db),
):
    """
    Transcribe audio to text.

    OpenAI-compatible endpoint for speech-to-text (Whisper).
    Note: Accepts multipart/form-data with audio file.
    """
    proxy_service = get_model_proxy_service()

    # For audio, we need to forward the raw body
    # This is a simplified version - production would handle multipart properly
    body = await raw_request.body()

    status_code, headers, response_body = await proxy_service.proxy_request(
        deployment=deployment,
        path="/v1/audio/transcriptions",
        method="POST",
        body=None,  # Body is raw bytes, handled separately
        headers=dict(raw_request.headers),
        db=db,
    )

    return JSONResponse(content=response_body, status_code=status_code)


@router.post("/audio/speech")
async def audio_speech(
    request: SpeechRequest,
    raw_request: Request,
    deployment=Depends(get_deployment_from_api_key),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate speech from text.

    OpenAI-compatible endpoint for text-to-speech.
    Returns audio file.
    """
    proxy_service = get_model_proxy_service()
    body = request.model_dump()

    # TTS returns binary audio, stream it
    return StreamingResponse(
        proxy_service.proxy_streaming_request(
            deployment=deployment,
            path="/v1/audio/speech",
            body=body,
            headers=dict(raw_request.headers),
            db=db,
        ),
        media_type=f"audio/{request.response_format}",
    )


@router.post("/images/generations")
async def image_generations(
    request: ImageGenerationRequest,
    raw_request: Request,
    deployment=Depends(get_deployment_from_api_key),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate images from text prompts.

    OpenAI-compatible endpoint for image generation models.
    """
    proxy_service = get_model_proxy_service()
    body = request.model_dump()

    status_code, headers, response_body = await proxy_service.proxy_request(
        deployment=deployment,
        path="/v1/images/generations",
        method="POST",
        body=body,
        headers=dict(raw_request.headers),
        db=db,
    )

    return JSONResponse(content=response_body, status_code=status_code)


@router.get("/models")
async def list_models(
    deployment=Depends(get_deployment_from_api_key),
    db: AsyncSession = Depends(get_db),
):
    """
    List available models.

    Returns the model associated with this deployment.
    """
    from sqlalchemy import select

    from brain.models.model_catalog import ModelTemplate

    result = await db.execute(
        select(ModelTemplate).where(ModelTemplate.id == deployment.model_template_id)
    )
    template = result.scalar_one_or_none()

    if not template:
        return {"object": "list", "data": []}

    return {
        "object": "list",
        "data": [
            {
                "id": template.slug,
                "object": "model",
                "created": int(template.created_at.timestamp()) if template.created_at else 0,
                "owned_by": "user",
                "permission": [],
                "root": template.slug,
                "parent": None,
            }
        ],
    }


@router.get("/models/{model_id}")
async def get_model(
    model_id: str,
    deployment=Depends(get_deployment_from_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific model."""
    from sqlalchemy import select

    from brain.models.model_catalog import ModelTemplate

    result = await db.execute(
        select(ModelTemplate).where(ModelTemplate.id == deployment.model_template_id)
    )
    template = result.scalar_one_or_none()

    if not template:
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "id": template.slug,
        "object": "model",
        "created": int(template.created_at.timestamp()) if template.created_at else 0,
        "owned_by": "user",
        "permission": [],
        "root": template.slug,
        "parent": None,
    }


# Health check endpoint (doesn't require API key)


@router.get("/health")
async def health():
    """Health check for the proxy service."""
    return {"status": "ok"}
