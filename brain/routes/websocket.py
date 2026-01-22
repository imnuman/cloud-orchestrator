"""
WebSocket Routes for Real-Time Voice Streaming.

Provides WebSocket endpoints for:
- Real-time speech-to-text (Whisper)
- Real-time text-to-speech
- Bidirectional voice agent communication

Authentication via query parameter: ?api_key=sk_xxxx
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from brain.models.base import async_session_maker
from brain.models.model_catalog import (
    Deployment,
    DeploymentStatus,
    ModelUsageLog,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket"])


async def authenticate_websocket(api_key: str) -> Optional[Deployment]:
    """
    Authenticate WebSocket connection using API key.

    Args:
        api_key: Deployment API key

    Returns:
        Deployment or None if invalid
    """
    async with async_session_maker() as db:
        result = await db.execute(
            select(Deployment).where(Deployment.api_key == api_key)
        )
        deployment = result.scalar_one_or_none()

        if not deployment:
            return None

        if deployment.status != DeploymentStatus.READY:
            return None

        return deployment


class VoiceStreamManager:
    """
    Manages WebSocket connections for voice streaming.

    Handles:
    - Connection lifecycle
    - Audio chunk processing
    - Bidirectional communication
    """

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, deployment_id: str, websocket: WebSocket):
        """Accept and track a WebSocket connection."""
        await websocket.accept()
        self.active_connections[deployment_id] = websocket
        logger.info(f"WebSocket connected for deployment {deployment_id}")

    def disconnect(self, deployment_id: str):
        """Remove a WebSocket connection."""
        if deployment_id in self.active_connections:
            del self.active_connections[deployment_id]
            logger.info(f"WebSocket disconnected for deployment {deployment_id}")

    async def send_json(self, deployment_id: str, data: dict):
        """Send JSON data to a connection."""
        if deployment_id in self.active_connections:
            await self.active_connections[deployment_id].send_json(data)

    async def send_bytes(self, deployment_id: str, data: bytes):
        """Send binary data to a connection."""
        if deployment_id in self.active_connections:
            await self.active_connections[deployment_id].send_bytes(data)


# Global manager instance
voice_manager = VoiceStreamManager()


@router.websocket("/audio/transcriptions")
async def websocket_transcription(
    websocket: WebSocket,
    api_key: str = Query(..., description="Deployment API key"),
):
    """
    Real-time speech-to-text WebSocket endpoint.

    Protocol:
    1. Connect with ?api_key=sk_xxxx
    2. Send audio chunks as binary messages
    3. Receive transcription results as JSON:
       {"type": "transcription", "text": "...", "is_final": true/false}
    4. Send {"type": "end"} to signal end of audio

    Audio format: PCM 16-bit, 16kHz, mono
    """
    # Authenticate
    deployment = await authenticate_websocket(api_key)
    if not deployment:
        await websocket.close(code=4001, reason="Invalid API key")
        return

    await voice_manager.connect(deployment.id, websocket)

    try:
        # Buffer for accumulating audio chunks
        audio_buffer = bytearray()

        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message:
                # Binary audio data
                audio_chunk = message["bytes"]
                audio_buffer.extend(audio_chunk)

                # Process in chunks (e.g., every 0.5 seconds of audio)
                # 16kHz * 2 bytes * 0.5s = 16000 bytes
                while len(audio_buffer) >= 16000:
                    chunk = bytes(audio_buffer[:16000])
                    audio_buffer = audio_buffer[16000:]

                    # In production, forward to Whisper model
                    # For now, simulate response
                    # result = await forward_to_whisper(deployment, chunk)

                    await websocket.send_json({
                        "type": "transcription",
                        "text": "",  # Partial transcription
                        "is_final": False,
                    })

            elif "text" in message:
                # JSON control message
                data = json.loads(message["text"])

                if data.get("type") == "end":
                    # Process remaining audio
                    if audio_buffer:
                        # Final transcription with remaining audio
                        await websocket.send_json({
                            "type": "transcription",
                            "text": "[Final transcription would appear here]",
                            "is_final": True,
                        })
                    break

                elif data.get("type") == "config":
                    # Client configuration (language, etc.)
                    logger.info(f"Transcription config: {data}")

        # Log usage
        await _log_websocket_usage(deployment.id, deployment.user_id, "transcription")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {deployment.id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        voice_manager.disconnect(deployment.id)


@router.websocket("/audio/speech")
async def websocket_speech(
    websocket: WebSocket,
    api_key: str = Query(..., description="Deployment API key"),
):
    """
    Real-time text-to-speech WebSocket endpoint.

    Protocol:
    1. Connect with ?api_key=sk_xxxx
    2. Send text as JSON: {"type": "synthesize", "text": "Hello", "voice": "alloy"}
    3. Receive audio chunks as binary messages
    4. Receive {"type": "end"} when synthesis completes

    Audio format: PCM 16-bit, 24kHz, mono
    """
    deployment = await authenticate_websocket(api_key)
    if not deployment:
        await websocket.close(code=4001, reason="Invalid API key")
        return

    await voice_manager.connect(deployment.id, websocket)

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "text" in message:
                data = json.loads(message["text"])

                if data.get("type") == "synthesize":
                    text = data.get("text", "")
                    voice = data.get("voice", "alloy")

                    # In production, forward to TTS model and stream audio back
                    # For now, simulate response
                    # async for chunk in forward_to_tts(deployment, text, voice):
                    #     await websocket.send_bytes(chunk)

                    # Send placeholder audio (silence)
                    silence = bytes(2400)  # 50ms of silence at 24kHz
                    for _ in range(10):  # Send 0.5s total
                        await websocket.send_bytes(silence)
                        await asyncio.sleep(0.05)

                    await websocket.send_json({"type": "end"})

                elif data.get("type") == "close":
                    break

        await _log_websocket_usage(deployment.id, deployment.user_id, "speech")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {deployment.id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        voice_manager.disconnect(deployment.id)


@router.websocket("/voice-agent")
async def websocket_voice_agent(
    websocket: WebSocket,
    api_key: str = Query(..., description="Deployment API key"),
):
    """
    Bidirectional voice agent WebSocket endpoint.

    Combines STT + LLM + TTS for real-time voice conversations.

    Protocol:
    1. Connect with ?api_key=sk_xxxx
    2. Send audio chunks as binary (user speech)
    3. Receive JSON events:
       - {"type": "transcription", "text": "..."} - User speech transcribed
       - {"type": "response", "text": "..."} - LLM response text
       - {"type": "audio_start"} - TTS audio starting
       - {"type": "audio_end"} - TTS audio complete
    4. Receive binary audio (agent speech)

    Audio format: PCM 16-bit, 16kHz (input), 24kHz (output)
    """
    deployment = await authenticate_websocket(api_key)
    if not deployment:
        await websocket.close(code=4001, reason="Invalid API key")
        return

    await voice_manager.connect(deployment.id, websocket)

    try:
        audio_buffer = bytearray()
        conversation_history = []

        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message:
                # User audio input
                audio_chunk = message["bytes"]
                audio_buffer.extend(audio_chunk)

                # Process when we have enough audio (~1 second)
                if len(audio_buffer) >= 32000:  # 16kHz * 2 bytes * 1s
                    # 1. Transcribe user speech (STT)
                    user_text = "[User speech transcribed here]"  # In production: Whisper
                    await websocket.send_json({
                        "type": "transcription",
                        "text": user_text,
                    })

                    # 2. Get LLM response
                    conversation_history.append({"role": "user", "content": user_text})
                    assistant_text = "[LLM response here]"  # In production: LLM
                    conversation_history.append({"role": "assistant", "content": assistant_text})

                    await websocket.send_json({
                        "type": "response",
                        "text": assistant_text,
                    })

                    # 3. Synthesize and stream response (TTS)
                    await websocket.send_json({"type": "audio_start"})

                    # In production: Stream TTS audio
                    silence = bytes(2400)
                    for _ in range(20):
                        await websocket.send_bytes(silence)
                        await asyncio.sleep(0.05)

                    await websocket.send_json({"type": "audio_end"})

                    # Clear buffer for next turn
                    audio_buffer.clear()

            elif "text" in message:
                data = json.loads(message["text"])

                if data.get("type") == "config":
                    # Configuration (system prompt, voice, etc.)
                    system_prompt = data.get("system_prompt")
                    if system_prompt:
                        conversation_history.insert(0, {
                            "role": "system",
                            "content": system_prompt,
                        })

                elif data.get("type") == "interrupt":
                    # User interrupted agent - stop current TTS
                    await websocket.send_json({"type": "interrupted"})

                elif data.get("type") == "close":
                    break

        await _log_websocket_usage(deployment.id, deployment.user_id, "voice-agent")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {deployment.id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        voice_manager.disconnect(deployment.id)


async def _log_websocket_usage(
    deployment_id: str,
    user_id: str,
    endpoint_type: str,
) -> None:
    """Log WebSocket session usage."""
    try:
        async with async_session_maker() as db:
            log = ModelUsageLog(
                deployment_id=deployment_id,
                user_id=user_id,
                endpoint=f"/ws/audio/{endpoint_type}",
                method="WEBSOCKET",
                status_code=200,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
            )
            db.add(log)
            await db.commit()
    except Exception as e:
        logger.error(f"Failed to log WebSocket usage: {e}")
