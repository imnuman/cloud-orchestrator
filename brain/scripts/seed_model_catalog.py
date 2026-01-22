"""
Seed script for the Model Catalog.

Run this script to populate the database with pre-built model templates.

Usage:
    python -m brain.scripts.seed_model_catalog
"""

import asyncio
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from brain.models.base import async_session_maker, init_db
from brain.models.model_catalog import (
    ModelTemplate,
    ModelCategory,
    ServingBackend,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Template Definitions
# =============================================================================

MODEL_TEMPLATES = [
    # =========================================================================
    # TEXT/LLM MODELS
    # =========================================================================
    {
        "name": "Llama 3.1 8B Instruct",
        "slug": "llama-3.1-8b",
        "version": "3.1",
        "category": ModelCategory.TEXT,
        "serving_backend": ServingBackend.VLLM,
        "min_vram_gb": 16,
        "recommended_vram_gb": 24,
        "min_gpu_count": 1,
        "recommended_gpu": "RTX 4090",
        "supports_quantization": True,
        "docker_image": "vllm/vllm-openai",
        "docker_tag": "latest",
        "default_env": {
            "MODEL": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "MAX_MODEL_LEN": "8192",
        },
        "default_ports": {"8000": "API"},
        "health_check_url": "/health",
        "startup_timeout_seconds": 300,
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "model_source": "huggingface",
        "api_type": "openai",
        "openai_compatible": True,
        "description": "Meta's Llama 3.1 8B Instruct model. Fast and efficient for most tasks.",
        "documentation_url": "https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct",
        "license": "Llama 3.1 Community License",
        "is_featured": True,
        "display_order": 1,
    },
    {
        "name": "Llama 3.1 70B Instruct",
        "slug": "llama-3.1-70b",
        "version": "3.1",
        "category": ModelCategory.TEXT,
        "serving_backend": ServingBackend.VLLM,
        "min_vram_gb": 140,  # 70B × 2 bytes = 140GB for FP16, or use quantization
        "recommended_vram_gb": 160,
        "min_gpu_count": 2,  # Requires 2x A100 80GB or 4x A100 40GB
        "recommended_gpu": "2x A100 80GB",
        "supports_quantization": True,
        "docker_image": "vllm/vllm-openai",
        "docker_tag": "latest",
        "default_env": {
            "MODEL": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "TENSOR_PARALLEL_SIZE": "1",
            "MAX_MODEL_LEN": "8192",
        },
        "default_ports": {"8000": "API"},
        "health_check_url": "/health",
        "startup_timeout_seconds": 600,
        "model_id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "model_source": "huggingface",
        "api_type": "openai",
        "openai_compatible": True,
        "description": "Meta's Llama 3.1 70B Instruct. State-of-the-art open model for complex reasoning.",
        "documentation_url": "https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct",
        "license": "Llama 3.1 Community License",
        "is_featured": True,
        "display_order": 2,
    },
    {
        "name": "Mistral 7B Instruct v0.3",
        "slug": "mistral-7b-instruct",
        "version": "0.3",
        "category": ModelCategory.TEXT,
        "serving_backend": ServingBackend.VLLM,
        "min_vram_gb": 16,
        "recommended_vram_gb": 24,
        "min_gpu_count": 1,
        "recommended_gpu": "RTX 4090",
        "supports_quantization": True,
        "docker_image": "vllm/vllm-openai",
        "docker_tag": "latest",
        "default_env": {
            "MODEL": "mistralai/Mistral-7B-Instruct-v0.3",
            "MAX_MODEL_LEN": "32768",
        },
        "default_ports": {"8000": "API"},
        "health_check_url": "/health",
        "startup_timeout_seconds": 300,
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "model_source": "huggingface",
        "api_type": "openai",
        "openai_compatible": True,
        "description": "Mistral's 7B Instruct model with 32K context. Excellent for long-context tasks.",
        "documentation_url": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3",
        "license": "Apache 2.0",
        "is_featured": True,
        "display_order": 3,
    },
    {
        "name": "Qwen2.5 72B Instruct",
        "slug": "qwen2.5-72b",
        "version": "2.5",
        "category": ModelCategory.TEXT,
        "serving_backend": ServingBackend.VLLM,
        "min_vram_gb": 144,  # 72B × 2 bytes = 144GB for FP16
        "recommended_vram_gb": 160,
        "min_gpu_count": 2,  # Requires 2x A100 80GB
        "recommended_gpu": "2x A100 80GB",
        "supports_quantization": True,
        "docker_image": "vllm/vllm-openai",
        "docker_tag": "latest",
        "default_env": {
            "MODEL": "Qwen/Qwen2.5-72B-Instruct",
            "MAX_MODEL_LEN": "32768",
        },
        "default_ports": {"8000": "API"},
        "health_check_url": "/health",
        "startup_timeout_seconds": 600,
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "model_source": "huggingface",
        "api_type": "openai",
        "openai_compatible": True,
        "description": "Alibaba's Qwen2.5 72B. Strong multilingual and coding capabilities.",
        "documentation_url": "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct",
        "license": "Qwen License",
        "is_featured": True,
        "display_order": 4,
    },
    {
        "name": "Llama 3.1 70B Instruct (AWQ 4-bit)",
        "slug": "llama-3.1-70b-awq",
        "version": "3.1",
        "category": ModelCategory.TEXT,
        "serving_backend": ServingBackend.VLLM,
        "min_vram_gb": 40,  # 4-bit quantized fits in ~40GB
        "recommended_vram_gb": 48,
        "min_gpu_count": 1,
        "recommended_gpu": "A100 40GB",
        "supports_quantization": True,
        "docker_image": "vllm/vllm-openai",
        "docker_tag": "latest",
        "default_env": {
            "MODEL": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
            "QUANTIZATION": "awq",
            "MAX_MODEL_LEN": "8192",
        },
        "default_ports": {"8000": "API"},
        "health_check_url": "/health",
        "startup_timeout_seconds": 600,
        "model_id": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "model_source": "huggingface",
        "api_type": "openai",
        "openai_compatible": True,
        "description": "Llama 3.1 70B with AWQ 4-bit quantization. Runs on single A100 40GB.",
        "documentation_url": "https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "license": "Llama 3.1 Community License",
        "is_featured": True,
        "display_order": 3,
    },
    {
        "name": "DeepSeek-V2.5",
        "slug": "deepseek-v2.5",
        "version": "2.5",
        "category": ModelCategory.TEXT,
        "serving_backend": ServingBackend.VLLM,
        "min_vram_gb": 40,
        "recommended_vram_gb": 80,
        "min_gpu_count": 1,
        "recommended_gpu": "A100 80GB",
        "supports_quantization": True,
        "docker_image": "vllm/vllm-openai",
        "docker_tag": "latest",
        "default_env": {
            "MODEL": "deepseek-ai/DeepSeek-V2.5",
            "TRUST_REMOTE_CODE": "true",
        },
        "default_ports": {"8000": "API"},
        "health_check_url": "/health",
        "startup_timeout_seconds": 600,
        "model_id": "deepseek-ai/DeepSeek-V2.5",
        "model_source": "huggingface",
        "api_type": "openai",
        "openai_compatible": True,
        "description": "DeepSeek's V2.5 MoE model. Excellent reasoning at efficient cost.",
        "documentation_url": "https://huggingface.co/deepseek-ai/DeepSeek-V2.5",
        "license": "DeepSeek License",
        "is_featured": False,
        "display_order": 5,
    },
    # =========================================================================
    # CODE MODELS
    # =========================================================================
    {
        "name": "DeepSeek Coder V2",
        "slug": "deepseek-coder-v2",
        "version": "2.0",
        "category": ModelCategory.CODE,
        "serving_backend": ServingBackend.VLLM,
        "min_vram_gb": 24,
        "recommended_vram_gb": 48,
        "min_gpu_count": 1,
        "recommended_gpu": "RTX 4090",
        "supports_quantization": True,
        "docker_image": "vllm/vllm-openai",
        "docker_tag": "latest",
        "default_env": {
            "MODEL": "deepseek-ai/DeepSeek-Coder-V2-Instruct",
            "TRUST_REMOTE_CODE": "true",
        },
        "default_ports": {"8000": "API"},
        "health_check_url": "/health",
        "startup_timeout_seconds": 600,
        "model_id": "deepseek-ai/DeepSeek-Coder-V2-Instruct",
        "model_source": "huggingface",
        "api_type": "openai",
        "openai_compatible": True,
        "description": "DeepSeek's specialized code model. State-of-the-art code generation.",
        "documentation_url": "https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct",
        "license": "DeepSeek License",
        "is_featured": True,
        "display_order": 10,
    },
    # =========================================================================
    # IMAGE GENERATION MODELS
    # =========================================================================
    {
        "name": "SDXL 1.0",
        "slug": "sdxl",
        "version": "1.0",
        "category": ModelCategory.IMAGE,
        "serving_backend": ServingBackend.COMFYUI,
        "min_vram_gb": 12,
        "recommended_vram_gb": 16,
        "min_gpu_count": 1,
        "recommended_gpu": "RTX 4090",
        "supports_quantization": False,
        "docker_image": "ghcr.io/ai-dock/comfyui",
        "docker_tag": "latest",
        "default_env": {
            "WEB_ENABLE_AUTH": "false",
            "CF_QUICK_TUNNELS": "false",
        },
        "default_ports": {"8188": "API", "7860": "UI"},
        "health_check_url": "/",
        "startup_timeout_seconds": 300,
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "model_source": "huggingface",
        "api_type": "comfyui",
        "openai_compatible": False,
        "description": "Stable Diffusion XL 1.0. High-quality 1024x1024 image generation.",
        "documentation_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
        "license": "CreativeML Open RAIL++-M License",
        "is_featured": True,
        "display_order": 20,
    },
    {
        "name": "Flux.1 Dev",
        "slug": "flux-dev",
        "version": "1.0",
        "category": ModelCategory.IMAGE,
        "serving_backend": ServingBackend.COMFYUI,
        "min_vram_gb": 24,
        "recommended_vram_gb": 48,
        "min_gpu_count": 1,
        "recommended_gpu": "A100 40GB",
        "supports_quantization": False,
        "docker_image": "ghcr.io/ai-dock/comfyui",
        "docker_tag": "latest",
        "default_env": {
            "WEB_ENABLE_AUTH": "false",
        },
        "default_ports": {"8188": "API", "7860": "UI"},
        "health_check_url": "/",
        "startup_timeout_seconds": 600,
        "model_id": "black-forest-labs/FLUX.1-dev",
        "model_source": "huggingface",
        "api_type": "comfyui",
        "openai_compatible": False,
        "description": "Black Forest Labs Flux.1 Dev. State-of-the-art image generation.",
        "documentation_url": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
        "license": "FLUX.1 [dev] Non-Commercial License",
        "is_featured": True,
        "display_order": 21,
    },
    # =========================================================================
    # AUDIO/VOICE MODELS (Phase 2C)
    # =========================================================================
    {
        "name": "Whisper Large V3",
        "slug": "whisper-large-v3",
        "version": "3.0",
        "category": ModelCategory.AUDIO,
        "serving_backend": ServingBackend.WHISPER,
        "min_vram_gb": 8,
        "recommended_vram_gb": 12,
        "min_gpu_count": 1,
        "recommended_gpu": "RTX 3080",
        "supports_quantization": False,
        "docker_image": "fedirz/faster-whisper-server",
        "docker_tag": "latest",
        "default_env": {
            "WHISPER__MODEL": "Systran/faster-whisper-large-v3",
            "WHISPER__INFERENCE_DEVICE": "cuda",
        },
        "default_ports": {"8000": "API"},
        "health_check_url": "/health",
        "startup_timeout_seconds": 180,
        "model_id": "openai/whisper-large-v3",
        "model_source": "huggingface",
        "api_type": "openai",
        "openai_compatible": True,
        "description": "OpenAI Whisper Large V3. Best-in-class speech-to-text transcription.",
        "documentation_url": "https://huggingface.co/openai/whisper-large-v3",
        "license": "MIT",
        "is_featured": True,
        "display_order": 30,
    },
    {
        "name": "Coqui XTTS v2",
        "slug": "coqui-xtts-v2",
        "version": "2.0",
        "category": ModelCategory.AUDIO,
        "serving_backend": ServingBackend.TTS,
        "min_vram_gb": 6,
        "recommended_vram_gb": 8,
        "min_gpu_count": 1,
        "recommended_gpu": "RTX 3070",
        "supports_quantization": False,
        "docker_image": "ghcr.io/coqui-ai/tts-server",
        "docker_tag": "latest",
        "default_env": {
            "TTS_MODEL": "tts_models/multilingual/multi-dataset/xtts_v2",
            "USE_CUDA": "true",
        },
        "default_ports": {"5002": "API"},
        "health_check_url": "/",
        "startup_timeout_seconds": 180,
        "model_id": "coqui/XTTS-v2",
        "model_source": "coqui",
        "api_type": "coqui",
        "openai_compatible": False,
        "description": "Coqui XTTS v2. Zero-shot voice cloning and multilingual TTS.",
        "documentation_url": "https://huggingface.co/coqui/XTTS-v2",
        "license": "Coqui Public Model License",
        "is_featured": True,
        "display_order": 31,
    },
    {
        "name": "Bark",
        "slug": "bark",
        "version": "1.0",
        "category": ModelCategory.AUDIO,
        "serving_backend": ServingBackend.TTS,
        "min_vram_gb": 12,
        "recommended_vram_gb": 16,
        "min_gpu_count": 1,
        "recommended_gpu": "RTX 4090",
        "supports_quantization": False,
        "docker_image": "ghcr.io/suno-ai/bark-server",
        "docker_tag": "latest",
        "default_env": {
            "SUNO_USE_SMALL_MODELS": "false",
        },
        "default_ports": {"8000": "API"},
        "health_check_url": "/health",
        "startup_timeout_seconds": 300,
        "model_id": "suno/bark",
        "model_source": "huggingface",
        "api_type": "custom",
        "openai_compatible": False,
        "description": "Suno Bark. Text-to-audio with music, sound effects, and voice.",
        "documentation_url": "https://huggingface.co/suno/bark",
        "license": "MIT",
        "is_featured": False,
        "display_order": 32,
    },
    {
        "name": "Parler TTS Mini",
        "slug": "parler-tts-mini",
        "version": "1.0",
        "category": ModelCategory.AUDIO,
        "serving_backend": ServingBackend.TTS,
        "min_vram_gb": 4,
        "recommended_vram_gb": 8,
        "min_gpu_count": 1,
        "recommended_gpu": "RTX 3060",
        "supports_quantization": False,
        "docker_image": "ghcr.io/huggingface/parler-tts",
        "docker_tag": "latest",
        "default_env": {
            "MODEL": "parler-tts/parler-tts-mini-v1",
        },
        "default_ports": {"8000": "API"},
        "health_check_url": "/health",
        "startup_timeout_seconds": 120,
        "model_id": "parler-tts/parler-tts-mini-v1",
        "model_source": "huggingface",
        "api_type": "custom",
        "openai_compatible": False,
        "description": "Parler TTS Mini. Lightweight, fast text-to-speech.",
        "documentation_url": "https://huggingface.co/parler-tts/parler-tts-mini-v1",
        "license": "Apache 2.0",
        "is_featured": False,
        "display_order": 33,
    },
    # =========================================================================
    # MULTIMODAL MODELS
    # =========================================================================
    {
        "name": "LLaVA 1.6 34B",
        "slug": "llava-1.6-34b",
        "version": "1.6",
        "category": ModelCategory.MULTIMODAL,
        "serving_backend": ServingBackend.VLLM,
        "min_vram_gb": 40,
        "recommended_vram_gb": 80,
        "min_gpu_count": 1,
        "recommended_gpu": "A100 80GB",
        "supports_quantization": True,
        "docker_image": "vllm/vllm-openai",
        "docker_tag": "latest",
        "default_env": {
            "MODEL": "liuhaotian/llava-v1.6-34b",
            "TRUST_REMOTE_CODE": "true",
        },
        "default_ports": {"8000": "API"},
        "health_check_url": "/health",
        "startup_timeout_seconds": 600,
        "model_id": "liuhaotian/llava-v1.6-34b",
        "model_source": "huggingface",
        "api_type": "openai",
        "openai_compatible": True,
        "description": "LLaVA 1.6 34B. Vision-language model for image understanding.",
        "documentation_url": "https://huggingface.co/liuhaotian/llava-v1.6-34b",
        "license": "LLaMA 2 Community License",
        "is_featured": False,
        "display_order": 40,
    },
    # =========================================================================
    # EMBEDDING MODELS
    # =========================================================================
    {
        "name": "BGE Large EN v1.5",
        "slug": "bge-large-en",
        "version": "1.5",
        "category": ModelCategory.EMBEDDING,
        "serving_backend": ServingBackend.TGI,
        "min_vram_gb": 4,
        "recommended_vram_gb": 8,
        "min_gpu_count": 1,
        "recommended_gpu": "RTX 3060",
        "supports_quantization": False,
        "docker_image": "ghcr.io/huggingface/text-embeddings-inference",
        "docker_tag": "latest",
        "default_env": {
            "MODEL_ID": "BAAI/bge-large-en-v1.5",
        },
        "default_ports": {"80": "API"},
        "health_check_url": "/health",
        "startup_timeout_seconds": 120,
        "model_id": "BAAI/bge-large-en-v1.5",
        "model_source": "huggingface",
        "api_type": "tei",
        "openai_compatible": False,
        "description": "BGE Large English. High-quality text embeddings for search.",
        "documentation_url": "https://huggingface.co/BAAI/bge-large-en-v1.5",
        "license": "MIT",
        "is_featured": False,
        "display_order": 50,
    },
]


async def seed_model_catalog(db: AsyncSession) -> dict:
    """
    Seed the model catalog with pre-built templates.

    Args:
        db: Database session

    Returns:
        Summary of seed operation
    """
    created = 0
    updated = 0
    skipped = 0

    for template_data in MODEL_TEMPLATES:
        slug = template_data["slug"]

        # Check if exists
        result = await db.execute(
            select(ModelTemplate).where(ModelTemplate.slug == slug)
        )
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing template
            for key, value in template_data.items():
                setattr(existing, key, value)
            updated += 1
            logger.info(f"Updated template: {slug}")
        else:
            # Create new template
            template = ModelTemplate(**template_data)
            db.add(template)
            created += 1
            logger.info(f"Created template: {slug}")

    await db.commit()

    return {
        "created": created,
        "updated": updated,
        "skipped": skipped,
        "total": len(MODEL_TEMPLATES),
    }


async def main():
    """Main entry point."""
    logger.info("Initializing database...")
    await init_db()

    logger.info("Seeding model catalog...")
    async with async_session_maker() as db:
        result = await seed_model_catalog(db)

    logger.info(f"Seed complete: {result}")


if __name__ == "__main__":
    asyncio.run(main())
