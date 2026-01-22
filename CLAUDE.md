# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU Cloud Orchestrator - A rent-to-rent GPU arbitrage platform aggregating capacity from multiple cloud providers (RunPod, Lambda Labs, Vast.ai) with unified API and enhanced services.

**Architecture:** Control Plane (Brain) + Data Plane (Agents)
- **Brain:** FastAPI + PostgreSQL + Redis/Celery - user auth, billing, pod scheduling, GPU sourcing
- **Agent:** Python daemon on GPU nodes - GPU detection, Docker management, heartbeats

## Common Commands

```bash
# Development setup
cp .env.example .env
docker-compose up -d

# Run Brain locally (requires postgres + redis)
source venv/bin/activate
pip install -r requirements.brain.txt
docker-compose up -d postgres redis
uvicorn brain.main:app --reload

# Run Celery worker (background tasks)
celery -A brain.tasks.celery_app worker -B -l info

# Run Agent (on GPU host)
pip install -r requirements.agent.txt
python -m agent.agent

# Tests
pytest tests/ -v
pytest tests/ --cov=brain --cov=agent
pytest tests/test_file.py::test_function  # single test

# Code quality
black .
ruff check --fix .
mypy .

# Database migrations
alembic revision --autogenerate -m "description"
alembic upgrade head

# Seed model catalog (Phase 2B)
python -m brain.scripts.seed_model_catalog
```

## Architecture

```
/brain                    # Control Plane (FastAPI)
├── main.py              # App entry, lifespan, routers mounted
├── config.py            # Pydantic settings (all env vars)
├── /middleware          # Rate limiting (60/min), security headers
├── /models              # SQLAlchemy AsyncORM (UUID PKs)
├── /routes              # FastAPI routers (auth, pods, nodes, users, providers, models, proxy, websocket)
├── /tasks               # Celery (billing 60s, health 30s, sourcing, model deployment)
├── /services            # Business logic (multi_provider, sourcing, provisioning, payouts, model_deployment)
└── /adapters            # Provider integrations (vast_ai, runpod, lambda_labs - each with client, mock, schemas)

/agent                    # Data Plane (Python daemon)
├── agent.py             # Main loop: register → heartbeat → process commands
├── config.py            # GPU_AGENT_* env vars
├── gpu_detector.py      # nvidia-smi parsing
├── system_info.py       # OS/CPU/RAM detection
└── docker_manager.py    # Docker SDK container management

/shared                   # API contracts (Pydantic models for Brain↔Agent)
/alembic                  # Database migrations (async SQLAlchemy)
/tests                    # Pytest (asyncio_mode="auto")
```

## Key Technical Details

- **Python 3.11+** required, async throughout (SQLAlchemy AsyncORM, httpx AsyncClient)
- **Auth:** Users via JWT (HS256, 24h), Agents via API key (`X-Agent-API-Key`)
- **API Base:** `/api/v1`
- **Tool Config:** Black/Ruff line-length=100, MyPy strict, pytest asyncio_mode="auto"

## Multi-Provider System

| Provider | API Type | Mock Support |
|----------|----------|--------------|
| Vast.ai | REST | ✅ |
| RunPod | GraphQL | ✅ |
| Lambda Labs | REST | ✅ |

**Pricing Strategies:** `LOWEST_PRICE`, `BEST_VALUE`, `HIGHEST_RELIABILITY`, `BALANCED`

**Critical Settings (all safe defaults):**
- `use_mock_providers: True` - No real API calls without changing
- `sourcing_enabled: False` - Must explicitly enable
- `auto_provisioning_enabled: False` - Search only, no auto-create

**Adding a Provider:**
1. Create `brain/adapters/{provider}/` with schemas.py, mock.py, client.py
2. Implement `BaseProviderAdapter` interface
3. Add to `ProviderType` enum in `shared/schemas.py`
4. Register in `MultiProviderService._get_all_clients()`

## Phase 2 Features

### Phase 2A: Community GPU Marketplace
- Provider registration, earnings dashboard, payout system
- Revenue split tiers: Basic (80%), Verified (85%), Pro (90%)
- Routes: `/providers/*`

### Phase 2B: AI Model Catalog
- One-click model deployment (Llama, Mistral, SDXL, Whisper, etc.)
- Routes: `/models/*`
- Seed: `python -m brain.scripts.seed_model_catalog`

### Phase 2C: Voice Agent Support
- OpenAI-compatible REST API: `/v1/chat/completions`, `/v1/embeddings`, etc.
- WebSocket endpoints: `/ws/audio/transcriptions`, `/ws/voice-agent`

## Testing Requirements

Tests require a running PostgreSQL instance:
```bash
# Start test database
docker-compose up -d postgres

# Run tests (auto-creates gpu_orchestrator_test DB)
pytest tests/ -v
```

Test fixtures in `tests/conftest.py` provide `db_session`, `client`, `test_user_data`, `test_gpu_offer`, `test_pod_request`.
