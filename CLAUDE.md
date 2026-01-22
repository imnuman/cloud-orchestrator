# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU Cloud Orchestrator - A rent-to-rent GPU arbitrage platform that aggregates GPU capacity from multiple cloud providers (RunPod, Lambda Labs, Vast.ai) and resells with enhanced services through a unified API.

**Architecture:** Control Plane (Brain) / Data Plane (Agents)
- **Brain:** FastAPI + PostgreSQL + Redis/Celery - handles user auth, billing, pod scheduling, node management, GPU sourcing
- **Agent:** Python daemon on GPU nodes - GPU detection, Docker container management, heartbeats

## Common Commands

```bash
# Development setup (all services via Docker)
cp .env.example .env
docker-compose up -d

# Run Brain locally
python -m venv venv && source venv/bin/activate
pip install -r requirements.brain.txt
docker-compose up -d postgres redis
uvicorn brain.main:app --reload

# Run Celery worker (for background tasks)
celery -A brain.tasks.celery_app worker -B -l info

# Run Agent (on GPU host)
pip install -r requirements.agent.txt
python -m agent.agent

# Tests
pytest tests/ -v
pytest tests/ --cov=brain --cov=agent  # with coverage
pytest tests/test_specific.py::test_name  # single test

# Code quality
black .
ruff check --fix .
mypy .

# Database migrations
alembic revision --autogenerate -m "description"
alembic upgrade head
```

## Code Architecture

```
/brain                    # Control Plane (FastAPI)
├── main.py              # App entry, routers mounted
├── config.py            # Pydantic settings (env vars)
├── /models              # SQLAlchemy AsyncORM models
│   ├── base.py          # DB session, Base class
│   ├── user.py          # User auth, balance
│   ├── node.py          # GPU worker nodes
│   ├── pod.py           # Container instances
│   ├── billing.py       # Transactions, UsageRecord
│   └── provisioned_instance.py  # Auto-provisioned GPU instances
├── /routes              # FastAPI routers
│   ├── auth.py          # /auth/* - JWT login/register
│   ├── nodes.py         # /nodes/* - agent registration/heartbeat + install.sh
│   ├── pods.py          # /pods/* - pod CRUD
│   └── users.py         # /users/* - balance, transactions
├── /tasks               # Celery background tasks
│   ├── celery_app.py    # Celery config + beat schedule
│   ├── billing.py       # bill_running_pods (every 60s)
│   ├── health.py        # check_node_health (every 30s)
│   └── sourcing.py      # GPU sourcing tasks (search, provision, cost tracking)
├── /services            # Business logic services
│   ├── sourcing.py      # SourcingService - GPU offer discovery
│   └── provisioning.py  # ProvisioningService - instance lifecycle
└── /adapters            # Provider integrations
    ├── base.py          # BaseProviderAdapter abstract class
    └── /vast_ai         # Vast.ai integration
        ├── client.py    # VastClient (auto-mocks when no API key)
        ├── mock.py      # MockVastClient for development
        └── schemas.py   # Vast.ai Pydantic models

/agent                    # Data Plane (Python daemon)
├── agent.py             # Main loop: register → heartbeat → process commands
├── config.py            # Settings (GPU_AGENT_* env vars, provider info)
├── gpu_detector.py      # nvidia-smi parsing
├── system_info.py       # OS/CPU/RAM detection
└── docker_manager.py    # Docker SDK container management

/shared                   # API contracts
└── schemas.py           # Pydantic models for Brain↔Agent communication
```

## Key Technical Details

**Python:** 3.11+ required, async throughout (SQLAlchemy AsyncORM, httpx AsyncClient)

**Authentication:**
- Users: JWT Bearer tokens (HS256), 24h expiry
- Agents: API key header (`X-Agent-API-Key`)

**Database:** PostgreSQL with UUID primary keys, all timestamps tracked

**API Base:** `/api/v1`

**Tool Configuration:**
- Black: line-length=100, target py311/py312
- Ruff: line-length=100, rules E/F/I/N/W/UP
- MyPy: strict mode enabled
- Pytest: asyncio_mode="auto"

## GPU Sourcing System

The sourcing system discovers and provisions GPU instances from providers automatically.

**Key Settings (all disabled by default for safety):**
- `sourcing_enabled: False` - Must explicitly enable
- `auto_provisioning_enabled: False` - Search only, no auto-create
- `use_mock_providers: True` - Uses mock data, no real API calls
- `sourcing_max_instances: 5` - Hard limit on auto-provisioned nodes

**Provisioning Lifecycle:**
1. `PENDING` → `CREATING` → `STARTING` → `INSTALLING` → `WAITING_REGISTRATION` → `ACTIVE`
2. Agent installs via onstart script, registers with Brain
3. ProvisioningService matches registration to ProvisionedInstance by `provider_id`

**Adding a New Provider:**
1. Create `brain/adapters/{provider}/` with schemas.py, mock.py, client.py
2. Implement `BaseProviderAdapter` interface
3. Add provider to `ProviderType` enum in `shared/schemas.py`
4. Update `SourcingService` to use new client

## Business Context

Phase 1 (current): Arbitrage model - rent from providers, add 20-30% markup
Phase 2 (planned): Marketplace - community providers list GPUs, platform takes transaction fee
