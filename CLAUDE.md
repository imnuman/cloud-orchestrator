# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU Cloud Orchestrator - A rent-to-rent GPU arbitrage platform that aggregates GPU capacity from multiple cloud providers (RunPod, Lambda Labs, Vast.ai) and resells with enhanced services through a unified API.

**Architecture:** Control Plane (Brain) / Data Plane (Agents)
- **Brain:** FastAPI + PostgreSQL + Redis/Celery - handles user auth, billing, pod scheduling, node management
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
│   └── billing.py       # Transactions, UsageRecord
├── /routes              # FastAPI routers
│   ├── auth.py          # /auth/* - JWT login/register
│   ├── nodes.py         # /nodes/* - agent registration/heartbeat
│   ├── pods.py          # /pods/* - pod CRUD
│   └── users.py         # /users/* - balance, transactions
├── /tasks               # Celery background tasks
│   ├── celery_app.py    # Celery config
│   ├── billing.py       # bill_running_pods (every 60s)
│   └── health.py        # check_node_health (every 30s)
└── /adapters            # Provider integrations (stubs)

/agent                    # Data Plane (Python daemon)
├── agent.py             # Main loop: register → heartbeat → process commands
├── config.py            # Settings (GPU_AGENT_* env vars)
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

## Business Context

Phase 1 (current): Arbitrage model - rent from providers, add 20-30% markup
Phase 2 (planned): Marketplace - community providers list GPUs, platform takes transaction fee

Provider adapters in `/brain/adapters/` are stubs awaiting implementation.
