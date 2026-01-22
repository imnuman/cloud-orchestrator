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
├── /middleware          # Custom middleware
│   ├── rate_limit.py    # Rate limiting (60/min, 1000/hr)
│   └── security.py      # Security headers middleware
├── /models              # SQLAlchemy AsyncORM models
│   ├── base.py          # DB session, Base class
│   ├── user.py          # User auth, balance
│   ├── node.py          # GPU worker nodes
│   ├── pod.py           # Container instances
│   ├── billing.py       # Transactions, UsageRecord
│   ├── provisioned_instance.py  # Auto-provisioned GPU instances
│   ├── provider.py      # Provider, Payout, ProviderEarning (Phase 2A)
│   └── model_catalog.py # ModelTemplate, Deployment (Phase 2B)
├── /routes              # FastAPI routers
│   ├── auth.py          # /auth/* - JWT login/register
│   ├── dashboard.py     # /dashboard/* - unified dashboard API
│   ├── nodes.py         # /nodes/* - agent registration/heartbeat + install.sh
│   ├── pods.py          # /pods/* - pod CRUD
│   ├── users.py         # /users/* - balance, transactions
│   ├── providers.py     # /providers/* - provider portal (Phase 2A)
│   └── models.py        # /models/* - model catalog & deployments (Phase 2B)
├── /tasks               # Celery background tasks
│   ├── celery_app.py    # Celery config + beat schedule
│   ├── billing.py       # bill_running_pods (every 60s)
│   ├── health.py        # check_node_health (every 30s)
│   ├── sourcing.py      # GPU sourcing tasks (search, provision, cost tracking)
│   └── model_health.py  # Model deployment tasks (Phase 2B)
├── /services            # Business logic services
│   ├── multi_provider.py   # MultiProviderService - cross-provider aggregation
│   ├── sourcing.py         # SourcingService - GPU offer discovery
│   ├── provisioning.py     # ProvisioningService - instance lifecycle
│   ├── payouts.py          # PayoutService - provider payouts (Phase 2A)
│   └── model_deployment.py # ModelDeploymentService (Phase 2B)
├── /scripts             # Management scripts
│   └── seed_model_catalog.py  # Seed model templates
└── /adapters            # Provider integrations
    ├── base.py          # BaseProviderAdapter abstract class
    ├── /vast_ai         # Vast.ai integration
    │   ├── client.py    # VastClient (auto-mocks when no API key)
    │   ├── mock.py      # MockVastClient for development
    │   └── schemas.py   # Vast.ai Pydantic models
    ├── /runpod          # RunPod integration
    │   ├── client.py    # RunPodClient (GraphQL API)
    │   ├── mock.py      # MockRunPodClient for development
    │   └── schemas.py   # RunPod Pydantic models
    └── /lambda_labs     # Lambda Labs integration
        ├── client.py    # LambdaClient (REST API)
        ├── mock.py      # MockLambdaClient for development
        └── schemas.py   # Lambda Labs Pydantic models

/agent                    # Data Plane (Python daemon)
├── agent.py             # Main loop: register → heartbeat → process commands
├── config.py            # Settings (GPU_AGENT_* env vars, provider info)
├── gpu_detector.py      # nvidia-smi parsing
├── system_info.py       # OS/CPU/RAM detection
└── docker_manager.py    # Docker SDK container management

/shared                   # API contracts
└── schemas.py           # Pydantic models for Brain↔Agent communication

/alembic                  # Database migrations
├── env.py               # Async SQLAlchemy migration env
└── versions/            # Migration files

/tests                    # Test suite
├── adapters/            # Provider adapter tests
├── services/            # Service tests
└── conftest.py          # Pytest fixtures
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

## Multi-Provider System

The platform aggregates GPU offers from three providers:

| Provider | API Type | Mock Support | Key Features |
|----------|----------|--------------|--------------|
| Vast.ai | REST | ✅ | Marketplace, lowest prices |
| RunPod | GraphQL | ✅ | Secure + Community cloud |
| Lambda Labs | REST | ✅ | Enterprise, InfiniBand |

**Pricing Strategies:**
- `LOWEST_PRICE` - Optimize for cheapest options
- `BEST_VALUE` - Price per GB VRAM
- `HIGHEST_RELIABILITY` - Prioritize reliable providers
- `BALANCED` - Mix of price, value, and reliability

**Auto-Failover:**
The system automatically tries alternative providers if one fails:
```python
# Example: Create instance with failover
instance, provider = await multi_provider.create_instance_with_failover(
    gpu_type="RTX 4090",
    config=instance_config,
    max_price=1.0,
    preferred_providers=["runpod", "vast_ai"],
)
```

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
4. Register in `MultiProviderService._get_all_clients()`
5. Update `SourcingService` to use new client

## Dashboard API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /dashboard/overview` | Overview statistics |
| `GET /dashboard/provider-health` | Provider health status |
| `GET /dashboard/gpu-availability` | GPU availability across providers |
| `GET /dashboard/gpu-offers` | Available GPU offers with pricing |
| `GET /dashboard/cost-analytics` | Cost analytics (admin only) |
| `GET /dashboard/price-comparison` | Price comparison across providers |

## Security Features

**Rate Limiting:**
- 60 requests/minute per client
- 1000 requests/hour per client
- Based on API key or IP address
- Excludes health check endpoints

**Security Headers:**
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Strict-Transport-Security (production only)
- Content-Security-Policy

**Input Validation:**
- SQL injection pattern detection
- XSS pattern detection
- Content length limits
- Content type validation

## Business Context

**Phase 1 (current):** Arbitrage model
- Rent GPUs from Vast.ai, RunPod, Lambda Labs
- Apply 25% markup (configurable)
- Unified API and billing
- Automatic failover across providers

**Phase 2 (implemented):** Community GPU Marketplace + AI Model Serving

### Phase 2A: Community GPU Onboarding

Community providers (crypto miners, GPU owners) can rent out their hardware:

```bash
# One-liner install for providers
curl -sSL https://brain-url/api/v1/nodes/install.sh | bash -s -- --provider-key=pk_xxxx
```

**Provider API Endpoints:**
| Endpoint | Description |
|----------|-------------|
| `POST /providers/register` | Register as a GPU provider |
| `GET /providers/dashboard` | Earnings, node status, analytics |
| `GET /providers/nodes` | List provider's nodes |
| `PATCH /providers/nodes/{id}/pricing` | Set custom pricing |
| `GET /providers/payouts` | Payout history |
| `POST /providers/payouts/request` | Request payout |

**Revenue Split Tiers:**
| Tier | Platform Fee | Provider Gets | Requirements |
|------|--------------|---------------|--------------|
| Basic | 20% | 80% | Email verified |
| Verified | 15% | 85% | Identity verified |
| Pro | 10% | 90% | Business verified |

**Payout Methods:** Crypto (USDC/USDT) or PayPal

### Phase 2B: AI Model Catalog

One-click deployment of open-source AI models with pay-per-GPU-hour billing:

**Model API Endpoints:**
| Endpoint | Description |
|----------|-------------|
| `GET /models` | List available models |
| `GET /models/{slug}` | Model details |
| `POST /models/{slug}/deploy` | Deploy model instance |
| `GET /models/deployments` | User's active deployments |
| `DELETE /models/deployments/{id}` | Stop deployment |
| `GET /models/deployments/{id}/metrics` | GPU utilization, requests/sec |

**Pre-built Model Templates:**
| Category | Models | Docker Image | Min VRAM |
|----------|--------|--------------|----------|
| Text/LLM | Llama 3.1, Mistral, Qwen, DeepSeek | vLLM | 16-80GB |
| Image Gen | SDXL, Flux | ComfyUI | 12-48GB |
| Voice/Audio | Whisper, Coqui TTS, Bark | Custom | 6-16GB |
| Code | DeepSeek Coder V2 | vLLM | 24-48GB |
| Multimodal | LLaVA | vLLM | 40-80GB |
| Embedding | BGE Large | TEI | 4-8GB |

**Seed the Model Catalog:**
```bash
python -m brain.scripts.seed_model_catalog
```

### Phase 2C: Voice Agent Support

Voice pipeline models for real-time voice agents:
- **Whisper Large V3** - Speech-to-text transcription
- **Coqui XTTS v2** - Zero-shot voice cloning and multilingual TTS
- **Bark** - Text-to-audio with music and sound effects
- **Parler TTS** - Lightweight, fast text-to-speech

### User-Facing API (Model Proxy)

Users access their deployed models via OpenAI-compatible API endpoints:

**Authentication:** Include deployment API key in requests:
- `Authorization: Bearer sk_xxxx`
- `X-API-Key: sk_xxxx`

**OpenAI-Compatible REST Endpoints:**
| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions (LLMs) |
| `POST /v1/completions` | Text completions |
| `POST /v1/embeddings` | Text embeddings |
| `POST /v1/audio/transcriptions` | Speech-to-text (Whisper) |
| `POST /v1/audio/speech` | Text-to-speech |
| `POST /v1/images/generations` | Image generation |
| `GET /v1/models` | List available models |

**Example Usage (Python):**
```python
from openai import OpenAI

# Point to your deployment
client = OpenAI(
    base_url="https://your-platform.com/v1",
    api_key="sk_xxxx",  # Deployment API key
)

response = client.chat.completions.create(
    model="llama-3.1-70b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

**WebSocket Endpoints (Real-Time Voice):**
| Endpoint | Description |
|----------|-------------|
| `WS /ws/audio/transcriptions?api_key=sk_xxx` | Real-time STT |
| `WS /ws/audio/speech?api_key=sk_xxx` | Real-time TTS |
| `WS /ws/voice-agent?api_key=sk_xxx` | Bidirectional voice agent |

**Voice Agent WebSocket Protocol:**
```javascript
// Connect
const ws = new WebSocket("wss://platform.com/ws/voice-agent?api_key=sk_xxx");

// Configure
ws.send(JSON.stringify({type: "config", system_prompt: "You are a helpful assistant."}));

// Send audio chunks (PCM 16-bit, 16kHz)
ws.send(audioChunk);

// Receive events
ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
        const msg = JSON.parse(event.data);
        // msg.type: "transcription", "response", "audio_start", "audio_end"
    } else {
        // Binary audio data (agent speech)
        playAudio(event.data);
    }
};
```

## New Models (Phase 2)

```
/brain/models/
├── provider.py          # Provider, Payout, ProviderEarning
├── model_catalog.py     # ModelTemplate, Deployment, ModelUsageLog

/brain/routes/
├── providers.py         # /providers/* - Provider portal
├── models.py            # /models/* - Model catalog & deployments
├── proxy.py             # /v1/* - OpenAI-compatible model API
├── websocket.py         # /ws/* - Real-time voice streaming

/brain/services/
├── payouts.py           # Payout processing (crypto + PayPal)
├── model_deployment.py  # Model deployment orchestration
├── model_proxy.py       # Request routing to deployed models

/brain/tasks/
├── model_health.py      # Deployment provisioning, health checks, billing

/brain/scripts/
├── seed_model_catalog.py  # Seed 15+ model templates
```

## Celery Beat Schedule (Updated)

| Task | Schedule | Description |
|------|----------|-------------|
| `bill-running-pods` | 60s | Charge for pod usage |
| `check-node-health` | 30s | Mark offline nodes |
| `provision-pending-deployments` | 30s | Provision model deployments |
| `check-deployment-health` | 30s | Health check deployments |
| `update-deployment-statuses` | 15s | Sync deployment/pod status |
| `bill-model-deployments` | 60s | Bill for model runtime |
