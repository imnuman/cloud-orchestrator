# GPU Cloud Orchestrator

A rent-to-rent GPU arbitrage platform that aggregates GPU capacity from multiple providers and resells with enhanced services.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CONTROL PLANE ("Brain")                     │
│  FastAPI + PostgreSQL + Redis/Celery                            │
│                                                                 │
│  - User authentication & billing                                │
│  - Pod scheduling & management                                  │
│  - Node registration & health monitoring                        │
│  - Provider adapters (RunPod, Lambda, Vast.ai)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                    Tailscale/WireGuard VPN
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PLANE ("Agents")                      │
│  Python daemon on GPU nodes                                     │
│                                                                 │
│  - GPU detection (nvidia-smi)                                   │
│  - Docker container management                                  │
│  - Heartbeat & telemetry                                        │
│  - Pod deployment execution                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- (For agents) NVIDIA GPU + drivers + Container Toolkit

### Development Setup

1. Clone the repository:
```bash
cd cloud-orchestrator
```

2. Copy environment file:
```bash
cp .env.example .env
```

3. Start services:
```bash
docker-compose up -d
```

4. Access the API:
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Running the Brain Locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.brain.txt

# Start PostgreSQL and Redis (via Docker)
docker-compose up -d postgres redis

# Run the API server
uvicorn brain.main:app --reload
```

### Running the Agent

On a GPU host:

```bash
# Install
curl -sSL https://your-domain.com/install.sh | sudo bash

# Or manually
pip install -r requirements.agent.txt
python -m agent.agent
```

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login and get token
- `GET /api/v1/auth/me` - Get current user

### Nodes (Agent communication)
- `POST /api/v1/nodes/register` - Register GPU node
- `POST /api/v1/nodes/heartbeat` - Send heartbeat
- `GET /api/v1/nodes/` - List all nodes
- `GET /api/v1/nodes/available` - List available nodes

### Pods (User operations)
- `POST /api/v1/pods/` - Create new pod
- `GET /api/v1/pods/` - List user's pods
- `GET /api/v1/pods/{id}` - Get pod details
- `POST /api/v1/pods/{id}/stop` - Stop pod
- `DELETE /api/v1/pods/{id}` - Terminate pod

### Users
- `GET /api/v1/users/me` - Get user info
- `GET /api/v1/users/balance` - Get balance
- `POST /api/v1/users/deposit` - Add funds
- `GET /api/v1/users/transactions` - Transaction history

## Project Structure

```
/cloud-orchestrator
├── /brain              # FastAPI Backend (Control Plane)
│   ├── main.py         # Entry point
│   ├── config.py       # Settings
│   ├── /models         # SQLAlchemy models
│   │   ├── base.py     # Database setup
│   │   ├── user.py     # User model
│   │   ├── node.py     # Node model
│   │   ├── pod.py      # Pod model
│   │   └── billing.py  # Transaction models
│   ├── /routes         # API endpoints
│   │   ├── auth.py     # Authentication
│   │   ├── nodes.py    # Node management
│   │   ├── pods.py     # Pod management
│   │   └── users.py    # User management
│   ├── /tasks          # Celery background tasks
│   │   ├── celery_app.py
│   │   ├── billing.py  # Billing meter
│   │   └── health.py   # Health checks
│   └── /adapters       # Provider adapters (TODO)
│       ├── runpod.py
│       ├── lambda_labs.py
│       └── vast_ai.py
├── /agent              # Worker Daemon (Data Plane)
│   ├── agent.py        # Main daemon
│   ├── config.py       # Settings
│   ├── gpu_detector.py # nvidia-smi wrapper
│   ├── system_info.py  # System detection
│   ├── docker_manager.py # Container management
│   └── install.sh      # Installation script
├── /shared             # Shared Pydantic schemas
│   └── schemas.py      # API contracts
├── docker-compose.yml  # Development environment
├── Dockerfile.brain    # Brain container
├── Dockerfile.agent    # Agent container
└── requirements.*.txt  # Dependencies
```

## Business Model

### Phase 1: Arbitrage
- Rent GPUs from providers (RunPod, Lambda Labs, Vast.ai)
- Add 20-30% markup
- Provide unified API, better UX, single billing

### Phase 2: Marketplace
- Allow users to list their own GPUs
- Take 10-20% transaction fee
- Build community supply

## Configuration

### Environment Variables

See `.env.example` for all available options.

Key settings:
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `JWT_SECRET_KEY` - Secret for JWT tokens
- `DEFAULT_MARKUP_PERCENT` - Markup on provider prices
- `BILLING_INTERVAL_SECONDS` - How often to charge running pods

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black .
ruff check --fix .
```

### Database Migrations

```bash
alembic revision --autogenerate -m "description"
alembic upgrade head
```

## License

MIT
