# Production Deployment Guide

## Your Mission

Build a production GPU cloud platform where:
1. **Day 1**: You use upstream providers (RunPod, Vast.ai, Lambda) for your own 100+ GPU hrs/month
2. **Phase 2**: Attract community GPU providers to expand supply
3. **Scale**: Build a self-sustaining GPU marketplace

---

## Part 1: Your Personal Savings Analysis

### Current Spend (Direct on RunPod)
```
100 GPU hours/month on RunPod
├── RTX 4090: $0.34/hr × 100 = $34/mo
├── A100 40GB: $1.89/hr × 100 = $189/mo
└── H100 80GB: $3.29/hr × 100 = $329/mo
```

### With Your Platform (25% Markup for Other Users)
As the platform owner, you can set your markup to 0% for yourself:

```python
# In brain/config.py or per-user setting
# Your account gets provider cost (0% markup)
# Other users pay 25% markup (your profit)
```

**Your effective cost**: Same as upstream providers
**Revenue from others**: 25% of their GPU spend

### Break-Even Calculation
```
Your infrastructure cost: ~$100/mo
To break even, you need other users spending:
$100 ÷ 0.25 = $400/mo in GPU usage

That's just 4 users at your usage level (100 hrs × $1/hr average)
```

---

## Part 2: Production Infrastructure

### Option A: Single VPS (Budget - $50-100/mo)

Best for: Starting out, <500 users

```
┌─────────────────────────────────────────────┐
│            Hetzner/Contabo VPS              │
│            (8 vCPU, 16GB RAM)               │
│                  ~$40/mo                    │
├─────────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌───────────────┐  │
│  │ FastAPI │ │ Celery  │ │ Celery Beat   │  │
│  │  :8000  │ │ Worker  │ │  (scheduler)  │  │
│  └─────────┘ └─────────┘ └───────────────┘  │
│                                             │
│  ┌─────────────────┐ ┌─────────────────┐    │
│  │   PostgreSQL    │ │     Redis       │    │
│  │     :5432       │ │     :6379       │    │
│  └─────────────────┘ └─────────────────┘    │
└─────────────────────────────────────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │   Cloudflare    │
            │  (SSL + CDN)    │
            │     FREE        │
            └─────────────────┘
```

**Monthly Cost:**
| Component | Service | Cost |
|-----------|---------|------|
| VPS | Hetzner CPX31 | €15 (~$16) |
| VPS | Contabo VPS L | $13 |
| Domain | Cloudflare | $10/yr |
| SSL | Cloudflare | FREE |
| **Total** | | **~$20-50/mo** |

### Option B: Managed Services (Recommended - $100-200/mo)

Best for: Production reliability, 500-5000 users

```
┌──────────────────────────────────────────────────────────────┐
│                        Render / Railway                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   FastAPI   │  │   Celery    │  │   Celery    │          │
│  │   Service   │  │   Worker    │  │    Beat     │          │
│  │   $25/mo    │  │   $15/mo    │  │   $10/mo    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │   Neon DB   │     │   Upstash   │     │ Cloudflare  │
   │ PostgreSQL  │     │    Redis    │     │  Tunnel/CDN │
   │   $25/mo    │     │   $10/mo    │     │    FREE     │
   └─────────────┘     └─────────────┘     └─────────────┘
```

**Monthly Cost:**
| Component | Service | Cost |
|-----------|---------|------|
| API Server | Render Web Service | $25 |
| Celery Worker | Render Background Worker | $15 |
| Celery Beat | Render Background Worker | $10 |
| PostgreSQL | Neon Pro | $25 |
| Redis | Upstash Pro | $10 |
| **Total** | | **~$85/mo** |

### Option C: Kubernetes (Scale - $300+/mo)

Best for: 5000+ users, enterprise

```
┌────────────────────────────────────────────────────────────────┐
│                    DigitalOcean/Linode K8s                     │
│                        (~$200/mo base)                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Ingress (Traefik)                     │  │
│  │                    SSL Termination                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                 │
│       ┌──────────────────────┼──────────────────────┐          │
│       ▼                      ▼                      ▼          │
│  ┌─────────┐           ┌─────────┐           ┌─────────┐       │
│  │ FastAPI │           │ FastAPI │           │ FastAPI │       │
│  │ Pod (1) │           │ Pod (2) │           │ Pod (3) │       │
│  └─────────┘           └─────────┘           └─────────┘       │
│       │                                                        │
│       ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Celery Workers (Auto-scaling)              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
       ┌─────────────┐                 ┌─────────────┐
       │ Managed DB  │                 │ Managed     │
       │ PostgreSQL  │                 │ Redis       │
       │  $50+/mo    │                 │  $30+/mo    │
       └─────────────┘                 └─────────────┘
```

---

## Part 3: Production Configuration

### Step 1: Get Provider API Keys

```bash
# RunPod: https://www.runpod.io/console/user/settings
# Vast.ai: https://cloud.vast.ai/account/
# Lambda Labs: https://cloud.lambda.ai/api-keys
```

### Step 2: Create Production .env

```bash
# /root/runpod/cloud-orchestrator/.env.production

# =============================================================================
# PRODUCTION SETTINGS
# =============================================================================

# Application
DEBUG=false
ENVIRONMENT=production

# Database (Use managed PostgreSQL)
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/gpu_orchestrator

# Redis (Use managed Redis)
REDIS_URL=redis://:password@host:6379/0

# JWT - GENERATE A STRONG SECRET
# python -c "import secrets; print(secrets.token_urlsafe(64))"
JWT_SECRET_KEY=YOUR_GENERATED_64_BYTE_SECRET_HERE
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440

# =============================================================================
# PROVIDER API KEYS (CRITICAL FOR PRODUCTION)
# =============================================================================

RUNPOD_API_KEY=your_runpod_api_key_here
VAST_AI_API_KEY=your_vast_ai_api_key_here
LAMBDA_LABS_API_KEY=your_lambda_labs_api_key_here

# =============================================================================
# PRODUCTION MODE - ENABLE REAL PROVIDERS
# =============================================================================

USE_MOCK_PROVIDERS=false
SOURCING_ENABLED=true
AUTO_PROVISIONING_ENABLED=true

# Sourcing limits (start conservative)
SOURCING_MAX_INSTANCES=10
SOURCING_MAX_PRICE_PER_HOUR=5.00
SOURCING_INTERVAL_SECONDS=60

# =============================================================================
# BILLING
# =============================================================================

DEFAULT_MARKUP_PERCENT=25
BILLING_INTERVAL_SECONDS=60
MINIMUM_BALANCE_FOR_POD=5.0
LOW_BALANCE_WARNING_THRESHOLD=20.0

# =============================================================================
# BRAIN PUBLIC URL (for agent callbacks)
# =============================================================================

BRAIN_PUBLIC_URL=https://api.yourdomain.com
```

### Step 3: Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: "3.8"

services:
  brain:
    image: your-registry/gpu-orchestrator-brain:latest
    container_name: gpu-brain
    restart: always
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - RUNPOD_API_KEY=${RUNPOD_API_KEY}
      - VAST_AI_API_KEY=${VAST_AI_API_KEY}
      - LAMBDA_LABS_API_KEY=${LAMBDA_LABS_API_KEY}
      - USE_MOCK_PROVIDERS=false
      - DEBUG=false
    ports:
      - "8000:8000"
    command: >
      gunicorn brain.main:app
      --workers 4
      --worker-class uvicorn.workers.UvicornWorker
      --bind 0.0.0.0:8000
      --timeout 120
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  celery-worker:
    image: your-registry/gpu-orchestrator-brain:latest
    container_name: gpu-celery-worker
    restart: always
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - RUNPOD_API_KEY=${RUNPOD_API_KEY}
      - VAST_AI_API_KEY=${VAST_AI_API_KEY}
      - LAMBDA_LABS_API_KEY=${LAMBDA_LABS_API_KEY}
      - USE_MOCK_PROVIDERS=false
    command: celery -A brain.tasks.celery_app worker --loglevel=info --concurrency=4

  celery-beat:
    image: your-registry/gpu-orchestrator-brain:latest
    container_name: gpu-celery-beat
    restart: always
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    command: celery -A brain.tasks.celery_app beat --loglevel=info
```

---

## Part 4: Production Code Changes Needed

### Critical Changes for Production

| File | Change | Priority |
|------|--------|----------|
| `brain/config.py` | Add `SOURCING_MAX_PRICE_PER_HOUR` tiers for different GPUs | HIGH |
| `brain/adapters/*/client.py` | Remove mock fallback in production mode | HIGH |
| `brain/models/model_catalog.py` | Fix 70B model VRAM requirements | MEDIUM |
| `brain/routes/auth.py` | Add email verification | MEDIUM |
| `brain/services/payouts.py` | Integrate real crypto/PayPal APIs | LOW (Phase 2) |

### Immediate Code Fix: Enable Real Providers

```python
# brain/config.py - Add these settings

# GPU-specific max prices (for sourcing)
sourcing_gpu_max_prices: dict = {
    "RTX 4090": 0.50,
    "RTX 3090": 0.40,
    "A100 40GB": 2.50,
    "A100 80GB": 3.50,
    "H100 80GB": 5.00,
}
```

---

## Part 5: Go-Live Checklist

### Before Launch

- [ ] **Database**: Run migrations `alembic upgrade head`
- [ ] **Models**: Seed catalog `python -m brain.scripts.seed_model_catalog`
- [ ] **API Keys**: Add RunPod, Vast.ai, Lambda Labs keys
- [ ] **JWT Secret**: Generate and set strong secret
- [ ] **Domain**: Point DNS to your server
- [ ] **SSL**: Configure via Cloudflare or Traefik
- [ ] **Monitoring**: Set up health checks
- [ ] **Backups**: Configure PostgreSQL backups

### Launch Day

1. Deploy with `docker-compose -f docker-compose.prod.yml up -d`
2. Create your admin account via API
3. Add initial balance to your account
4. Deploy your first model
5. Test the full flow

### Post-Launch

- [ ] Monitor error logs
- [ ] Check billing accuracy
- [ ] Test failover between providers
- [ ] Set up alerting (Sentry, PagerDuty)

---

## Part 6: Community Provider Marketing Strategy

### Phase 1: Seed Supply (Month 1-2)

**Target**: Crypto miners with idle GPUs

```
Channels:
├── Reddit: r/gpumining, r/EtherMining, r/NiceHash
├── Discord: Mining communities, AI/ML servers
├── Twitter/X: #GPUmining, #cryptomining
└── Forums: Bitcointalk, mining-specific forums
```

**Value Proposition**:
```
"Your GPUs are sitting idle. Earn $0.30-$3.00/hour passively.
One command to start: curl https://yourplatform.com/install | bash
```

### Phase 2: Growth (Month 3-6)

**Referral Program**:
```
- Provider refers provider: Both get 5% bonus for 3 months
- User refers user: Referrer gets $10 credit per active user
```

**Content Marketing**:
```
- "How I Make $500/mo With My Gaming PC" blog posts
- YouTube tutorials on setup
- Case studies from early providers
```

### Phase 3: Enterprise (Month 6+)

**Target**: Small data centers, hosting companies

```
Benefits:
├── Bulk onboarding (10+ GPUs)
├── Dedicated account manager
├── Custom SLA options
└── Lower platform fee (10% vs 20%)
```

---

## Part 7: Financial Projections

### Month 1 (Just You)

```
Revenue: $0 (you're the only user)
Your GPU spend: ~$150 (100hrs at avg $1.50/hr)
Infrastructure: ~$50
Net: -$50/mo (but you're using GPUs at cost)
```

### Month 3 (10 Active Users)

```
Total GPU spend: 10 users × $50 avg = $500/mo
Your revenue (25% markup): $125/mo
Infrastructure: ~$75
Net: +$50/mo profit
```

### Month 6 (50 Active Users + 5 Community Providers)

```
Total GPU spend: $5,000/mo
├── Via upstream (RunPod/Vast): $3,000 → Your cut: $750
├── Via community GPUs: $2,000 → Your cut (20%): $400
Total revenue: $1,150/mo
Infrastructure: ~$150
Net: +$1,000/mo profit
```

### Month 12 (200 Users + 20 Providers)

```
Total GPU spend: $30,000/mo
├── Upstream: $10,000 → Your cut: $2,500
├── Community: $20,000 → Your cut: $4,000
Total revenue: $6,500/mo
Infrastructure: ~$300
Net: +$6,200/mo profit
```

---

## Quick Start Commands

```bash
# 1. Set up production environment
cp .env.example .env.production
# Edit .env.production with real values

# 2. Build production images
docker build -f Dockerfile.brain -t gpu-brain:latest .

# 3. Start services
docker-compose -f docker-compose.prod.yml up -d

# 4. Run migrations
docker exec gpu-brain alembic upgrade head

# 5. Seed model catalog
docker exec gpu-brain python -m brain.scripts.seed_model_catalog

# 6. Create your admin account (via API)
curl -X POST https://api.yourdomain.com/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@email.com", "password": "securepassword"}'

# 7. Add balance and start using!
```

---

## Sources

- [RunPod API Keys](https://docs.runpod.io/get-started/api-keys)
- [FastAPI Production Best Practices](https://render.com/articles/fastapi-production-deployment-best-practices)
- [FastAPI Docker Guide](https://fastapi.tiangolo.com/deployment/docker/)
