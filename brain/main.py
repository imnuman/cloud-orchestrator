"""
GPU Cloud Orchestrator - Brain (Control Plane)
Main FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from brain.config import get_settings
from brain.middleware.rate_limit import RateLimitMiddleware
from brain.middleware.security import SecurityHeadersMiddleware
from brain.models.base import init_db
from brain.routes import (
    auth_router,
    dashboard_router,
    nodes_router,
    pods_router,
    users_router,
    providers_router,
    models_router,
    proxy_router,
    websocket_router,
    billing_router,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print(f"Starting {settings.app_name} v{settings.app_version}")
    await init_db()
    print("Database initialized")
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="GPU Cloud Orchestrator - Control Plane API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Security headers middleware (first in chain)
app.add_middleware(
    SecurityHeadersMiddleware,
    enable_hsts=settings.environment == "production",
    enable_csp=True,
)

# Rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    exclude_paths=["/", "/health", "/docs", "/redoc", "/openapi.json"],
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix=settings.api_prefix)
app.include_router(dashboard_router, prefix=settings.api_prefix)
app.include_router(nodes_router, prefix=settings.api_prefix)
app.include_router(pods_router, prefix=settings.api_prefix)
app.include_router(users_router, prefix=settings.api_prefix)
app.include_router(providers_router, prefix=settings.api_prefix)
app.include_router(models_router, prefix=settings.api_prefix)
app.include_router(billing_router, prefix=settings.api_prefix)

# Model proxy routes (OpenAI-compatible API)
# Mounted at /api/v1/v1/* to provide /api/v1/v1/chat/completions etc.
# Also mount at root for direct API access
app.include_router(proxy_router, prefix=settings.api_prefix)
app.include_router(proxy_router, prefix="")  # Also at root for direct access

# WebSocket routes for real-time streaming
app.include_router(websocket_router, prefix=settings.api_prefix)
app.include_router(websocket_router, prefix="")  # Also at root for direct access


@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "healthy",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.environment,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "brain.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
