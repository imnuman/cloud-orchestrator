# API Routes
from brain.routes.auth import router as auth_router
from brain.routes.dashboard import router as dashboard_router
from brain.routes.nodes import router as nodes_router
from brain.routes.pods import router as pods_router
from brain.routes.users import router as users_router
from brain.routes.providers import router as providers_router
from brain.routes.models import router as models_router
from brain.routes.proxy import router as proxy_router
from brain.routes.websocket import router as websocket_router
from brain.routes.billing import router as billing_router
from brain.routes.api_keys import router as api_keys_router
from brain.routes.admin import router as admin_router

__all__ = [
    "auth_router",
    "dashboard_router",
    "nodes_router",
    "pods_router",
    "users_router",
    "providers_router",
    "models_router",
    "proxy_router",
    "websocket_router",
    "billing_router",
    "api_keys_router",
    "admin_router",
]
