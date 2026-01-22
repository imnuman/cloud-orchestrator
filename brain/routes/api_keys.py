"""
API Key Management Routes.

Allows users to:
- Create multiple API keys with custom names and scopes
- List and manage their API keys
- Revoke keys
- Rotate keys (revoke old, create new)
- View usage statistics
"""

import hashlib
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from brain.config import get_settings
from brain.models.api_key import APIKey, APIKeyScope
from brain.models.base import get_db
from brain.models.user import User
from brain.routes.auth import get_current_user

settings = get_settings()
router = APIRouter(prefix="/api-keys", tags=["API Keys"])


# Constants
API_KEY_PREFIX = "sk_"
API_KEY_LENGTH = 48  # 48 bytes = 64 base64 chars


# Request/Response Models


class CreateAPIKeyRequest(BaseModel):
    """Request to create a new API key."""

    name: str = Field(..., min_length=1, max_length=255, description="Key name for identification")
    description: Optional[str] = Field(None, max_length=1000, description="Optional description")
    scopes: list[str] = Field(
        default=["read", "write"],
        description="Permission scopes: read, write, admin, inference",
    )
    expires_in_days: Optional[int] = Field(
        None, ge=1, le=365, description="Days until expiration (optional)"
    )
    rate_limit_per_minute: Optional[int] = Field(
        None, ge=1, le=10000, description="Per-minute rate limit"
    )
    allowed_ips: Optional[list[str]] = Field(None, description="Restrict to specific IPs")


class APIKeyResponse(BaseModel):
    """API key information (without the actual key)."""

    id: str
    name: str
    description: Optional[str]
    key_prefix: str
    scopes: list[str]
    is_active: bool
    expires_at: Optional[str]
    last_used_at: Optional[str]
    last_used_ip: Optional[str]
    request_count: int
    rate_limit_per_minute: Optional[int]
    rate_limit_per_hour: Optional[int]
    allowed_ips: Optional[list[str]]
    created_at: str


class APIKeyCreatedResponse(BaseModel):
    """Response when creating a new API key (includes the key once)."""

    id: str
    name: str
    key: str  # Only returned once on creation!
    key_prefix: str
    scopes: list[str]
    expires_at: Optional[str]
    message: str = "Save this key - it won't be shown again!"


class APIKeyListResponse(BaseModel):
    """List of API keys."""

    keys: list[APIKeyResponse]
    total: int


class UpdateAPIKeyRequest(BaseModel):
    """Request to update an API key."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    scopes: Optional[list[str]] = None
    is_active: Optional[bool] = None
    rate_limit_per_minute: Optional[int] = Field(None, ge=1, le=10000)
    allowed_ips: Optional[list[str]] = None


class RotateAPIKeyResponse(BaseModel):
    """Response when rotating an API key."""

    old_key_id: str
    new_key: APIKeyCreatedResponse
    message: str = "Old key has been revoked. Save the new key - it won't be shown again!"


class APIKeyUsageResponse(BaseModel):
    """API key usage statistics."""

    id: str
    name: str
    request_count: int
    last_used_at: Optional[str]
    last_used_ip: Optional[str]


# Helper Functions


def generate_api_key() -> tuple[str, str, str]:
    """
    Generate a new API key.

    Returns:
        tuple: (full_key, key_prefix, key_hash)
    """
    # Generate random bytes
    random_bytes = secrets.token_bytes(API_KEY_LENGTH)

    # Create the full key with prefix
    key_suffix = secrets.token_urlsafe(API_KEY_LENGTH)
    full_key = f"{API_KEY_PREFIX}{key_suffix}"

    # Create prefix for identification (first 12 chars after prefix)
    key_prefix = full_key[:16]

    # Hash the key for storage
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()

    return full_key, key_prefix, key_hash


def hash_api_key(key: str) -> str:
    """Hash an API key for comparison."""
    return hashlib.sha256(key.encode()).hexdigest()


def validate_scopes(scopes: list[str]) -> list[str]:
    """Validate and normalize scopes."""
    valid_scopes = {s.value for s in APIKeyScope}
    normalized = []
    for scope in scopes:
        if scope.lower() in valid_scopes:
            normalized.append(scope.lower())
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid scope: {scope}. Valid scopes: {', '.join(valid_scopes)}",
            )
    return list(set(normalized))  # Remove duplicates


# Endpoints


@router.post("", response_model=APIKeyCreatedResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: CreateAPIKeyRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Create a new API key.

    The full key is only returned once - save it securely!

    **Scopes:**
    - `read`: Read-only access to resources
    - `write`: Create and modify resources (pods, deployments)
    - `admin`: Full access including billing and account settings
    - `inference`: Model inference only (for deployed models)
    """
    # Validate scopes
    scopes = validate_scopes(request.scopes)

    # Check key limit (max 10 keys per user)
    result = await db.execute(
        select(APIKey).where(APIKey.user_id == current_user.id)
    )
    existing_keys = result.scalars().all()
    if len(existing_keys) >= 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum of 10 API keys per user. Please revoke an existing key first.",
        )

    # Generate key
    full_key, key_prefix, key_hash = generate_api_key()

    # Calculate expiration
    expires_at = None
    if request.expires_in_days:
        from datetime import timedelta

        expires_at = datetime.now(timezone.utc) + timedelta(days=request.expires_in_days)

    # Create API key record
    api_key = APIKey(
        name=request.name,
        description=request.description,
        key_hash=key_hash,
        key_prefix=key_prefix,
        user_id=current_user.id,
        scopes=scopes,
        expires_at=expires_at,
        rate_limit_per_minute=request.rate_limit_per_minute,
        allowed_ips=request.allowed_ips,
    )
    db.add(api_key)
    await db.flush()
    await db.refresh(api_key)

    return {
        "id": api_key.id,
        "name": api_key.name,
        "key": full_key,  # Only returned once!
        "key_prefix": key_prefix,
        "scopes": scopes,
        "expires_at": expires_at.isoformat() if expires_at else None,
        "message": "Save this key - it won't be shown again!",
    }


@router.get("", response_model=APIKeyListResponse)
async def list_api_keys(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    List all API keys for the current user.

    Note: The actual key values are not returned - only metadata.
    """
    result = await db.execute(
        select(APIKey)
        .where(APIKey.user_id == current_user.id)
        .order_by(APIKey.created_at.desc())
    )
    keys = result.scalars().all()

    return {
        "keys": [
            APIKeyResponse(
                id=k.id,
                name=k.name,
                description=k.description,
                key_prefix=k.key_prefix,
                scopes=k.scopes,
                is_active=k.is_active,
                expires_at=k.expires_at.isoformat() if k.expires_at else None,
                last_used_at=k.last_used_at.isoformat() if k.last_used_at else None,
                last_used_ip=k.last_used_ip,
                request_count=k.request_count,
                rate_limit_per_minute=k.rate_limit_per_minute,
                rate_limit_per_hour=k.rate_limit_per_hour,
                allowed_ips=k.allowed_ips,
                created_at=k.created_at.isoformat() if k.created_at else "",
            )
            for k in keys
        ],
        "total": len(keys),
    }


@router.get("/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIKeyResponse:
    """Get details of a specific API key."""
    result = await db.execute(
        select(APIKey).where(APIKey.id == key_id, APIKey.user_id == current_user.id)
    )
    key = result.scalar_one_or_none()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    return APIKeyResponse(
        id=key.id,
        name=key.name,
        description=key.description,
        key_prefix=key.key_prefix,
        scopes=key.scopes,
        is_active=key.is_active,
        expires_at=key.expires_at.isoformat() if key.expires_at else None,
        last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
        last_used_ip=key.last_used_ip,
        request_count=key.request_count,
        rate_limit_per_minute=key.rate_limit_per_minute,
        rate_limit_per_hour=key.rate_limit_per_hour,
        allowed_ips=key.allowed_ips,
        created_at=key.created_at.isoformat() if key.created_at else "",
    )


@router.patch("/{key_id}", response_model=APIKeyResponse)
async def update_api_key(
    key_id: str,
    request: UpdateAPIKeyRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIKeyResponse:
    """Update an API key's settings."""
    result = await db.execute(
        select(APIKey).where(APIKey.id == key_id, APIKey.user_id == current_user.id)
    )
    key = result.scalar_one_or_none()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    # Update fields
    if request.name is not None:
        key.name = request.name
    if request.description is not None:
        key.description = request.description
    if request.scopes is not None:
        key.scopes = validate_scopes(request.scopes)
    if request.is_active is not None:
        key.is_active = request.is_active
    if request.rate_limit_per_minute is not None:
        key.rate_limit_per_minute = request.rate_limit_per_minute
    if request.allowed_ips is not None:
        key.allowed_ips = request.allowed_ips

    await db.flush()
    await db.refresh(key)

    return APIKeyResponse(
        id=key.id,
        name=key.name,
        description=key.description,
        key_prefix=key.key_prefix,
        scopes=key.scopes,
        is_active=key.is_active,
        expires_at=key.expires_at.isoformat() if key.expires_at else None,
        last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
        last_used_ip=key.last_used_ip,
        request_count=key.request_count,
        rate_limit_per_minute=key.rate_limit_per_minute,
        rate_limit_per_hour=key.rate_limit_per_hour,
        allowed_ips=key.allowed_ips,
        created_at=key.created_at.isoformat() if key.created_at else "",
    )


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Revoke (delete) an API key.

    This action is permanent and cannot be undone.
    """
    result = await db.execute(
        select(APIKey).where(APIKey.id == key_id, APIKey.user_id == current_user.id)
    )
    key = result.scalar_one_or_none()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    await db.execute(delete(APIKey).where(APIKey.id == key_id))
    await db.flush()


@router.post("/{key_id}/rotate", response_model=RotateAPIKeyResponse)
async def rotate_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Rotate an API key.

    This revokes the old key and creates a new one with the same settings.
    The new key is only returned once - save it securely!
    """
    result = await db.execute(
        select(APIKey).where(APIKey.id == key_id, APIKey.user_id == current_user.id)
    )
    old_key = result.scalar_one_or_none()

    if not old_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    # Generate new key
    full_key, key_prefix, key_hash = generate_api_key()

    # Create new key with same settings
    new_key = APIKey(
        name=old_key.name,
        description=old_key.description,
        key_hash=key_hash,
        key_prefix=key_prefix,
        user_id=current_user.id,
        scopes=old_key.scopes,
        expires_at=old_key.expires_at,
        rate_limit_per_minute=old_key.rate_limit_per_minute,
        rate_limit_per_hour=old_key.rate_limit_per_hour,
        allowed_ips=old_key.allowed_ips,
        allowed_origins=old_key.allowed_origins,
    )
    db.add(new_key)

    # Revoke old key
    await db.execute(delete(APIKey).where(APIKey.id == key_id))

    await db.flush()
    await db.refresh(new_key)

    return {
        "old_key_id": key_id,
        "new_key": {
            "id": new_key.id,
            "name": new_key.name,
            "key": full_key,
            "key_prefix": key_prefix,
            "scopes": new_key.scopes,
            "expires_at": new_key.expires_at.isoformat() if new_key.expires_at else None,
            "message": "Save this key - it won't be shown again!",
        },
        "message": "Old key has been revoked. Save the new key - it won't be shown again!",
    }


@router.get("/{key_id}/usage", response_model=APIKeyUsageResponse)
async def get_api_key_usage(
    key_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIKeyUsageResponse:
    """Get usage statistics for an API key."""
    result = await db.execute(
        select(APIKey).where(APIKey.id == key_id, APIKey.user_id == current_user.id)
    )
    key = result.scalar_one_or_none()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    return APIKeyUsageResponse(
        id=key.id,
        name=key.name,
        request_count=key.request_count,
        last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
        last_used_ip=key.last_used_ip,
    )


# Authentication helper for API key-based auth


async def get_user_from_api_key(
    api_key: str,
    db: AsyncSession,
    required_scope: Optional[str] = None,
    client_ip: Optional[str] = None,
) -> User:
    """
    Authenticate a user via API key.

    Args:
        api_key: The API key from the request
        db: Database session
        required_scope: Optional scope requirement
        client_ip: Client IP for tracking and restrictions

    Returns:
        User: The authenticated user

    Raises:
        HTTPException: If authentication fails
    """
    if not api_key or not api_key.startswith(API_KEY_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format",
        )

    # Hash the key and look it up
    key_hash = hash_api_key(api_key)

    result = await db.execute(
        select(APIKey).where(APIKey.key_hash == key_hash)
    )
    key_record = result.scalar_one_or_none()

    if not key_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # Check if key is valid
    if not key_record.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is disabled",
        )

    if key_record.is_expired():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired",
        )

    # Check IP restrictions
    if client_ip and not key_record.is_ip_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="IP address not allowed for this API key",
        )

    # Check scope
    if required_scope and not key_record.has_scope(required_scope):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"API key does not have required scope: {required_scope}",
        )

    # Update usage tracking
    key_record.last_used_at = datetime.now(timezone.utc)
    key_record.last_used_ip = client_ip
    key_record.request_count += 1
    await db.flush()

    # Get user
    user_result = await db.execute(
        select(User).where(User.id == key_record.user_id)
    )
    user = user_result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled",
        )

    return user
