"""
Authentication routes for user login, registration, and email verification.
"""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from brain.config import get_settings
from brain.models.base import get_db
from brain.models.user import User
from brain.services.email import get_email_service
from shared.schemas import UserCreate, UserResponse, TokenResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])
settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.api_prefix}/auth/login")


# Additional response models
class MessageResponse(BaseModel):
    """Generic message response."""

    message: str
    success: bool = True


class VerificationStatusResponse(BaseModel):
    """Email verification status."""

    is_verified: bool
    email: str
    verification_sent: bool = False


class ResendVerificationRequest(BaseModel):
    """Request to resend verification email."""

    email: EmailStr


class ForgotPasswordRequest(BaseModel):
    """Request to initiate password reset."""

    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """Request to reset password with token."""

    token: str
    new_password: str


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.jwt_access_token_expire_minutes)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def generate_verification_token() -> str:
    """Generate a secure verification token."""
    return secrets.token_urlsafe(32)


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Get the current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_current_verified_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Get current verified user (for endpoints requiring email verification)."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    if settings.email_verification_required and not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required. Please verify your email address.",
        )
    return current_user


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """
    Register a new user.

    A verification email will be sent if email verification is enabled.
    """
    # Check if email already exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Generate verification token
    verification_token = generate_verification_token()

    # Create user
    user = User(
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        name=user_data.name,
        verification_token=verification_token,
        is_verified=False,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)

    # Send verification email
    email_service = get_email_service()
    await email_service.send_verification_email(user.email, verification_token)

    return user


@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """
    Login and get access token.

    If email verification is required, unverified users cannot login.
    """
    # Find user
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    # Check email verification if required
    if settings.email_verification_required and not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified. Please check your email for the verification link.",
        )

    # Update last login
    user.last_login_at = datetime.now(timezone.utc)
    await db.flush()

    # Create token
    access_token = create_access_token(data={"sub": user.id})
    expires_in = settings.jwt_access_token_expire_minutes * 60

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": expires_in,
    }


@router.get("/verify-email", response_model=MessageResponse)
async def verify_email(
    token: str = Query(..., description="Verification token from email"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Verify email address using token from verification email.

    This endpoint is typically accessed via the link in the verification email.
    """
    # Find user by verification token
    result = await db.execute(
        select(User).where(User.verification_token == token)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token",
        )

    if user.is_verified:
        return {"message": "Email already verified", "success": True}

    # Verify user
    user.is_verified = True
    user.verification_token = None  # Clear token after use
    await db.flush()

    # Send welcome email
    email_service = get_email_service()
    await email_service.send_welcome_email(user.email, user.name)

    return {"message": "Email verified successfully. Welcome!", "success": True}


@router.post("/resend-verification", response_model=MessageResponse)
async def resend_verification(
    request: ResendVerificationRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Resend verification email.

    Use this if the original verification email was not received.
    """
    # Find user
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    # Don't reveal if email exists or not
    if not user:
        return {
            "message": "If this email is registered, a verification link will be sent.",
            "success": True,
        }

    if user.is_verified:
        return {"message": "Email is already verified.", "success": True}

    # Generate new token
    user.verification_token = generate_verification_token()
    await db.flush()

    # Send verification email
    email_service = get_email_service()
    await email_service.send_verification_email(user.email, user.verification_token)

    return {
        "message": "If this email is registered, a verification link will be sent.",
        "success": True,
    }


@router.get("/verification-status", response_model=VerificationStatusResponse)
async def get_verification_status(
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """Get current user's email verification status."""
    return {
        "is_verified": current_user.is_verified,
        "email": current_user.email,
        "verification_sent": current_user.verification_token is not None,
    }


@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(
    request: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Initiate password reset process.

    Sends a password reset email if the email is registered.
    """
    # Find user
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    # Don't reveal if email exists or not
    if not user:
        return {
            "message": "If this email is registered, a password reset link will be sent.",
            "success": True,
        }

    # Generate reset token (reuse verification_token field)
    user.verification_token = generate_verification_token()
    await db.flush()

    # Send reset email
    email_service = get_email_service()
    await email_service.send_password_reset_email(user.email, user.verification_token)

    return {
        "message": "If this email is registered, a password reset link will be sent.",
        "success": True,
    }


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(
    request: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Reset password using token from reset email.
    """
    # Find user by token
    result = await db.execute(
        select(User).where(User.verification_token == request.token)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    # Update password
    user.hashed_password = get_password_hash(request.new_password)
    user.verification_token = None  # Clear token after use
    await db.flush()

    return {"message": "Password reset successfully. You can now login.", "success": True}


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> User:
    """Get current user info."""
    return current_user
