"""
Security utilities: password hashing (bcrypt) and JWT token creation.
This is what used to be called auth.py — renamed to security.py since
that's the conventional name for this concern in larger FastAPI projects.

Two kinds of tokens are issued at login:
- access token  (short-lived, 30 min)  — sent on every protected request
- refresh token (long-lived, 7 days)   — used ONLY to get a new access
  token via /refresh, when the access token has expired

A third kind is issued via OAuth2 Client Credentials, for service /
automation accounts (e.g. n8n) rather than human employees:
- service token (short-lived) — proves "this request comes from a
  trusted automation client," not any particular employee

All three are signed with the same SECRET_KEY but carry a "type" claim
so one can never be used in place of another.
"""
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.core.config import settings


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8")
    )


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")


def generate_client_credentials() -> tuple[str, str]:
    """Generates a new (client_id, client_secret) pair for an ApiClient.
    The secret is returned in plain text ONCE — only its hash is stored."""
    client_id = f"client_{secrets.token_hex(8)}"
    client_secret = secrets.token_urlsafe(32)
    return client_id, client_secret


def _create_token(data: dict, expires_delta: timedelta, token_type: str) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire, "type": token_type})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_access_token(data: dict) -> str:
    return _create_token(
        data,
        timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        token_type="access",
    )


def create_refresh_token(data: dict) -> str:
    return _create_token(
        data,
        timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        token_type="refresh",
    )


def create_service_token(data: dict) -> str:
    return _create_token(
        data,
        timedelta(minutes=settings.SERVICE_TOKEN_EXPIRE_MINUTES),
        token_type="service",
    )


def decode_token(token: str) -> dict:
    """
    Raises jwt.PyJWTError (caught by callers) if the token is invalid,
    tampered with, or expired.
    """
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
