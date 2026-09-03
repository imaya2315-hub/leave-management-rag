"""
Shared FastAPI dependencies, used across more than one router.
Route files import get_db and get_current_user from here rather than
duplicating this logic or importing directly from core/database.
"""
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db  # re-exported for convenience
from app.core.security import decode_token
from app.models.api_client import ApiClient
from app.models.employee import Employee

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_PREFIX}/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Employee:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials or token expired",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_token(token)
        email: str = payload.get("sub")
        # A refresh token must never be accepted as an access token —
        # otherwise a leaked refresh token could be used directly on
        # every protected endpoint instead of only /refresh.
        if email is None or payload.get("type") != "access":
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = db.query(Employee).filter(Employee.email == email).first()
    if user is None:
        raise credentials_exception

    return user


def require_manager(current_user: Employee = Depends(get_current_user)) -> Employee:
    """
    Use as a dependency on any route that only managers should access
    (e.g. approving or rejecting leave requests).
    """
    if current_user.role != "manager":
        raise HTTPException(
            status_code=403,
            detail="Only managers can perform this action",
        )
    return current_user


def require_service_client(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> ApiClient:
    """
    Use as a dependency on any route meant for trusted automation
    clients (e.g. n8n) authenticated via OAuth2 Client Credentials —
    not a human employee login. Rejects employee access/refresh tokens
    just as strictly as it rejects invalid ones.
    """
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate service credentials or token expired",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_token(token)
        client_id: str = payload.get("sub")
        if client_id is None or payload.get("type") != "service":
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    client = db.query(ApiClient).filter(ApiClient.client_id == client_id).first()
    if client is None or not client.is_active:
        raise credentials_exception

    return client
