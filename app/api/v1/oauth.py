from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_manager
from app.core.config import settings
from app.core.security import create_service_token, verify_password
from app.crud.api_client import get_client_by_client_id
from app.models.employee import Employee
from app.schemas.api_client import ApiClientCreateRequest, ApiClientCreated
from app.schemas.token import ClientCredentialsToken
from app.services import api_client_service

router = APIRouter(prefix="/oauth", tags=["OAuth2 (Service Clients)"])


@router.post("/clients", response_model=ApiClientCreated)
def create_client(
    request: ApiClientCreateRequest,
    db: Session = Depends(get_db),
    manager: Employee = Depends(require_manager),
):
    """
    Manager-only. Registers a new automation client (e.g. n8n) and
    returns its client_id + client_secret. The secret is shown ONCE —
    store it immediately, since it cannot be retrieved again.
    """
    return api_client_service.register_client(db, request)


@router.post("/token", response_model=ClientCredentialsToken)
def issue_token(
    grant_type: str = Form(...),
    client_id: str = Form(...),
    client_secret: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    Standard OAuth2 Client Credentials token endpoint — this exact
    shape (form-encoded grant_type/client_id/client_secret in, JSON
    access_token/token_type/expires_in out) is what n8n's built-in
    OAuth2 "Client Credentials" credential type expects, so it can be
    configured to fetch and refresh tokens automatically with no
    custom workflow logic needed.
    """
    if grant_type != "client_credentials":
        raise HTTPException(status_code=400, detail="Only grant_type=client_credentials is supported")

    client = get_client_by_client_id(db, client_id)
    if not client or not client.is_active or not verify_password(client_secret, client.hashed_client_secret):
        raise HTTPException(status_code=401, detail="Invalid client_id or client_secret")

    access_token = create_service_token(data={"sub": client.client_id})
    return ClientCredentialsToken(
        access_token=access_token,
        expires_in=settings.SERVICE_TOKEN_EXPIRE_MINUTES * 60,
    )
