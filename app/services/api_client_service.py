from sqlalchemy.orm import Session

from app import crud
from app.core.security import generate_client_credentials, get_password_hash
from app.schemas.api_client import ApiClientCreateRequest, ApiClientCreated


def register_client(db: Session, request: ApiClientCreateRequest) -> ApiClientCreated:
    client_id, client_secret = generate_client_credentials()
    hashed_secret = get_password_hash(client_secret)

    crud.api_client.create_client(db, request.name, client_id, hashed_secret)

    # The plain-text secret is returned ONLY here — it is never stored
    # and can never be retrieved again after this response.
    return ApiClientCreated(client_id=client_id, client_secret=client_secret, name=request.name)
