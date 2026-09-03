from sqlalchemy.orm import Session

from app.models.api_client import ApiClient


def get_client_by_client_id(db: Session, client_id: str) -> ApiClient | None:
    return db.query(ApiClient).filter(ApiClient.client_id == client_id).first()


def create_client(db: Session, name: str, client_id: str, hashed_secret: str) -> ApiClient:
    new_client = ApiClient(
        name=name,
        client_id=client_id,
        hashed_client_secret=hashed_secret,
    )
    db.add(new_client)
    db.commit()
    db.refresh(new_client)
    return new_client
