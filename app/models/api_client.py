from sqlalchemy import Column, Integer, String, Boolean

from app.core.database import Base


class ApiClient(Base):
    """
    A service account for machine-to-machine access (e.g. n8n), using
    OAuth2 Client Credentials — entirely separate from Employee logins.
    The client_secret is never stored in plain text, only its hash.
    """
    __tablename__ = "api_clients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)  # human-readable label, e.g. "n8n automation"
    client_id = Column(String, unique=True, index=True)
    hashed_client_secret = Column(String)
    is_active = Column(Boolean, default=True)
