from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class ClientCredentialsToken(BaseModel):
    """Standard OAuth2 Client Credentials response shape — this is the
    exact format n8n's built-in OAuth2 credential type expects."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
