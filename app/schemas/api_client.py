from pydantic import BaseModel, ConfigDict


class ApiClientCreateRequest(BaseModel):
    name: str


class ApiClientCreated(BaseModel):
    """Returned exactly once, at creation time — the plain-text secret
    is never retrievable again after this response."""
    client_id: str
    client_secret: str
    name: str


class ApiClientOut(BaseModel):
    id: int
    name: str
    client_id: str
    is_active: bool

    model_config = ConfigDict(from_attributes=True)
