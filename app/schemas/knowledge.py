from datetime import datetime

from pydantic import BaseModel, ConfigDict


class KnowledgeDocumentCreate(BaseModel):
    title: str
    content: str  # raw policy text; chunked automatically on ingestion


class KnowledgeDocumentOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    chunk_count: int

    model_config = ConfigDict(from_attributes=True)
