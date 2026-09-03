"""
Importing all models here ensures they're registered on the shared
Base's metadata before `Base.metadata.create_all()` runs in main.py —
and it lets other files do `from app.models import Employee, Leave, ApiClient`.
"""
from app.models.employee import Employee
from app.models.leave import Leave
from app.models.api_client import ApiClient
from app.models.knowledge import KnowledgeDocument, KnowledgeChunk

__all__ = ["Employee", "Leave", "ApiClient", "KnowledgeDocument", "KnowledgeChunk"]
