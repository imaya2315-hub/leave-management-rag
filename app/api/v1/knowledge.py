from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_manager
from app.models.employee import Employee
from app.schemas.knowledge import KnowledgeDocumentCreate, KnowledgeDocumentOut
from app.services import knowledge_service

router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])


@router.post("/documents", response_model=KnowledgeDocumentOut)
def add_document(
    document: KnowledgeDocumentCreate,
    db: Session = Depends(get_db),
    manager: Employee = Depends(require_manager),
):
    """
    Manager-only. Ingests a policy document (plain text — leave policy,
    eligibility rules, holiday calendar, handbook section, etc.) into
    the knowledge base the assistant retrieves from.
    """
    return knowledge_service.add_document(db, document.title, document.content)


@router.get("/documents", response_model=list[KnowledgeDocumentOut])
def list_documents(
    db: Session = Depends(get_db),
    manager: Employee = Depends(require_manager),
):
    """Manager-only. Lists documents currently in the knowledge base."""
    return knowledge_service.list_documents(db)


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    manager: Employee = Depends(require_manager),
):
    """Manager-only. Removes a document (and its chunks) from the knowledge base."""
    knowledge_service.delete_document(db, document_id)
    return {"status": "deleted"}
