from fastapi import HTTPException
from sqlalchemy.orm import Session

from app import crud
from app.models.knowledge import KnowledgeDocument
from app.rag.ingest import ingest_document


def add_document(db: Session, title: str, content: str) -> dict:
    try:
        document = ingest_document(db, title, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _to_summary(document)


def list_documents(db: Session) -> list[dict]:
    return [_to_summary(document) for document in crud.knowledge.get_documents(db)]


def delete_document(db: Session, document_id: int) -> None:
    document = crud.knowledge.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Knowledge document not found")
    crud.knowledge.delete_document(db, document)


def _to_summary(document: KnowledgeDocument) -> dict:
    return {
        "id": document.id,
        "title": document.title,
        "created_at": document.created_at,
        "chunk_count": len(document.chunks),
    }
