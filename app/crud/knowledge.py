"""
CRUD layer: direct database operations only. No chunking, no
retrieval logic here — that belongs in app/rag/. This layer just
knows how to read/write KnowledgeDocument and KnowledgeChunk rows.
"""
from sqlalchemy.orm import Session, joinedload

from app.models.knowledge import KnowledgeDocument, KnowledgeChunk


def create_document(db: Session, title: str, chunk_texts: list[str]) -> KnowledgeDocument:
    document = KnowledgeDocument(title=title)
    db.add(document)
    db.flush()  # assigns document.id before the chunks are created below

    for index, text in enumerate(chunk_texts):
        db.add(KnowledgeChunk(document_id=document.id, chunk_index=index, content=text))

    db.commit()
    db.refresh(document)
    return document


def get_documents(db: Session) -> list[KnowledgeDocument]:
    return db.query(KnowledgeDocument).options(joinedload(KnowledgeDocument.chunks)).all()


def get_document(db: Session, document_id: int) -> KnowledgeDocument | None:
    return db.query(KnowledgeDocument).filter(KnowledgeDocument.id == document_id).first()


def delete_document(db: Session, document: KnowledgeDocument) -> None:
    db.delete(document)
    db.commit()


def get_all_chunks(db: Session) -> list[KnowledgeChunk]:
    return db.query(KnowledgeChunk).options(joinedload(KnowledgeChunk.document)).all()
