"""
Turns raw policy text into stored, retrievable chunks. This is the
"Knowledge Base" ingestion step — leave policies, eligibility rules,
carry-forward rules, holiday calendars, handbook sections, etc. all
go through this before the assistant can retrieve them.
"""
from sqlalchemy.orm import Session

from app import crud
from app.models.knowledge import KnowledgeDocument


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
    """
    Splits text into overlapping word-count chunks.

    Word-count chunking (rather than a fixed character count) keeps
    chunks readable and rarely splits mid-sentence. The overlap keeps a
    rule that happens to fall at a chunk boundary from being cut in
    half and losing retrievability.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap

    return chunks


def ingest_document(db: Session, title: str, content: str) -> KnowledgeDocument:
    chunks = chunk_text(content)
    if not chunks:
        raise ValueError("Document content is empty — nothing to ingest")
    return crud.knowledge.create_document(db, title, chunks)
