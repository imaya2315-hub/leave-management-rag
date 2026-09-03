from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship

from app.core.database import Base


class KnowledgeDocument(Base):
    """
    One ingested policy document (e.g. 'Leave Policy 2026',
    'Holiday Calendar'). The raw text isn't stored here — only its
    title and the chunks it was split into for retrieval.
    """
    __tablename__ = "knowledge_documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    chunks = relationship(
        "KnowledgeChunk", back_populates="document", cascade="all, delete-orphan"
    )


class KnowledgeChunk(Base):
    """
    A retrievable slice of a document's text. The assistant's RAG
    retriever searches over these rows, not whole documents.
    """
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("knowledge_documents.id"), index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)

    document = relationship("KnowledgeDocument", back_populates="chunks")
