"""
Stage 2: Chunking.

Splits document text into overlapping chunks. Deliberately standalone
(no import from app/rag/ingest.py) — this lab has no dependency on the
FastAPI app, so every stage can be run and inspected in isolation.

Why chunk at all: an embedding of an entire policy document blurs
every rule in it into one vector, so a specific question ("carry
forward") retrieves the whole document with no way to tell which part
answered it. Chunking gives each rule its own, separately retrievable
vector.
"""


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
    """
    Splits into overlapping word-count chunks. The overlap keeps a
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


def chunk_documents(documents: list[dict], chunk_size: int = 120, overlap: int = 20) -> list[dict]:
    """Returns a flat list of {"title", "chunk_index", "content"} dicts across all documents."""
    chunks = []
    for doc in documents:
        for i, text in enumerate(chunk_text(doc["content"], chunk_size, overlap)):
            chunks.append({"title": doc["title"], "chunk_index": i, "content": text})
    return chunks


if __name__ == "__main__":
    # Run directly to see what a real chunk looks like:
    # python -m rag_lab.chunking
    from rag_lab.ingestion import load_documents

    docs = load_documents("rag_lab/documents")
    chunks = chunk_documents(docs)
    print(f"{len(docs)} documents -> {len(chunks)} chunks\n")
    for chunk in chunks[:2]:
        print(f"[{chunk['title']} #{chunk['chunk_index']}]")
        print(chunk["content"][:200] + "...\n")
