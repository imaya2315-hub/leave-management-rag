"""
Stage 1: Document ingestion.

Loads raw policy documents from disk. Kept separate from chunking so
each stage of the pipeline can be run and inspected on its own —
that's the whole point of this lab.
"""
from pathlib import Path


def load_documents(folder: str) -> list[dict]:
    """Returns [{"title": ..., "filename": ..., "source": ..., "content": ...}, ...] for every .txt file in folder."""
    documents = []
    for path in sorted(Path(folder).glob("*.txt")):
        title = path.stem.replace("_", " ").title()
        documents.append({
            "title": title,
            "filename": path.name,
            "source": str(path).replace("\\", "/"),
            "content": path.read_text(encoding="utf-8"),
        })
    return documents



if __name__ == "__main__":
    # Run directly to inspect what got loaded: python -m rag_lab.ingestion
    docs = load_documents("rag_lab/documents")
    for doc in docs:
        print(f"- {doc['title']}  ({len(doc['content'].split())} words)")
