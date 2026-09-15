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


def derive_metadata(doc: dict, chunk_text: str, chunk_index: int) -> dict:
    """
    Derives grounded metadata from policy document and chunk text.
    Categories and leave types are derived strictly from policy documents:
      - Leave Types and Eligibility -> category: "eligibility", leave_type: "annual" / "sick" / "casual" / "general"
      - Carry Forward Rules -> category: "carry_forward", leave_type: "annual" / "sick" / "casual" / "general"
      - Holiday Calendar -> category: "holiday", leave_type: "general"
      - Approval Escalation -> category: "approval", leave_type: "general"
    """
    title = doc.get("title", "")
    filename = doc.get("filename", "")
    if not filename and title:
        filename = title.lower().replace(" ", "_") + ".txt"

    source = doc.get("source", filename or title)

    norm_title = title.lower()
    norm_fn = filename.lower()
    if "eligibility" in norm_title or "types" in norm_title or "leave_types" in norm_fn:
        category = "eligibility"
    elif "carry" in norm_title or "carry_forward" in norm_fn:
        category = "carry_forward"
    elif "holiday" in norm_title or "holiday_calendar" in norm_fn:
        category = "holiday"
    elif "approval" in norm_title or "escalation" in norm_title or "approval_escalation" in norm_fn:
        category = "approval"
    else:
        category = "general"

    text_lower = chunk_text.lower()
    if category in ("holiday", "approval"):
        leave_type = "general"
    else:
        ac = text_lower.count("annual leave")
        sc = text_lower.count("sick leave")
        cc = text_lower.count("casual leave")

        if "unused annual leave:" in text_lower or "annual leave:" in text_lower:
            leave_type = "annual" if (ac >= sc and ac >= cc) else "general"
        elif "unused sick leave:" in text_lower or "sick leave:" in text_lower:
            leave_type = "sick" if (sc >= ac and sc >= cc) else "general"
        elif "unused casual leave:" in text_lower or "casual leave:" in text_lower:
            leave_type = "casual" if (cc >= ac and cc >= sc) else "general"
        elif ac > sc and ac > cc and ac >= 2:
            leave_type = "annual"
        elif sc > ac and sc > cc and sc >= 2:
            leave_type = "sick"
        elif cc > ac and cc > sc and cc >= 2:
            leave_type = "casual"
        else:
            leave_type = "general"


    doc_version = "2026.1" if "2026" in text_lower or "2026" in norm_title else "v1.0"

    return {
        "source": source,
        "filename": filename,
        "policy_category": category,
        "leave_type": leave_type,
        "document_version": doc_version,
        "chunk_index": chunk_index,
        "text": chunk_text,
    }


def chunk_documents(documents: list[dict], chunk_size: int = 120, overlap: int = 20) -> list[dict]:
    """Returns a flat list of chunk dicts across all documents with enriched metadata."""
    chunks = []
    for doc in documents:
        for i, text in enumerate(chunk_text(doc["content"], chunk_size, overlap)):
            meta = derive_metadata(doc, text, i)
            chunk_dict = {
                "title": doc["title"],
                "chunk_index": i,
                "content": text,
                **meta,
            }
            chunks.append(chunk_dict)
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
