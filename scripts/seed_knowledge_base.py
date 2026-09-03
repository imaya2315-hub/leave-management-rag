"""
Loads the sample policy documents shipped in app/rag/policies/ into the
knowledge base, so the AI assistant has something to retrieve from out
of the box.

Usage:
    python scripts/seed_knowledge_base.py

Safe to re-run: it skips any document whose title is already in the
knowledge base instead of duplicating it.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.core.database import Base, SessionLocal, engine
from app import crud
from app.rag.ingest import ingest_document

POLICIES_DIR = pathlib.Path(__file__).resolve().parent.parent / "app" / "rag" / "policies"

# Maps filename -> display title for the knowledge base.
DOCUMENTS = {
    "leave_types_and_eligibility.txt": "Leave Types and Eligibility",
    "carry_forward_rules.txt": "Leave Carry-Forward Rules",
    "holiday_calendar.txt": "Company Holiday Calendar 2026",
    "approval_escalation.txt": "Leave Approval and Escalation Procedure",
}


def main():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        existing_titles = {doc.title for doc in crud.knowledge.get_documents(db)}

        for filename, title in DOCUMENTS.items():
            if title in existing_titles:
                print(f"Skipping '{title}' — already in the knowledge base")
                continue

            content = (POLICIES_DIR / filename).read_text(encoding="utf-8")
            document = ingest_document(db, title, content)
            print(f"Ingested '{title}' ({len(document.chunks)} chunks)")

    finally:
        db.close()


if __name__ == "__main__":
    main()
