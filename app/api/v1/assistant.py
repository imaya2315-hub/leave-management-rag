from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.models.employee import Employee
from app.schemas.assistant import AssistantAskRequest, AssistantAskResponse
from app.services import assistant_service

router = APIRouter(prefix="/assistant", tags=["Assistant"])


@router.post("/ask", response_model=AssistantAskResponse)
def ask_assistant(
    request: AssistantAskRequest,
    db: Session = Depends(get_db),
    current_user: Employee = Depends(get_current_user),
):
    """
    Ask the leave assistant a question in plain English, e.g.
    "How many casual leaves can I take in a year?" or, for a live
    eligibility check, pair the question with leave_type/start_date/
    end_date, e.g. "Can I take 3 days off next week?"

    Retrieves the relevant policy (RAG) and — when leave_type,
    start_date, and end_date are all supplied — also runs a live,
    non-mutating eligibility check against the current employee's
    balance, without creating an actual leave request.
    """
    return assistant_service.ask(db, current_user, request)
