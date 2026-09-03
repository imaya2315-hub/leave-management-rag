from datetime import date
from typing import Optional

from pydantic import BaseModel


class AssistantAskRequest(BaseModel):
    question: str
    # Optional — when the employee is asking about a *specific* request
    # ("can I take 3 days next week"), supplying these lets the assistant
    # also run a live, non-mutating eligibility check via leave_service,
    # instead of only answering from policy text.
    leave_type: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class AssistantSource(BaseModel):
    """A policy chunk the answer was grounded in, for transparency."""
    title: str
    snippet: str


class AssistantEligibility(BaseModel):
    eligible: bool
    reason: str
    requested_days: Optional[int] = None
    remaining_balance_after: Optional[int] = None


class AssistantAskResponse(BaseModel):
    answer: str
    sources: list[AssistantSource]
    eligibility: Optional[AssistantEligibility] = None
