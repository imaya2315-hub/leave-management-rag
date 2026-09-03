"""
Orchestrates the AI assistant: retrieves relevant policy context (RAG),
combines it with the employee's live leave data when relevant, and
produces one grounded answer.

This is the "Knowledge-Based AI" layer described in the architecture:

    Employee -> AI Assistant -> Intent Understanding
             -> RAG Knowledge Base + Employee/Leave Data
             -> Rules/Policy Validation -> Response

"Intent understanding" here is deliberately simple: rather than an
extra classification call, the caller (the API request) states intent
directly — a plain question is answered from policy alone; a question
paired with leave_type/start_date/end_date also triggers the live
eligibility check. leave_service remains the single source of truth
for balances and eligibility — the LLM only ever explains and phrases
what leave_service and the retriever already determined, never
computes it itself.
"""
from sqlalchemy.orm import Session

from app.models.employee import Employee
from app.rag import llm
from app.rag.retriever import retrieve_relevant_chunks
from app.schemas.assistant import AssistantAskRequest, AssistantAskResponse, AssistantSource
from app.services import leave_service

SNIPPET_LENGTH = 220


def ask(db: Session, employee: Employee, request: AssistantAskRequest) -> AssistantAskResponse:
    # 1. RAG: retrieve relevant policy chunks for the question.
    results = retrieve_relevant_chunks(db, request.question, top_k=4)
    policy_context = "\n\n".join(f"[{title}] {content}" for title, content, _ in results)
    sources = [
        AssistantSource(title=title, snippet=_snippet(content))
        for title, content, _ in results
    ]

    # 2. Knowledge-based check: only run the deterministic eligibility
    # check when the employee is asking about a specific request.
    eligibility = None
    if request.leave_type and request.start_date and request.end_date:
        eligibility = leave_service.check_eligibility(
            db, employee, request.leave_type, request.start_date, request.end_date
        )

    employee_context = _employee_context_summary(employee, request, eligibility)

    # 3. Generate the final answer, grounded in both contexts.
    answer = llm.generate_answer(request.question, policy_context, employee_context)

    return AssistantAskResponse(answer=answer, sources=sources, eligibility=eligibility)


def _employee_context_summary(employee: Employee, request: AssistantAskRequest, eligibility: dict | None) -> str:
    lines = [
        f"Name: {employee.first_name} {employee.last_name}",
        f"Department: {employee.department}",
        f"Annual leave balance: {employee.annual_leave_balance} days",
        f"Sick leave balance: {employee.sick_leave_balance} days",
        f"Casual leave balance: {employee.casual_leave_balance} days",
    ]

    if eligibility is not None:
        lines.append(
            f"Requested: {request.leave_type} leave from {request.start_date} to {request.end_date}"
        )
        lines.append(f"Eligible: {eligibility['eligible']}")
        lines.append(f"Reason: {eligibility['reason']}")
        if eligibility["requested_days"] is not None:
            lines.append(f"Requested working days: {eligibility['requested_days']}")
        if eligibility["remaining_balance_after"] is not None:
            lines.append(f"Balance remaining if approved: {eligibility['remaining_balance_after']}")

    return "\n".join(lines)


def _snippet(content: str) -> str:
    if len(content) <= SNIPPET_LENGTH:
        return content
    return content[:SNIPPET_LENGTH].rsplit(" ", 1)[0] + "..."
