"""
Streamlit chat front end for the leave assistant.

    STREAMLIT (this file)
    Chat interface
         |
         v
    Agent Router
       /        \\           \\
      v          v            v
  Policy      Live Data     Leave Action
  Question    (FastAPI)     (FastAPI)
      |
      v
  rag_lab.agent  (decomposition -> semantic retrieval ->
                  merge/dedup -> reranking -> grounded generation)

Three hard rules this file enforces:
1. The LLM never calls a backend endpoint directly — only the
   deterministic handle_action/handle_live_data functions do, using
   requests. Policy knowledge and live employee data / actions never
   mix in the same code path.
2. Policy questions don't require login (RAG has no notion of "your"
   data). Live data and actions do.
3. The LLM tool router distinguishes ASKING about a rule ("can I cancel
   an approved leave?" -> policy) from ASKING the system to DO
   something ("cancel my pending leave" -> action). Only imperative,
   first-person phrasing about the user's own live request triggers
   an action.

Run with:
    streamlit run streamlit_app.py

Requires the FastAPI backend running separately:
    uvicorn app.main:app --reload
"""
import json
import os
import re
from datetime import date, datetime, timedelta

import requests
import streamlit as st

from rag_lab.agent import answer_policy_question

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_BASE = "http://localhost:8000/api/v1"

LEAVE_TYPES = ("annual", "sick", "casual")
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

# Phrases that make a message a question ABOUT a rule, not a request
# to DO something — checked first so e.g. "can I cancel an approved
# leave?" never reaches the action handler.
_POLICY_QUESTION_MARKERS = (
    "can i", "could i", "am i able to", "is it possible", "what is",
    "what are", "how many", "how much", "how long", "do i need",
    "does", "is there", "are there", "what happens",
)

_APPLY_MARKERS = ("apply", "request leave", "book leave", "take leave", "submit leave")
_CANCEL_MARKERS = ("cancel my", "cancel the", "cancel leave", "cancel pending", "cancel request")
_LIVE_DATA_MARKERS = (
    "my balance", "my leave", "my status", "my history", "remaining balance",
    "how many do i have", "leave balance", "previous leaves", "past leaves",
)

# ---------------------------------------------------------------------------
# LLM tool-calling router
# ---------------------------------------------------------------------------
# The LLM chooses exactly one high-level tool for each new user message.
# The tool implementation below remains deterministic and calls FastAPI only
# from application code. The LLM never receives database/API credentials.
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "answer_policy_question",
            "description": (
                "Answer a general company leave-policy question using grounded RAG. "
                "Use for rules, eligibility policy, carry-forward, holidays, notice periods, "
                "approval rules, and similar policy information. Do not use for live employee data or actions."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_leave_balance",
            "description": (
                "Retrieve the authenticated employee's current Annual, Sick, and Casual leave balances."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_leave_history",
            "description": (
                "Retrieve the authenticated employee's own leave request history and statuses."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pending_leave_requests",
            "description": (
                "Manager-only tool. Retrieve all currently pending leave requests for the team. "
                "Use when a manager asks to show/list/view pending requests or what requests need action. "
                "The result is displayed using employee names, leave types, and dates; backend IDs are internal."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_leave",
            "description": (
                "Create a leave request for the authenticated employee when they are actually asking "
                "the system to request time off. Understand natural language such as 'I need Monday off' "
                "or 'I'll be away for three days starting Tuesday'. Extract only details explicitly supported "
                "by the message. Do not invent a missing leave type or date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "leave_type": {
                        "type": ["string", "null"],
                        "enum": ["Annual", "Sick", "Casual", None],
                        "description": "Leave type explicitly stated by the user, or null if missing.",
                    },
                    "start_date": {
                        "type": ["string", "null"],
                        "description": "Date mentioned by the user, or null if missing.",
                    },
                    "end_date": {
                        "type": ["string", "null"],
                        "description": "End date if explicitly specified or clearly implied by a duration, otherwise null.",
                    },
                },
                "required": ["leave_type", "start_date", "end_date"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_leave",
            "description": (
                "Cancel a leave request belonging to the authenticated employee. "
                "Use when the user wants to withdraw/remove/cancel their request. "
                "A backend leave ID may be used only when the user explicitly supplied it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "leave_id": {
                        "type": ["integer", "null"],
                        "description": "Backend leave ID only if explicitly supplied by the user, otherwise null.",
                    }
                },
                "required": ["leave_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "approve_leave",
            "description": (
                "Manager-only. Approve one pending leave request using the employee's name. "
                "Use start_date and/or leave_type when available to disambiguate multiple requests "
                "for the same employee. Never ask the manager to know the backend leave ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_name": {
                        "type": "string",
                        "description": "Employee name mentioned by the manager.",
                    },
                    "start_date": {
                        "type": ["string", "null"],
                        "description": "Requested leave start date if the manager specifies it.",
                    },
                    "leave_type": {
                        "type": ["string", "null"],
                        "enum": ["Annual", "Sick", "Casual", None],
                        "description": "Leave type if the manager specifies it.",
                    },
                },
                "required": ["employee_name", "start_date", "leave_type"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reject_leave",
            "description": (
                "Manager-only. Reject one pending leave request using the employee's name. "
                "Use start_date and/or leave_type when available to disambiguate multiple requests. "
                "Never ask the manager to know the backend leave ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_name": {
                        "type": "string",
                        "description": "Employee name mentioned by the manager.",
                    },
                    "start_date": {
                        "type": ["string", "null"],
                        "description": "Requested leave start date if the manager specifies it.",
                    },
                    "leave_type": {
                        "type": ["string", "null"],
                        "enum": ["Annual", "Sick", "Casual", None],
                        "description": "Leave type if the manager specifies it.",
                    },
                },
                "required": ["employee_name", "start_date", "leave_type"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_ROUTER_SYSTEM_PROMPT = """
You are the intent-and-tool router for a leave-management assistant.

Choose exactly ONE tool. Do not answer the user yourself.

Policy:
- answer_policy_question = general company policy/rules.
Employee live data:
- get_leave_balance = user's own current balances.
- get_leave_history = user's own history/status.
Manager live data:
- get_pending_leave_requests = a manager asks to list/show/view pending requests.
Actions:
- apply_leave = user is actually asking the system to request time off, including natural paraphrases like
  "I need Monday off", "I'll be away next Tuesday", or "I'd like some time off".
- cancel_leave = user wants to withdraw/remove/cancel their own request.
- approve_leave / reject_leave = manager wants to act on a pending request.

For manager approve/reject:
- Identify the employee by name.
- If several pending requests match, the application will ask the manager for leave type or start date and continue the same manager action.
- Extract start_date and leave_type only when the manager provides them.
- Never invent a backend leave ID.
- Never tell the manager they need to know the backend ID.

For apply_leave:
- Extract only information actually present in the user's message.
- Do not invent a leave type.
- Do not invent dates.
- Relative dates may be represented naturally; application code will normalize them safely.
- Return exactly one tool call.

Do not use an action tool for a hypothetical/policy question such as "Can I take Monday off?"
"""



def _get_tool_router_client():
    if Groq is None:
        return None

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def choose_tool(message: str) -> tuple[str, dict]:
    """Use the LLM to select one tool and extract structured arguments."""
    client = _get_tool_router_client()
    if client is None:
        raise RuntimeError("GROQ_API_KEY is not configured; LLM tool routing is unavailable.")

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=250,
        tool_choice="required",
        tools=TOOL_DEFINITIONS,
        messages=[
            {
                "role": "system",
                "content": (
                    TOOL_ROUTER_SYSTEM_PROMPT
                    + f"\nToday's date is {date.today().isoformat()}."
                ),
            },
            {"role": "user", "content": message},
        ],
    )

    tool_calls = response.choices[0].message.tool_calls or []
    if not tool_calls:
        raise RuntimeError("The LLM returned no tool call.")

    if len(tool_calls) > 1:
        # Never execute multiple leave mutations from a single user message.
        mutation_tools = {
            "apply_leave", "cancel_leave", "approve_leave", "reject_leave"
        }
        mutating = [tc for tc in tool_calls if tc.function.name in mutation_tools]
        if len(mutating) > 1:
            raise RuntimeError("The LLM returned multiple leave actions for one message; no action was executed.")
        call = mutating[0] if mutating else tool_calls[0]
    else:
        call = tool_calls[0]

    try:
        arguments = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"The LLM returned invalid tool arguments: {exc}") from exc

    return call.function.name, arguments

# ---------------------------------------------------------------------------
# Backend API helpers
# ---------------------------------------------------------------------------
def api_login(email: str, password: str) -> str:
    response = requests.post(f"{API_BASE}/login", data={"username": email, "password": password})
    response.raise_for_status()
    return response.json()["access_token"]


def api_get(path: str, token: str):
    response = requests.get(f"{API_BASE}{path}", headers={"Authorization": f"Bearer {token}"})
    response.raise_for_status()
    return response.json()


def api_post(path: str, token: str, payload: dict):
    response = requests.post(
        f"{API_BASE}{path}", json=payload, headers={"Authorization": f"Bearer {token}"}
    )
    response.raise_for_status()
    return response.json()
def api_put(path: str, token: str):
    """Send an authenticated PUT request to FastAPI."""

    response = requests.put(
        f"{API_BASE}{path}",
        headers={
            "Authorization": f"Bearer {token}",
        },
        timeout=30,
    )

    response.raise_for_status()
    return response.json()


def get_current_user(token: str) -> dict:
    """Return the authenticated user's profile."""
    return api_get("/employees/me/", token)


def is_manager(token: str) -> bool:
    """Return True only when the authenticated user has manager role."""
    profile = get_current_user(token)
    return str(profile.get("role", "")).strip().lower() == "manager"




# ---------------------------------------------------------------------------
# Agent router
# ---------------------------------------------------------------------------
def route_message(message: str) -> str:
    """
    Backward-compatible route label.

    New requests are classified by the LLM through tool calling rather than
    keyword matching. Pending multi-turn actions still bypass the LLM so that
    a reply such as "34" can safely complete a previously requested choice.
    """
    if st.session_state.get("pending_action"):
        return "pending_action"
    return "llm_tool"


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------
def _next_weekday(target_weekday: int, strictly_next_week: bool = False) -> date:
    today = date.today()
    days_ahead = (target_weekday - today.weekday()) % 7
    if days_ahead == 0 and not strictly_next_week:
        days_ahead = 0  # "Monday" said on a Monday means today
    elif days_ahead == 0 and strictly_next_week:
        days_ahead = 7
    if strictly_next_week and days_ahead < 7:
        days_ahead += 7
    return today + timedelta(days=days_ahead)


def parse_date_phrase(message: str) -> date | None:
    """
    Supports: today, tomorrow, <weekday>, next <weekday>,
    YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY.
    """
    lowered = message.lower()

    if "today" in lowered:
        return date.today()
    if "tomorrow" in lowered:
        return date.today() + timedelta(days=1)

    for i, weekday_name in enumerate(WEEKDAYS):
        if f"next {weekday_name}" in lowered:
            return _next_weekday(i, strictly_next_week=True)
    for i, weekday_name in enumerate(WEEKDAYS):
        if weekday_name in lowered:
            return _next_weekday(i, strictly_next_week=False)

    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", message)
    if iso_match:
        return date.fromisoformat(iso_match.group(1))

    slash_or_dash_match = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", message)
    if slash_or_dash_match:
        day, month, year = (int(part) for part in slash_or_dash_match.groups())
        return date(year, month, day)

    return None


def extract_leave_type(message: str) -> str | None:
    lowered = message.lower()
    for leave_type in LEAVE_TYPES:
        if leave_type in lowered:
            return leave_type.capitalize()
    return None

def extract_date_range(
    message: str,
) -> tuple[date, date] | None:
    """
    Extract a leave date range from natural language.

    Supports:
        today
        tomorrow
        Monday / Tuesday / ...
        this Monday / this Tuesday / ...
        next Monday / next Tuesday / ...
        YYYY-MM-DD
        DD/MM/YYYY
        DD-MM-YYYY
        2nd September
        September 2nd
        3 days from next Monday
        three days from next Tuesday
        5 days from 2026-09-14

    Duration is the total number of calendar days requested.
    The FastAPI backend remains responsible for validating
    weekends and holidays.
    """

    lowered = message.lower().strip()
    today = date.today()

    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    def parse_duration(text: str) -> int | None:
        match = re.search(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
            r"\s+(?:calendar\s+|working\s+)?days?\b",
            text,
        )
        if not match:
            return None

        value = match.group(1)
        return int(value) if value.isdigit() else number_words[value]

    def make_range(
        start: date,
        duration: int | None,
    ) -> tuple[date, date]:
        if duration is None:
            return start, start

        return start, start + timedelta(days=duration - 1)

    duration = parse_duration(lowered)

    # --------------------------------------------------------------
    # Explicit YYYY-MM-DD
    # --------------------------------------------------------------
    match = re.search(
        r"\b(\d{4}-\d{2}-\d{2})\b",
        message,
    )

    if match:
        try:
            parsed = date.fromisoformat(match.group(1))
            return make_range(parsed, duration)
        except ValueError:
            return None

    # --------------------------------------------------------------
    # Explicit DD/MM/YYYY or DD-MM-YYYY
    # --------------------------------------------------------------
    match = re.search(
        r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b",
        message,
    )

    if match:
        try:
            day, month, year = map(int, match.groups())
            parsed = date(year, month, day)
            return make_range(parsed, duration)
        except ValueError:
            return None

    # --------------------------------------------------------------
    # Natural date: "2nd September"
    # --------------------------------------------------------------
    months = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }

    match = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(january|february|march|april|may|june|july|august|"
        r"september|sept|october|november|december|jan|feb|mar|apr|jun|jul|aug|oct|nov|dec)"
        r"(?:\s+(\d{4}))?\b",
        lowered,
    )

    if match:
        day = int(match.group(1))
        month = months[match.group(2)]
        year = int(match.group(3)) if match.group(3) else today.year

        try:
            parsed = date(year, month, day)

            if (
                match.group(3) is None
                and parsed < today
            ):
                parsed = date(today.year + 1, month, day)

            return make_range(parsed, duration)
        except ValueError:
            return None

    # --------------------------------------------------------------
    # Natural date: "September 2nd"
    # --------------------------------------------------------------
    match = re.search(
        r"\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|"
        r"july|jul|august|aug|september|sept|october|oct|november|nov|december|dec)\s+"
        r"(\d{1,2})(?:st|nd|rd|th)?"
        r"(?:\s+(\d{4}))?\b",
        lowered,
    )

    if match:
        month = months[match.group(1)]
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else today.year

        try:
            parsed = date(year, month, day)

            if (
                match.group(3) is None
                and parsed < today
            ):
                parsed = date(today.year + 1, month, day)

            return make_range(parsed, duration)
        except ValueError:
            return None

    # --------------------------------------------------------------
    # Today / tomorrow
    # --------------------------------------------------------------
    if re.search(r"\btoday\b", lowered):
        return make_range(today, duration)

    if re.search(r"\btomorrow\b", lowered):
        return make_range(
            today + timedelta(days=1),
            duration,
        )

    # --------------------------------------------------------------
    # Weekdays
    # --------------------------------------------------------------
    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    for name, target_weekday in weekdays.items():

        if not re.search(
            rf"\b(?:this|next)?\s*{name}\b",
            lowered,
        ):
            continue

        days_ahead = (
            target_weekday - today.weekday()
        ) % 7

        is_next = bool(
            re.search(
                rf"\bnext\s+{name}\b",
                lowered,
            )
        )

        # "next Tuesday" on Tuesday means one full week ahead.
        # If the target weekday is already later this week, use it.
        if is_next and days_ahead == 0:
            days_ahead = 7

        parsed = today + timedelta(days=days_ahead)

        return make_range(
            parsed,
            duration,
        )

    # --------------------------------------------------------------
    # Next week
    # --------------------------------------------------------------
    if re.search(
        r"\bnext\s+week\b",
        lowered,
    ):
        return make_range(
            today + timedelta(days=7),
            duration,
        )

    return None

# ---------------------------------------------------------------------------
# Action handling — deterministic, never delegated to the LLM
# ---------------------------------------------------------------------------
def handle_action(
    message: str,
    token: str,
) -> str:
    """
    Handle actual leave actions.

    Supports:
        - Apply leave
        - Cancel pending leave
        - Multi-turn cancellation by ID
    """

    text = message.lower().strip()

    # ============================================================
    # CONTINUE MANAGER APPROVE / REJECT BY ID
    # ============================================================

    pending = st.session_state.get(
        "pending_action",
        {},
    )

    if pending.get("kind") == "manager_decision":
        return _continue_manager_decision(message, token)

    if pending.get("kind") == "cancel_choice":

        match = re.fullmatch(
            r"(?:leave\s*)?(\d+)",
            text,
        )

        if not match:
            return (
                "Please enter one of the leave request IDs "
                "shown above."
            )

        leave_id = int(
            match.group(1)
        )

        valid_ids = {
            int(leave["id"])
            for leave in pending.get(
                "pending_leaves",
                [],
            )
        }

        if leave_id not in valid_ids:

            return (
                "That is not one of the pending leave IDs. "
                "Please choose from: "
                + ", ".join(
                    str(x)
                    for x in sorted(valid_ids)
                )
            )

        try:

            leave = api_put(
                f"/leaves/{leave_id}/cancel",
                token,
            )

            st.session_state.pending_action = {}

            return (
                f"Cancelled: {leave['leave_type']} leave "
                f"from {leave['start_date']} to "
                f"{leave['end_date']} — "
                f"status **{leave['status']}**."
            )

        except requests.HTTPError as exc:

            try:
                detail = exc.response.json().get(
                    "detail",
                    exc.response.text,
                )
            except Exception:
                detail = str(exc)

            return (
                f"Couldn't cancel request {leave_id}: "
                f"{detail}"
            )

    # ============================================================
    # DETECT NEW CANCELLATION REQUEST
    # ============================================================

    is_cancel = (
        "cancel" in text
        and (
            "leave" in text
            or "request" in text
            or "pending" in text
        )
    )

    if is_cancel:

        try:

            history = api_get(
                "/leaves/me/",
                token,
            )

        except requests.HTTPError as exc:

            try:
                detail = exc.response.json().get(
                    "detail",
                    exc.response.text,
                )
            except Exception:
                detail = str(exc)

            return (
                f"Couldn't fetch your leave requests: "
                f"{detail}"
            )

        pending_leaves = [
            leave
            for leave in history
            if str(
                leave.get("status", "")
            ).lower() == "pending"
        ]

        # --------------------------------------------------------
        # No pending requests
        # --------------------------------------------------------

        if not pending_leaves:

            return (
                "You do not have any pending "
                "leave requests to cancel."
            )

        # --------------------------------------------------------
        # Exactly one pending request
        # --------------------------------------------------------

        if len(pending_leaves) == 1:

            leave = pending_leaves[0]

            try:

                cancelled = api_put(
                    f"/leaves/{leave['id']}/cancel",
                    token,
                )

                st.session_state.pending_action = {}

                return (
                    f"Cancelled: "
                    f"{cancelled['leave_type']} leave "
                    f"from {cancelled['start_date']} to "
                    f"{cancelled['end_date']} — "
                    f"status **{cancelled['status']}**."
                )

            except requests.HTTPError as exc:

                try:
                    detail = exc.response.json().get(
                        "detail",
                        exc.response.text,
                    )
                except Exception:
                    detail = str(exc)

                return (
                    f"Couldn't cancel request "
                    f"{leave['id']}: {detail}"
                )

        # --------------------------------------------------------
        # Multiple pending requests
        # --------------------------------------------------------

        st.session_state.pending_action = {
            "kind": "cancel_choice",
            "pending_leaves": pending_leaves,
        }

        lines = [
            "You have multiple pending requests:"
        ]

        for leave in pending_leaves:

            lines.append(
                f"- ID {leave['id']}: "
                f"{leave['leave_type']} leave, "
                f"{leave['start_date']} to "
                f"{leave['end_date']}"
            )

        lines.append(
            "Which ID would you like to cancel?"
        )

        return "\n".join(lines)

    # ============================================================
    # APPLY LEAVE
    # ============================================================

    pending = dict(
        st.session_state.get(
            "pending_action",
            {},
        )
    )

    extracted_leave_type = extract_leave_type(
        message
    )

    extracted_date_range = extract_date_range(
        message
    )

    leave_type = (
        extracted_leave_type
        or pending.get("leave_type")
    )

    date_range = (
        extracted_date_range
        or (
            (
                pending["start_date"],
                pending["end_date"],
            )
            if (
                "start_date" in pending
                and "end_date" in pending
            )
            else None
        )
    )

    if extracted_date_range:

        start_date, end_date = extracted_date_range

        pending["start_date"] = start_date
        pending["end_date"] = end_date

    if leave_type is None:

        pending["kind"] = "apply"

        st.session_state.pending_action = pending

        return (
            "Sure. Which leave type would you like to use — "
            "Annual, Sick, or Casual?"
        )

    pending["leave_type"] = leave_type
    pending["kind"] = "apply"

    if date_range is None:

        st.session_state.pending_action = pending

        return (
            f"Got it, {leave_type} leave. "
            "What date would you like to take it? "
            "(e.g. today, tomorrow, Monday, "
            "31/08/2026, or YYYY-MM-DD)"
        )

    start_date, end_date = date_range

    payload = {
        "leave_type": leave_type,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }

    try:

        leave = api_post(
            "/leaves/",
            token,
            payload,
        )

        st.session_state.pending_action = {}

        return (
            f"Submitted: {leave['leave_type']} leave "
            f"from {leave['start_date']} to "
            f"{leave['end_date']} — "
            f"status **{leave['status']}**, "
            "pending manager approval."
        )

    except requests.HTTPError as exc:

        st.session_state.pending_action = pending

        try:
            detail = exc.response.json().get(
                "detail",
                exc.response.text,
            )
        except Exception:
            detail = str(exc)

        return (
            f"Couldn't submit that request: "
            f"{detail}"
        )


def _handle_cancel(message: str, token: str) -> str:
    """
    Cancel workflow, per spec:
      0 pending  -> tell the user there's nothing to cancel
      1 pending  -> cancel it directly
      2+ pending -> list them (ID, type, dates) and ask which ID,
                    then wait for the user's next message to pick one
    """
    pending_choice = st.session_state.pending_action

    # If we already asked "which ID?" last turn, this message should be the ID.
    if pending_choice.get("kind") == "cancel_choice":
        id_match = re.search(r"\b(\d+)\b", message)
        if not id_match:
            return "Please reply with just the leave request ID you'd like to cancel."
        leave_id = int(id_match.group(1))
        st.session_state.pending_action = {}
        return _cancel_leave_by_id(leave_id, token)

    try:
        history = api_get("/leaves/me/", token)
    except requests.HTTPError as exc:
        return f"Couldn't fetch your leave history: {_extract_error_detail(exc)}"

    pending_requests = [leave for leave in history if leave["status"] == "Pending"]

    if not pending_requests:
        return "You have no pending leave requests to cancel."

    if len(pending_requests) == 1:
        return _cancel_leave_by_id(pending_requests[0]["id"], token)

    st.session_state.pending_action = {"kind": "cancel_choice"}
    lines = [
        f"- ID {leave['id']}: {leave['leave_type']} leave, {leave['start_date']} to {leave['end_date']}"
        for leave in pending_requests
    ]
    return "You have multiple pending requests:\n" + "\n".join(lines) + "\n\nWhich ID would you like to cancel?"


def _cancel_leave_by_id(leave_id: int, token: str) -> str:
    try:
        leave = api_put(f"/leaves/{leave_id}/cancel", token)
        return (
            f"Cancelled: {leave['leave_type']} leave from {leave['start_date']} to "
            f"{leave['end_date']} (request ID {leave['id']})."
        )
    except requests.HTTPError as exc:
        return f"Couldn't cancel request {leave_id}: {_extract_error_detail(exc)}"


def _extract_error_detail(exc: requests.HTTPError) -> str:
    try:
        return exc.response.json().get("detail", str(exc))
    except ValueError:
        return str(exc)


# ---------------------------------------------------------------------------
# Tool execution — local, deterministic implementations
# ---------------------------------------------------------------------------
def _tool_apply_leave(args: dict, original_message: str, token: str) -> str:
    """
    Execute an apply-leave tool call safely.

    The LLM decides that the user intends to apply for leave and may extract
    leave_type, but date normalization is performed from the ORIGINAL USER
    MESSAGE by our deterministic parser. This prevents the LLM from
    hallucinating or mis-normalizing relative dates such as "Monday" or
    "next Tuesday". FastAPI remains the final validation authority.
    """
    leave_type = args.get("leave_type")

    # ------------------------------------------------------------
    # IMPORTANT: Parse dates from the user's actual words.
    # Do NOT trust an LLM-generated date for transactional operations.
    # ------------------------------------------------------------
    date_range = extract_date_range(original_message)

    if date_range is not None:
        start_date, end_date = date_range

        synthetic_message = (
            f"Apply leave {leave_type}"
            if leave_type
            else "Apply leave"
        )
        synthetic_message += (
            f" from {start_date.isoformat()} to {end_date.isoformat()}"
        )

        return handle_action(synthetic_message, token)

    # No date was found in the original user message. Preserve the existing
    # multi-turn behavior and let the deterministic handler ask for the date.
    synthetic_message = (
        f"Apply leave {leave_type}"
        if leave_type
        else "Apply leave"
    )

    return handle_action(synthetic_message, token)


def _tool_cancel_leave(leave_id, token: str) -> str:
    if leave_id is not None:
        return _cancel_leave_by_id(int(leave_id), token)

    try:
        history = api_get("/leaves/me/", token)
    except requests.HTTPError as exc:
        return f"Couldn't fetch your leave requests: {_extract_error_detail(exc)}"

    pending_leaves = [
        leave for leave in history
        if str(leave.get("status", "")).lower() == "pending"
    ]

    if not pending_leaves:
        return "You do not have any pending leave requests to cancel."

    if len(pending_leaves) == 1:
        return _cancel_leave_by_id(int(pending_leaves[0]["id"]), token)

    st.session_state.pending_action = {
        "kind": "cancel_choice",
        "pending_leaves": pending_leaves,
    }

    lines = ["You have multiple pending requests:"]
    for leave in pending_leaves:
        lines.append(
            f"- ID {leave['id']}: {leave['leave_type']} leave, "
            f"{leave['start_date']} to {leave['end_date']}"
        )
    lines.append("Which ID would you like to cancel?")
    return "\n".join(lines)


def _tool_manager_action(action: str, leave_id, token: str) -> str:
    try:
        if not is_manager(token):
            return f"Manager authorization is required to {action} leave requests."
    except requests.HTTPError as exc:
        return f"Couldn't verify your manager authorization: {_extract_error_detail(exc)}"
    except Exception as exc:
        return f"Couldn't verify your manager authorization: {exc}"

    if leave_id is None:
        st.session_state.pending_action = {
            "kind": "manager_decision",
            "action": action,
        }
        return f"Please provide the leave request ID you want to {action}."

    leave_id = int(leave_id)
    endpoint = f"/leaves/{leave_id}/{action}"

    try:
        result = api_put(endpoint, token)
        status = result.get("status", "Approved" if action == "approve" else "Rejected")
        leave_type = result.get("leave_type", "Leave")
        start_date = result.get("start_date", "")
        end_date = result.get("end_date", "")
        return (
            f"Request {leave_id} {status.lower()}: {leave_type} leave from "
            f"{start_date} to {end_date}."
        )
    except requests.HTTPError as exc:
        return f"Couldn't {action} leave request {leave_id}: {_extract_error_detail(exc)}"
    except Exception as exc:
        return f"Couldn't {action} leave request {leave_id}: {exc}"



def _normalize_person_name(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (value or "").lower()).strip()


def _employee_name_from_leave(leave: dict) -> str:
    employee = leave.get("employee")
    if isinstance(employee, dict):
        full = " ".join(
            str(employee.get(k) or "").strip()
            for k in ("first_name", "last_name")
            if str(employee.get(k) or "").strip()
        )
        if full:
            return full

    for key in ("employee_name", "employee_full_name", "full_name"):
        value = leave.get(key)
        if value:
            return str(value).strip()

    first = str(leave.get("first_name") or "").strip()
    last = str(leave.get("last_name") or "").strip()
    full = " ".join(x for x in (first, last) if x)
    return full or f"Employee {leave.get('employee_id', 'Unknown')}"


def _format_manager_pending_leave(leave: dict) -> str:
    return (
        f"- {_employee_name_from_leave(leave)} — "
        f"{leave.get('leave_type', 'Leave')} leave, "
        f"{leave.get('start_date', '')} to {leave.get('end_date', '')}"
    )


def _get_pending_for_manager(token: str) -> list[dict]:
    return api_get("/leaves/pending/", token)


def _resolve_manager_candidates(
    token: str,
    employee_name: str,
    start_date: str | None = None,
    leave_type: str | None = None,
) -> list[dict]:
    pending = _get_pending_for_manager(token)
    wanted = _normalize_person_name(employee_name)

    candidates = []
    for leave in pending:
        actual = _normalize_person_name(_employee_name_from_leave(leave))
        if wanted == actual or wanted in actual or actual in wanted:
            candidates.append(leave)

    if leave_type:
        candidates = [
            leave for leave in candidates
            if str(leave.get("leave_type", "")).lower() == leave_type.lower()
        ]

    if start_date:
        candidates = [
            leave for leave in candidates
            if str(leave.get("start_date")) == str(start_date)
        ]

    return candidates


def _resolve_manager_request(
    token: str,
    employee_name: str,
    start_date: str | None = None,
    leave_type: str | None = None,
) -> tuple[dict | None, str | None]:
    candidates = _resolve_manager_candidates(
        token, employee_name, start_date, leave_type
    )

    if len(candidates) == 1:
        return candidates[0], None

    if not candidates:
        return None, (
            f"I couldn't find a pending leave request for {employee_name}."
        )

    details = "\n".join(
        _format_manager_pending_leave(x) for x in candidates
    )

    return None, (
        f"{employee_name} has multiple matching pending requests:\n"
        f"{details}\n"
        "Please specify the leave type or start date."
    )


def _extract_manager_choice(message: str) -> tuple[str | None, str | None, str | None]:
    """Extract leave type and date information from a manager clarification."""
    lowered = message.lower().strip()

    leave_type = None
    for value in ("annual", "sick", "casual"):
        if re.search(rf"\b{value}\b", lowered):
            leave_type = value.capitalize()
            break

    # First support explicit ISO dates, including a two-date range.
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", message)
    if dates:
        return leave_type, dates[0], dates[-1]

    # Also support natural dates such as "7th September", "7 Sep",
    # "September 7", and relative weekdays such as "next Monday".
    natural_range = extract_date_range(message)
    if natural_range:
        start, end = natural_range
        return leave_type, start.isoformat(), end.isoformat()

    return leave_type, None, None


def _extract_choice_number(message: str, count: int) -> int | None:
    """Accept a visible option number without exposing backend leave IDs."""
    text = message.lower().strip()

    match = re.fullmatch(r"(?:option\s*)?(\d+)", text)
    if match:
        value = int(match.group(1))
        return value if 1 <= value <= count else None

    ordinals = {
        "first": 1, "1st": 1,
        "second": 2, "2nd": 2,
        "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4,
        "fifth": 5, "5th": 5,
        "sixth": 6, "6th": 6,
        "seventh": 7, "7th": 7,
        "eighth": 8, "8th": 8,
        "ninth": 9, "9th": 9,
        "tenth": 10, "10th": 10,
    }

    for word, value in ordinals.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            return value if 1 <= value <= count else None

    return None


def _continue_manager_decision(message: str, token: str) -> str:
    pending = st.session_state.get("pending_action", {})

    action = str(pending.get("action", "")).lower()
    employee_name = pending.get("employee_name")
    candidates = pending.get("candidates", [])

    if action not in {"approve", "reject"} or not employee_name or not candidates:
        st.session_state.pending_action = {}
        return "The manager action state is invalid. Please repeat the approve/reject request."

    # First allow the manager to select a visible option number.
    choice_number = _extract_choice_number(message, len(candidates))
    if choice_number is not None:
        filtered = [candidates[choice_number - 1]]
    else:
        leave_type, start_date, end_date = _extract_manager_choice(message)
        filtered = candidates

        if leave_type:
            filtered = [
                leave for leave in filtered
                if str(leave.get("leave_type", "")).lower() == leave_type.lower()
            ]

        if start_date:
            filtered = [
                leave for leave in filtered
                if str(leave.get("start_date")) == start_date
            ]

        if end_date and end_date != start_date:
            filtered = [
                leave for leave in filtered
                if str(leave.get("end_date")) == end_date
            ]

    if len(filtered) != 1:
        details = "\n".join(
            f"{idx}. {_format_manager_pending_leave(x)}"
            for idx, x in enumerate(candidates, start=1)
        )
        return (
            f"I still need a unique request from {employee_name}.\n\n"
            f"{details}\n\n"
            "These requests are identical, so choose an option number "
            "(for example, 2)."
        )

    leave = filtered[0]
    leave_id = leave.get("id")

    if leave_id is None:
        st.session_state.pending_action = {}
        return "The matching request has no internal ID, so I could not safely complete the action."

    try:
        updated = api_put(f"/leaves/{leave_id}/{action}", token)
    except requests.HTTPError as exc:
        return (
            f"Couldn't {action} {_employee_name_from_pending_leave(leave)}'s "
            f"request: {_extract_error_detail(exc)}"
        )
    except Exception as exc:
        return f"Couldn't {action} the request: {exc}"

    st.session_state.pending_action = {}

    result = updated or leave
    name = _employee_name_from_leave(result)
    leave_type_text = result.get("leave_type", "Leave")
    start = result.get("start_date", "")
    end = result.get("end_date", "")

    return (
        f"{action.capitalize()}d {name}'s {leave_type_text} leave "
        f"from {start} to {end}."
    )


def _employee_name_from_pending_leave(leave: dict) -> str:
    return _employee_name_from_leave(leave)



def _tool_get_pending_leave_requests(token: str) -> str:
    try:
        if not is_manager(token):
            return "Only managers can view pending leave requests."
        pending = _get_pending_for_manager(token)
    except requests.HTTPError as exc:
        return f"Couldn't fetch pending leave requests: {_extract_error_detail(exc)}"
    except Exception as exc:
        return f"Couldn't fetch pending leave requests: {exc}"

    if not pending:
        return "There are no pending leave requests."

    lines = ["Pending leave requests:"]
    lines.extend(_format_manager_pending_leave(leave) for leave in pending)
    return "\n".join(lines)


def _tool_manager_name_action(
    action: str,
    employee_name: str,
    start_date: str | None,
    leave_type: str | None,
    token: str,
) -> str:
    try:
        if not is_manager(token):
            return f"Manager authorization is required to {action} leave requests."
    except requests.HTTPError as exc:
        return f"Couldn't verify your manager authorization: {_extract_error_detail(exc)}"
    except Exception as exc:
        return f"Couldn't verify your manager authorization: {exc}"

    if not employee_name:
        return "Please provide the employee name for the leave request."

    try:
        candidates = _resolve_manager_candidates(
            token,
            employee_name,
            start_date,
            leave_type,
        )
    except requests.HTTPError as exc:
        return f"Couldn't fetch pending leave requests: {_extract_error_detail(exc)}"
    except Exception as exc:
        return f"Couldn't fetch pending leave requests: {exc}"

    if len(candidates) == 0:
        return f"I couldn't find a pending leave request for {employee_name}."

    if len(candidates) > 1:
        st.session_state.pending_action = {
            "kind": "manager_decision",
            "action": action,
            "employee_name": employee_name,
            "candidates": candidates,
        }

        details = "\n".join(
            f"{idx}. {_format_manager_pending_leave(x)}"
            for idx, x in enumerate(candidates, start=1)
        )

        return (
            f"{employee_name} has multiple matching pending requests:\n\n"
            f"{details}\n\n"
            "Choose the option number (for example, 2), or specify the leave type/start date."
        )

    leave = candidates[0]
    leave_id = leave.get("id")

    if leave_id is None:
        return "The pending request did not contain its internal ID, so I could not safely complete the action."

    try:
        updated = api_put(f"/leaves/{leave_id}/{action}", token)
    except requests.HTTPError as exc:
        return (
            f"Couldn't {action} {_employee_name_from_leave(leave)}'s "
            f"{leave.get('leave_type', 'leave')} request: "
            f"{_extract_error_detail(exc)}"
        )
    except Exception as exc:
        return f"Couldn't {action} the request: {exc}"

    st.session_state.pending_action = {}

    result = updated or leave
    return (
        f"{action.capitalize()}d {_employee_name_from_leave(result)}'s "
        f"{result.get('leave_type', 'Leave')} leave from "
        f"{result.get('start_date', '')} to {result.get('end_date', '')}."
    )



def execute_tool(tool_name: str, args: dict, original_message: str, token: str | None) -> str:
    """Execute a tool selected by the LLM. The LLM itself never touches FastAPI."""
    protected_tools = {
        "get_leave_balance", "get_leave_history", "get_pending_leave_requests",
        "apply_leave", "cancel_leave", "approve_leave", "reject_leave",
    }

    if tool_name in protected_tools and token is None:
        return "Please log in first (sidebar) — this needs your account's live data."

    if tool_name == "answer_policy_question":
        answer, sources = answer_policy_question(original_message)
        if sources:
            answer += f"\n\n*Sources: {', '.join(sources)}*"
        return answer

    if tool_name == "get_leave_balance":
        profile = api_get("/employees/me/", token)
        return (
            f"Your current balances — Annual: {profile['annual_leave_balance']} days, "
            f"Sick: {profile['sick_leave_balance']} days, "
            f"Casual: {profile['casual_leave_balance']} days."
        )

    if tool_name == "get_leave_history":
        history = api_get("/leaves/me/", token)
        if not history:
            return "You have no leave requests on record yet."
        lines = [
            f"- {leave['leave_type']} leave, {leave['start_date']} to {leave['end_date']} — {leave['status']}"
            for leave in history
        ]
        return "Your leave history:\n" + "\n".join(lines)

    if tool_name == "get_pending_leave_requests":
        return _tool_get_pending_leave_requests(token)

    if tool_name == "apply_leave":
        return _tool_apply_leave(args, original_message, token)

    if tool_name == "cancel_leave":
        return _tool_cancel_leave(args.get("leave_id"), token)

    if tool_name == "approve_leave":
        return _tool_manager_name_action(
            "approve",
            args.get("employee_name"),
            args.get("start_date"),
            args.get("leave_type"),
            token,
        )

    if tool_name == "reject_leave":
        return _tool_manager_name_action(
            "reject",
            args.get("employee_name"),
            args.get("start_date"),
            args.get("leave_type"),
            token,
        )

    raise RuntimeError(f"Unknown tool selected by the LLM: {tool_name}")


# ---------------------------------------------------------------------------
# Live data — always the backend, never RAG
# ---------------------------------------------------------------------------
def handle_live_data(message: str, token: str) -> str:
    lowered = message.lower()
    if "history" in lowered or "previous" in lowered or "past" in lowered:
        history = api_get("/leaves/me/", token)
        if not history:
            return "You have no leave requests on record yet."
        lines = [
            f"- {leave['leave_type']} leave, {leave['start_date']} to {leave['end_date']} — {leave['status']}"
            for leave in history
        ]
        return "Your leave history:\n" + "\n".join(lines)

    profile = api_get("/employees/me/", token)
    return (
        f"Your current balances — "
        f"Annual: {profile['annual_leave_balance']} days, "
        f"Sick: {profile['sick_leave_balance']} days, "
        f"Casual: {profile['casual_leave_balance']} days."
    )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Leave Assistant", page_icon="🗓️")
st.title("🗓️ Leave Assistant")
st.caption("Natural-language requests are interpreted by the LLM and mapped to safe application tools.")

if "token" not in st.session_state:
    st.session_state.token = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_action" not in st.session_state:
    st.session_state.pending_action = {}

with st.sidebar:
    st.subheader("Login")
    st.caption("Needed for live data and employee/manager actions — policy questions work without login.")
    if st.session_state.token is None:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Log in"):
            try:
                st.session_state.token = api_login(email, password)
                st.success("Logged in.")
            except requests.HTTPError:
                st.error("Login failed — check email/password and that the backend is running.")
    else:
        st.success("Logged in.")
        if st.button("Log out"):
            st.session_state.token = None
            st.session_state.messages = []
            st.session_state.pending_action = {}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about leave policy, your balance, or manage leave requests...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        if st.session_state.get("pending_action"):
            # Multi-turn completion is deterministic. The LLM already chose
            # the action on the previous turn; this message supplies the
            # missing field (for example, a leave request ID or leave type).
            if st.session_state.token is None:
                answer = "Please log in first (sidebar) — this needs your account's live data."
            else:
                answer = handle_action(prompt, st.session_state.token)
        else:
            tool_name, tool_args = choose_tool(prompt)
            answer = execute_tool(
                tool_name,
                tool_args,
                prompt,
                st.session_state.token,
            )
    except requests.HTTPError as exc:
        answer = f"The backend request failed: {_extract_error_detail(exc)}"
    except Exception as exc:
        answer = f"I couldn't route that request safely: {exc}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
