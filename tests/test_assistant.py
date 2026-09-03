import uuid
from datetime import date, timedelta

from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

client = TestClient(app)
PREFIX = settings.API_V1_PREFIX


def _create_employee(role="employee"):
    unique_id = str(uuid.uuid4())[:8]
    payload = {
        "first_name": "Assistant",
        "last_name": "Tester",
        "email": f"assistant_{role}_{unique_id}@gmail.com",
        "department": "Engineering",
        "password": "securepassword123",
        "role": role,
    }
    response = client.post(f"{PREFIX}/employees/", json=payload)
    assert response.status_code == 200
    return payload["email"], payload["password"]


def _token(role="employee"):
    email, password = _create_employee(role)
    login = client.post(f"{PREFIX}/login", data={"username": email, "password": password})
    return login.json()["access_token"]


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def _next_weekday(days_ahead_minimum: int = 1) -> date:
    """Returns the next weekday (Mon-Fri) at least days_ahead_minimum days out, so tests never land on a weekend."""
    candidate = date.today() + timedelta(days=days_ahead_minimum)
    while candidate.weekday() >= 5:  # 5=Saturday, 6=Sunday
        candidate += timedelta(days=1)
    return candidate


def _ingest_sample_policy(manager_token):
    client.post(
        f"{PREFIX}/knowledge/documents",
        json={
            "title": f"Casual Leave Policy {uuid.uuid4().hex[:6]}",
            "content": (
                "Every employee is entitled to 5 Casual Leave days per calendar year. "
                "Casual Leave cannot be taken for more than 2 consecutive working days "
                "at a time without manager approval in advance. "
            )
            * 10,
        },
        headers=_auth_headers(manager_token),
    )


def test_assistant_requires_authentication():
    response = client.post(f"{PREFIX}/assistant/ask", json={"question": "How many casual leaves do I get?"})
    assert response.status_code == 401


def test_assistant_answers_a_policy_question():
    manager_token = _token("manager")
    _ingest_sample_policy(manager_token)

    employee_token = _token("employee")
    response = client.post(
        f"{PREFIX}/assistant/ask",
        json={"question": "How many casual leaves can I take in a year?"},
        headers=_auth_headers(employee_token),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert isinstance(body["sources"], list)
    assert body["eligibility"] is None


def _two_consecutive_weekdays(days_ahead_minimum: int = 7) -> tuple[date, date]:
    """Returns (start, end) — two consecutive weekdays, guaranteed not to straddle a weekend."""
    start = _next_weekday(days_ahead_minimum)
    end = start + timedelta(days=1)
    if end.weekday() >= 5:
        start = _next_weekday(days_ahead_minimum + 1)
        end = start + timedelta(days=1)
    return start, end


def test_assistant_runs_eligibility_check_when_dates_given():
    employee_token = _token("employee")
    start, end = _two_consecutive_weekdays()

    response = client.post(
        f"{PREFIX}/assistant/ask",
        json={
            "question": "Can I take this leave?",
            "leave_type": "Casual",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        headers=_auth_headers(employee_token),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["eligibility"] is not None
    assert body["eligibility"]["eligible"] is True
    assert body["eligibility"]["requested_days"] == 2


def test_assistant_eligibility_check_flags_insufficient_balance():
    employee_token = _token("employee")
    start = date.today() + timedelta(days=7)
    end = start + timedelta(days=10)  # far more than the default 5-day casual balance

    response = client.post(
        f"{PREFIX}/assistant/ask",
        json={
            "question": "Can I take this leave?",
            "leave_type": "Casual",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        headers=_auth_headers(employee_token),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["eligibility"]["eligible"] is False
