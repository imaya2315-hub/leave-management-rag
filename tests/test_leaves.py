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
        "first_name": "Leave",
        "last_name": "Tester",
        "email": f"leave_{role}_{unique_id}@gmail.com",
        "department": "Engineering",
        "password": "securepassword123",
        "role": role,
    }
    response = client.post(f"{PREFIX}/employees/", json=payload)
    assert response.status_code == 200
    return response.json(), payload["email"], payload["password"]


def _login(email, password):
    response = client.post(
        f"{PREFIX}/login",
        data={"username": email, "password": password},
    )
    assert response.status_code == 200
    return response.json()


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def test_apply_leave_starts_pending():
    employee, email, password = _create_employee()
    token = _login(email, password)["access_token"]

    start = date.today() + timedelta(days=10)
    end = start + timedelta(days=2)

    response = client.post(
        f"{PREFIX}/leaves/",
        json={"leave_type": "Annual", "start_date": start.isoformat(), "end_date": end.isoformat()},
        headers=_auth_headers(token),
    )
    assert response.status_code == 200
    leave = response.json()
    assert leave["status"] == "Pending"
    assert leave["employee_id"] == employee["id"]

    profile = client.get(f"{PREFIX}/employees/{employee['id']}").json()
    assert profile["annual_leave_balance"] == employee["annual_leave_balance"]


def test_apply_leave_requires_auth():
    start = date.today() + timedelta(days=10)
    end = start + timedelta(days=2)
    response = client.post(
        f"{PREFIX}/leaves/",
        json={"leave_type": "Annual", "start_date": start.isoformat(), "end_date": end.isoformat()},
    )
    assert response.status_code == 401


def test_manager_can_approve_and_balance_is_deducted():
    employee, email, password = _create_employee(role="employee")
    _, manager_email, manager_password = _create_employee(role="manager")
    starting_balance = employee["annual_leave_balance"]

    employee_token = _login(email, password)["access_token"]
    start = date.today() + timedelta(days=15)
    end = start + timedelta(days=2)
    leave = client.post(
        f"{PREFIX}/leaves/",
        json={"leave_type": "Annual", "start_date": start.isoformat(), "end_date": end.isoformat()},
        headers=_auth_headers(employee_token),
    ).json()

    manager_token = _login(manager_email, manager_password)["access_token"]
    approve_response = client.put(
        f"{PREFIX}/leaves/{leave['id']}/approve",
        headers=_auth_headers(manager_token),
    )
    assert approve_response.status_code == 200
    assert approve_response.json()["status"] == "Approved"

    profile = client.get(f"{PREFIX}/employees/{employee['id']}").json()
    assert profile["annual_leave_balance"] < starting_balance


def test_non_manager_cannot_approve():
    employee, email, password = _create_employee(role="employee")
    _create_employee(role="employee")

    employee_token = _login(email, password)["access_token"]
    start = date.today() + timedelta(days=20)
    end = start + timedelta(days=1)
    leave = client.post(
        f"{PREFIX}/leaves/",
        json={"leave_type": "Casual", "start_date": start.isoformat(), "end_date": end.isoformat()},
        headers=_auth_headers(employee_token),
    ).json()

    response = client.put(
        f"{PREFIX}/leaves/{leave['id']}/approve",
        headers=_auth_headers(employee_token),
    )
    assert response.status_code == 403


def test_manager_can_reject_with_no_balance_change():
    employee, email, password = _create_employee(role="employee")
    _, manager_email, manager_password = _create_employee(role="manager")
    starting_balance = employee["sick_leave_balance"]

    employee_token = _login(email, password)["access_token"]
    start = date.today() + timedelta(days=25)
    end = start + timedelta(days=1)
    leave = client.post(
        f"{PREFIX}/leaves/",
        json={"leave_type": "Sick", "start_date": start.isoformat(), "end_date": end.isoformat()},
        headers=_auth_headers(employee_token),
    ).json()

    manager_token = _login(manager_email, manager_password)["access_token"]
    reject_response = client.put(
        f"{PREFIX}/leaves/{leave['id']}/reject",
        headers=_auth_headers(manager_token),
    )
    assert reject_response.status_code == 200
    assert reject_response.json()["status"] == "Rejected"

    profile = client.get(f"{PREFIX}/employees/{employee['id']}").json()
    assert profile["sick_leave_balance"] == starting_balance


def test_cancel_approved_leave_refunds_balance():
    employee, email, password = _create_employee(role="employee")
    _, manager_email, manager_password = _create_employee(role="manager")
    starting_balance = employee["annual_leave_balance"]

    employee_token = _login(email, password)["access_token"]
    start = date.today() + timedelta(days=30)
    end = start + timedelta(days=2)
    leave = client.post(
        f"{PREFIX}/leaves/",
        json={"leave_type": "Annual", "start_date": start.isoformat(), "end_date": end.isoformat()},
        headers=_auth_headers(employee_token),
    ).json()

    manager_token = _login(manager_email, manager_password)["access_token"]
    client.put(f"{PREFIX}/leaves/{leave['id']}/approve", headers=_auth_headers(manager_token))

    cancel_response = client.put(f"{PREFIX}/leaves/{leave['id']}/cancel")
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "Cancelled"

    profile = client.get(f"{PREFIX}/employees/{employee['id']}").json()
    assert profile["annual_leave_balance"] == starting_balance


def test_insufficient_balance_rejected_at_apply():
    employee, email, password = _create_employee()
    token = _login(email, password)["access_token"]

    start = date.today() + timedelta(days=40)
    end = start + timedelta(days=60)

    response = client.post(
        f"{PREFIX}/leaves/",
        json={"leave_type": "Annual", "start_date": start.isoformat(), "end_date": end.isoformat()},
        headers=_auth_headers(token),
    )
    assert response.status_code == 400


def _next_weekday(days_ahead_minimum: int = 1) -> date:
    """Returns the next weekday (Mon-Fri) at least days_ahead_minimum days out, so tests never land on a weekend."""
    candidate = date.today() + timedelta(days=days_ahead_minimum)
    while candidate.weekday() >= 5:  # 5=Saturday, 6=Sunday
        candidate += timedelta(days=1)
    return candidate


def test_my_leave_history():
    employee, email, password = _create_employee()
    token = _login(email, password)["access_token"]

    start = _next_weekday(50)
    end = start  # single working day — avoids landing on a weekend as a 2-day span could
    client.post(
        f"{PREFIX}/leaves/",
        json={"leave_type": "Casual", "start_date": start.isoformat(), "end_date": end.isoformat()},
        headers=_auth_headers(token),
    )

    history = client.get(f"{PREFIX}/leaves/me/", headers=_auth_headers(token))
    assert history.status_code == 200
    assert len(history.json()) == 1
