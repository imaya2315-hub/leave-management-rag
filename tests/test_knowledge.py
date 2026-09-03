import uuid

from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

client = TestClient(app)
PREFIX = settings.API_V1_PREFIX


def _create_employee(role="employee"):
    unique_id = str(uuid.uuid4())[:8]
    payload = {
        "first_name": "Knowledge",
        "last_name": "Tester",
        "email": f"knowledge_{role}_{unique_id}@gmail.com",
        "department": "HR",
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


def test_only_manager_can_ingest_a_document():
    employee_token = _token("employee")
    response = client.post(
        f"{PREFIX}/knowledge/documents",
        json={"title": "Should Fail", "content": "irrelevant content"},
        headers=_auth_headers(employee_token),
    )
    assert response.status_code == 403


def test_manager_can_ingest_list_and_delete_a_document():
    manager_token = _token("manager")
    unique_id = str(uuid.uuid4())[:8]
    title = f"Test Policy {unique_id}"

    create_response = client.post(
        f"{PREFIX}/knowledge/documents",
        json={
            "title": title,
            "content": "Employees get 20 annual leave days per year. " * 20,
        },
        headers=_auth_headers(manager_token),
    )
    assert create_response.status_code == 200
    document = create_response.json()
    assert document["title"] == title
    assert document["chunk_count"] >= 1

    list_response = client.get(f"{PREFIX}/knowledge/documents", headers=_auth_headers(manager_token))
    assert list_response.status_code == 200
    assert any(doc["id"] == document["id"] for doc in list_response.json())

    delete_response = client.delete(
        f"{PREFIX}/knowledge/documents/{document['id']}", headers=_auth_headers(manager_token)
    )
    assert delete_response.status_code == 200

    list_after_delete = client.get(f"{PREFIX}/knowledge/documents", headers=_auth_headers(manager_token))
    assert all(doc["id"] != document["id"] for doc in list_after_delete.json())


def test_empty_document_is_rejected():
    manager_token = _token("manager")
    response = client.post(
        f"{PREFIX}/knowledge/documents",
        json={"title": "Empty Doc", "content": "   "},
        headers=_auth_headers(manager_token),
    )
    assert response.status_code == 400
