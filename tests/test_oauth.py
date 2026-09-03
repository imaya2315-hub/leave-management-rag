import uuid

from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

client = TestClient(app)
PREFIX = settings.API_V1_PREFIX


def _create_manager():
    unique_id = str(uuid.uuid4())[:8]
    payload = {
        "first_name": "Oauth",
        "last_name": "Manager",
        "email": f"oauth_manager_{unique_id}@gmail.com",
        "department": "IT",
        "password": "securepassword123",
        "role": "manager",
    }
    response = client.post(f"{PREFIX}/employees/", json=payload)
    assert response.status_code == 200
    return payload["email"], payload["password"]


def _manager_token():
    email, password = _create_manager()
    login = client.post(f"{PREFIX}/login", data={"username": email, "password": password})
    return login.json()["access_token"]


def _register_client(manager_token, name="n8n automation"):
    response = client.post(
        f"{PREFIX}/oauth/clients",
        json={"name": name},
        headers={"Authorization": f"Bearer {manager_token}"},
    )
    assert response.status_code == 200
    return response.json()


def test_only_manager_can_register_a_client():
    response = client.post(f"{PREFIX}/oauth/clients", json={"name": "unauthorized attempt"})
    assert response.status_code == 401


def test_manager_can_register_client_and_get_secret_once():
    manager_token = _manager_token()
    created = _register_client(manager_token)
    assert created["client_id"].startswith("client_")
    assert len(created["client_secret"]) > 20
    assert created["name"] == "n8n automation"


def test_client_credentials_token_issued_correctly():
    manager_token = _manager_token()
    created = _register_client(manager_token)

    token_response = client.post(
        f"{PREFIX}/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": created["client_id"],
            "client_secret": created["client_secret"],
        },
    )
    assert token_response.status_code == 200
    data = token_response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] == settings.SERVICE_TOKEN_EXPIRE_MINUTES * 60


def test_wrong_client_secret_rejected():
    manager_token = _manager_token()
    created = _register_client(manager_token)

    response = client.post(
        f"{PREFIX}/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": created["client_id"],
            "client_secret": "wrong-secret",
        },
    )
    assert response.status_code == 401


def test_unsupported_grant_type_rejected():
    manager_token = _manager_token()
    created = _register_client(manager_token)

    response = client.post(
        f"{PREFIX}/oauth/token",
        data={
            "grant_type": "password",
            "client_id": created["client_id"],
            "client_secret": created["client_secret"],
        },
    )
    assert response.status_code == 400


def test_service_token_can_register_employee_via_service_endpoint():
    manager_token = _manager_token()
    created = _register_client(manager_token)

    token_response = client.post(
        f"{PREFIX}/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": created["client_id"],
            "client_secret": created["client_secret"],
        },
    )
    service_token = token_response.json()["access_token"]

    unique_id = str(uuid.uuid4())[:8]
    new_employee_payload = {
        "first_name": "Automated",
        "last_name": "Employee",
        "email": f"automated_{unique_id}@gmail.com",
        "department": "Ops",
        "password": "securepassword123",
    }
    response = client.post(
        f"{PREFIX}/employees/service-register",
        json=new_employee_payload,
        headers={"Authorization": f"Bearer {service_token}"},
    )
    assert response.status_code == 200
    assert response.json()["email"] == new_employee_payload["email"]


def test_service_token_rejected_on_employee_only_route():
    """A service token must NOT work on routes meant for human employee access tokens."""
    manager_token = _manager_token()
    created = _register_client(manager_token)

    token_response = client.post(
        f"{PREFIX}/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": created["client_id"],
            "client_secret": created["client_secret"],
        },
    )
    service_token = token_response.json()["access_token"]

    response = client.get(
        f"{PREFIX}/employees/me/",
        headers={"Authorization": f"Bearer {service_token}"},
    )
    assert response.status_code == 401


def test_employee_access_token_rejected_on_service_only_route():
    """A human employee's access token must NOT work on service-only routes."""
    email, password = _create_manager()
    login = client.post(f"{PREFIX}/login", data={"username": email, "password": password})
    access_token = login.json()["access_token"]

    response = client.post(
        f"{PREFIX}/employees/service-register",
        json={
            "first_name": "X", "last_name": "Y", "email": "x@y.com",
            "department": "Z", "password": "securepassword123",
        },
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert response.status_code == 401
