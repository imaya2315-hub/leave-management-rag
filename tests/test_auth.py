import uuid

import jwt
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

client = TestClient(app)
PREFIX = settings.API_V1_PREFIX


def _create_employee():
    unique_id = str(uuid.uuid4())[:8]
    payload = {
        "first_name": "Auth",
        "last_name": "Tester",
        "email": f"auth_{unique_id}@gmail.com",
        "department": "Engineering",
        "password": "securepassword123",
    }
    response = client.post(f"{PREFIX}/employees/", json=payload)
    assert response.status_code == 200
    return payload["email"], payload["password"]


def test_login_returns_access_and_refresh_tokens():
    email, password = _create_employee()
    response = client.post(f"{PREFIX}/login", data={"username": email, "password": password})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


def test_refresh_token_issues_new_working_access_token():
    email, password = _create_employee()
    login_data = client.post(f"{PREFIX}/login", data={"username": email, "password": password}).json()

    refresh_response = client.post(f"{PREFIX}/refresh", json={"refresh_token": login_data["refresh_token"]})
    assert refresh_response.status_code == 200
    new_tokens = refresh_response.json()
    assert "access_token" in new_tokens
    assert "refresh_token" in new_tokens

    # The new access token should actually work on a protected route
    profile_response = client.get(
        f"{PREFIX}/employees/me/",
        headers={"Authorization": f"Bearer {new_tokens['access_token']}"},
    )
    assert profile_response.status_code == 200
    assert profile_response.json()["email"] == email


def test_access_token_rejected_on_refresh_endpoint():
    """An access token must NOT work as a refresh token — only a real refresh token should."""
    email, password = _create_employee()
    login_data = client.post(f"{PREFIX}/login", data={"username": email, "password": password}).json()

    response = client.post(f"{PREFIX}/refresh", json={"refresh_token": login_data["access_token"]})
    assert response.status_code == 401


def test_refresh_token_rejected_as_access_token():
    """A refresh token must NOT work on protected routes meant for access tokens."""
    email, password = _create_employee()
    login_data = client.post(f"{PREFIX}/login", data={"username": email, "password": password}).json()

    response = client.get(
        f"{PREFIX}/employees/me/",
        headers={"Authorization": f"Bearer {login_data['refresh_token']}"},
    )
    assert response.status_code == 401


def test_expired_refresh_token_rejected():
    email, password = _create_employee()
    client.post(f"{PREFIX}/login", data={"username": email, "password": password})

    # Manually craft an already-expired refresh token to simulate real expiry
    expired_token = jwt.encode(
        {"sub": email, "type": "refresh", "exp": 0},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )

    response = client.post(f"{PREFIX}/refresh", json={"refresh_token": expired_token})
    assert response.status_code == 401


def test_garbage_refresh_token_rejected():
    response = client.post(f"{PREFIX}/refresh", json={"refresh_token": "not-a-real-token"})
    assert response.status_code == 401
