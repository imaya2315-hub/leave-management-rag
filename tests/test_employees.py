import uuid

from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

client = TestClient(app)
PREFIX = settings.API_V1_PREFIX


def test_health_check():
    """Test that the API is awake and responding"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "The Leave Management API is up and running."}


def test_create_employee():
    """Test that we can successfully create a new employee with a unique email"""
    unique_id = str(uuid.uuid4())[:8]
    test_email = f"tester_{unique_id}@gmail.com"

    payload = {
        "first_name": "Test",
        "last_name": "Automation",
        "email": test_email,
        "department": "Quality Assurance",
        "password": "securepassword123"
    }

    response = client.post(f"{PREFIX}/employees/", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert data["email"] == test_email
    assert data["first_name"] == "Test"
    assert "id" in data
    assert data["is_active"] is True


def test_duplicate_email_rejected():
    """Test that registering the same email twice is rejected"""
    unique_id = str(uuid.uuid4())[:8]
    test_email = f"dup_{unique_id}@gmail.com"
    payload = {
        "first_name": "Dup",
        "last_name": "Test",
        "email": test_email,
        "department": "QA",
        "password": "securepassword123"
    }
    first = client.post(f"{PREFIX}/employees/", json=payload)
    assert first.status_code == 200

    second = client.post(f"{PREFIX}/employees/", json=payload)
    assert second.status_code == 400
