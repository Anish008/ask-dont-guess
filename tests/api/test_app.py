"""Tests for FastAPI application."""

import pytest
from adg.api.app import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data
    assert "device" in data
    assert "version" in data


def test_chat_new_session(client):
    """Test chat endpoint with new session."""
    response = client.post(
        "/v1/chat",
        json={"message": "What is the capital of France?"},
    )
    assert response.status_code == 200

    data = response.json()
    assert "session_id" in data
    assert "message" in data
    assert "action" in data
    assert data["action"] in ["ANSWER", "ASK", "ABSTAIN"]
    assert "decision" in data
    assert "turn_number" in data


def test_chat_existing_session(client):
    """Test chat with existing session."""
    # First message
    response1 = client.post(
        "/v1/chat",
        json={"session_id": "test-session", "message": "Hello"},
    )
    assert response1.status_code == 200
    data1 = response1.json()
    session_id = data1["session_id"]

    # Second message in same session
    response2 = client.post(
        "/v1/chat",
        json={"session_id": session_id, "message": "What is 2+2?"},
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["session_id"] == session_id
    assert data2["turn_number"] > data1["turn_number"]


def test_analyze(client):
    """Test analyze endpoint."""
    response = client.post(
        "/v1/analyze",
        json={"query": "What is the capital of Georgia?"},
    )
    assert response.status_code == 200

    data = response.json()
    assert "query" in data
    assert "decision" in data
    assert "response" in data


def test_analyze_with_overrides(client):
    """Test analyze with parameter overrides."""
    response = client.post(
        "/v1/analyze",
        json={
            "query": "What is 2+2?",
            "n_samples": 4,
            "temperature": 0.5,
        },
    )
    assert response.status_code == 200


def test_get_session(client):
    """Test get session endpoint."""
    # Create session first
    chat_response = client.post(
        "/v1/chat",
        json={"session_id": "test-get", "message": "Hello"},
    )
    assert chat_response.status_code == 200

    # Get session
    response = client.get("/v1/sessions/test-get")
    assert response.status_code == 200

    data = response.json()
    assert data["session_id"] == "test-get"
    assert "turn_number" in data


def test_get_session_not_found(client):
    """Test get non-existent session."""
    response = client.get("/v1/sessions/nonexistent")
    assert response.status_code == 404


def test_delete_session(client):
    """Test delete session endpoint."""
    # Create session
    chat_response = client.post(
        "/v1/chat",
        json={"session_id": "test-delete", "message": "Hello"},
    )
    assert chat_response.status_code == 200

    # Delete session
    response = client.delete("/v1/sessions/test-delete")
    assert response.status_code == 204

    # Verify deleted
    get_response = client.get("/v1/sessions/test-delete")
    assert get_response.status_code == 404


def test_stats(client):
    """Test stats endpoint."""
    response = client.get("/v1/stats")
    assert response.status_code == 200

    data = response.json()
    assert "total_requests" in data
    assert "decisions_by_action" in data
    assert "mean_latency_ms" in data


def test_chat_validation(client):
    """Test chat request validation."""
    # Empty message
    response = client.post(
        "/v1/chat",
        json={"message": ""},
    )
    assert response.status_code == 422  # Validation error


def test_analyze_validation(client):
    """Test analyze request validation."""
    # Invalid n_samples (too high)
    response = client.post(
        "/v1/analyze",
        json={"query": "test", "n_samples": 100},
    )
    assert response.status_code == 422

    # Invalid temperature (negative)
    response = client.post(
        "/v1/analyze",
        json={"query": "test", "temperature": -1.0},
    )
    assert response.status_code == 422
