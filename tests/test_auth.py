"""The estimate endpoint's bearer auth, and the env var name that turns it on."""
from fastapi.testclient import TestClient

from src import main
from src.config import Settings


def test_token_is_read_from_bidagent_api_token(monkeypatch):
    monkeypatch.setenv("BIDAGENT_API_TOKEN", "s3cret")
    monkeypatch.delenv("API_TOKEN", raising=False)
    assert Settings(_env_file=None).api_token == "s3cret"


def test_estimate_refuses_a_missing_or_wrong_token(monkeypatch):
    monkeypatch.setattr(main.settings, "api_token", "s3cret")
    client = TestClient(main.app)
    assert client.post("/api/v1/estimate", data={"requested_services": "x"}).status_code == 401
    assert client.post("/api/v1/estimate", data={"requested_services": "x"},
                       headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/healthz").json()["auth"] is True
