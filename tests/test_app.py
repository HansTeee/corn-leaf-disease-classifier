"""Smoke test of the Streamlit UI (local-model mode), using Streamlit's own test harness."""
from streamlit.testing.v1 import AppTest

from .conftest import ROOT


def test_app_starts_without_errors():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception


def test_app_classifies_an_example():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    at.selectbox[0].select("rust_1").run()
    assert not at.exception
    assert at.subheader[0].value == "Common rust"


def test_app_gets_predictions_from_the_api_when_configured(monkeypatch):
    """In Docker, API_URL is set and the UI must call FastAPI instead of loading the model."""
    import requests
    from fastapi.testclient import TestClient

    from api.main import app as api_app

    calls = []
    with TestClient(api_app) as client:
        def fake_post(url, files, timeout):
            calls.append(url)
            return client.post("/predict", files=files)

        monkeypatch.setenv("API_URL", "http://api.test")
        monkeypatch.setattr(requests, "post", fake_post)
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        at.selectbox[0].select("healthy_1").run()

    assert not at.exception
    assert calls == ["http://api.test/predict"]
    assert at.subheader[0].value == "Healthy"
