import pytest
from fastapi.testclient import TestClient

from api.main import app

from .conftest import EXAMPLES, ROOT, expected_label


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # 'with' runs the lifespan, which loads the model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


@pytest.mark.parametrize("path", EXAMPLES[::2], ids=lambda p: p.stem)
def test_predict_example(client, path):
    r = client.post("/predict", files={"file": (path.name, path.read_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == expected_label(path)
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4
    assert list(body["probabilities"])[0] == body["label"]  # sorted highest first
    assert body["low_confidence"] is False


def test_predict_rgba_png(client):
    path = ROOT / "docs" / "non_corn" / "logo2.png"
    r = client.post("/predict", files={"file": (path.name, path.read_bytes(), "image/png")})
    assert r.status_code == 200


def test_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_rejects_empty_file(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"", "image/jpeg")})
    assert r.status_code == 400


def test_requires_file(client):
    assert client.post("/predict").status_code == 422
