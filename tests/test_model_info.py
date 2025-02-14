from fastapi.testclient import TestClient
from app.src.main import app

client = TestClient(app)


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    # assert response.json() == ...
    # структура ответа может быть любой
