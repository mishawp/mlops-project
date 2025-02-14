from fastapi.testclient import TestClient
from app.src.main import app

client = TestClient(app)


def test_predict():
    response = client.post(
        "/predict",
        json=dict(
            fixed_acidity=7.4,
            volatile_acidity=0.7,
            citric_acid=0.0,
            residual_sugar=1.9,
            chlorides=0.076,
            free_sulfur_dioxide=11.0,
            total_sulfur_dioxide=34.0,
            density=0.9978,
            pH=3.51,
            sulphates=0.56,
            alcohol=9.4,
        ),
    )
    quality = response.json()
    assert response.status_code == 200
    assert list(quality.keys())[0] == "predicted_quality"
    assert isinstance(quality["predicted_quality"], float)
