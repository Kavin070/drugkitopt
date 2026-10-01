from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_analyze_endpoint():
    response = client.post(
        "/api/analyze",
        json={
            "symptom": "fever",
            "condition": "malaria",
            "region": "tropical",
            "mission_type": "jungle",
            "environment": "humid",
            "allergies": ["penicillin"],
            "current_medications": [],
            "predicted_drug": "ExampleDrug",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "coordinator_result" in data
    assert "explanation" in data
    assert data["requires_medical_review"] is True
