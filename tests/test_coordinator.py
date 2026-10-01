import pytest

from agents.coordinator import AICoordinator


def test_coordinator_calls_all_specialist_agents():
    coordinator = AICoordinator()

    result = coordinator.analyze(
        symptom="fever",
        condition="malaria",
        region="tropical",
        mission_type="jungle",
        environment="humid",
        allergies=["penicillin"],
        current_medications=[],
        predicted_drug="ExampleDrug",
    )

    assert "medical_assessment" in result
    assert "regional_risk_assessment" in result
    assert "clinical_safety_validation" in result
    assert result["coordinator_summary"]["agents_called"]
    assert result["requires_medical_review"] is True


def test_coordinator_can_call_existing_ml_predictor():
    coordinator = AICoordinator()

    def fake_ml(inputs):
        assert inputs["region"] == "desert"
        return {"drug": "ExampleDrug", "confidence": 0.80}

    result = coordinator.analyze(
        condition="heat exhaustion",
        region="desert",
        mission_type="desert",
        predicted_drug="ExampleDrug",
        ml_predictor=fake_ml,
    )

    assert result["ml_prediction"]["status"] == "SUCCESS"
    assert result["ml_prediction"]["prediction"]["drug"] == "ExampleDrug"
    assert result["coordinator_summary"]["ml_called"] is True


def test_coordinator_does_not_modify_existing_ml_when_predictor_missing():
    coordinator = AICoordinator()

    result = coordinator.analyze(
        symptom="headache",
        region="high altitude",
        mission_type="mountain",
    )

    assert result["ml_prediction"]["status"] == "NOT_CALLED"
