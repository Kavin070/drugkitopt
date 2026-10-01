from agents.safety_agent import ClinicalSafetyValidationAgent


def test_safety_agent_requires_review_for_missing_drug_and_context():
    agent = ClinicalSafetyValidationAgent()
    result = agent.validate(
        predicted_drug="",
        allergies="",
        medical_history="",
        current_medications="",
        condition="field illness",
    )
    assert result["status"] == "INSUFFICIENT_INFORMATION"
    assert result["requires_medical_review"] is True
    assert result["predicted_drug"] == "Not supplied"


def test_safety_agent_flags_drug_named_in_allergy_information():
    agent = ClinicalSafetyValidationAgent()
    result = agent.validate(
        predicted_drug="amoxicillin",
        allergies="Allergy to amoxicillin",
        medical_history="No known chronic history",
        current_medications="None",
        condition="infection",
    )
    assert result["status"] == "FLAGGED"
    assert result["allergy_warnings"]
    assert result["requires_medical_review"] is True


def test_safety_agent_returns_structured_output_with_evidence():
    agent = ClinicalSafetyValidationAgent()
    result = agent.validate(
        predicted_drug="oral rehydration solution",
        allergies="No known medication allergies",
        medical_history="No relevant history reported",
        current_medications="None",
        condition="dehydration",
        symptoms="thirst and fatigue",
        region="desert",
        mission_type="desert patrol",
    )
    assert result["status"] in {"SAFE_FOR_REVIEW", "FLAGGED", "INSUFFICIENT_INFORMATION"}
    assert isinstance(result["evidence"], list)
    assert result["requires_medical_review"] is True
