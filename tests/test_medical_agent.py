import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.medical_agent import MedicalAssessmentAgent
from rag.retriever import LocalRAGRetriever


def test_medical_agent_returns_structured_grounded_assessment():
    agent = MedicalAssessmentAgent(LocalRAGRetriever().build())
    result = agent.assess(
        symptoms="fever and dehydration",
        region="tropical",
        mission_type="jungle",
        medical_history="none provided",
        allergies="none known",
    )

    assert result["status"] == "REVIEW_REQUIRED"
    assert result["requires_medical_review"] is True
    assert result["retrieved_evidence"]
    assert "symptom_summary" in result


def test_medical_agent_flags_red_flag_symptom():
    agent = MedicalAssessmentAgent(LocalRAGRetriever().build())
    result = agent.assess(
        symptoms="breathing difficulty",
        medical_history="not provided",
        allergies="not provided",
    )

    assert result["status"] == "RED_FLAG_REVIEW_REQUIRED"
    assert result["red_flags"]
    assert result["requires_medical_review"] is True
