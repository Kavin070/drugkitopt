from pathlib import Path

from agents.regional_agent import RegionalRiskAssessmentAgent
from rag.retriever import LocalRAGRetriever


ROOT = Path(__file__).resolve().parents[1]


def test_regional_agent_identifies_tropical_context():
    retriever = LocalRAGRetriever(ROOT / "knowledge_base").build()
    agent = RegionalRiskAssessmentAgent(retriever)

    result = agent.assess(
        region="Tropical",
        mission_type="Jungle Ops",
        environment="hot humid jungle with mosquito exposure",
    )

    assert result["status"] in {"REVIEW_REQUIRED", "HIGH_CONTEXT_RISK_REVIEW_REQUIRED"}
    assert result["risk_level"] in {"MODERATE_CONTEXT_RISK", "HIGH_CONTEXT_RISK"}
    assert result["identified_risks"]
    assert result["environmental_factors"]
    assert result["retrieved_evidence"]
    assert result["requires_medical_review"] is True


def test_regional_agent_handles_missing_context():
    retriever = LocalRAGRetriever(ROOT / "knowledge_base").build()
    agent = RegionalRiskAssessmentAgent(retriever)

    result = agent.assess()

    assert result["status"] == "INSUFFICIENT_INFORMATION"
    assert result["risk_level"] == "UNKNOWN"
    assert "Region" in result["missing_information"]
    assert "Mission type" in result["missing_information"]
    assert "Environment/operational conditions" in result["missing_information"]
    assert result["requires_medical_review"] is True


def test_regional_agent_can_use_rag_for_altitude_context():
    retriever = LocalRAGRetriever(ROOT / "knowledge_base").build()
    agent = RegionalRiskAssessmentAgent(retriever)

    result = agent.assess(
        region="Mountain",
        mission_type="High Altitude Patrol",
        environment="high altitude cold mountain terrain",
    )

    evidence_text = " ".join(item["text"].lower() for item in result["retrieved_evidence"])
    assert "altitude" in evidence_text or "cold" in evidence_text
    assert result["risk_level"] in {"MODERATE_CONTEXT_RISK", "HIGH_CONTEXT_RISK"}
