"""Medical Assessment Agent for the Military Drug Kit Optimizer.

Stage 2 intentionally keeps this agent deterministic and local. It uses the
existing RAG retriever as a grounding tool and does not call an LLM or make a
diagnosis/prescription. A later Coordinator Agent can call this module as one
of its tools.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from rag.retriever import LocalRAGRetriever, get_retriever


@dataclass
class MedicalAssessment:
    """Structured output returned by the Medical Assessment Agent."""

    status: str
    symptom_summary: str
    identified_concerns: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    retrieved_evidence: List[Dict] = field(default_factory=list)
    requires_medical_review: bool = True
    disclaimer: str = (
        "Educational prototype only. This assessment does not diagnose a "
        "condition or prescribe medication and must be reviewed by qualified "
        "medical personnel."
    )

    def to_dict(self) -> Dict:
        return asdict(self)


class MedicalAssessmentAgent:
    """Assess symptoms using simple rules plus grounded RAG retrieval."""

    # These are screening terms only; they are deliberately not diagnoses.
    RED_FLAG_TERMS = {
        "breathing difficulty": "Breathing difficulty requires prompt medical assessment.",
        "shortness of breath": "Shortness of breath requires prompt medical assessment.",
        "altered mental status": "Altered mental status is a red-flag symptom requiring prompt assessment.",
        "unconscious": "Loss of consciousness requires prompt medical assessment.",
        "severe weakness": "Severe weakness requires prompt medical assessment.",
        "severe bleeding": "Severe bleeding requires urgent medical assessment.",
        "chest pain": "Chest pain requires prompt medical assessment.",
    }

    def __init__(self, retriever: Optional[LocalRAGRetriever] = None):
        self.retriever = retriever or get_retriever()

    @staticmethod
    def _normalise(value: Optional[str]) -> str:
        return " ".join((value or "").strip().lower().split())

    def _find_red_flags(self, symptoms: str) -> List[str]:
        return [
            message
            for term, message in self.RED_FLAG_TERMS.items()
            if term in symptoms
        ]

    def assess(
        self,
        symptoms: str,
        condition: str = "",
        medical_history: str = "",
        allergies: str = "",
        region: str = "",
        mission_type: str = "",
        top_k: int = 4,
    ) -> Dict:
        """Return a grounded, non-diagnostic medical assessment.

        The method gathers context, identifies missing safety information,
        retrieves relevant knowledge, and returns structured evidence for a
        future supervisor/coordinator agent.
        """
        symptoms_clean = " ".join((symptoms or "").strip().split())
        condition_clean = " ".join((condition or "").strip().split())
        history_clean = " ".join((medical_history or "").strip().split())
        allergies_clean = " ".join((allergies or "").strip().split())
        region_clean = " ".join((region or "").strip().split())
        mission_clean = " ".join((mission_type or "").strip().split())

        missing: List[str] = []
        if not symptoms_clean and not condition_clean:
            missing.append("Symptoms or a condition description")
        if not history_clean:
            missing.append("Relevant medical history")
        if not allergies_clean:
            missing.append("Known medication allergies")

        query_parts = [
            condition_clean,
            symptoms_clean,
            history_clean,
            region_clean,
            mission_clean,
        ]
        query = " ".join(part for part in query_parts if part)

        evidence = (
            self.retriever.retrieve(
                query=query,
                top_k=top_k,
                category="diseases",
            )
            if query
            else []
        )
        red_flags = self._find_red_flags(self._normalise(symptoms_clean))

        concerns: List[str] = []
        if condition_clean:
            concerns.append(f"User-provided condition: {condition_clean}")
        if symptoms_clean:
            concerns.append(f"User-reported symptoms: {symptoms_clean}")
        if region_clean:
            concerns.append(f"Region supplied for context: {region_clean}")
        if mission_clean:
            concerns.append(f"Mission type supplied for context: {mission_clean}")

        if not evidence and query:
            concerns.append("No sufficiently relevant knowledge chunk was retrieved.")

        status = "REVIEW_REQUIRED"
        if red_flags:
            status = "RED_FLAG_REVIEW_REQUIRED"
        elif not symptoms_clean and not condition_clean:
            status = "INSUFFICIENT_INFORMATION"

        result = MedicalAssessment(
            status=status,
            symptom_summary=symptoms_clean or condition_clean or "No symptom/condition information supplied.",
            identified_concerns=concerns,
            missing_information=missing,
            red_flags=red_flags,
            retrieved_evidence=evidence,
            requires_medical_review=True,
        )
        return result.to_dict()
