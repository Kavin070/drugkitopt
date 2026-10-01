"""Regional Risk Assessment Agent for the Military Drug Kit Optimizer.

Stage 3 adds a deterministic regional-risk specialist that uses the existing
local RAG retriever. It does not prescribe medication, call an LLM, or alter
the existing ML prediction engine. A later Coordinator Agent can call this
module as one of its tools.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from rag.retriever import LocalRAGRetriever, get_retriever


@dataclass
class RegionalRiskAssessment:
    """Structured output returned by the Regional Risk Assessment Agent."""

    status: str
    region: str
    mission_type: str
    environment: str
    identified_risks: List[str] = field(default_factory=list)
    relevant_conditions: List[str] = field(default_factory=list)
    environmental_factors: List[str] = field(default_factory=list)
    retrieved_evidence: List[Dict] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    risk_level: str = "UNKNOWN"
    requires_medical_review: bool = True
    disclaimer: str = (
        "Educational prototype only. Regional risk screening is not a field "
        "medical diagnosis or operational medical directive and must be "
        "reviewed by qualified medical personnel."
    )

    def to_dict(self) -> Dict:
        return asdict(self)


class RegionalRiskAssessmentAgent:
    """Assess environmental and mission context using grounded RAG retrieval."""

    # Screening keywords are deliberately broad. They identify context to
    # investigate rather than making a clinical diagnosis or prescription.
    RISK_PATTERNS = {
        "tropical": (
            "Tropical environment may increase heat, dehydration and vector-"
            "exposure considerations."
        ),
        "jungle": (
            "Jungle operations may involve heat, humidity, water exposure and "
            "increased vector/environmental exposure."
        ),
        "desert": (
            "Desert operations may increase heat stress, dehydration and "
            "dust/sand exposure considerations."
        ),
        "arctic": (
            "Arctic/cold environments may increase cold-exposure and "
            "hypothermia/frostbite considerations."
        ),
        "cold": (
            "Cold environments require attention to cold exposure and "
            "environmental protection."
        ),
        "high altitude": (
            "High-altitude operations may involve reduced oxygen availability "
            "and altitude-related health considerations."
        ),
        "altitude": (
            "Altitude is an environmental factor that may require additional "
            "medical and acclimatization assessment."
        ),
        "mountain": (
            "Mountain operations may involve altitude, cold, terrain and "
            "dehydration considerations."
        ),
        "water": (
            "Field water exposure can create hydration and water-safety "
            "considerations."
        ),
    }

    ENVIRONMENT_TERMS = {
        "heat": "Heat exposure",
        "hot": "Hot conditions",
        "humidity": "High humidity",
        "cold": "Cold exposure",
        "snow": "Snow/ice exposure",
        "altitude": "High altitude",
        "mountain": "Mountain terrain",
        "desert": "Dry/desert environment",
        "jungle": "Jungle/forest environment",
        "water": "Water exposure",
        "dust": "Dust/sand exposure",
        "vector": "Potential vector exposure",
        "mosquito": "Potential mosquito exposure",
    }

    def __init__(self, retriever: Optional[LocalRAGRetriever] = None):
        self.retriever = retriever or get_retriever()

    @staticmethod
    def _clean(value: Optional[str]) -> str:
        return " ".join((value or "").strip().split())

    @staticmethod
    def _normalise(value: str) -> str:
        return " ".join(value.lower().split())

    def _identify_risks(self, context: str) -> List[str]:
        normalized = self._normalise(context)
        risks: List[str] = []
        for term, message in self.RISK_PATTERNS.items():
            if term in normalized and message not in risks:
                risks.append(message)
        return risks

    def _identify_environmental_factors(self, context: str) -> List[str]:
        normalized = self._normalise(context)
        factors: List[str] = []
        for term, label in self.ENVIRONMENT_TERMS.items():
            if term in normalized and label not in factors:
                factors.append(label)
        return factors

    def _risk_level(self, risks: List[str], evidence: List[Dict], missing: List[str]) -> str:
        """Return a screening level, not a clinical severity classification."""
        if not risks and not evidence:
            return "UNKNOWN"
        if len(risks) >= 3:
            return "HIGH_CONTEXT_RISK"
        if risks:
            return "MODERATE_CONTEXT_RISK"
        if missing:
            return "UNKNOWN"
        return "LOW_CONTEXT_RISK"

    def assess(
        self,
        region: str = "",
        mission_type: str = "",
        environment: str = "",
        symptoms: str = "",
        condition: str = "",
        top_k: int = 5,
    ) -> Dict:
        """Return a grounded, non-diagnostic regional risk assessment."""
        region_clean = self._clean(region)
        mission_clean = self._clean(mission_type)
        environment_clean = self._clean(environment)
        symptoms_clean = self._clean(symptoms)
        condition_clean = self._clean(condition)

        missing: List[str] = []
        if not region_clean:
            missing.append("Region")
        if not mission_clean:
            missing.append("Mission type")
        if not environment_clean:
            missing.append("Environment/operational conditions")

        query_parts = [
            region_clean,
            mission_clean,
            environment_clean,
            symptoms_clean,
            condition_clean,
            "regional environmental risk",
        ]
        query = " ".join(part for part in query_parts if part)
        evidence = (
            self.retriever.retrieve(
                query=query,
                top_k=top_k,
                category="regional_risks",
            )
            if query
            else []
        )

        context = " ".join(
            part for part in [region_clean, mission_clean, environment_clean] if part
        )
        risks = self._identify_risks(context)
        factors = self._identify_environmental_factors(context)

        relevant_conditions: List[str] = []
        if symptoms_clean:
            relevant_conditions.append(f"User-reported symptoms: {symptoms_clean}")
        if condition_clean:
            relevant_conditions.append(f"User-provided condition: {condition_clean}")
        if not evidence and query:
            relevant_conditions.append(
                "No sufficiently relevant regional knowledge chunk was retrieved."
            )

        risk_level = self._risk_level(risks, evidence, missing)
        status = "REVIEW_REQUIRED"
        if not region_clean and not mission_clean and not environment_clean:
            status = "INSUFFICIENT_INFORMATION"
        elif risk_level == "HIGH_CONTEXT_RISK":
            status = "HIGH_CONTEXT_RISK_REVIEW_REQUIRED"

        result = RegionalRiskAssessment(
            status=status,
            region=region_clean or "Not supplied",
            mission_type=mission_clean or "Not supplied",
            environment=environment_clean or "Not supplied",
            identified_risks=risks,
            relevant_conditions=relevant_conditions,
            environmental_factors=factors,
            retrieved_evidence=evidence,
            missing_information=missing,
            risk_level=risk_level,
            requires_medical_review=True,
        )
        return result.to_dict()
