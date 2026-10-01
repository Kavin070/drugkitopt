"""Clinical Safety Validation Agent for the Military Drug Kit Optimizer.

Stage 4 adds a conservative safety-validation specialist. It validates an
existing ML-generated recommendation against user-provided allergies,
contraindication/interaction context, and grounded RAG evidence. It does not
prescribe medication, diagnose conditions, or declare a drug clinically safe.
A later Coordinator Agent can call this module after the ML prediction tool.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import re

from rag.retriever import LocalRAGRetriever, get_retriever


@dataclass
class SafetyValidation:
    """Structured output returned by the Clinical Safety Agent."""

    status: str
    predicted_drug: str
    warnings: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    interaction_warnings: List[str] = field(default_factory=list)
    allergy_warnings: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    evidence: List[Dict] = field(default_factory=list)
    requires_medical_review: bool = True
    disclaimer: str = (
        "Educational prototype only. This validation does not establish that a "
        "medication is clinically safe or appropriate. A qualified medical "
        "professional must review the recommendation before use."
    )

    def to_dict(self) -> Dict:
        return asdict(self)


class ClinicalSafetyValidationAgent:
    """Conservative, RAG-grounded safety screening for an ML prediction."""

    # Generic terms used only to detect that the supplied allergy/context may
    # need attention. They are not a substitute for a validated drug database.
    ALLERGY_TERMS = {
        "allergy",
        "allergic",
        "anaphylaxis",
        "anaphylactic",
        "hypersensitivity",
    }
    INTERACTION_TERMS = {
        "interaction",
        "interacts",
        "drug interaction",
        "medication interaction",
        "concomitant",
    }
    CONTRAINDICATION_TERMS = {
        "contraindication",
        "contraindicated",
        "avoid",
        "should not be used",
    }

    def __init__(self, retriever: Optional[LocalRAGRetriever] = None):
        self.retriever = retriever or get_retriever()

    @staticmethod
    def _clean(value: Optional[str]) -> str:
        return " ".join((value or "").strip().split())

    @staticmethod
    def _normalise(value: str) -> str:
        return " ".join(value.lower().split())

    def _retrieve_safety_evidence(
        self,
        predicted_drug: str,
        allergies: str,
        medical_history: str,
        current_medications: str,
        condition: str,
        symptoms: str,
        top_k: int,
    ) -> List[Dict]:
        query = " ".join(
            part
            for part in [
                predicted_drug,
                allergies,
                medical_history,
                current_medications,
                condition,
                symptoms,
                "safety allergy contraindication drug interaction",
            ]
            if part
        )
        return (
            self.retriever.retrieve(
                query=query,
                top_k=top_k,
                category="safety",
            )
            if query
            else []
        )

    def _scan_evidence(self, evidence: List[Dict]) -> tuple[List[str], List[str], List[str]]:
        allergy_warnings: List[str] = []
        contraindications: List[str] = []
        interaction_warnings: List[str] = []

        for item in evidence:
            text = item.get("text", "")
            lower = self._normalise(text)
            source = item.get("source", "knowledge base")
            if any(term in lower for term in self.ALLERGY_TERMS):
                allergy_warnings.append(f"Review allergy/hypersensitivity information in {source}.")
            if any(term in lower for term in self.CONTRAINDICATION_TERMS):
                contraindications.append(f"Review contraindication information in {source}.")
            if any(term in lower for term in self.INTERACTION_TERMS):
                interaction_warnings.append(f"Review interaction information in {source}.")

        return (
            list(dict.fromkeys(allergy_warnings)),
            list(dict.fromkeys(contraindications)),
            list(dict.fromkeys(interaction_warnings)),
        )

    def validate(
        self,
        predicted_drug: str = "",
        allergies: str = "",
        medical_history: str = "",
        current_medications: str = "",
        condition: str = "",
        symptoms: str = "",
        region: str = "",
        mission_type: str = "",
        top_k: int = 5,
    ) -> Dict:
        """Return a conservative safety screening result.

        An empty or incomplete safety record is never treated as approval.
        The method only flags issues and identifies evidence that needs review.
        """
        drug = self._clean(predicted_drug)
        allergy_text = self._clean(allergies)
        history = self._clean(medical_history)
        medications = self._clean(current_medications)
        condition_clean = self._clean(condition)
        symptoms_clean = self._clean(symptoms)
        region_clean = self._clean(region)
        mission_clean = self._clean(mission_type)

        missing: List[str] = []
        if not drug:
            missing.append("ML-predicted drug/recommendation")
        if not allergy_text:
            missing.append("Known medication allergies")
        if not history:
            missing.append("Relevant medical history")
        if not medications:
            missing.append("Current medications or confirmation that none are being taken")

        evidence = self._retrieve_safety_evidence(
            predicted_drug=drug,
            allergies=allergy_text,
            medical_history=history,
            current_medications=medications,
            condition=condition_clean,
            symptoms=symptoms_clean,
            top_k=top_k,
        )

        allergy_warnings, contraindications, interaction_warnings = self._scan_evidence(evidence)
        warnings: List[str] = []

        if allergy_text:
            warnings.append("User supplied allergy information must be checked against the predicted drug and its ingredients.")
        else:
            warnings.append("No allergy information was supplied; allergy safety cannot be established.")

        if medications:
            warnings.append("Current medications were supplied; interaction review is required before use.")
        else:
            warnings.append("Current medication information is incomplete; interaction safety cannot be established.")

        if evidence:
            warnings.append("Retrieved knowledge is contextual evidence only and requires qualified clinical review.")
        else:
            warnings.append("No relevant safety evidence was retrieved from the local knowledge base.")

        # Explicitly flag a possible allergy concern when the user's text itself
        # contains allergy/hypersensitivity language. We do not infer that a
        # particular drug is allergic without validated drug-specific evidence.
        if any(term in self._normalise(allergy_text) for term in self.ALLERGY_TERMS):
            allergy_warnings.append("Allergy/hypersensitivity history was reported; verify compatibility with the predicted drug.")

        # If the user explicitly names the predicted drug in an allergy field,
        # flag it conservatively. This is string matching only, not a clinical claim.
        if drug and allergy_text and re.search(rf"\b{re.escape(drug.lower())}\b", allergy_text.lower()):
            allergy_warnings.append("The predicted drug name appears in the supplied allergy information; do not approve without medical review.")

        allergy_warnings = list(dict.fromkeys(allergy_warnings))
        contraindications = list(dict.fromkeys(contraindications))
        interaction_warnings = list(dict.fromkeys(interaction_warnings))
        warnings = list(dict.fromkeys(warnings))

        if not drug or not evidence:
            status = "INSUFFICIENT_INFORMATION"
        elif allergy_warnings or contraindications or interaction_warnings:
            status = "FLAGGED"
        else:
            # Deliberately conservative: absence of a retrieved warning is not
            # equivalent to a clinical safety approval.
            status = "SAFE_FOR_REVIEW"

        return SafetyValidation(
            status=status,
            predicted_drug=drug or "Not supplied",
            warnings=warnings,
            contraindications=contraindications,
            interaction_warnings=interaction_warnings,
            allergy_warnings=allergy_warnings,
            missing_information=missing,
            evidence=evidence,
            requires_medical_review=True,
        ).to_dict()


# Friendly alias for the future Coordinator Agent.
SafetyAgent = ClinicalSafetyValidationAgent
