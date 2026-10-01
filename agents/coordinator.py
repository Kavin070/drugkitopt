"""
AI Coordinator / Supervisor Agent.

This module orchestrates the three specialist agents and the existing ML
prediction function. It is intentionally deterministic at this stage so the
project can be tested without an external LLM/API key.

The coordinator:
1. Validates the input.
2. Runs the Medical Assessment Agent.
3. Runs the Regional Risk Agent.
4. Runs the Clinical Safety Agent.
5. Optionally calls an existing ML prediction callable.
6. Combines the results into one structured response.

The future LLM layer can consume this structured result without changing the
specialist agents.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .medical_agent import MedicalAssessmentAgent
from .regional_agent import RegionalRiskAssessmentAgent
from .safety_agent import ClinicalSafetyValidationAgent
from rag.retriever import retrieve_drug_candidates


class AICoordinator:
    """Supervisor that coordinates the project's specialist agents."""

    def __init__(
        self,
        medical_agent: Optional[MedicalAssessmentAgent] = None,
        regional_agent: Optional[RegionalRiskAssessmentAgent] = None,
        safety_agent: Optional[ClinicalSafetyValidationAgent] = None,
    ):
        self.medical_agent = medical_agent or MedicalAssessmentAgent()
        self.regional_agent = regional_agent or RegionalRiskAssessmentAgent()
        self.safety_agent = safety_agent or ClinicalSafetyValidationAgent()

    @staticmethod
    def _clean_list(value: Any) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [value]

    @staticmethod
    def _run_ml(
        ml_predictor: Optional[Callable[..., Any]],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Safely call the existing ML prediction function.

        Supports both:
        1. Predictors accepting keyword arguments.
        2. Predictors accepting the complete inputs dictionary.
        """

        if ml_predictor is None:
            return {
                "status": "NOT_CALLED",
                "prediction": None,
                "message": (
                    "No ML predictor was supplied to the coordinator. "
                    "The existing ML model has not been modified."
                ),
            }

        # First support the project's existing keyword-based ML adapter.
        try:
            prediction = ml_predictor(
                symptom=inputs.get("symptom", ""),
                condition=inputs.get("condition", ""),
                region=inputs.get("region", ""),
                mission_type=inputs.get("mission_type", ""),
            )

            return {
                "status": "SUCCESS",
                "prediction": prediction,
                "message": "Existing ML predictor returned a result.",
            }

        except TypeError:
            # Backward compatibility with predictors that accept
            # the complete input dictionary.
            try:
                prediction = ml_predictor(inputs)

                return {
                    "status": "SUCCESS",
                    "prediction": prediction,
                    "message": "Existing ML predictor returned a result.",
                }

            except Exception as exc:
                return {
                    "status": "ERROR",
                    "prediction": None,
                    "message": f"ML predictor could not be called: {exc}",
                }

        except Exception as exc:
            return {
                "status": "ERROR",
                "prediction": None,
                "message": f"ML predictor could not be called: {exc}",
            }

    def analyze(
        self,
        *,
        symptom: str = "",
        condition: str = "",
        region: str = "",
        mission_type: str = "",
        environment: str = "",
        medical_history: Optional[list] = None,
        allergies: Optional[list] = None,
        current_medications: Optional[list] = None,
        predicted_drug: str = "",
        ml_predictor: Optional[Callable[..., Any]] = None,
    ) -> Dict[str, Any]:
        """Run the complete multi-agent analysis workflow."""

        allergies = self._clean_list(allergies)
        medical_history = self._clean_list(medical_history)
        current_medications = self._clean_list(current_medications)

        inputs = {
            "symptom": symptom,
            "condition": condition,
            "region": region,
            "mission_type": mission_type,
            "environment": environment,
            "medical_history": medical_history,
            "allergies": allergies,
            "current_medications": current_medications,
            "predicted_drug": predicted_drug,
        }

        medical_result = self.medical_agent.assess(
            symptoms=symptom,
            condition=condition,
            medical_history=", ".join(medical_history),
            allergies=", ".join(allergies),
            region=region,
            mission_type=mission_type,
        )

        regional_result = self.regional_agent.assess(
            region=region,
            mission_type=mission_type,
            environment=environment,
            symptoms=symptom,
            condition=condition,
        )
                # Retrieve unique drug candidates from the local RAG knowledge base.
        # These are evidence candidates from project data, not prescriptions.
        rag_query = " ".join(
            part
            for part in [
                symptom,
                condition,
                region,
                mission_type,
                environment,
            ]
            if part
        )

        try:
            rag_drug_candidates = retrieve_drug_candidates(
                rag_query,
                top_k=5,
            )
        except Exception as exc:
            rag_drug_candidates = []
            rag_error = str(exc)
        else:
            rag_error = ""
        # Run the existing ML model first.
        ml_result = self._run_ml(ml_predictor, inputs)

        # Extract the ML prediction for downstream safety validation.
        ml_prediction = ml_result.get("prediction")

        if isinstance(ml_prediction, dict):
            predicted_drug_for_safety = ml_prediction.get("prediction") or predicted_drug
        else:
            predicted_drug_for_safety = ml_prediction or predicted_drug

        # Validate the ML result using the Clinical Safety Agent.
        safety_result = self.safety_agent.validate(
            predicted_drug=str(predicted_drug_for_safety or ""),
            allergies=", ".join(allergies),
            medical_history=", ".join(medical_history),
            current_medications=", ".join(current_medications),
            condition=condition,
            symptoms=symptom,
            region=region,
            mission_type=mission_type,
        )

        # Determine a conservative overall state.
        safety_status = str(safety_result.get("status", "")).upper()
        requires_review = (
            bool(medical_result.get("requires_medical_review", True))
            or bool(regional_result.get("requires_medical_review", True))
            or bool(safety_result.get("requires_medical_review", True))
            or safety_status in {"FLAGGED", "INSUFFICIENT_INFORMATION"}
        )

        if safety_status == "FLAGGED":
            overall_status = "FLAGGED_FOR_MEDICAL_REVIEW"
        elif safety_status == "INSUFFICIENT_INFORMATION":
            overall_status = "INSUFFICIENT_INFORMATION"
        else:
            overall_status = "ANALYSIS_COMPLETE_MEDICAL_REVIEW_REQUIRED"

        return {
            "status": overall_status,
            "requires_medical_review": requires_review,
            "inputs": inputs,
            "rag_drug_candidates": rag_drug_candidates,
            "rag_drug_candidate_error": rag_error,
            "ml_prediction": ml_result,
            "medical_assessment": medical_result,
            "regional_risk_assessment": regional_result,
            "clinical_safety_validation": safety_result,
            "coordinator_summary": {
                "agents_called": [
                    "Medical Assessment Agent",
                    "Regional Risk Assessment Agent",
                    "Clinical Safety Validation Agent",
                ],
                "ml_called": ml_predictor is not None,
                "rag_used_through_agents": True,
                "next_step": (
                    "Use the structured result for an LLM explanation layer "
                    "or display it directly in the application."
                ),
            },
            "medical_disclaimer": (
                "Academic prototype only. This output is not a diagnosis, "
                "prescription, or substitute for qualified medical review."
            ),
        }


# Convenient functional API for the application.
_default_coordinator: Optional[AICoordinator] = None


def get_coordinator() -> AICoordinator:
    global _default_coordinator
    if _default_coordinator is None:
        _default_coordinator = AICoordinator()
    return _default_coordinator


def run_coordinator(**kwargs: Any) -> Dict[str, Any]:
    """
    Run the coordinator. If no predictor is explicitly supplied, attempt to
    use the existing project's ML adapter. This keeps direct callers and tests
    backward compatible.
    """
    if kwargs.get("ml_predictor") is None:
        try:
            from ml_adapter import predict_with_existing_model
            kwargs["ml_predictor"] = predict_with_existing_model
        except Exception:
            # Preserve the coordinator's safe no-predictor behavior.
            pass
    return get_coordinator().analyze(**kwargs)
