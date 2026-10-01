"""
FastAPI backend for the Military Drug Kit Optimizer.

Stage 7 exposes the Stage 5 coordinator and Stage 6 explanation service
through a clean API without modifying the existing Streamlit application.

This is an academic prototype. The API does not provide clinical advice.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.coordinator import run_coordinator
from agents.llm_explainer import generate_explanation


app = FastAPI(
    title="Military Drug Kit Optimizer API",
    description=(
        "Backend API for the ML + RAG + multi-agent academic prototype."
    ),
    version="1.0.0",
)


class AnalysisRequest(BaseModel):
    symptom: str = ""
    condition: str = ""
    region: str = ""
    mission_type: str = ""
    environment: str = ""
    medical_history: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    predicted_drug: str = ""


class AnalysisResponse(BaseModel):
    status: str
    requires_medical_review: bool
    coordinator_result: Dict[str, Any]
    explanation: Dict[str, Any]
    medical_disclaimer: str


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "Military Drug Kit Optimizer API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": ["/health", "/api/analyze"],
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest) -> AnalysisResponse:
    try:
        coordinator_result = run_coordinator(
            symptom=request.symptom,
            condition=request.condition,
            region=request.region,
            mission_type=request.mission_type,
            environment=request.environment,
            medical_history=request.medical_history,
            allergies=request.allergies,
            current_medications=request.current_medications,
            predicted_drug=request.predicted_drug,
        )

        explanation = generate_explanation(coordinator_result)

        disclaimer = coordinator_result.get(
            "medical_disclaimer",
            "Academic prototype only. Qualified medical review is required.",
        )

        return AnalysisResponse(
            status=coordinator_result.get("status", "UNKNOWN"),
            requires_medical_review=bool(
                coordinator_result.get("requires_medical_review", True)
            ),
            coordinator_result=coordinator_result,
            explanation=explanation,
            medical_disclaimer=disclaimer,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc
