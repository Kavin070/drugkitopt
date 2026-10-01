"""
LLM Explanation Layer.

This stage intentionally separates explanation from decision making.

The LLM receives structured results produced by the coordinator and returns
an explanation. It is not allowed to independently select or prescribe a
drug. A deterministic fallback is provided so the project remains runnable
without an API key.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


SYSTEM_INSTRUCTION = """
You are the explanation component of an academic medical-support prototype.

You receive structured outputs from:
- an existing ML prediction model,
- a Medical Assessment Agent,
- a Regional Risk Assessment Agent,
- a Clinical Safety Validation Agent,
- RAG-retrieved evidence.

Rules:
1. Do not independently prescribe, diagnose, or invent medication.
2. Do not override the clinical safety result.
3. Clearly distinguish ML prediction from retrieved knowledge.
4. Use only information present in the supplied structured results.
5. If information is missing or conflicting, say so.
6. Preserve safety warnings and medical-review requirements.
7. Do not invent dosages or treatment instructions.
8. Keep the explanation concise and understandable.
9. State that this is an academic prototype and requires qualified medical
   review for real-world use.
"""


def _safe_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def build_explanation_prompt(coordinator_result: Dict[str, Any]) -> str:
    """Build a grounded prompt from coordinator output."""
    return (
        SYSTEM_INSTRUCTION.strip()
        + "\n\nSTRUCTURED COORDINATOR RESULT:\n"
        + _safe_json(coordinator_result)
        + "\n\nWrite a grounded explanation using only this information."
    )


def deterministic_explanation(coordinator_result: Dict[str, Any]) -> str:
    """API-free fallback explanation."""
    status = coordinator_result.get("status", "UNKNOWN")
    ml = coordinator_result.get("ml_prediction", {})
    medical = coordinator_result.get("medical_assessment", {})
    regional = coordinator_result.get("regional_risk_assessment", {})
    safety = coordinator_result.get("clinical_safety_validation", {})

    prediction = ml.get("prediction")
    safety_status = safety.get("status", "UNKNOWN")
    rag_candidates = coordinator_result.get("rag_drug_candidates", [])

    lines = [
        "### AI Analysis Summary",
        f"- Overall status: **{status}**",
        f"- ML prediction: **{prediction if prediction is not None else 'Not supplied'}**",
        f"- Clinical safety status: **{safety_status}**",
    ]

    if rag_candidates:
        candidate_names = [
            str(item.get("drug_name", "Unknown"))
            for item in rag_candidates
        ]
        lines.append(
            "- RAG-retrieved drug candidates: "
            + ", ".join(candidate_names)
        )
    else:
        lines.append(
            "- RAG-retrieved drug candidates: None available"
        )
    condition = medical.get("condition_analysis")
    if condition:
        lines.append(f"- Medical assessment: {condition}")

    risks = regional.get("identified_risks") or regional.get("risks")
    if risks:
        lines.append(f"- Regional risks: {', '.join(map(str, risks))}")

    warnings = safety.get("warnings", [])
    if warnings:
        lines.append(f"- Safety warnings: {', '.join(map(str, warnings))}")

    lines.append(
        "- This is an academic prototype. Qualified medical review is required "
        "before any real-world clinical decision."
    )
    return "\n".join(lines)


class LLMExplanationService:
    """Generate a grounded explanation using a configured LLM provider."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "fallback")).lower()

    def explain(self, coordinator_result: Dict[str, Any]) -> Dict[str, Any]:
        prompt = build_explanation_prompt(coordinator_result)

        if self.provider in {"fallback", "none", "disabled"}:
            return {
                "status": "FALLBACK",
                "provider": "deterministic",
                "explanation": deterministic_explanation(coordinator_result),
                "prompt": prompt,
            }

        if self.provider == "gemini":
            return self._gemini(coordinator_result, prompt)

        if self.provider == "openai":
            return self._openai(coordinator_result, prompt)

        if self.provider == "anthropic":
            return self._anthropic(coordinator_result, prompt)

        return {
            "status": "FALLBACK",
            "provider": "deterministic",
            "explanation": deterministic_explanation(coordinator_result),
            "prompt": prompt,
            "warning": f"Unsupported LLM_PROVIDER={self.provider!r}; fallback used.",
        }

    def _gemini(self, coordinator_result: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        try:
            from google import genai
        except ImportError:
            return self._fallback_with_warning(
                coordinator_result, prompt, "google-genai is not installed."
            )

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return self._fallback_with_warning(
                coordinator_result, prompt, "GEMINI_API_KEY is not configured."
            )

        try:
            client = genai.Client(api_key=api_key)
            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            text = getattr(response, "text", None) or str(response)
            return {
                "status": "SUCCESS",
                "provider": "gemini",
                "model": model,
                "explanation": text,
                "prompt": prompt,
            }
        except Exception as exc:
            return self._fallback_with_warning(
                coordinator_result, prompt, f"Gemini call failed: {exc}"
            )

    def _openai(self, coordinator_result: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError:
            return self._fallback_with_warning(
                coordinator_result, prompt, "openai package is not installed."
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._fallback_with_warning(
                coordinator_result, prompt, "OPENAI_API_KEY is not configured."
            )

        try:
            client = OpenAI(api_key=api_key)
            model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
            response = client.responses.create(
                model=model,
                instructions=SYSTEM_INSTRUCTION,
                input=prompt,
            )
            text = getattr(response, "output_text", None) or str(response)
            return {
                "status": "SUCCESS",
                "provider": "openai",
                "model": model,
                "explanation": text,
                "prompt": prompt,
            }
        except Exception as exc:
            return self._fallback_with_warning(
                coordinator_result, prompt, f"OpenAI call failed: {exc}"
            )

    def _anthropic(self, coordinator_result: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        try:
            from anthropic import Anthropic
        except ImportError:
            return self._fallback_with_warning(
                coordinator_result, prompt, "anthropic package is not installed."
            )

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return self._fallback_with_warning(
                coordinator_result, prompt, "ANTHROPIC_API_KEY is not configured."
            )

        try:
            client = Anthropic(api_key=api_key)
            model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                system=SYSTEM_INSTRUCTION,
                messages=[{"role": "user", "content": prompt}],
            )
            text_parts = []
            for block in getattr(response, "content", []):
                block_text = getattr(block, "text", None)
                if block_text:
                    text_parts.append(block_text)
            text = "\n".join(text_parts) or str(response)
            return {
                "status": "SUCCESS",
                "provider": "anthropic",
                "model": model,
                "explanation": text,
                "prompt": prompt,
            }
        except Exception as exc:
            return self._fallback_with_warning(
                coordinator_result, prompt, f"Anthropic call failed: {exc}"
            )

    @staticmethod
    def _fallback_with_warning(
        coordinator_result: Dict[str, Any], prompt: str, warning: str
    ) -> Dict[str, Any]:
        return {
            "status": "FALLBACK",
            "provider": "deterministic",
            "explanation": deterministic_explanation(coordinator_result),
            "prompt": prompt,
            "warning": warning,
        }


_default_service: Optional[LLMExplanationService] = None


def get_llm_service() -> LLMExplanationService:
    global _default_service
    if _default_service is None:
        _default_service = LLMExplanationService()
    return _default_service


def generate_explanation(coordinator_result: Dict[str, Any]) -> Dict[str, Any]:
    return get_llm_service().explain(coordinator_result)
