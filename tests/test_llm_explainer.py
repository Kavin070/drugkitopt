from agents.coordinator import AICoordinator
from agents.llm_explainer import (
    LLMExplanationService,
    build_explanation_prompt,
    deterministic_explanation,
)


def sample_result():
    return AICoordinator().analyze(
        symptom="fever",
        condition="malaria",
        region="tropical",
        mission_type="jungle",
        allergies=["penicillin"],
        predicted_drug="ExampleDrug",
    )


def test_prompt_contains_structured_result():
    result = sample_result()
    prompt = build_explanation_prompt(result)

    assert "STRUCTURED COORDINATOR RESULT" in prompt
    assert "ExampleDrug" in prompt
    assert "do not independently prescribe" in prompt.lower()


def test_deterministic_fallback_works_without_api():
    result = sample_result()
    explanation = deterministic_explanation(result)

    assert "AI Analysis Summary" in explanation
    assert "academic prototype" in explanation.lower()


def test_fallback_provider_is_available():
    result = sample_result()
    service = LLMExplanationService(provider="fallback")
    output = service.explain(result)

    assert output["status"] == "FALLBACK"
    assert output["provider"] == "deterministic"
    assert output["explanation"]


def test_unknown_provider_falls_back():
    result = sample_result()
    service = LLMExplanationService(provider="unknown-provider")
    output = service.explain(result)

    assert output["status"] == "FALLBACK"
    assert "Unsupported LLM_PROVIDER" in output["warning"]
