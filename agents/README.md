# Stage 5 — AI Coordinator / Supervisor

This stage adds the central AI Coordinator that orchestrates the three
specialist agents:

- Medical Assessment Agent
- Regional Risk Assessment Agent
- Clinical Safety Validation Agent

The coordinator also provides an adapter for the existing ML prediction
function. The existing ML model is not replaced or retrained by this stage.

## Current workflow

User input
    -> AI Coordinator
        -> Medical Agent -> RAG
        -> Regional Agent -> RAG
        -> Safety Agent -> RAG
        -> Existing ML predictor (when supplied)
    -> Structured combined result

## Important design choice

Stage 5 is intentionally API-free and deterministic. It creates the
multi-agent orchestration layer before adding an LLM. This makes the system
easier to test and debug.

The next stage can consume `AICoordinator.analyze(...)` output and use an LLM
to produce a grounded natural-language explanation.

## Example

```python
from agents.coordinator import run_coordinator

result = run_coordinator(
    symptom="fever",
    condition="malaria",
    region="tropical",
    mission_type="jungle",
    allergies=["penicillin"],
    predicted_drug="ExampleDrug",
)

print(result["status"])
print(result["medical_assessment"])
print(result["regional_risk_assessment"])
print(result["clinical_safety_validation"])
```

Medical disclaimer: this is an academic prototype and must not be used as a
real-world clinical decision system.
