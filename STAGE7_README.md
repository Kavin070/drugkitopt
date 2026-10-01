# Stage 7 — FastAPI Backend

Stage 7 adds a FastAPI backend around the existing Coordinator + RAG +
specialist agents + LLM explanation layer.

## API architecture

Streamlit (existing UI)
        |
        v
FastAPI
        |
        v
AI Coordinator
   |       |       |
Medical Regional Safety
        |
        v
Existing ML + RAG
        |
        v
LLM Explanation
        |
        v
JSON response

## Endpoints

### GET /
Basic service information.

### GET /health
Health check.

### POST /api/analyze
Runs the complete coordinator and explanation workflow.

Example request:

```json
{
  "symptom": "fever",
  "condition": "malaria",
  "region": "tropical",
  "mission_type": "jungle",
  "environment": "humid",
  "medical_history": [],
  "allergies": ["penicillin"],
  "current_medications": [],
  "predicted_drug": "ExampleDrug"
}
```

## Run locally

From the project root:

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload
```

Then open the API docs at:

```text
http://127.0.0.1:8000/docs
```

The current API does not automatically call the existing Streamlit ML
prediction function because the exact Streamlit-side model invocation should
be integrated in the next stage rather than guessed or duplicated.

Medical disclaimer: academic prototype only; qualified medical review is
required for real-world use.
