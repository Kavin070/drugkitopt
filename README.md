# Military Drug Kit Optimizer — Final ML + RAG + Agentic AI Prototype

## Architecture
Streamlit -> FastAPI -> AI Coordinator/Supervisor -> Medical Agent + Regional Agent + Safety Agent -> Existing ML + RAG -> LLM Explanation -> Streamlit

## Main components
- Existing Streamlit frontend
- Existing ML model and prediction workflow
- RAG knowledge retrieval
- Medical Assessment Agent
- Regional Risk Assessment Agent
- Clinical Safety Validation Agent
- AI Coordinator/Supervisor Agent
- LLM Explanation Service
- FastAPI backend

## Windows setup
```powershell
python -m venv venv
.env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run
Terminal 1:
```powershell
python -m uvicorn api.main:app --reload
```
Terminal 2:
```powershell
python -m streamlit run app\app.py
```

API docs: http://127.0.0.1:8000/docs

## Optional LLM
Default `LLM_PROVIDER=fallback` works without an API key.
For Gemini: `pip install google-genai` then set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY`.
For OpenAI: `pip install openai` then set `LLM_PROVIDER=openai` and `OPENAI_API_KEY`.
For Claude: `pip install anthropic` then set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`.

Never commit API keys.

## Tests
```powershell
python -m pytest -q
```

## Medical disclaimer
Academic prototype only. It is not a clinical decision-support system and must not be used to diagnose, prescribe, or make real-world medical decisions. Qualified medical review and authoritative protocols are required.
