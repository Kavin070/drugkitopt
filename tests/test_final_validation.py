from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_final_structure():
    files=[
        "app/app.py","api/main.py","agents/coordinator.py",
        "agents/medical_agent.py","agents/regional_agent.py",
        "agents/safety_agent.py","agents/llm_explainer.py",
        "rag/retriever.py","ml_adapter.py","models/model.pkl",
        "data/dataset.csv","requirements.txt","README.md",".env.example"
    ]
    assert all((ROOT/f).exists() for f in files)
def test_streamlit_integration():
    t=(ROOT/"app/app.py").read_text(encoding="utf-8")
    assert "Agentic AI Medical Analysis" in t
    assert "Run Agentic AI Analysis" in t
