import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag.retriever import LocalRAGRetriever


def test_rag_builds_and_retrieves():
    retriever = LocalRAGRetriever().build()
    assert len(retriever.chunks) > 0

    results = retriever.retrieve("tropical heat humidity insect exposure", top_k=3)
    assert results
    assert any("tropical" in r["text"].lower() for r in results)
