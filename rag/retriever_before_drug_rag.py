"""Lightweight local retrieval layer for the Military Drug Kit Optimizer.

This module intentionally does not call an LLM. It retrieves relevant knowledge
chunks that a future agent/LLM can use as grounded context.

The default implementation uses TF-IDF + cosine similarity so the existing
project remains lightweight and works without an external API or paid service.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge_base"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "rag" / "index"


@dataclass
class KnowledgeChunk:
    """A retrievable piece of knowledge."""

    chunk_id: str
    source: str
    category: str
    text: str


class LocalRAGRetriever:
    """Simple local RAG retriever based on TF-IDF vectors.

    It is deliberately separated from generation: retrieve() only returns
    evidence. A future agent/LLM is responsible for using that evidence.
    """

    def __init__(self, knowledge_dir: Path | str = DEFAULT_KNOWLEDGE_DIR):
        self.knowledge_dir = Path(knowledge_dir)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.chunks: List[KnowledgeChunk] = []

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        return text

    def load_documents(self) -> List[KnowledgeChunk]:
        """Load markdown/text knowledge files and create simple chunks."""
        chunks: List[KnowledgeChunk] = []
        for path in sorted(self.knowledge_dir.rglob("*")):
            if path.suffix.lower() not in {".txt", ".md"}:
                continue

            category = path.parent.name
            raw = path.read_text(encoding="utf-8")
            paragraphs = [self._clean_text(p) for p in re.split(r"\n\s*\n", raw)]
            paragraphs = [p for p in paragraphs if p]

            for idx, paragraph in enumerate(paragraphs):
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=f"{path.stem}-{idx}",
                        source=str(path.relative_to(self.knowledge_dir)),
                        category=category,
                        text=paragraph,
                    )
                )

        if not chunks:
            raise FileNotFoundError(
                f"No .txt or .md knowledge files found under {self.knowledge_dir}"
            )
        return chunks

    def build(self) -> "LocalRAGRetriever":
        """Build the in-memory retrieval index."""
        self.chunks = self.load_documents()
        texts = [chunk.text for chunk in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(texts)
        return self

    def save(self, index_dir: Path | str = DEFAULT_INDEX_DIR) -> None:
        """Persist the lightweight index metadata for inspection/reuse."""
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("Build the retriever before saving it.")

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        # We intentionally store the source documents and metadata, then rebuild
        # the TF-IDF matrix on load. This keeps the persisted format simple.
        metadata = [asdict(chunk) for chunk in self.chunks]
        (index_dir / "chunks.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
    ) -> List[Dict]:
        """Return the most relevant knowledge chunks for a query."""
        if not query or not query.strip():
            return []
        if self.vectorizer is None or self.matrix is None:
            self.build()

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()

        candidates = np.argsort(scores)[::-1]
        results = []
        for idx in candidates:
            chunk = self.chunks[int(idx)]
            if category and chunk.category.lower() != category.lower():
                continue
            score = float(scores[int(idx)])
            if score <= 0:
                continue
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "category": chunk.category,
                    "text": chunk.text,
                    "score": round(score, 4),
                }
            )
            if len(results) >= top_k:
                break
        return results


def get_retriever() -> LocalRAGRetriever:
    """Create and build a retriever using the project's knowledge base."""
    return LocalRAGRetriever().build()


def retrieve_context(query: str, top_k: int = 5, category: Optional[str] = None) -> List[Dict]:
    """Convenience function for future agents."""
    return get_retriever().retrieve(query=query, top_k=top_k, category=category)
