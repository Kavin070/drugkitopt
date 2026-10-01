"""Lightweight local retrieval layer for the Military Drug Kit Optimizer.

Retrieves evidence from:
1. Existing Markdown/text knowledge base.
2. Structured drug records from data/dataset.csv.

The retriever does NOT diagnose or prescribe. It only returns grounded
retrieval evidence for downstream agents/coordinator/LLM components.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge_base"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "dataset.csv"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "rag" / "index"


@dataclass
class KnowledgeChunk:
    """A retrievable piece of knowledge."""

    chunk_id: str
    source: str
    category: str
    text: str


class LocalRAGRetriever:
    """Local TF-IDF retriever over knowledge files and structured drug data."""

    def __init__(
        self,
        knowledge_dir: Path | str = DEFAULT_KNOWLEDGE_DIR,
        dataset_path: Path | str = DEFAULT_DATASET_PATH,
    ):
        self.knowledge_dir = Path(knowledge_dir)
        self.dataset_path = Path(dataset_path)

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.chunks: List[KnowledgeChunk] = []

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def load_documents(self) -> List[KnowledgeChunk]:
        """Load Markdown/text knowledge files and CSV drug records."""
        chunks: List[KnowledgeChunk] = []

        # ---------------------------------------------------------
        # 1. Existing Markdown / text knowledge base
        # ---------------------------------------------------------
        for path in sorted(self.knowledge_dir.rglob("*")):
            if path.suffix.lower() not in {".txt", ".md"}:
                continue

            category = path.parent.name
            raw = path.read_text(encoding="utf-8")

            paragraphs = [
                self._clean_text(p)
                for p in re.split(r"\n\s*\n", raw)
            ]
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

        # ---------------------------------------------------------
        # 2. Structured drug dataset
        # ---------------------------------------------------------
        if self.dataset_path.exists():
            df = pd.read_csv(self.dataset_path)

            # Remove exact duplicate rows and rows without drug name.
            df = df.drop_duplicates()
            df = df.dropna(subset=["Drug_Name"])

            expected_columns = [
                "Drug_Name",
                "Generic_Name",
                "Use_Case",
                "When_To_Use",
                "Dosage_Form",
                "Typical_Dosage",
                "Quantity_in_Kit",
                "Storage_Notes",
                "Region_Notes",
                "Mission_Type",
            ]

            available_columns = [
                col for col in expected_columns if col in df.columns
            ]

            for row_number, (_, row) in enumerate(df.iterrows()):
                parts = []

                for column in available_columns:
                    value = self._clean_text(row.get(column, ""))
                    if value:
                        parts.append(f"{column}: {value}")

                if not parts:
                    continue

                drug_name = self._clean_text(row.get("Drug_Name", ""))

                chunks.append(
                    KnowledgeChunk(
                        chunk_id=f"drug-record-{row_number}",
                        source=str(
                            self.dataset_path.relative_to(PROJECT_ROOT)
                        ),
                        category="drugs",
                        text=" | ".join(parts),
                    )
                )

        if not chunks:
            raise FileNotFoundError(
                "No .txt/.md knowledge files or drug dataset records found."
            )

        return chunks

    def build(self) -> "LocalRAGRetriever":
        """Build the in-memory TF-IDF retrieval index."""
        self.chunks = self.load_documents()

        texts = [chunk.text for chunk in self.chunks]

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
        )

        self.matrix = self.vectorizer.fit_transform(texts)

        return self

    def save(self, index_dir: Path | str = DEFAULT_INDEX_DIR) -> None:
        """Persist chunk metadata for inspection/reuse."""
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("Build the retriever before saving it.")

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        metadata = [asdict(chunk) for chunk in self.chunks]

        (index_dir / "chunks.json").write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
    ) -> List[Dict]:
        """Return the most relevant knowledge chunks."""
        if not query or not query.strip():
            return []

        if self.vectorizer is None or self.matrix is None:
            self.build()

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(
            query_vector,
            self.matrix,
        ).ravel()

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
    """Create and build the project's local retriever."""
    return LocalRAGRetriever().build()


def retrieve_context(
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
) -> List[Dict]:
    """Convenience function for agents and other modules."""
    return get_retriever().retrieve(
        query=query,
        top_k=top_k,
        category=category,
    )
    
def retrieve_drug_candidates(
    query: str,
    top_k: int = 5,
) -> List[Dict]:
    """Retrieve unique drug candidates from the structured drug records.

    The best matching record for each drug is retained, so repeated
    region/mission rows for the same drug do not dominate the results.
    """
    retriever = get_retriever()

    results = retriever.retrieve(
        query=query,
        top_k=max(top_k * 5, 20),
        category="drugs",
    )

    candidates: Dict[str, Dict] = {}

    for item in results:
        match = re.search(
            r"Drug_Name:\s*([^|]+)",
            item["text"],
        )

        if not match:
            continue

        drug_name = match.group(1).strip()

        if drug_name not in candidates:
            candidates[drug_name] = {
                "drug_name": drug_name,
                "score": item["score"],
                "source": item["source"],
                "chunk_id": item["chunk_id"],
                "evidence": item["text"],
            }

    return list(candidates.values())[:top_k]