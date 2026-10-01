"""Build/validate the local RAG knowledge index."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag.retriever import LocalRAGRetriever, DEFAULT_INDEX_DIR


def main() -> None:
    retriever = LocalRAGRetriever()
    retriever.build()
    retriever.save(DEFAULT_INDEX_DIR)
    print(f"Indexed {len(retriever.chunks)} knowledge chunks.")
    print(f"Metadata saved to: {DEFAULT_INDEX_DIR}")


if __name__ == "__main__":
    main()
