"""Command-line smoke test for the RAG layer."""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag.retriever import get_retriever


def main() -> None:
    query = input("Enter a medical knowledge query: ").strip()
    retriever = get_retriever()
    results = retriever.retrieve(query, top_k=5)

    if not results:
        print("No relevant knowledge found.")
        return

    print("\nRetrieved evidence:\n")
    for i, item in enumerate(results, start=1):
        print(f"[{i}] {item['source']} | score={item['score']}")
        print(item["text"])
        print()


if __name__ == "__main__":
    main()
