# RAG Layer - Stage 1

This stage adds a local retrieval layer without changing the existing ML model or Streamlit UI.

## Run the smoke test

From the project root:

```bash
python rag/query.py
```

Example query:

```text
tropical environment malaria insect exposure
```

## Build the metadata index

```bash
python rag/ingest.py
```

## Run the test

```bash
python -m pytest tests/test_rag.py
```

The current implementation uses TF-IDF + cosine similarity so it does not require an API key or paid LLM. The retrieved passages are the grounding context that later agents/LLMs will consume.
