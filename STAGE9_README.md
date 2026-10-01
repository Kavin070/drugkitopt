# Stage 9 — Existing ML Integration

This stage connects the existing ML prediction layer to the Agentic AI
Coordinator through a dedicated adapter.

## Key rule

The existing ML source and trained model are not rewritten, retrained,
replaced, or deleted.

The adapter imports the existing `src.predict` module and attempts to call a
compatible prediction function. If the existing module exposes no reusable
function, the adapter reports `UNAVAILABLE` rather than inventing a second ML
pipeline.

## New flow

Streamlit
   |
FastAPI
   |
AI Coordinator
   |
   +--> Medical Agent --> RAG
   +--> Regional Agent --> RAG
   +--> Safety Agent --> RAG
   +--> Existing ML Adapter --> existing src.predict
   |
LLM Explanation

## Why use an adapter?

The current Streamlit application may keep some ML inputs/model state inside
the UI. An adapter lets us integrate the real model safely without copying
the model code into the agent layer.

## Next stage

The finalization stage should inspect the actual `src.predict` API and, if
needed, make only the smallest compatible refactor so the exact existing
prediction can be called by both Streamlit and FastAPI.
