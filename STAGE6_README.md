# Stage 6 — LLM Explanation Layer

This stage adds a grounded LLM explanation layer on top of the Stage 5
Coordinator.

## Architecture

Existing ML
  +
Medical Agent
  +
Regional Agent
  +
Clinical Safety Agent
  +
RAG evidence
       |
       v
AI Coordinator
       |
       v
LLM Explanation Service
       |
       v
Human-readable explanation

## Important design rule

The LLM is an explanation layer. It does not independently choose or
prescribe medication and must not override the safety agent.

## Providers

Supported providers:

- Gemini
- OpenAI
- Anthropic Claude
- deterministic fallback

The default is `fallback`, so the project can run without an API key.

Set in `.env` or the environment:

```text
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key
```

or:

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
```

or:

```text
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key
```

The project should load secrets through environment variables and never
commit API keys.

Medical disclaimer: this is an academic prototype and must not be used as a
real-world clinical decision system.
