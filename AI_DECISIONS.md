# FRONTLINE — AI Decisions Note

## Purpose

FRONTLINE turns unstructured customer-support messages into a consistent triage decision: category, priority, concise summary, next action, human-review flag, and confidence. The goal is not to replace support staff; it is to make the first pass fast, safe, and auditable.

## Model and tools

The application uses the local Ollama model `qwen3:4b` through `http://localhost:11434/api/chat`. Ollama is configured to return JSON using the same schema as the Pydantic `TriageOutput` model. Pydantic then strictly validates every response before it can be displayed. Local inference has no cloud API cost. If Ollama or the model is unavailable, the app switches immediately to a deterministic, conservative fallback so a batch never fails.

## Prompt and output strategy

The system prompt defines the priority scale and explicitly treats every customer message as untrusted data. It tells the model to ignore instructions inside customer text that attempt to change its role, reveal prompts, skip rules, or control the JSON result. It also prohibits inventing facts such as account history, payment amounts, or causes not stated in the message. The structured output contract prevents prose-only answers and keeps downstream handling predictable.

## Uncertainty and bad input

The model is instructed to set `confidence` to `low` and `needs_human` to `true` when essential details are missing, a message is ambiguous, adversarial, angry, or contains multiple issues. The fallback applies the same review-first principle for prompt injection, security/account-takeover signals, unrecognized charges, complaints, short vague messages, and non-English text. Invalid model JSON, network failures, missing Ollama models, and malformed uploads are contained per row or surfaced clearly without crashing a complete batch.

## Evidence it works

The repository includes `data/frontline_messages.csv` and an equivalent JSON file with 40 realistic messages covering billing, technical problems, complaints, security, ambiguity, adversarial input, out-of-scope questions, and non-English text. `evaluator.py` contains 10 hand-labelled cases and reports category, priority, and human-review agreement alongside token counts, latency, and cost. The Streamlit dashboard supports upload/paste flows, highlights P0 and human-review rows, and exports results as both CSV and JSON.

## What I would improve with more time

I would add labelled regression cases from real support outcomes, measure agreement by category rather than only aggregate fields, allow controlled parallel local inference, and add a human feedback loop to capture corrected triage decisions for future evaluation.
