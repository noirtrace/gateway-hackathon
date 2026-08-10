# FRONTLINE — One-Day AI Build Challenge

FRONTLINE turns raw customer-support messages into safe, validated triage decisions. It runs locally with **Ollama** and **qwen3:4b**—no cloud API, API key, or paid credits required.

## What it does

For every customer message, FRONTLINE produces structured JSON with:

- `category`
- `priority` (`P0`–`P3`)
- `summary`
- `suggested_action`
- `needs_human`
- `confidence` (`high`, `medium`, or `low`)

The Streamlit dashboard accepts pasted messages and CSV/JSON uploads, highlights urgent/human-review rows, shows processing metrics, and exports results as CSV and JSON.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- Ollama model: `qwen3:4b`

## Quick start

1. Install and start Ollama.

2. Download the local model once:

   ```powershell
   ollama pull qwen3:4b
   ```

3. Create and activate a Python virtual environment, then install project dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

4. Start the dashboard:

   ```powershell
   .\.venv\Scripts\streamlit.exe run app.py
   ```

5. Open `http://localhost:8501`, select **Use the built-in sample batch**, then click **Run Triage**.

## Configuration

The default local settings are in `.env` (or copy `.env.example` first):

```env
TRIAGE_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT_SECONDS=180
```

No API key is required. Local Ollama inference records token and latency metrics, while cloud API cost remains `$0.00`.

## Input formats

- Paste one customer message per line.
- Upload CSV rows with a `message`, `customer_message`, `text`, or `content` column.
- Upload a JSON array of records using one of those same fields.

Each run supports up to 250 messages and results can be downloaded as CSV or JSON.

## Included dataset and submission note

- `data/frontline_messages.csv` contains 40 representative customer-support messages.
- `data/frontline_messages.json` contains the same 40 messages in JSON format.
- `AI_DECISIONS.md` is the one-page explanation of the model, safeguards, uncertainty handling, evaluation, and next improvements.

## Reliability and safety

- Ollama is asked to return JSON constrained by the existing Pydantic `TriageOutput` schema.
- The system prompt treats customer text as untrusted data and rejects prompt-injection attempts.
- The result is strictly revalidated by Pydantic before it is accepted.
- If Ollama, its server, or `qwen3:4b` is unavailable, FRONTLINE immediately uses its built-in deterministic fallback rather than crashing.
- Ambiguous, adversarial, angry, multi-issue, security-sensitive, and non-English messages are conservatively routed to human review.

## Evaluation

Run the included hand-labelled evaluation set:

```powershell
.\.venv\Scripts\python.exe evaluator.py
```

It reports completion, category/priority/human-review agreement, token counts, latency, and estimated cost.
