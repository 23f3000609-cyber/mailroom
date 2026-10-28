# Safe AI Mailroom — `ga5-mailroom-action-gate/v2`

A FastAPI service implementing the `propose` and `commit` operations, durable SQLite state, canonical JSON hashing, content-addressed proposal caching, and Ed25519 receipt verification.

## Important design notes

- The single `POST /` endpoint dispatches by `operation`.
- `GET /health` is only a health check; submit the root URL to the grader.
- All external mail/source text is untrusted input. The model proposes; deterministic code validates exact action schemas, targets, payload keys, and evidence IDs.
- `send_approved_notice` is never permitted by fallback behavior. Model output must cite evidence and pass schema validation.
- A valid receipt is verified against the public JWK saved for that evaluation and the exact evaluation ID, input digest, receipt fields, call ID, action, and persisted proposal digest.
- No external email or record-changing system is integrated. The `executed`/`rejected` outcomes report the grader's accepted decision lifecycle as requested by this benchmark; they do not actually send mail or mutate a customer system.
- Stable proposal cache keys are canonical dossier-content SHA-256 fingerprints. `callId` is deterministic from that fingerprint so stable content keeps the same call ID.
- The service rejects changed content for a previously used evaluation ID with HTTP 409.
- Invalid commits are verified in full before any receipt or terminal result is written.

## Prerequisites

- Python 3.12+
- An OpenAI API key for hosted inference
- A public host with persistent storage

## Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env  # Windows; on macOS/Linux use cp
```

Set `OPENAI_API_KEY` in your shell/environment. Do not commit `.env` or put keys in the URL.

PowerShell example:
```powershell
$env:OPENAI_API_KEY = "your-key"
$env:DATABASE_PATH = "data/mailroom.sqlite3"
uvicorn main:app --host 0.0.0.0 --port 8000
```

macOS/Linux:
```bash
export OPENAI_API_KEY="your-key"
export DATABASE_PATH="data/mailroom.sqlite3"
uvicorn main:app --host 0.0.0.0 --port 8000
```

Health check: `http://127.0.0.1:8000/health`. The grader URL must be public HTTPS, not localhost.

## Tests

```bash
pytest -q
```

The tests use a fake model and generated Ed25519 keys. They do not spend API credits. They cover canonical hashing, proposal replay, changed-content conflicts, successful receipt verification, invalid signature rejection, duplicate dossier IDs, and unknown commits.

## Deploy to Render

1. Push this folder to a private or public Git repository.
2. In Render, create a **Web Service** from the repository. The included `render.yaml` describes a Docker service with a persistent disk at `/var/data`.
3. Use a plan that supports persistent disks. A free ephemeral service is not sufficient for SQLite state that must survive restarts.
4. Add `OPENAI_API_KEY` as a secret environment variable. Set `OPENAI_MODEL` to a model available to your API account, default `gpt-4.1-mini`.
5. Deploy and wait for `GET /health` to return `{"status":"ok",...}`.
6. Submit the base HTTPS service URL, ending in `/` if required by the grader. Do not include a key, query string, or fragment.

If your chosen host does not support persistent local disks, use its supported managed PostgreSQL service and adapt the storage layer before grading. Do not assume SQLite on an ephemeral filesystem is durable.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none | Hosted AI API credential |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Model name |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base |
| `DATABASE_PATH` | `data/mailroom.sqlite3` | Persistent SQLite path |
| `MODEL_TIMEOUT_SECONDS` | `38` | Model request timeout, less than 55-second request budget |
| `MODEL_MAX_OUTPUT_TOKENS` | `14000` | Bounded output |
| `MAX_BODY_BYTES` | 2 MiB | Request body cap |
| `MAX_MODEL_BATCH_CHARS` | 260000 | Model input safety bound |

## Protocol details

### Propose
- Strictly parses JSON and rejects duplicate JSON keys.
- Rejects duplicate dossier IDs and duplicate line IDs within a dossier.
- `inputDigest` is SHA-256 over canonical JSON of the normalized `dossiers` array.
- Returns exactly one proposal per dossier, with a stable `callId`.
- Proposal digest uses only `dossierId`, `callId`, `action`, `target`, `payload`, and sorted `evidence`.

### Commit
- Requires a known evaluation and exact `inputDigest`.
- Requires exactly one receipt per proposal.
- Checks the persisted call ID, action, and proposal digest.
- Verifies every signature before writing any receipt/outcome.
- Uses the evaluation's stored `receiptVerifier.publicKeyJwk`.
- `accepted: true` becomes `executed` only after successful signature/scope validation; false becomes `rejected`.
- Exact commit replay returns the stored terminal response; changed replay returns HTTP 409.

## Before submission: required manual checks

1. Deploy and call `/health`.
2. Run `pytest -q`.
3. Run one complete grader Check and inspect the category hints.
4. Fix decision/evidence errors; do not use static responses or keyword-only rules.
5. Make sure the host's disk persists across restarts and that the grader reaches the same service instance/state.
6. Never submit until the endpoint URL is valid and the hosted API key is configured.

## Limitations to address if the grader exposes them

- AI decisions are probabilistic; hidden-case accuracy cannot be guaranteed without the actual evaluation feedback.
- The included conservative fallback exists to keep service availability and avoid unauthorized actions, but it may not earn decision-accuracy marks if the hosted model fails.
- This is a single-process SQLite deployment. For horizontally scaled deployment, use a shared transactional database and distributed request coordination.
