# Planning delegate

WQF7002 Artificial Intelligence Techniques, Assignment 1: *an intelligent agent that knows its limits*.

You chat with an agent that plans your week around travel time, opening hours, prerequisites,
deadlines and your own pace. The LLM chooses the tool calls. Python validates each call and
asks you before anything is deleted. Every call is shown inline in the chat, rejected ones included.

## Run it

All settings are environment variables with the `PLANNER_` prefix, or lines in a local `.env`
(see `.env.example`). The tables below list every one.

**Locally with Ollama (default).**

```bash
conda activate wqf7002_assignment1        # Python >= 3.12
pip install -e . --group dev
ollama pull qwen3:4b-instruct             # default model (Qwen3-4B-Instruct-2507)
planner                                   # or: python -m planner -> http://127.0.0.1:7860
```

**Locally on a GPU, without Ollama.** Runs Qwen3 in-process with transformers, as the Space does.

```bash
pip install -e . --group space
PLANNER_BACKEND=zerogpu planner
```

**On the Space.** `space/app.py` sets `PLANNER_BACKEND=zerogpu`; see [Deploy](#deploy-hugging-face-space).

**With your own endpoint.** In the sidebar (gear), choose "Your own endpoint" and give an
OpenAI-compatible base URL, model name and key. The URL must be https on port 443 and publicly
routable; the key stays on the server for your tab and is never exported. To test against a local
server such as Ollama, start the app with `PLANNER_ALLOW_PRIVATE_ENDPOINTS=true`, which also allows
http and private addresses; never set it on a public deployment (see
[docs/decisions.md](docs/decisions.md)).

The app starts empty on the real clock; "Try a demo" loads the sample persona. Tabs: Chat (every
tool call appears inline as a collapsible row), Tasks (with read-only places and travel) and Pace
(all pace settings with a live chart). The header exports and imports the session and downloads a
transcript.

## Settings

Defaults are in `src/planner/adapters/config.py`, which is the source of truth.

| Variable | Default | Meaning |
|---|---|---|
| `PLANNER_BACKEND` | `endpoints` | Default model: `endpoints` (the list below) or `zerogpu` (Qwen3 in-process). |
| `PLANNER_LLM_ENDPOINTS` | one local Ollama endpoint | JSON list of `{name, base_url, model, api_key?, timeout_s?}`, tried in order until one answers. |
| `PLANNER_MAX_REJECTIONS` | `3` | Rejected tool calls in a row before the agent gives up the turn. |
| `PLANNER_ZEROGPU_MODEL_ID` | `Qwen/Qwen3-4B-Instruct-2507` | Hugging Face model the `zerogpu` backend loads. |
| `PLANNER_ZEROGPU_MODEL_REVISION` | unset (latest) | Commit sha or tag of that model to load. |
| `PLANNER_ZEROGPU_MAX_NEW_TOKENS` | `1024` | Most tokens generated for one reply. |
| `PLANNER_ZEROGPU_GPU_DURATION_S` | `60` | Seconds of GPU time requested per generation. |
| `PLANNER_PREFIX_CACHE` | `false` | `zerogpu` only: reuse the cached first turn (rules and tool schemas) across generations. Much faster first token, but outputs can differ slightly (bf16 rounding); untested on a real ZeroGPU Space. |
| `PLANNER_TOOL_CALL_OPEN` / `PLANNER_TOOL_CALL_CLOSE` | `<tool_call>` / `</tool_call>` | Markers around a tool call in the model's reply (Hermes format; the payload between them must still be Hermes JSON). |
| `PLANNER_THINKING_ROTATE_S` | `4` | Seconds of silence before the pending line ("Thinking…") changes its text. |
| `PLANNER_STATE_TTL_S` | `3600` | Seconds an idle tab's conversation is kept on the server. |
| `PLANNER_CONCURRENCY_LIMIT` | `4` | Chat turns run at once; the rest wait in the queue. |
| `PLANNER_QUEUE_MAX_SIZE` | `20` | Events allowed to wait; beyond it visitors are told the app is busy. |
| `PLANNER_CUSTOM_TIMEOUT_S` | `30` | Seconds a visitor's own endpoint has to answer before the turn fails. |
| `PLANNER_ALLOW_PRIVATE_ENDPOINTS` | `false` | Lets a visitor's endpoint use http and private addresses. Local testing only. |
| `PLANNER_PROVIDER_ERROR_MAX_CHARS` | `300` | Longest provider error text shown in the chat. |
| `PLANNER_MESSAGE_MAX_CHARS` | `4000` | Longest chat message accepted. |
| `PLANNER_IMPORT_MAX_BYTES` | `1000000` | Largest session file the import button accepts. |
| `PLANNER_UPLOAD_MAX_BYTES` | `10000000` | Transport backstop for any upload (Gradio `max_file_size`); must exceed the import limit. |
| `PLANNER_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. |
| `PLANNER_LOG_FORMAT` | `json` | `json` (one object per line) or `text`. |

## Logging

The app logs to stderr under the `planner.*` loggers, as JSON lines by default. Every line has
`ts`, `level`, `logger` and `event`; lines inside a turn also carry `turn_id` and `session`. Events
cover tool checks (tool, accepted, issue codes), confirmations, backend calls (backend, host, model,
tokens, duration), limit hits and errors with stack traces. Each graph step and its timing is
logged at `DEBUG` only.

Privacy rules: API keys are never logged (known keys are scrubbed from every line, in any
spelling). User message text and tool arguments appear only at `DEBUG`, so keep `DEBUG` off on a
public deployment. For a custom endpoint only the host and model are logged, not the full URL, and
a rejected endpoint is logged without its host. Below `DEBUG` an error's stack is logged but not its
message, which can quote what the user typed.

## How it works

```
src/planner/
  domain/    models: tasks, places, travel table, pace (no framework imports)
  app/       tool argument models, two-stage validation, feasibility, scheduler, guardrails
  adapters/  LangGraph agent, LLM client chain, session, Gradio UI, eval runner
```

Imports point inward only, and `import-linter` enforces it. The agent graph runs
`call_model → check → (confirm) → act → call_model`:

- **check**: an allowlist, then a Pydantic model per tool (the same model is the JSON schema
  the LLM sees), then checks against the current state (known IDs and places, 7-day horizon,
  dependency cycles). A rejection goes back to the model with a reason it can fix. After 3
  rejections in a row the turn ends with an apology.
- **confirm**: `delete_task` always pauses on a LangGraph `interrupt()` until the user answers.
- **act**: the tool returns a new immutable state plus a message for the model.

Design scenarios: [docs/scenarios.md](docs/scenarios.md). Architecture decisions and their
trade-offs: [docs/decisions.md](docs/decisions.md).

## Checks

```bash
ruff check . && ruff format --check . && mypy && lint-imports && pytest --cov
```

CI runs these on every pull request to `main` and never calls an LLM. Live behaviour:

```bash
python -m planner.adapters.evals evals/cases.toml            # all challenge cases
python -m planner.adapters.evals evals/cases.toml --only 3-feasible
```

The runner uses the configured default backend (so `PLANNER_BACKEND` and `PLANNER_LLM_ENDPOINTS`
apply) and records a provider outage as an error, not a failed case. Reports go to
`evals/results/` (gitignored).

## Deploy (Hugging Face Space)

`space/` holds a Gradio SDK Space on ZeroGPU: Qwen3-4B-Instruct-2507 runs in-process with
transformers (`PLANNER_BACKEND=zerogpu`, set by `space/app.py`), on each visitor's own GPU quota.
Visitors can instead give their own https OpenAI-compatible endpoint for the session. Log in once
with `hf auth login`, then:

```bash
python scripts/deploy_space.py <user>/<space>
```

## Versions

- `v1-baseline`: to be tagged.
- `v2-improved`: to be tagged.
