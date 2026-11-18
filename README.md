# Planning delegate

WQF7002 Artificial Intelligence Techniques, Assignment 1: *an intelligent agent that knows its limits*.

You chat with an agent that plans your week around travel time, opening hours, prerequisites,
deadlines and your own pace. The LLM chooses the tool calls. Python validates each call and
asks you before anything is deleted. Every call is shown in a log, rejected ones included.

## Run it

```bash
conda activate wqf7002_assignment1        # Python >= 3.12
pip install -e . --group dev
ollama pull qwen3:4b-instruct             # default model (Qwen3-4B-Instruct-2507)
planner                                   # or: python -m planner -> http://127.0.0.1:7860
```

The app starts empty on the real clock; "Try a demo" loads the sample persona. Tabs: Chat (every
tool call appears inline as a collapsible row), Tasks (with read-only places and travel) and Pace
(all pace settings with a live chart). The header exports and imports the session and downloads a
transcript; the gear opens the model sidebar, where you can use your own OpenAI-compatible
endpoint (your key stays on the server for your tab and is never exported).

Backends are an ordered fallback list, configured in `.env` (see `.env.example`):

```bash
PLANNER_LLM_ENDPOINTS=[{"name":"local-gpu","base_url":"http://localhost:11434/v1","model":"qwen3:4b-instruct"}]
```

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

Design scenarios and decisions: [docs/scenarios.md](docs/scenarios.md).

## Checks

```bash
ruff check . && ruff format --check . && mypy && lint-imports && pytest --cov
```

CI runs these on every pull request to `main` and never calls an LLM. Live behaviour:

```bash
python -m planner.adapters.evals evals/cases.toml            # all challenge cases
python -m planner.adapters.evals evals/cases.toml --only 3-feasible
```

Reports go to `evals/results/` (gitignored).

## Deploy (Hugging Face Space)

`space/` holds a Gradio SDK Space on ZeroGPU: Qwen3-4B-Instruct-2507 runs in-process with
transformers (`PLANNER_BACKEND=zerogpu`, set by `space/app.py`), on each visitor's own GPU quota.
Visitors can instead give their own https OpenAI-compatible endpoint for the session. Log in once
with `hf auth login`, then:

```bash
python scripts/deploy_space.py <user>/<space>
```

## Versions

- `v1-baseline`: the build tested in Part 3.
- `v2-improved`: the one Part 4 change (feasibility reasons state the real opening hours).
