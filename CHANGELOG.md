# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). No version has been released yet.

## [Unreleased]

### Added
- Hugging Face Space on ZeroGPU: Qwen3-4B-Instruct-2507 runs in-process with transformers
  (`PLANNER_BACKEND=zerogpu`), with a visitor's own https OpenAI-compatible endpoint per session
  (#25, #26, #27).
- UI redesign: Chat, Tasks and Pace tabs, streaming tool rows, a model sidebar and a transcript
  download (#26).
- Eval runner and the challenge cases (`python -m planner.adapters.evals`), and a second persona
  (`persona_saturday`).
- Settings for the model id, revision, token limit, GPU duration, tool-call markers, state TTL,
  concurrency, queue size, endpoint timeout, and message, import, upload and provider-error sizes,
  with a clear message for the message and import limits and the endpoint timeout (#32, #33).
- Structured logging: JSON lines (or text) under `planner.*`, a turn id across each turn, timings
  and token counts; `PLANNER_LOG_LEVEL` and `PLANNER_LOG_FORMAT` (#35).
- `pandas` as a direct dependency and `pandas-stubs` for type checks (#29); package metadata
  (license, readme, authors, URLs) and exact dev tool pins (#30).
- More import-linter forbidden modules; mypy covers `scripts` and `space/app.py`; Dependabot
  watches `/space`; CI runs on `workflow_dispatch`, with read-only permissions and superseded runs
  cancelled (#30).
- `docs/decisions.md`, the settings table in the README, and `.agency/` and `.readout/` in
  `.gitignore`.

### Changed
- The default model is Qwen3-4B-Instruct-2507; the Space moved from Docker with Ollama to the
  Gradio SDK on ZeroGPU, on Python 3.12 (#25, #27).
- `PLANNER_MAX_REJECTIONS` now reaches the agent; the model id, token limit and tool-call markers
  are settings instead of literals (#32).
- Task ids are never reused after a delete (`last_task_number` in the session; older exports still
  load) (#31).
- Evals build the backend once per run, use the configured default backend, and record a provider
  outage as an error instead of a failed case (#33).
- Domain models validate more strictly: positive durations and travel times, unique places, routes
  and task ids, and a read-only pace table (#31).

### Fixed
- A fixed event now needs a start time, and an update is checked on the merged task (#31).
- The deadline quota on the last day when a deadline falls at 00:00 (#31).
- Missing drive legs between the library, workshop, grocery, post office and pharmacy (#31).
- Unexpected-argument rejections list the valid fields.
- Feasibility reasons state the real opening hours of the place.

### Security
- A visitor's own endpoint is checked before use: https on port 443 only, no credentials in the
  URL, no loopback, private, link-local, reserved or multicast addresses. The address is resolved
  at request time and pinned, and redirects are not followed (#34).
- Provider error text echoed to the chat is capped and has the key removed (#34).
- Logs never contain API keys; user text and tool arguments are logged only at `DEBUG` (#35).

### Removed
- The Docker Space and its Ollama image (#25, #27).
- The call log panel and sidebar task list; calls now show inline in the chat (#26).
