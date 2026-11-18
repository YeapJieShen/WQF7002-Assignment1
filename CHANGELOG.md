# Changelog

## Unreleased
- UI: chat-only main view; settings and the task list/call log in collapsible sidebars; Soft theme.
- Unexpected-argument rejections list the valid fields.
- Eval runner and the challenge cases (`python -m planner.adapters.evals`).
- Second persona (`persona_saturday`).
- Hugging Face Docker Space (`space/`, `scripts/deploy_space.py`); default model Qwen3-4B-Instruct-2507.
- Report draft, figures and Part 4 evidence (`docs/report/`).

## v2-improved
- Feasibility reasons state the real opening hours ("Car workshop opening hours on Wed: 09:00-18:00; the visit would need 17:45-19:45"), so the model no longer reports the needed window as the hours.

## v1-baseline
- The build tested in Part 3.

- Repo scaffold.
- Core foundation: domain models, tool argument models, two-stage call validation, packaged base data and persona loader.
- Planning: scenario list, task feasibility check, task state tools, day planner and "what now" recommendation.
