# Decisions

Short records of choices that are not obvious from the code: context, decision, consequence.

## Layered design enforced by import-linter
- Context: the agent logic should be testable without an LLM or a UI.
- Decision: `domain` <- `app` <- `adapters`; imports point inward only. `domain` and `app` never
  import langgraph, langchain, gradio, openai, transformers, torch, spaces or pandas.
- Consequence: `lint-imports` fails CI on a violation. Framework code lives in `adapters`.

## Immutable state
- Context: a rejected or declined call must not leave half-applied state.
- Decision: models are frozen; every tool returns a new `SessionState`.
- Consequence: a rejection leaves the old state untouched, at the cost of copying on each edit.

## Pydantic argument models are the one schema and shape check
- Context: the JSON schema the LLM sees and the Python validation could drift apart.
- Decision: one model per tool in `app/tools/args.py` gives the schema and checks the call's
  shape; meaning checks (known ids, horizon, cycles) live in `app/validation.py`.
- Consequence: a new tool is one model plus one handler.

## One tool call per turn step
- Context: parallel calls need ordering and partial-failure rules.
- Decision: tools are bound with `parallel_tool_calls=False` and the graph reads `tool_calls[0]`.
- Consequence: multi-step requests take several model calls, each validated alone.

## Only delete asks for confirmation; accepted risk
- Context: a pasted message can hold instructions aimed at the model (prompt injection).
- Decision: `delete_task` pauses on a LangGraph `interrupt()` until the user answers. Other tools
  run without asking.
- Consequence (accepted): an injection can still add or edit the visitor's own tasks. The damage
  stays inside that visitor's session and shows in the task list and the inline tool rows.

## Paste fence
- Context: Ctrl+V pastes are the main route for third-party text.
- Decision: `forwarded()` in `app/prompt.py` removes any `<<<` from the text, wraps it between
  `<<<` fences and tells the model it is untrusted data, to summarise it and to ask before adding
  a task.
- Consequence: a mitigation, not a boundary; see the accepted risk above.

## ZeroGPU in-process transformers, not the Inference API; Python 3.12 on the Space
- Context: the Hugging Face Inference API needs paid credits ($0 available); ZeroGPU gives each
  signed-in visitor a GPU quota.
- Decision: the Space runs Qwen3-4B-Instruct-2507 in-process (`PLANNER_BACKEND=zerogpu`);
  visitors can bring their own endpoint instead. The Space sets `python_version: "3.12"`, so the
  package supports `>=3.12`.
- Consequence: the quota is per visitor and a cold start loads the model. CI never calls an LLM.

## `TemplateChat` implements `_generate` only
- Context: the in-process model needs a LangChain chat model that returns tool calls.
- Decision: `TemplateChat` renders the chat template, generates, and parses tool calls from the
  text. It pushes visible text to the callback manager from `_generate`; there is no `_stream`.
- Consequence: one code path to test; text is streamed, tool-call text is held back until parsed.

## Tool-call markers are Settings
- Context: Qwen3 writes tool calls in the Hermes format between `<tool_call>` markers; other
  models use other markers.
- Decision: `PLANNER_TOOL_CALL_OPEN` and `PLANNER_TOOL_CALL_CLOSE`; the payload between them must
  still be Hermes JSON.
- Consequence: other markers work by configuration; another payload format does not.

## Only `/chat` is a public API endpoint
- Context: Gradio exposes every event as an API route.
- Decision: every event except chat uses `api_visibility="private"`, and the visitor's key is not
  an input of the public endpoint.
- Consequence: export, import and endpoint setup are reachable only from the page.

## Custom endpoint guard: SSRF checks and IP pinning
- Context: a visitor's endpoint URL makes the server send requests, possibly into its own network.
- Decision: `adapters/endpoint_guard.py` accepts only https on 443 without credentials and rejects
  loopback, private, link-local, reserved and multicast addresses. It resolves at request time,
  pins the connection to the vetted address (TLS still checks the original host name) and follows
  no redirects. Operator endpoints from Settings are trusted and not checked.
- Consequence: a private or http endpoint needs `PLANNER_ALLOW_PRIVATE_ENDPOINTS=true`, which must
  stay off on a public deployment.

## Limits are Settings with clear messages
- Context: hard-coded limits cannot be tuned per deployment, and Gradio's own errors are vague.
- Decision: message length, import and upload size, queue size, concurrency, endpoint timeout and
  provider error length are `Settings`. Our handlers state the limit and the amount.
- Consequence: the upload limit is only a backstop above the import limit. Gradio's own "busy"
  and launch file-size texts are not customisable.

## Logging privacy
- Context: logs are kept by the host and others may read them.
- Decision: structured JSON lines; keys are never logged and are scrubbed from every line; user
  text and tool arguments only at `DEBUG`; a custom endpoint is logged as host and model only (a
  rejected one without its host, since a key pasted into the URL box would be the host); an
  exception's message, which can quote user input, only at `DEBUG` (below it: stack and class).
- Consequence: the default `INFO` is safe for the Space; `DEBUG` is for local diagnosis.

## Monotonic task ids
- Context: reusing a deleted id such as `T7` would make an old reference point at a new task.
- Decision: `SessionState.last_task_number` holds the highest number handed out; the next id is
  one above it.
- Consequence: ids can have gaps. Older exports load with the mark taken from their tasks.

## A duration or deadline must quote the user
- Context: the rules say never guess, yet the model invented `duration_min=60` and a deadline
  for "Add a haircut at the salon in Bangsar this week" in 10 of 10 runs; nothing in code
  stopped it.
- Decision: `add_task` and `update_task` carry `duration_quote` and `deadline_quote`, the user's
  words that state the value. Validation rejects a duration or deadline whose quote is missing,
  blank, or not found (ignoring case and spacing) in the user's messages since the assistant's
  last plain reply, and tells the model to ask instead of guessing. A pasted message (wrapped by
  `forwarded`) is someone else's words and never counts. The graph hands the turn's texts to
  `validate_call` as plain strings, so `app` stays framework-free. Quotes are never stored; an
  update that changes neither value needs none.
  The quote must also say something: a duration quote needs an amount of time (digits or a word
  from one to twelve, "half", "a"/"an", then a unit such as min, hour, h), so "salon" or "a" fails;
  a deadline quote needs 3 or more characters with a letter or digit, so the user's own relative
  phrase ("this week", "tonight") passes without a date pattern.
  The duration must also be the amount its quote states (within a tenth); the rejection gives
  the number ("'2 hours' is 120 minutes, not 60: use 120"). Only a plain amount is checked: a
  quote with a second number ("an hour and a half", "1-2 hours", "2 hours 30 minutes") is
  accepted as is. A quoted amount over the 720-minute task maximum allows any part up to it.
  A duration over that maximum is rejected with the quote's minutes and how to go on (several
  tasks, then `plan_day`; the daily limit is the user's setting, not a reason to refuse). It
  replaces pydantic's bare "less than or equal to 720": for "Schedule 14 hours of assignment
  work on Friday" the model sent 1440 quoting "14 hours", read 720 as the daily limit, and refused.
- Consequence: a value given in an earlier turn must be restated. The parts of work too long for
  one task are not summed against the quote. A deadline quote proves only that the words were
  said, not the date. Personas and loaded files are not checked.

## Datetimes as the model writes them
- Context: the model adds a "Z" or sends a date-only deadline; both were rejected and used up
  the retry cap.
- Decision: a trailing "Z" is dropped and the time read on the session's naive clock (the model
  adds it by habit, not to mean UTC). A date-only deadline means 23:59 that day, as in the
  personas. A date-only start or check time is still rejected: its time cannot be inferred.
- Consequence: other offsets such as `+08:00` are still rejected.
