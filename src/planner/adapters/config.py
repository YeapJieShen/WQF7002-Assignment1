# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
"""Runtime settings, read from environment variables or a local `.env` file."""

from __future__ import annotations

from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Sent as the bearer token when none is given: Ollama ignores it but the client requires one.
DUMMY_API_KEY = "ollama"


class Endpoint(BaseModel):
    """One OpenAI-compatible chat backend.

    Attributes:
        name: A label for logs, such as "local-gpu".
        base_url: The API root, such as http://localhost:11434/v1.
        model: The model name the backend knows.
        api_key: Sent as the bearer token. Ollama ignores it but the client requires one.
        timeout_s: Seconds to wait before this endpoint counts as failed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    base_url: str
    model: str
    api_key: SecretStr = SecretStr(DUMMY_API_KEY)
    timeout_s: float = Field(default=120, gt=0)


LOCAL_GPU = Endpoint(
    name="local-gpu", base_url="http://localhost:11434/v1", model="qwen3:4b-instruct"
)


class Settings(BaseSettings):
    """Everything that changes between machines or deployments.

    Environment variables use the `PLANNER_` prefix. `PLANNER_LLM_ENDPOINTS` is a JSON list,
    tried in order until one answers.

    Attributes:
        backend: The default model: "endpoints" (the list below) or "zerogpu" (Qwen3 in-process
            on a Hugging Face ZeroGPU Space).
        llm_endpoints: The chat backends, best first.
        max_rejections: Rejected tool calls in a row before the agent gives up.
        zerogpu_model_id: The Hugging Face model the "zerogpu" backend loads.
        zerogpu_model_revision: A commit sha or tag of that model to load; None loads the latest.
        zerogpu_max_new_tokens: Most tokens the model may generate for one reply.
        zerogpu_gpu_duration_s: Seconds of GPU time asked for per generation (`@spaces.GPU`).
        prefix_cache: Reuse the attention cache of the fixed first turn (rules and tool schemas)
            across generations on the "zerogpu" backend. Cuts the time to the first token from
            10-40 s to under a second on a 12 GB card, but prefilling in two parts rounds
            differently in bf16, so wording (rarely a tool choice) can differ from the uncached
            run. Off by default: untested on a real ZeroGPU Space, where cached GPU memory may
            not survive between `@spaces.GPU` calls.
        tool_call_open: Text that opens a tool call in the model's reply. The default is the
            Hermes format Qwen3 uses; the payload between the markers must still be Hermes JSON,
            `{"name": ..., "arguments": {...}}`, whatever the markers are.
        tool_call_close: Text that closes a tool call; generation stops at it.
        thinking_rotate_s: Seconds of silence after which the chat's pending line ("Thinking…")
            changes its text.
        state_ttl_s: Seconds the server keeps an idle browser tab's conversation.
        concurrency_limit: Chat turns the app runs at once; the rest wait in the queue.
        queue_max_size: Events allowed to wait in the queue; beyond it visitors are told the
            app is busy. Without a value Gradio allows one on ZeroGPU.
        custom_timeout_s: Seconds a visitor's own endpoint has to answer before the turn fails.
        allow_private_endpoints: Lets a visitor's own endpoint use http and addresses inside the
            server's network, such as http://localhost:11434/v1 for Ollama. For local testing
            only: on a public deployment it would let visitors reach the server's network.
        provider_error_max_chars: Longest provider error text shown in the chat, in characters.
        message_max_chars: Longest chat message accepted, in characters.
        import_max_bytes: Largest session file the import button accepts, in bytes. Checked in
            the import handler before the file is read, so the visitor sees a clear message.
        upload_max_bytes: Hard transport limit on any upload, passed to Gradio's `max_file_size`.
            Gradio rejects a larger file with a generic error, so keep it above
            `import_max_bytes`; it is only a backstop.
        log_level: Lowest severity the app logs: DEBUG, INFO, WARNING or ERROR. INFO records
            events, timings and counts; DEBUG adds user message text and tool arguments, so keep
            it off on a public deployment.
        log_format: "json" (one JSON object per line, for log collectors) or "text" (readable).
    """

    model_config = SettingsConfigDict(
        env_prefix="PLANNER_", env_file=".env", extra="ignore", frozen=True
    )

    backend: Literal["endpoints", "zerogpu"] = "endpoints"
    llm_endpoints: tuple[Endpoint, ...] = Field(default=(LOCAL_GPU,), min_length=1)
    max_rejections: int = Field(default=3, ge=1)
    zerogpu_model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    zerogpu_model_revision: str | None = None
    zerogpu_max_new_tokens: int = Field(default=1024, ge=1)
    zerogpu_gpu_duration_s: int = Field(default=60, ge=1)
    prefix_cache: bool = False
    tool_call_open: str = Field(default="<tool_call>", min_length=1)
    tool_call_close: str = Field(default="</tool_call>", min_length=1)
    thinking_rotate_s: float = Field(default=4.0, gt=0)
    state_ttl_s: int = Field(default=3600, ge=1)
    concurrency_limit: int = Field(default=4, ge=1)
    queue_max_size: int = Field(default=20, ge=1)
    custom_timeout_s: float = Field(default=30, gt=0)
    allow_private_endpoints: bool = False
    provider_error_max_chars: int = Field(default=300, ge=1)
    message_max_chars: int = Field(default=4000, ge=1)
    import_max_bytes: int = Field(default=1_000_000, ge=1)
    upload_max_bytes: int = Field(default=10_000_000, ge=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_case_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _backstop_above_import_limit(self) -> Self:
        if self.upload_max_bytes <= self.import_max_bytes:
            msg = "upload_max_bytes must exceed import_max_bytes, or Gradio hides our message"
            raise ValueError(msg)
        return self

    def summary(self) -> dict[str, Any]:
        """Return the settings as plain values for the start-up log, without any API key."""
        out = self.model_dump(exclude={"llm_endpoints"})
        out["llm_endpoints"] = [
            {"name": e.name, "host": urlsplit(e.base_url).hostname, "model": e.model}
            for e in self.llm_endpoints
        ]
        return out
