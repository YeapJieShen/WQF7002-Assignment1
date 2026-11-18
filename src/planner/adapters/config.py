"""Runtime settings, read from environment variables or a local `.env` file."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    api_key: SecretStr = SecretStr("ollama")
    timeout_s: float = Field(default=120, gt=0)


LOCAL_GPU = Endpoint(
    name="local-gpu", base_url="http://localhost:11434/v1", model="qwen3:4b-instruct"
)


class Settings(BaseSettings):
    """Everything that changes between machines or deployments.

    Environment variables use the `PLANNER_` prefix. `PLANNER_LLM_ENDPOINTS` is a JSON list,
    tried in order until one answers.

    Attributes:
        llm_endpoints: The chat backends, best first.
        max_rejections: Rejected tool calls in a row before the agent gives up.
    """

    model_config = SettingsConfigDict(
        env_prefix="PLANNER_", env_file=".env", extra="ignore", frozen=True
    )

    llm_endpoints: tuple[Endpoint, ...] = Field(default=(LOCAL_GPU,), min_length=1)
    max_rejections: int = Field(default=3, ge=1)
