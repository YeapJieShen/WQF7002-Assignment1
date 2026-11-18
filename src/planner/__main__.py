"""Run the app with `planner` or `python -m planner`.

For auto-reload while editing, run `gradio src/planner/__main__.py --watch-dirs src/planner`,
which looks for the module-level `demo`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from planner.adapters.config import Endpoint, Settings
from planner.adapters.llm import BACKEND_ERRORS, build_model
from planner.adapters.ui import CSS, THEME, build_app

if TYPE_CHECKING:
    from planner.adapters.graph import Model
    from planner.adapters.session import Usage

BACKEND_DOWN = "No language-model backend answered. Check that it is running."


class TemporaryBackend:
    """TEMPORARY: the configured endpoints behind the `Backend` seam, until `build_backend` lands.

    Replace with `planner.adapters.backends.build_backend(Settings())` when that module exists.
    """

    def model(self, custom: Endpoint | None, /) -> Model:
        """Build the configured chain, or a client for `custom`."""
        settings = Settings(llm_endpoints=(custom,)) if custom else Settings()
        return build_model(settings)

    def explain(self, error: Exception, /, *, signed_in: bool, custom: bool) -> str | None:  # noqa: ARG002
        """Say the backend is down for connection and rate-limit errors, else None."""
        return BACKEND_DOWN if isinstance(error, BACKEND_ERRORS) else None

    def cost(self, usage: Usage, custom: Endpoint | None, /) -> str | None:  # noqa: ARG002
        """Show no cost."""
        return None


demo = build_app(TemporaryBackend())  # pragma: no cover - building the layout starts nothing


def main() -> None:  # pragma: no cover - starts a server
    """Launch the web app."""
    demo.launch(theme=THEME, css=CSS, pwa=True)


if __name__ == "__main__":  # pragma: no cover
    main()
