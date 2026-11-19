"""Run the app with `planner` or `python -m planner`.

For auto-reload while editing, run `gradio src/planner/__main__.py --watch-dirs src/planner`,
which looks for the module-level `demo`.
"""

from __future__ import annotations

import logging

from planner.adapters.backends import build_backend
from planner.adapters.config import Settings
from planner.adapters.logs import configure_logging, emit
from planner.adapters.ui import CSS, THEME, build_app

# Built at import, once: the ZeroGPU backend loads its model and registers `@spaces.GPU` here.
settings = Settings()
# The one place logging is set up. It runs here, not in build_app, so loading the model is logged.
configure_logging(settings.log_level, settings.log_format)
emit(logging.getLogger("planner.app"), "app_start", **settings.summary())
demo = build_app(
    build_backend(settings), settings
)  # pragma: no cover - building the layout starts nothing


def main() -> None:  # pragma: no cover - starts a server
    """Launch the web app."""
    demo.launch(theme=THEME, css=CSS, pwa=True, max_file_size=settings.upload_max_bytes)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
