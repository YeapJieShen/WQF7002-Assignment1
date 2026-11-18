"""Run the app with `planner` or `python -m planner`.

For auto-reload while editing, run `gradio src/planner/__main__.py --watch-dirs src/planner`,
which looks for the module-level `demo`.
"""

from __future__ import annotations

from planner.adapters.backends import build_backend
from planner.adapters.config import Settings
from planner.adapters.ui import CSS, THEME, build_app

# Built at import, once: the ZeroGPU backend loads its model and registers `@spaces.GPU` here.
demo = build_app(build_backend(Settings()))  # pragma: no cover - building the layout starts nothing


def main() -> None:  # pragma: no cover - starts a server
    """Launch the web app."""
    demo.launch(theme=THEME, css=CSS, pwa=True)


if __name__ == "__main__":  # pragma: no cover
    main()
