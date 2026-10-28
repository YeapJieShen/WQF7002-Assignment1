"""Run the app with `planner` or `python -m planner`.

For auto-reload while editing, run `gradio src/planner/__main__.py --watch-dirs src/planner`,
which looks for the module-level `demo`.
"""

from planner.adapters.ui import build_app

demo = build_app()  # pragma: no cover - building the layout starts nothing


def main() -> None:  # pragma: no cover - starts a server
    """Launch the web app."""
    demo.launch()


if __name__ == "__main__":  # pragma: no cover
    main()
