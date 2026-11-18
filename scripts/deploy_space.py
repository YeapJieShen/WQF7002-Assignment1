"""Upload the app to a Hugging Face Docker Space.

Usage: python scripts/deploy_space.py <user>/<space>   (reads the token from HF_TOKEN)

The Space gets space/* at its root plus pyproject.toml and src/, nothing else.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")


def stage(into: Path) -> None:
    """Copy exactly the files the Space needs into `into`."""
    shutil.copytree(ROOT / "space", into, dirs_exist_ok=True)
    shutil.copy2(ROOT / "pyproject.toml", into)
    shutil.copytree(ROOT / "src", into / "src", ignore=SKIP)


def main() -> None:
    """Stage and upload in one commit."""
    if len(sys.argv) != 2:  # noqa: PLR2004 - one argument
        sys.exit(__doc__)
    repo_id = sys.argv[1]
    api = HfApi(token=os.environ["HF_TOKEN"])
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(
            folder_path=tmp,
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy from GitHub",
            delete_patterns=["src/**"],
        )
    print(f"Uploaded. Build log: https://huggingface.co/spaces/{repo_id}?logs=build")


if __name__ == "__main__":
    main()
