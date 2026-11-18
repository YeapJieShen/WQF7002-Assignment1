"""Upload the app to a Hugging Face Gradio Space running on ZeroGPU.

Usage: python scripts/deploy_space.py <user>/<space>   (log in first with `hf auth login`)

The Space gets space/* at its root plus src/ (code and its data files), nothing else.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[1]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")


def stage(into: Path) -> None:
    """Copy exactly the files the Space needs into `into`."""
    shutil.copytree(ROOT / "space", into, dirs_exist_ok=True, ignore=SKIP)
    shutil.copytree(ROOT / "src", into / "src", ignore=SKIP)


def main() -> None:
    """Stage and upload in one commit."""
    if len(sys.argv) != 2:  # noqa: PLR2004 - one argument
        sys.exit(__doc__)
    token = get_token()
    if token is None:
        sys.exit("Not logged in to Hugging Face; run `hf auth login` first.")
    repo_id = sys.argv[1]
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        HfApi(token=token).upload_folder(
            folder_path=tmp,
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy from GitHub",
            delete_patterns=["src/**"],
        )
    print(f"Uploaded. Build log: https://huggingface.co/spaces/{repo_id}?logs=build")


if __name__ == "__main__":
    main()
