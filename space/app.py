"""Hugging Face Space entry point: the planner on ZeroGPU, run by the Gradio SDK.

`deploy_space.py` uploads `src/` beside this file; nothing is pip-installed from it.
"""

import os
import sys
from pathlib import Path

import spaces  # noqa: F401 - ZeroGPU must patch CUDA before anything imports torch

sys.path.insert(0, str(Path(__file__).parent / "src"))
os.environ.setdefault("PLANNER_BACKEND", "zerogpu")

# Importing builds the app, which loads the model and registers its @spaces.GPU function.
from planner.__main__ import demo, main

__all__ = ["demo"]

if __name__ == "__main__":
    main()
