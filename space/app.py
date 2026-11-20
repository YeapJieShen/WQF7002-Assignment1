# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
"""Hugging Face Space entry point: the planner on ZeroGPU, run by the Gradio SDK.

The Space is a mirror of the repository (see the README's Deploy section) and runs this file from
`space/`, so `src/` is put on the path here; the package itself is not pip-installed.
"""

import os
import sys
from pathlib import Path

import spaces  # noqa: F401 - ZeroGPU must patch CUDA before anything imports torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("PLANNER_BACKEND", "zerogpu")

# Importing builds the app, which loads the model and registers its @spaces.GPU function.
from planner.__main__ import demo, main

__all__ = ["demo"]

if __name__ == "__main__":
    main()
