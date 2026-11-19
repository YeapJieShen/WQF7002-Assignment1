# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
"""The seam between the UI and the model backends.

The UI asks a `Backend` for a model and for the words to show about failures and cost; it never
builds clients itself. The default model is the one the deployment provides (ZeroGPU on the
Space, the configured endpoints elsewhere); a custom endpoint is the visitor's own, held only for
their session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from planner.adapters.config import Endpoint
    from planner.adapters.graph import Model
    from planner.adapters.session import Usage


class Backend(Protocol):
    """Builds models and explains what happened to them."""

    def model(self, custom: Endpoint | None, /) -> Model:
        """Return the default model when `custom` is None, otherwise a client for `custom`.

        Raises:
            ValueError: If `custom` is not acceptable, such as a URL that is not https.
        """
        ...

    def explain(self, error: Exception, /, *, signed_in: bool, custom: bool) -> str | None:
        """Return the chat message for a turn that failed in the backend, or None if it did not."""
        ...

    def cost(self, usage: Usage, custom: Endpoint | None, /) -> str | None:
        """Return a one-line token or cost note for a turn, or None when none should be shown."""
        ...
