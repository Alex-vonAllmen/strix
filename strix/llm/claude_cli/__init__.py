"""The ``claude-cli/`` execution lane: run Strix on a Claude Pro/Max subscription
login through the ``claude`` CLI / ``claude-agent-sdk`` instead of a metered API key.

This package is the *only* place that knows the ``claude_agent_sdk`` surface
(AD-7): ``dispatch`` imports it lazily so the default (API-key) lane never pays
for the dependency. This module is the lane's public surface — prefix parsing and
the admissibility gate — and imports nothing heavy, so ``run_strix_scan`` and the
CLI preflight can call the gate on every launch.

Unlike the ChatGPT/Codex subscription lane (``strix/config/codex.py``), the lane
is *not* an OpenAI-Agents ``Model``: a Pro/Max login has no raw model endpoint, so
Claude Code becomes the agent and Strix supplies its tools over MCP. The
``claude-cli/`` prefix must therefore never be added to ``StrixProvider``'s prefix
resolution.
"""

from __future__ import annotations

import os
import shutil


CLAUDE_CLI_PREFIX = "claude-cli/"

# Setting a metered Anthropic key would route the CLI through the paid API and
# defeat the whole point of the lane; the conflict is refused rather than
# silently billed.
_CONFLICTING_API_KEY = "ANTHROPIC_API_KEY"


class ClaudeCliLaneError(RuntimeError):
    """A ``claude-cli/`` lane is configured but cannot run.

    The message names the offending variable and/or the lane. Terminal — nothing
    in a run clears it, so it is raised before any scan activity.
    """


def claude_cli_lane_model(model_name: str | None) -> str | None:
    """The model slug behind a ``claude-cli/<model>`` ``STRIX_LLM``, or ``None``.

    Matched case-insensitively on the whole model string. An empty slug
    (``claude-cli/``) returns ``None`` here — it is not a usable lane selection;
    :func:`ensure_lane_admissible` turns that into a clear error.
    """
    name = (model_name or "").strip()
    if not name.lower().startswith(CLAUDE_CLI_PREFIX):
        return None
    return name[len(CLAUDE_CLI_PREFIX) :] or None


def is_claude_cli_lane(model_name: str | None) -> bool:
    """Whether ``model_name`` selects the lane at all (prefix present).

    True even for an empty slug, so callers can route an inadmissible
    ``claude-cli/`` to the gate instead of silently falling through to the
    default lane.
    """
    return (model_name or "").strip().lower().startswith(CLAUDE_CLI_PREFIX)


def ensure_lane_admissible(model_name: str | None) -> None:
    """Raise :class:`ClaudeCliLaneError` if a ``claude-cli/`` lane is configured
    but inadmissible. A no-op for any non-lane model.

    Checks in order (first failure wins, per the FRD error matrix):

    1. ``ANTHROPIC_API_KEY`` present together with a ``claude-cli/`` prefix.
    2. The ``claude`` CLI is not on ``PATH``.
    3. The slug after the prefix is empty.
    """
    if not is_claude_cli_lane(model_name):
        return

    if os.environ.get(_CONFLICTING_API_KEY):
        raise ClaudeCliLaneError(
            f"{_CONFLICTING_API_KEY} is set together with the claude-cli/ lane "
            f"(STRIX_LLM={model_name}). The claude-cli/ lane runs on your Claude "
            f"Pro/Max subscription login and must not use a metered API key. "
            f"Unset {_CONFLICTING_API_KEY} to use the subscription lane, or drop the "
            f"claude-cli/ prefix to use the metered API."
        )

    if shutil.which("claude") is None:
        raise ClaudeCliLaneError(
            "The claude-cli/ lane needs the `claude` CLI on your PATH, but it was "
            "not found. Install Claude Code (https://docs.claude.com/claude-code) "
            "and sign in with `claude` before running the claude-cli/ lane."
        )

    if claude_cli_lane_model(model_name) is None:
        raise ClaudeCliLaneError(
            f"STRIX_LLM={model_name} selects the claude-cli/ lane but names no model. "
            f"Use the shape STRIX_LLM=claude-cli/<model>, e.g. "
            f"STRIX_LLM=claude-cli/claude-opus-5."
        )


__all__ = [
    "CLAUDE_CLI_PREFIX",
    "ClaudeCliLaneError",
    "claude_cli_lane_model",
    "ensure_lane_admissible",
    "is_claude_cli_lane",
]
