"""claude-cli/ lane gate: prefix parsing and fail-fast admissibility (SC-3).

Covers the lane's public surface in ``strix.llm.claude_cli``: prefix detection,
slug extraction, and the ``ensure_lane_admissible`` gate ordering (API-key
conflict → missing CLI → empty slug). Also asserts the runner invokes the gate
before any scan activity (before the sandbox bundle is built).
"""

from __future__ import annotations

import shutil
from typing import Any

import pytest

from strix.core import runner
from strix.llm import claude_cli


def test_prefix_parsing_extracts_slug() -> None:
    assert claude_cli.claude_cli_lane_model("claude-cli/claude-opus-5") == "claude-opus-5"
    # Case-insensitive on the prefix.
    assert claude_cli.claude_cli_lane_model("Claude-CLI/claude-opus-5") == "claude-opus-5"
    # Surrounding whitespace is tolerated.
    assert claude_cli.claude_cli_lane_model("  claude-cli/x  ") == "x"


def test_non_lane_models_are_ignored() -> None:
    assert claude_cli.claude_cli_lane_model("openai/gpt-5") is None
    assert claude_cli.claude_cli_lane_model("chatgpt/gpt-5") is None
    assert claude_cli.claude_cli_lane_model(None) is None
    assert claude_cli.claude_cli_lane_model("") is None
    # Empty slug is not a usable selection...
    assert claude_cli.claude_cli_lane_model("claude-cli/") is None
    # ...but the prefix is still recognized as "the lane was requested".
    assert claude_cli.is_claude_cli_lane("claude-cli/") is True
    assert claude_cli.is_claude_cli_lane("openai/gpt-5") is False


def test_ensure_admissible_is_noop_for_non_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-matter")
    # No claude-cli/ prefix → the gate never fires, regardless of the env.
    claude_cli.ensure_lane_admissible("openai/gpt-5")
    claude_cli.ensure_lane_admissible(None)


def test_anthropic_api_key_conflict_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-live-xxxx")
    # Even if the CLI is present, the key conflict wins (first failure in order).
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/claude")

    with pytest.raises(claude_cli.ClaudeCliLaneError) as excinfo:
        claude_cli.ensure_lane_admissible("claude-cli/claude-opus-5")

    message = str(excinfo.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "claude-cli/" in message


def test_missing_cli_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    with pytest.raises(claude_cli.ClaudeCliLaneError) as excinfo:
        claude_cli.ensure_lane_admissible("claude-cli/claude-opus-5")

    assert "claude" in str(excinfo.value).lower()


def test_empty_slug_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/claude")

    with pytest.raises(claude_cli.ClaudeCliLaneError) as excinfo:
        claude_cli.ensure_lane_admissible("claude-cli/")

    assert "claude-cli/<model>" in str(excinfo.value)


def test_admissible_lane_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/claude")
    # No exception.
    claude_cli.ensure_lane_admissible("claude-cli/claude-opus-5")


async def test_runner_gates_before_building_the_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """SC-3: run_strix_scan raises the lane error before any sandbox/bundle work."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-live-xxxx")

    # If the gate is bypassed, this would be reached and fail the test loudly.
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("sandbox bundle was built before the lane gate fired")

    monkeypatch.setattr(runner.session_manager, "create_or_reuse", _boom)

    with pytest.raises(claude_cli.ClaudeCliLaneError) as excinfo:
        await runner.run_strix_scan(
            scan_config={},
            image="strix:test",
            model="claude-cli/claude-opus-5",
        )

    message = str(excinfo.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "claude-cli/" in message
