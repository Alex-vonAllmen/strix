"""claude-cli/ lane: tool invocations run under a ToolContext (issue #7).

Regression for the bug where the adapter handed tools a bare ``RunContextWrapper``,
so the SDK's tool machinery (including its error path) crashed on the missing
``run_config`` / ``tool_name`` and returned an ``AttributeError`` message to the model
instead of the real tool error — flailing the agent into ``MaxTurnsExceeded``.
"""

from __future__ import annotations

from typing import Any

import claude_agent_sdk
import pytest
from agents import RunConfig, function_tool
from agents.tool import FunctionTool

from strix.llm.claude_cli import server


_EMPTY_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def _capture_created_tools(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    captured: list[Any] = []

    def _fake_create(*, name: str, tools: list[Any] | None = None) -> object:
        captured.extend(tools or [])
        return object()

    monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server", _fake_create)
    return captured


async def test_tool_context_carries_run_config_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SC-2: the tool sees the run's run_config plus tool name/arguments."""
    seen: dict[str, Any] = {}

    async def _invoke(ctx: Any, args_json: str) -> str:
        seen["run_config"] = ctx.run_config
        seen["tool_name"] = ctx.tool_name
        seen["tool_arguments"] = ctx.tool_arguments
        return "ok"

    tool = FunctionTool(
        name="probe", description="d", params_json_schema=_EMPTY_SCHEMA, on_invoke_tool=_invoke
    )
    cfg = RunConfig()
    captured = _capture_created_tools(monkeypatch)
    server.build_tool_server([tool], context={}, run_config=cfg)

    result = await captured[0].handler({})
    assert result["is_error"] is False
    assert seen["run_config"] is cfg
    assert seen["tool_name"] == "probe"
    assert seen["tool_arguments"] == "{}"


async def test_run_config_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """SC-2: run_config is a safe None when the caller passes none."""
    seen: dict[str, Any] = {}

    async def _invoke(ctx: Any, args_json: str) -> str:
        seen["run_config"] = ctx.run_config
        return "ok"

    tool = FunctionTool(
        name="probe", description="d", params_json_schema=_EMPTY_SCHEMA, on_invoke_tool=_invoke
    )
    captured = _capture_created_tools(monkeypatch)
    server.build_tool_server([tool], context={})  # no run_config

    await captured[0].handler({})
    assert seen["run_config"] is None


async def test_erroring_tool_returns_real_error_not_attribute_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SC-1/SC-5: a tool that raises returns the SDK's real error result, and the
    context no longer crashes on a missing run_config."""

    @function_tool
    def boom(x: str) -> str:
        raise RuntimeError("kaboom")

    captured = _capture_created_tools(monkeypatch)
    server.build_tool_server([boom], context={}, run_config=None)

    result = await captured[0].handler({"x": "a"})
    text = result["content"][0]["text"]
    # The model sees the real tool error...
    assert "kaboom" in text
    # ...not the run_config AttributeError the bare wrapper produced.
    assert "run_config" not in text
    assert "has no attribute" not in text


def test_successful_tool_still_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    """SC-3: the success path is unchanged — tools are still exposed by name."""
    captured = _capture_created_tools(monkeypatch)

    async def _invoke(ctx: Any, args_json: str) -> str:
        return "fine"

    tool = FunctionTool(
        name="alpha", description="d", params_json_schema=_EMPTY_SCHEMA, on_invoke_tool=_invoke
    )
    server.build_tool_server([tool], context={}, run_config=RunConfig())
    assert [t.name for t in captured] == ["alpha"]
