"""claude-cli/ lane MCP tool adapter (SC-4).

Verifies the single adapter exposes every Strix ``FunctionTool`` as
``mcp__strix__*``, reusing each tool's ``.params_json_schema`` verbatim and
routing invocations to its ``.on_invoke_tool``. Non-``FunctionTool`` entries are
skipped. Real notes/proxy/reporting tools are asserted present.
"""

from __future__ import annotations

from typing import Any

import claude_agent_sdk
import pytest
from agents.tool import FunctionTool

from strix.llm.claude_cli import server
from strix.tools.notes.tools import create_note
from strix.tools.proxy.tools import list_requests
from strix.tools.reporting.tool import create_vulnerability_report


def _make_function_tool(name: str, sink: dict[str, Any]) -> FunctionTool:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    async def _invoke(_ctx: Any, args_json: str) -> str:
        sink["called_with"] = args_json
        return f"{name}-ok"

    return FunctionTool(
        name=name,
        description=f"desc for {name}",
        params_json_schema=schema,
        on_invoke_tool=_invoke,
    )


class _NotAFunctionTool:
    name = "custom_thing"


def _capture_created_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> list[Any]:
    captured: list[Any] = []

    def _fake_create(*, name: str, tools: list[Any] | None = None) -> object:  # noqa: ARG001
        captured.extend(tools or [])
        return object()

    monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server", _fake_create)
    return captured


def test_exposed_tool_names_are_mcp_strix_prefixed() -> None:
    sink: dict[str, Any] = {}
    tools = [_make_function_tool("alpha", sink), _make_function_tool("beta", sink)]
    assert server.exposed_tool_names(tools) == ["mcp__strix__alpha", "mcp__strix__beta"]


def test_non_function_tools_are_excluded_from_names() -> None:
    sink: dict[str, Any] = {}
    tools = [_make_function_tool("alpha", sink), _NotAFunctionTool()]
    assert server.exposed_tool_names(tools) == ["mcp__strix__alpha"]  # type: ignore[list-item]


def test_build_server_exposes_names_and_verbatim_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_created_tools(monkeypatch)
    sink: dict[str, Any] = {}
    tool_a = _make_function_tool("alpha", sink)
    tool_b = _make_function_tool("beta", sink)

    server.build_tool_server([tool_a, tool_b], context={})

    by_name = {t.name: t for t in captured}
    assert set(by_name) == {"alpha", "beta"}
    # Schema is reused verbatim (same object, not a re-derived copy).
    assert by_name["alpha"].input_schema is tool_a.params_json_schema
    assert by_name["beta"].input_schema is tool_b.params_json_schema
    assert by_name["alpha"].description == "desc for alpha"


def test_build_server_skips_non_function_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_created_tools(monkeypatch)
    sink: dict[str, Any] = {}
    server.build_tool_server(
        [_make_function_tool("alpha", sink), _NotAFunctionTool()],  # type: ignore[list-item]
        context={},
    )
    assert [t.name for t in captured] == ["alpha"]


async def test_handler_routes_to_on_invoke_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_created_tools(monkeypatch)
    sink: dict[str, Any] = {}
    server.build_tool_server([_make_function_tool("alpha", sink)], context={"k": "v"})

    handler = captured[0].handler
    result = await handler({"value": "hi"})

    # The tool's on_invoke_tool ran with the JSON-serialized args...
    assert sink["called_with"] == '{"value": "hi"}'
    # ...and its string return is the tool's text content, not an error.
    assert result["is_error"] is False
    assert result["content"][0]["text"] == "alpha-ok"


async def test_handler_returns_error_result_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_created_tools(monkeypatch)

    async def _boom(_ctx: Any, _args: str) -> str:
        raise RuntimeError("kaboom")

    tool = FunctionTool(
        name="explode",
        description="d",
        params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
        on_invoke_tool=_boom,
    )
    server.build_tool_server([tool], context={})

    result = await captured[0].handler({})
    assert result["is_error"] is True
    assert "explode" in result["content"][0]["text"]


def test_real_registry_tools_are_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    """notes / proxy / reporting FunctionTools are exposed by the adapter (SC-4)."""
    captured = _capture_created_tools(monkeypatch)
    tools = [create_note, list_requests, create_vulnerability_report]
    server.build_tool_server(tools, context={})

    names = {t.name for t in captured}
    for tool in tools:
        assert tool.name in names
        # Names surface to Claude Code as mcp__strix__<name>.
    exposed = server.exposed_tool_names(tools)
    assert f"mcp__strix__{create_note.name}" in exposed
