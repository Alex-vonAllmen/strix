"""MCP tool adapter: expose Strix's ``FunctionTool``s to Claude Code (AD-2).

A single in-process MCP server named ``strix`` is built from an agent's final,
already-wrapped tool list. Claude Code surfaces each tool as ``mcp__strix__<name>``.
There is no per-tool glue: every ``FunctionTool`` is exposed by reusing its
``.params_json_schema`` (verbatim) and ``.on_invoke_tool`` (the same callable the
default lane invokes), so tool behavior — bounding, coercion, error-as-result — is
identical across lanes.

Non-``FunctionTool`` entries (e.g. ``CustomTool``) are skipped in M1; the standard
Strix registry is entirely ``FunctionTool``s, so this is a safety net that logs
once. The sandbox shell/fs tools are not here — they are SDK capabilities bound
per session and are bridged separately in M2.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from agents.tool import FunctionTool, Tool
from agents.tool_context import ToolContext


if TYPE_CHECKING:
    from claude_agent_sdk import McpSdkServerConfig


logger = logging.getLogger(__name__)

# The MCP server name; Claude Code prefixes exposed tools with ``mcp__<name>__``.
SERVER_NAME = "strix"


def _text_result(text: str, *, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": is_error}


def _make_handler(tool: Any, context: dict[str, Any], run_config: Any) -> Any:
    """Build an async MCP handler that routes to ``tool.on_invoke_tool``.

    ``on_invoke_tool`` takes a JSON *string* of arguments and a ``ToolContext``
    carrying the run context — the same object shape the default lane hands its
    tools, so context-reading tools (child-agent spawn, reporting, notes, proxy)
    behave identically. The result string is returned as the tool's text content.
    Any failure is returned as an error result string; the handler never raises
    into the SDK loop.

    A ``ToolContext`` (not a bare ``RunContextWrapper``) is required: the SDK's tool
    machinery — including its error path (``agents.tool._on_handled_error``) — reads
    ``run_config``, ``tool_name`` and ``tool_call_id`` off the context, which the
    ``Runner`` populates on the default lane. Building a bare wrapper here made any
    tool that hit its error branch crash with ``AttributeError`` under the lane
    (issue #7). ``run_config`` may be ``None`` (the SDK guards ``run_config is None``).
    """

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            args_json = json.dumps(arguments or {})
        except (TypeError, ValueError) as exc:
            return _text_result(
                f"Could not serialize arguments for tool '{tool.name}': {exc}", is_error=True
            )
        tool_ctx: ToolContext[dict[str, Any]] = ToolContext(
            context=context,
            tool_name=tool.name,
            tool_call_id=uuid.uuid4().hex,
            tool_arguments=args_json,
            run_config=run_config,
        )
        try:
            result = await tool.on_invoke_tool(tool_ctx, args_json)
        except Exception as exc:  # noqa: BLE001 - error-as-result, mirrors SDK tool behavior.
            logger.debug("tool %s failed under claude-cli lane", tool.name, exc_info=True)
            return _text_result(f"Tool '{tool.name}' failed: {exc}", is_error=True)
        return _text_result("" if result is None else str(result))

    return handler


def build_tool_server(
    tools: list[Tool], context: dict[str, Any], run_config: Any = None
) -> McpSdkServerConfig:
    """Build the one in-process ``strix`` MCP server exposing the ``FunctionTool``s.

    ``tools`` is the agent's final wrapped tool list (§3.3); ``context`` is the
    run context dict handed to each invocation; ``run_config`` is the run's
    ``RunConfig``, stamped on each tool-invocation wrapper so the SDK's tool error
    path works under the lane (issue #7).
    """
    from claude_agent_sdk import (  # noqa: PLC0415 - lazy by design (AD-6).
        SdkMcpTool,
        create_sdk_mcp_server,
    )

    sdk_tools: list[SdkMcpTool[Any]] = []
    skipped: list[str] = []
    for tool in tools:
        if not isinstance(tool, FunctionTool):
            skipped.append(f"{type(tool).__name__}:{getattr(tool, 'name', '?')}")
            continue
        sdk_tools.append(
            SdkMcpTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.params_json_schema,
                handler=_make_handler(tool, context, run_config),
            )
        )

    if skipped:
        logger.warning(
            "claude-cli lane: skipped %d non-FunctionTool tool(s), not exposed over MCP: %s",
            len(skipped),
            ", ".join(skipped),
        )

    logger.info(
        "claude-cli lane: exposing %d Strix tool(s) as mcp__%s__*", len(sdk_tools), SERVER_NAME
    )
    return create_sdk_mcp_server(name=SERVER_NAME, tools=sdk_tools)


def exposed_tool_names(tools: list[Tool]) -> list[str]:
    """The ``mcp__strix__*`` names Claude Code will see for ``tools``.

    Used to build the lane's ``allowed_tools`` allow-list (Invariant I-1): Claude
    Code may auto-call exactly these and none of its built-in host tools.
    """
    return [f"mcp__{SERVER_NAME}__{tool.name}" for tool in tools if isinstance(tool, FunctionTool)]


__all__ = ["SERVER_NAME", "build_tool_server", "exposed_tool_names"]
