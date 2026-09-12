"""Sandbox MCP bridge for the ``claude-cli/`` lane (AD-3, M2).

Under Claude Code there is no OpenAI-Agents ``Shell``/``Filesystem`` capability
binding a live sandbox to the run, so this module hand-writes an in-process MCP
server named ``strix-sandbox`` that exposes shell/fs tools and executes them
through the sandbox session the run already holds. Every command runs inside the
existing per-scan Docker sandbox — with the Caido proxy env the session already
injects (SC-6 holds by construction) — never on the host.

Invariant I-2: the bridge is the lane's only shell/fs surface, and it is
sandbox-bound — all paths resolve inside the sandbox root and execution goes only
through ``session.exec`` (no direct Docker calls, no host paths). Tool failures are
returned as structured error strings, never raised into the SDK loop (mirrors the
host-tool adapter, §4.4).
"""

from __future__ import annotations

import io
import logging
import posixpath
import shlex
from functools import partial
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from strix.config import load_settings
from strix.tools.output_store import bound_text


if TYPE_CHECKING:
    from claude_agent_sdk import McpSdkServerConfig


logger = logging.getLogger(__name__)

SERVER_NAME = "strix-sandbox"

# The sandbox workspace root; every bridge path resolves inside it (I-2).
DEFAULT_WORKDIR = "/workspace"

# Hard ceiling on a single exec, independent of the model-supplied timeout.
_MAX_TIMEOUT_S = 600
_DEFAULT_TIMEOUT_S = 120
# Cap fs_read payloads so one call can't blow the context or the sandbox.
_DEFAULT_READ_MAX_BYTES = 200_000
_MAX_LIST_ENTRIES = 1000

_TOOL_NAMES = ("shell_exec", "fs_read", "fs_write", "fs_list")


class SandboxPathError(ValueError):
    """A bridge path escaped the sandbox root or named a host path."""


def _bound(text: str) -> str:
    context = load_settings().context
    return bound_text(
        text, max_lines=context.tool_output_max_lines, max_bytes=context.tool_output_max_bytes
    )


def _resolve_in_sandbox(path: str, root: str) -> str:
    """Resolve a bridge ``path`` to an absolute sandbox path inside ``root``.

    Relative paths join onto ``root``; absolute paths must already be within it.
    ``..`` traversal that escapes the root is refused (I-2).
    """
    root_norm = posixpath.normpath(root)
    candidate = path if posixpath.isabs(path) else posixpath.join(root_norm, path)
    resolved = posixpath.normpath(candidate)
    if resolved != root_norm and not resolved.startswith(root_norm + "/"):
        raise SandboxPathError(
            f"path {path!r} resolves outside the sandbox root {root_norm!r}; "
            "only paths inside the sandbox are allowed"
        )
    return resolved


def _text_result(text: str, *, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": is_error}


def _decode(raw: Any) -> str:
    if isinstance(raw, bytes | bytearray):
        return raw.decode("utf-8", "replace")
    return str(raw)


async def _shell_exec(session: Any, root: str, args: dict[str, Any]) -> dict[str, Any]:
    command = str(args.get("command", "")).strip()
    if not command:
        return _text_result("shell_exec: `command` must be a non-empty string.", is_error=True)
    try:
        timeout_s = int(args.get("timeout_s", _DEFAULT_TIMEOUT_S) or _DEFAULT_TIMEOUT_S)
    except (TypeError, ValueError):
        timeout_s = _DEFAULT_TIMEOUT_S
    timeout_s = max(1, min(timeout_s, _MAX_TIMEOUT_S))
    # Run inside the sandbox root; the session injects the Caido proxy env. shell=True
    # is safe here: this executes in the isolated Docker sandbox, never on the host (I-2).
    wrapped = f"cd {shlex.quote(root)} && {command}"
    try:
        # shell=True runs inside the isolated Docker sandbox, never on the host; a
        # sandbox shell is the tool's whole purpose (I-2).
        result = await session.exec(wrapped, timeout=timeout_s, shell=True)  # noqa: S604  # nosec B604
    except Exception as exc:  # noqa: BLE001 - error-as-result, never raise into the SDK loop.
        logger.debug("shell_exec failed", exc_info=True)
        return _text_result(f"shell_exec failed: {exc}", is_error=True)
    exit_code = int(getattr(result, "exit_code", -1) or 0)
    parts = [f"$ {command}", f"exit_code: {exit_code}"]
    out = _decode(getattr(result, "stdout", b""))
    err = _decode(getattr(result, "stderr", b""))
    if out:
        parts.append(f"stdout:\n{out}")
    if err:
        parts.append(f"stderr:\n{err}")
    return _text_result(_bound("\n".join(parts)))


async def _fs_read(session: Any, root: str, args: dict[str, Any]) -> dict[str, Any]:
    try:
        resolved = _resolve_in_sandbox(str(args.get("path", "")), root)
    except SandboxPathError as exc:
        return _text_result(f"fs_read: {exc}", is_error=True)
    try:
        max_bytes = int(args.get("max_bytes", _DEFAULT_READ_MAX_BYTES) or _DEFAULT_READ_MAX_BYTES)
    except (TypeError, ValueError):
        max_bytes = _DEFAULT_READ_MAX_BYTES
    max_bytes = max(1, min(max_bytes, _DEFAULT_READ_MAX_BYTES))
    try:
        stream = await session.read(PurePosixPath(resolved))
        raw = stream.read(max_bytes) if hasattr(stream, "read") else bytes(stream)
    except Exception as exc:  # noqa: BLE001 - error-as-result.
        logger.debug("fs_read failed", exc_info=True)
        return _text_result(f"fs_read failed for {resolved!r}: {exc}", is_error=True)
    return _text_result(_bound(_decode(raw)))


async def _fs_write(session: Any, root: str, args: dict[str, Any]) -> dict[str, Any]:
    try:
        resolved = _resolve_in_sandbox(str(args.get("path", "")), root)
    except SandboxPathError as exc:
        return _text_result(f"fs_write: {exc}", is_error=True)
    content = args.get("content", "")
    if not isinstance(content, str):
        return _text_result("fs_write: `content` must be a string.", is_error=True)
    data = content.encode("utf-8")
    try:
        await session.write(PurePosixPath(resolved), io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 - error-as-result.
        logger.debug("fs_write failed", exc_info=True)
        return _text_result(f"fs_write failed for {resolved!r}: {exc}", is_error=True)
    return _text_result(f"wrote {len(data)} bytes to {resolved}")


async def _fs_list(session: Any, root: str, args: dict[str, Any]) -> dict[str, Any]:
    try:
        resolved = _resolve_in_sandbox(str(args.get("path", ".")), root)
    except SandboxPathError as exc:
        return _text_result(f"fs_list: {exc}", is_error=True)
    try:
        entries = await session.ls(resolved)
    except Exception as exc:  # noqa: BLE001 - error-as-result.
        logger.debug("fs_list failed", exc_info=True)
        return _text_result(f"fs_list failed for {resolved!r}: {exc}", is_error=True)
    lines: list[str] = []
    for entry in list(entries)[:_MAX_LIST_ENTRIES]:
        is_dir = bool(getattr(entry, "kind", None)) and entry.is_dir()
        size = getattr(entry, "size", 0)
        lines.append(f"{'dir ' if is_dir else 'file'}  {size:>10}  {getattr(entry, 'path', '?')}")
    return _text_result(_bound("\n".join(lines) if lines else "(empty)"))


def _tool_specs() -> list[tuple[str, str, dict[str, Any], Any]]:
    return [
        (
            "shell_exec",
            "Run a shell command inside the Strix sandbox container (with the Caido "
            "proxy active). Returns stdout, stderr and the exit code.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                    "timeout_s": {
                        "type": "integer",
                        "description": f"Timeout in seconds (capped at {_MAX_TIMEOUT_S}).",
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            _shell_exec,
        ),
        (
            "fs_read",
            "Read a text file from inside the sandbox (path relative to the workspace).",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}, "max_bytes": {"type": "integer"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            _fs_read,
        ),
        (
            "fs_write",
            "Write a text file inside the sandbox (path relative to the workspace).",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            _fs_write,
        ),
        (
            "fs_list",
            "List a directory inside the sandbox (path relative to the workspace).",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}, "depth": {"type": "integer"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            _fs_list,
        ),
    ]


def build_sandbox_bridge(
    sandbox_session: Any, *, workdir: str = DEFAULT_WORKDIR
) -> McpSdkServerConfig:
    """In-process MCP server ``strix-sandbox``: shell/fs tools over ``session.exec``.

    ``sandbox_session`` is the run's existing sandbox session; the bridge holds no
    other capability. All tools run inside ``workdir`` (the sandbox root).
    """
    from claude_agent_sdk import (  # noqa: PLC0415 - lazy by design (AD-6).
        SdkMcpTool,
        create_sdk_mcp_server,
    )

    root = posixpath.normpath(workdir)
    tools = [
        SdkMcpTool(
            name=name,
            description=description,
            input_schema=schema,
            handler=partial(handler, sandbox_session, root),
        )
        for name, description, schema, handler in _tool_specs()
    ]
    return create_sdk_mcp_server(name=SERVER_NAME, tools=tools)


def sandbox_tool_names() -> list[str]:
    """The ``mcp__strix-sandbox__*`` names for the lane's allow-list (I-1/I-2)."""
    return [f"mcp__{SERVER_NAME}__{name}" for name in _TOOL_NAMES]


__all__ = ["SERVER_NAME", "SandboxPathError", "build_sandbox_bridge", "sandbox_tool_names"]
