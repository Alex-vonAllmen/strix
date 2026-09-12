"""claude-cli/ lane sandbox bridge (M2): shell/fs tools over session.exec.

Drives the bridge handlers with a fake sandbox session (no Docker) and asserts:
- shell_exec routes through session.exec inside the sandbox root and reports
  stdout/stderr/exit code,
- fs_read/fs_write/fs_list route through the session's read/write/ls,
- paths escaping the sandbox root are refused with an error string (I-2),
- a failing session call is returned as a structured error result, never raised.
"""

from __future__ import annotations

import io
from functools import partial
from pathlib import PurePosixPath
from typing import Any

import pytest

from strix.llm.claude_cli import sandbox_bridge


class _ExecResult:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class _Entry:
    def __init__(self, path: str, *, size: int = 0, is_dir: bool = False) -> None:
        self.path = path
        self.size = size
        self.kind = "dir" if is_dir else "file"

    def is_dir(self) -> bool:
        return self.kind == "dir"


class _FakeSession:
    def __init__(self) -> None:
        self.exec_calls: list[tuple[str, Any]] = []
        self.reads: list[PurePosixPath] = []
        self.writes: list[tuple[PurePosixPath, bytes]] = []
        self.ls_calls: list[str] = []
        self.raise_on_exec = False

    async def exec(self, command: str, *, timeout: float | None = None, shell: bool = True) -> Any:
        self.exec_calls.append((command, timeout))
        if self.raise_on_exec:
            raise RuntimeError("sandbox is gone")
        return _ExecResult(stdout=b"hello\n", stderr=b"", exit_code=0)

    async def read(self, path: PurePosixPath, *, user: Any = None) -> io.BytesIO:
        self.reads.append(path)
        return io.BytesIO(b"file contents")

    async def write(self, path: PurePosixPath, data: io.IOBase, *, user: Any = None) -> None:
        self.writes.append((path, data.read()))

    async def ls(self, path: str, *, user: Any = None) -> list[_Entry]:
        self.ls_calls.append(str(path))
        return [_Entry("/workspace/a.txt", size=12), _Entry("/workspace/sub", is_dir=True)]


def _handler(name: str, session: _FakeSession, workdir: str = "/workspace") -> Any:
    handlers = {
        "shell_exec": sandbox_bridge._shell_exec,
        "fs_read": sandbox_bridge._fs_read,
        "fs_write": sandbox_bridge._fs_write,
        "fs_list": sandbox_bridge._fs_list,
    }
    return partial(handlers[name], session, workdir)


def test_sandbox_tool_names() -> None:
    assert sandbox_bridge.sandbox_tool_names() == [
        "mcp__strix-sandbox__shell_exec",
        "mcp__strix-sandbox__fs_read",
        "mcp__strix-sandbox__fs_write",
        "mcp__strix-sandbox__fs_list",
    ]


async def test_shell_exec_runs_inside_sandbox_root() -> None:
    session = _FakeSession()
    result = await _handler("shell_exec", session)({"command": "ls -la"})

    assert result["is_error"] is False
    assert "exit_code: 0" in result["content"][0]["text"]
    assert "hello" in result["content"][0]["text"]
    # Command runs inside /workspace and preserves the model's command.
    command, timeout = session.exec_calls[0]
    assert command == "cd /workspace && ls -la"
    assert timeout == 120


async def test_shell_exec_caps_timeout() -> None:
    session = _FakeSession()
    await _handler("shell_exec", session)({"command": "sleep 1", "timeout_s": 999999})
    assert session.exec_calls[0][1] == sandbox_bridge._MAX_TIMEOUT_S


async def test_shell_exec_empty_command_is_error() -> None:
    session = _FakeSession()
    result = await _handler("shell_exec", session)({"command": "   "})
    assert result["is_error"] is True
    assert session.exec_calls == []


async def test_shell_exec_failure_is_error_result_not_raised() -> None:
    session = _FakeSession()
    session.raise_on_exec = True
    result = await _handler("shell_exec", session)({"command": "boom"})
    assert result["is_error"] is True
    assert "shell_exec failed" in result["content"][0]["text"]


async def test_fs_read_routes_through_session() -> None:
    session = _FakeSession()
    result = await _handler("fs_read", session)({"path": "notes.txt"})
    assert result["is_error"] is False
    assert result["content"][0]["text"] == "file contents"
    assert session.reads == [PurePosixPath("/workspace/notes.txt")]


async def test_fs_write_routes_through_session() -> None:
    session = _FakeSession()
    result = await _handler("fs_write", session)({"path": "out.txt", "content": "data"})
    assert result["is_error"] is False
    assert "wrote 4 bytes to /workspace/out.txt" in result["content"][0]["text"]
    assert session.writes[0][0] == PurePosixPath("/workspace/out.txt")
    assert session.writes[0][1] == b"data"


async def test_fs_list_formats_entries() -> None:
    session = _FakeSession()
    result = await _handler("fs_list", session)({"path": "."})
    text = result["content"][0]["text"]
    assert "/workspace/a.txt" in text
    assert "/workspace/sub" in text
    assert "dir " in text


@pytest.mark.parametrize("tool", ["fs_read", "fs_write", "fs_list"])
async def test_path_escape_is_refused(tool: str) -> None:
    session = _FakeSession()
    args = {"path": "../../etc/passwd"}
    if tool == "fs_write":
        args["content"] = "x"
    result = await _handler(tool, session)(args)
    assert result["is_error"] is True
    assert "outside the sandbox root" in result["content"][0]["text"]
    # The escaping path never reached the session.
    assert session.reads == []
    assert session.writes == []
    assert session.ls_calls == []


async def test_absolute_path_inside_root_is_allowed() -> None:
    session = _FakeSession()
    result = await _handler("fs_read", session)({"path": "/workspace/deep/f.txt"})
    assert result["is_error"] is False
    assert session.reads == [PurePosixPath("/workspace/deep/f.txt")]


async def test_absolute_host_path_is_refused() -> None:
    session = _FakeSession()
    result = await _handler("fs_read", session)({"path": "/etc/shadow"})
    assert result["is_error"] is True
    assert session.reads == []


def test_build_sandbox_bridge_returns_server_config() -> None:
    session = _FakeSession()
    config = sandbox_bridge.build_sandbox_bridge(session)
    # McpSdkServerConfig is a TypedDict: type "sdk", name "strix-sandbox".
    assert config["type"] == "sdk"
    assert config["name"] == "strix-sandbox"
