"""claude-cli/ lane wiring of the sandbox bridge into ClaudeCliStream (M2).

Asserts the executor attaches the ``strix-sandbox`` MCP server (and its tool names
to the allow-list) when the run holds a sandbox session, and omits it otherwise —
without touching M1's published duck-type contract.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from strix.llm.claude_cli import dispatch
from strix.llm.claude_cli.dispatch import ClaudeCliStream


class _FakeOptions:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeClient:
    def __init__(self, options: Any = None) -> None:
        self.options = options

    async def connect(self) -> None:
        return

    async def query(self, prompt: str, session_id: str = "default") -> None:
        return

    async def receive_response(self) -> Any:
        return
        yield  # pragma: no cover - async generator with no messages

    async def disconnect(self) -> None:
        return


class _FakeSDK:
    ClaudeAgentOptions = _FakeOptions
    ClaudeSDKClient = _FakeClient


def _agent() -> Any:
    return SimpleNamespace(name="root", instructions="sys", tools=[])


async def _captured_options(stream: ClaudeCliStream, monkeypatch: Any) -> _FakeOptions:
    monkeypatch.setattr(dispatch, "_import_sdk", lambda: _FakeSDK)
    captured: list[_FakeOptions] = []

    class _CapturingClient(_FakeClient):
        def __init__(self, options: Any = None) -> None:
            super().__init__(options)
            captured.append(options)

    monkeypatch.setattr(_FakeSDK, "ClaudeSDKClient", _CapturingClient)
    async for _ in stream.stream_events():
        pass
    return captured[0]


async def test_sandbox_bridge_attached_when_session_present(monkeypatch: Any) -> None:
    session = object()
    run_config = SimpleNamespace(
        model="claude-cli/claude-opus-5", sandbox=SimpleNamespace(session=session)
    )
    stream = ClaudeCliStream(
        _agent(),
        input="do it",
        run_config=run_config,
        context={},
        max_turns=5,
        session=None,
        hooks=None,
    )
    options = await _captured_options(stream, monkeypatch)

    assert set(options.kwargs["mcp_servers"]) == {"strix", "strix-sandbox"}
    assert "mcp__strix-sandbox__shell_exec" in options.kwargs["allowed_tools"]
    # No built-in host tools reachable (I-1/I-2).
    assert options.kwargs["tools"] == []


async def test_sandbox_bridge_from_context_fallback(monkeypatch: Any) -> None:
    run_config = SimpleNamespace(model="claude-cli/claude-opus-5", sandbox=None)
    stream = ClaudeCliStream(
        _agent(),
        input="do it",
        run_config=run_config,
        context={"sandbox_session": object()},
        max_turns=5,
        session=None,
        hooks=None,
    )
    options = await _captured_options(stream, monkeypatch)
    assert "strix-sandbox" in options.kwargs["mcp_servers"]


async def test_no_sandbox_means_no_bridge(monkeypatch: Any) -> None:
    run_config = SimpleNamespace(model="claude-cli/claude-opus-5", sandbox=None)
    stream = ClaudeCliStream(
        _agent(),
        input="do it",
        run_config=run_config,
        context={},
        max_turns=5,
        session=None,
        hooks=None,
    )
    options = await _captured_options(stream, monkeypatch)

    assert set(options.kwargs["mcp_servers"]) == {"strix"}
    assert not any("strix-sandbox" in t for t in options.kwargs["allowed_tools"])
