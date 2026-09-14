"""claude-cli/ lane dispatch at the Runner.run_streamed call site (SC-1, SC-7).

``_run_cycle`` must construct a ``ClaudeCliStream`` when the run's model carries a
``claude-cli/`` prefix and must otherwise take the unchanged ``Runner.run_streamed``
path — with the lane never constructed for a non-lane model (SC-7 default-path
regression).
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from agents import RunConfig

from strix.core import execution
from strix.core.agents import AgentCoordinator
from strix.llm.claude_cli import dispatch as lane_dispatch


class _FakeStream:
    """Minimal stand-in for a streamed run: yields nothing, no exception."""

    def __init__(self, source: str, **kwargs: Any) -> None:
        self.source = source
        self.kwargs = kwargs
        self.run_loop_exception: BaseException | None = None
        self.new_items: list[Any] = []
        self.final_output: str | None = None

    async def stream_events(self) -> Any:
        return
        yield  # pragma: no cover - makes this an async generator


class _RunnerRecorder:
    def __init__(self) -> None:
        self.calls = 0
        self.last_stream: _FakeStream | None = None

    def run_streamed(self, _agent: Any, **kwargs: Any) -> _FakeStream:
        self.calls += 1
        self.last_stream = _FakeStream("default", **kwargs)
        return self.last_stream


class _ClaudeCliStreamRecorder:
    constructed: ClassVar[list[_ClaudeCliStreamRecorder]] = []

    def __init__(self, _agent: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.run_loop_exception: BaseException | None = None
        self.new_items: list[Any] = []
        self.final_output: str | None = None
        _ClaudeCliStreamRecorder.constructed.append(self)

    async def stream_events(self) -> Any:
        return
        yield  # pragma: no cover - async generator


async def _drive_cycle(monkeypatch: pytest.MonkeyPatch, model: str) -> tuple[Any, _RunnerRecorder]:
    _ClaudeCliStreamRecorder.constructed = []
    recorder = _RunnerRecorder()
    monkeypatch.setattr(execution, "Runner", recorder)
    monkeypatch.setattr(lane_dispatch, "ClaudeCliStream", _ClaudeCliStreamRecorder)

    coordinator = AgentCoordinator()
    await coordinator.register("root", "strix", parent_id=None)

    result = await execution._run_cycle(
        object(),  # agent - unused by the fakes
        coordinator,
        "root",
        input_data=[],
        run_config=RunConfig(model=model),
        context={},
        max_turns=5,
        session=None,
        interactive=False,
        event_sink=None,
        hooks=None,
    )
    return result, recorder


async def test_claude_cli_model_dispatches_to_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    result, recorder = await _drive_cycle(monkeypatch, "claude-cli/claude-sonnet-4-5")

    assert isinstance(result, _ClaudeCliStreamRecorder)
    assert len(_ClaudeCliStreamRecorder.constructed) == 1
    # The default lane was not used.
    assert recorder.calls == 0
    # The lane received the run's wiring.
    assert _ClaudeCliStreamRecorder.constructed[0].kwargs["max_turns"] == 5


async def test_non_lane_model_takes_default_path(monkeypatch: pytest.MonkeyPatch) -> None:
    result, recorder = await _drive_cycle(monkeypatch, "openai/gpt-5")

    assert recorder.calls == 1
    assert result is recorder.last_stream
    # The lane was never constructed (SC-7: default path unchanged).
    assert _ClaudeCliStreamRecorder.constructed == []


# --- #24: lifecycle-tool detection for the stall diagnostic -------------------


def test_is_lifecycle_tool_matches_mcp_names() -> None:
    assert lane_dispatch._is_lifecycle_tool("mcp__strix__finish_scan")
    assert lane_dispatch._is_lifecycle_tool("mcp__strix__agent_finish")
    assert lane_dispatch._is_lifecycle_tool("mcp__strix__wait_for_agents")
    assert lane_dispatch._is_lifecycle_tool("mcp__strix__respond_to_user")
    assert lane_dispatch._is_lifecycle_tool("finish_scan")
    # Non-lifecycle tools are not matched.
    assert not lane_dispatch._is_lifecycle_tool("mcp__strix__get_threat_model")
    assert not lane_dispatch._is_lifecycle_tool("mcp__strix-sandbox__fs_read")
    assert not lane_dispatch._is_lifecycle_tool("create_agent")
