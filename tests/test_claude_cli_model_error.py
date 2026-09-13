"""claude-cli/ lane: fail fast on an invalid/inaccessible model (issue #5).

An errored CLI result (unknown/inaccessible model being the common case) must raise a
terminal ``ClaudeCliLaneError`` from ``ClaudeCliStream`` on the first cycle — carrying
the CLI's own message — instead of a tool-call-free turn that the recovery loop burns
into a misleading ``MaxTurnsExceeded``. A valid result must be unaffected.
"""

from __future__ import annotations

from typing import Any

import claude_agent_sdk as sdk
import pytest
from agents.exceptions import MaxTurnsExceeded

from strix.llm.claude_cli import ClaudeCliLaneError
from strix.llm.claude_cli.dispatch import ClaudeCliStream, _model_error_message


def _make_stream(model: str = "claude-cli/claude-opus-4.8") -> ClaudeCliStream:
    return ClaudeCliStream(
        type("Agent", (), {"name": "root", "instructions": "sys", "tools": []})(),
        input=[],
        run_config=type("RC", (), {"model": model, "sandbox": None})(),
        context={},
        max_turns=6,
        session=None,
        hooks=None,
    )


async def _drive(stream: ClaudeCliStream, message: Any) -> list[Any]:
    return [event async for event in stream._translate(sdk, message)]


def _result(*, is_error: bool, result: str | None) -> sdk.ResultMessage:
    return sdk.ResultMessage(
        subtype="success",  # the CLI reports "success" even on error — must be ignored
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id="s1",
        result=result,
        usage={"input_tokens": 1, "output_tokens": 1},
    )


# -- _model_error_message (SC-4) --------------------------------------------------


def test_message_hints_hyphenated_slug_on_dotted_model_not_found() -> None:
    msg = _model_error_message(
        "There's an issue with the selected model (claude-opus-4.8). It may not exist "
        "or you may not have access to it.",
        "model_not_found",
        "claude-opus-4.8",
    )
    assert "claude-cli lane:" in msg
    assert "claude-cli/claude-opus-4-8" in msg


def test_message_no_hint_when_slug_has_no_dot() -> None:
    msg = _model_error_message("some error", "model_not_found", "claude-opus-4-8")
    assert "Did you mean" not in msg
    assert "some error" in msg


def test_message_falls_back_when_result_empty() -> None:
    msg = _model_error_message(None, None, "opus")
    assert "the claude CLI returned an error" in msg


# -- fail-fast in the stream (SC-1, SC-2) -----------------------------------------


async def test_errored_result_raises_lane_error() -> None:
    stream = _make_stream()
    text = "There's an issue with the selected model (claude-opus-4.8). It may not exist."
    with pytest.raises(ClaudeCliLaneError) as excinfo:
        await _drive(stream, _result(is_error=True, result=text))
    assert "issue with the selected model" in str(excinfo.value)
    # The dotted-slug hint is included.
    assert "claude-cli/claude-opus-4-8" in str(excinfo.value)


async def test_assistant_error_on_first_cycle_then_errored_result_raises() -> None:
    stream = _make_stream()
    # First the assistant turn carrying the error code...
    assistant = sdk.AssistantMessage(
        content=[sdk.TextBlock(text="There's an issue with the selected model.")],
        model="claude-opus-4.8",
        error="model_not_found",
    )
    await _drive(stream, assistant)  # captured, not yet raised
    assert stream._assistant_error == "model_not_found"
    # ...then the errored result on the same (first) cycle raises.
    with pytest.raises(ClaudeCliLaneError):
        await _drive(stream, _result(is_error=True, result="model error"))


async def test_errored_result_is_not_maxturns() -> None:
    stream = _make_stream()
    try:
        await _drive(stream, _result(is_error=True, result="bad model"))
    except ClaudeCliLaneError:
        pass
    except MaxTurnsExceeded:  # pragma: no cover - the whole point is this does NOT fire
        pytest.fail("errored result raised MaxTurnsExceeded instead of ClaudeCliLaneError")


# -- no regression on a valid result (SC-3) ---------------------------------------


async def test_successful_result_does_not_raise() -> None:
    stream = _make_stream(model="claude-cli/claude-opus-4-8")
    events = await _drive(stream, _result(is_error=False, result="all done"))
    assert events == []
    assert stream.final_output == "all done"
