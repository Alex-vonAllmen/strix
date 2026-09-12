"""claude-cli/ lane stream translation and usage accounting.

Drives ``ClaudeCliStream`` translation with real ``claude_agent_sdk`` message
objects (no live CLI) and asserts:
- assistant text / tool-use / tool-result map to the Agents-SDK stream events the
  event sink and TUI already consume,
- partial text deltas map to a raw ``response.output_text.delta`` event,
- ``ResultMessage`` usage is fed through the run hooks' ``on_llm_end`` (SC-7) and
  the final output text is captured.
"""

from __future__ import annotations

from typing import Any, ClassVar

import claude_agent_sdk as sdk

from strix.llm.claude_cli.dispatch import ClaudeCliStream


class _Agent:
    name = "root"
    instructions = "sys"
    tools: ClassVar[list[Any]] = []


def _make_stream(hooks: Any = None) -> ClaudeCliStream:
    return ClaudeCliStream(
        _Agent(),
        input=[],
        run_config=type("RC", (), {"model": "claude-cli/claude-opus-5"})(),
        context={"agent_id": "root", "parent_id": None},
        max_turns=5,
        session=None,
        hooks=hooks,
    )


async def _collect(stream: ClaudeCliStream, message: Any) -> list[Any]:
    return [event async for event in stream._translate(sdk, message)]


async def test_assistant_text_and_tool_use_translate() -> None:
    stream = _make_stream()
    message = sdk.AssistantMessage(
        content=[
            sdk.TextBlock(text="hello world"),
            sdk.ToolUseBlock(id="call-1", name="create_note", input={"text": "x"}),
        ],
        model="claude-opus-5",
    )
    events = await _collect(stream, message)

    assert [e.type for e in events] == ["run_item_stream_event", "run_item_stream_event"]
    msg_event, tool_event = events
    assert msg_event.item.type == "message_output_item"
    assert tool_event.item.type == "tool_call_item"
    # Tool-use carries the invoked name (surfaces as mcp__strix__create_note upstream).
    assert tool_event.item.raw_item.name == "create_note"
    assert tool_event.item.raw_item.call_id == "call-1"

    # Transcript captures the turn for crash salvage (to_input_list).
    salvage = stream.to_input_list()
    assert {"role": "assistant", "content": "hello world"} in salvage
    assert any(i.get("type") == "function_call" for i in salvage)


async def test_tool_result_translates_to_output_event() -> None:
    stream = _make_stream()
    message = sdk.UserMessage(content=[sdk.ToolResultBlock(tool_use_id="call-1", content="done")])
    events = await _collect(stream, message)

    assert len(events) == 1
    assert events[0].item.type == "tool_call_output_item"
    assert events[0].item.output == "done"


async def test_partial_text_delta_translates_to_raw_event() -> None:
    stream = _make_stream()
    message = sdk.StreamEvent(
        uuid="u1",
        session_id="s1",
        event={
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "chunk"},
        },
    )
    events = await _collect(stream, message)

    assert len(events) == 1
    assert events[0].type == "raw_response_event"
    assert events[0].data.type == "response.output_text.delta"
    assert events[0].data.delta == "chunk"


async def test_result_message_records_usage_and_final_output() -> None:
    recorded: list[Any] = []

    class _Hooks:
        async def on_llm_end(self, _ctx: Any, _agent: Any, response: Any) -> None:
            recorded.append(response)

    stream = _make_stream(hooks=_Hooks())
    message = sdk.ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=3,
        session_id="s1",
        result="final answer",
        usage={
            "input_tokens": 100,
            "output_tokens": 40,
            "cache_read_input_tokens": 10,
            "cache_creation_input_tokens": 5,
        },
    )
    events = await _collect(stream, message)

    assert events == []  # a result message emits no stream event
    assert stream.final_output == "final answer"
    assert len(recorded) == 1
    usage = recorded[0].usage
    # input_tokens folds in cache read/write; total sums input + output.
    assert usage.input_tokens == 100 + 10 + 5
    assert usage.output_tokens == 40
    assert usage.total_tokens == 100 + 10 + 5 + 40
    assert usage.requests == 3


async def test_zero_usage_is_not_recorded() -> None:
    recorded: list[Any] = []

    class _Hooks:
        async def on_llm_end(self, _ctx: Any, _agent: Any, response: Any) -> None:
            recorded.append(response)

    stream = _make_stream(hooks=_Hooks())
    message = sdk.ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="s1",
        result=None,
        usage={"input_tokens": 0, "output_tokens": 0},
    )
    await _collect(stream, message)
    assert recorded == []
