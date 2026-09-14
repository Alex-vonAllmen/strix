"""claude-cli/ lane completion semantics (issue #9, M2).

- The lane ends the SDK run when a *terminal* lifecycle tool settles the agent (the
  lane's tool_use_behavior), instead of running to the CLI's max_turns. A blocking
  parking tool (``wait_for_agents``) parks the agent ``waiting`` but the cycle stays
  alive to resume it (#24).
- Empty-input cycles re-feed the session transcript so a fresh SDK client keeps its task
  (no bare "Continue.").
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

from strix.llm.claude_cli import dispatch
from strix.llm.claude_cli.dispatch import ClaudeCliStream


def _stream(
    context: dict[str, Any], *, session: Any = None, input_data: Any = None
) -> ClaudeCliStream:
    return ClaudeCliStream(
        SimpleNamespace(name="a", instructions="sys", tools=[]),
        input=input_data if input_data is not None else [],
        run_config=SimpleNamespace(model="claude-cli/claude-opus-4-8", sandbox=None),
        context=context,
        max_turns=6,
        session=session,
        hooks=None,
    )


# -- _lifecycle_settled (SC-3) ----------------------------------------------------


def test_lifecycle_settled_reads_coordinator_status() -> None:
    coord = SimpleNamespace(statuses={"root": "running"})
    stream = _stream({"coordinator": coord, "agent_id": "root"})
    assert stream._lifecycle_settled() is False
    coord.statuses["root"] = "completed"
    assert stream._lifecycle_settled() is True
    coord.statuses["root"] = "waiting"  # blocking parking tool — cycle must stay alive (#24)
    assert stream._lifecycle_settled() is False


def test_lifecycle_settled_false_without_coordinator() -> None:
    assert _stream({})._lifecycle_settled() is False


# -- stream ends the run when settled (SC-3) --------------------------------------


class _FakeClient:
    def __init__(self, options: Any = None, messages: list[Any] | None = None) -> None:
        self.interrupted = 0
        self._messages = messages or []
        self.consumed = 0

    async def connect(self) -> None:
        return

    async def query(self, prompt: str, session_id: str = "default") -> None:
        return

    async def receive_response(self) -> Any:
        for m in self._messages:
            self.consumed += 1
            yield m

    async def interrupt(self) -> None:
        self.interrupted += 1

    async def disconnect(self) -> None:
        return


class _Result:
    """Stands in for ``sdk.ResultMessage`` — carries cumulative usage + result."""

    is_error = False
    subtype = "success"
    terminal_reason = "aborted_streaming"
    num_turns = 5
    result = "done"
    usage: ClassVar[dict[str, int]] = {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


async def test_stream_settles_then_drains_to_result_message(monkeypatch: Any) -> None:
    # Settled from the start (a terminal lifecycle tool). The lane must interrupt
    # once, then DRAIN to the ResultMessage — recording its usage + final_output —
    # instead of breaking before it and losing them (#28).
    coord = SimpleNamespace(statuses={"root": "completed"})

    recorded: list[Any] = []

    class _Hooks:
        async def on_llm_end(self, _ctx: Any, _agent: Any, response: Any) -> None:
            recorded.append(response.usage)

    stream = ClaudeCliStream(
        SimpleNamespace(name="a", instructions="sys", tools=[]),
        input=[],
        run_config=SimpleNamespace(model="claude-cli/claude-opus-4-8", sandbox=None),
        context={"coordinator": coord, "agent_id": "root"},
        max_turns=6,
        session=None,
        hooks=_Hooks(),
    )

    created: list[_FakeClient] = []

    class _SDK:
        StreamEvent = type("StreamEvent", (), {})
        AssistantMessage = type("AssistantMessage", (), {})
        UserMessage = type("UserMessage", (), {})
        ResultMessage = _Result
        ClaudeAgentOptions = SimpleNamespace

        @staticmethod
        def ClaudeSDKClient(options: Any = None) -> _FakeClient:  # noqa: N802
            # settle-marker, the ResultMessage, then a trailing message that must
            # NOT be consumed (we stop AT the ResultMessage).
            c = _FakeClient(options, messages=[object(), _Result(), object()])
            created.append(c)
            return c

    monkeypatch.setattr(dispatch, "_import_sdk", lambda: _SDK)
    async for _ in stream.stream_events():
        pass

    client = created[0]
    assert client.interrupted == 1
    assert client.consumed == 2  # settle-marker + ResultMessage, then stop
    assert recorded and recorded[0].total_tokens == 120  # usage recorded (#28)
    assert stream.final_output == "done"


# -- session continuity (SC-4) ----------------------------------------------------


class _FakeSession:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    async def get_items(self) -> list[Any]:
        return self._items


async def test_seed_prompt_uses_input_when_present() -> None:
    stream = _stream({}, input_data="do the thing")
    assert await stream._seed_prompt() == "do the thing"


async def test_seed_prompt_refeeds_session_when_input_empty() -> None:
    session = _FakeSession(
        [
            {"role": "user", "content": "Your task: audit the login flow"},
            {"role": "assistant", "content": "Understood, starting."},
        ]
    )
    stream = _stream({}, session=session, input_data=[])
    prompt = await stream._seed_prompt()
    assert prompt != "Continue."
    assert "audit the login flow" in prompt


async def test_seed_prompt_falls_back_to_continue_without_session() -> None:
    stream = _stream({}, session=None, input_data=[])
    assert await stream._seed_prompt() == "Continue."
