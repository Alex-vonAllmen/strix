"""claude-cli/ lane completion semantics (issue #9, M2).

- The lane ends the SDK run when a lifecycle/parking tool settles the agent (the lane's
  tool_use_behavior), instead of running to the CLI's max_turns.
- Empty-input cycles re-feed the session transcript so a fresh SDK client keeps its task
  (no bare "Continue.").
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

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
    coord.statuses["root"] = "waiting"  # parking tool
    assert stream._lifecycle_settled() is True


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


async def test_stream_interrupts_and_stops_when_settled(monkeypatch: Any) -> None:
    coord = SimpleNamespace(statuses={"root": "completed"})  # already settled
    stream = _stream({"coordinator": coord, "agent_id": "root"})

    created: list[_FakeClient] = []

    class _SDK:
        StreamEvent = type("StreamEvent", (), {})
        AssistantMessage = type("AssistantMessage", (), {})
        UserMessage = type("UserMessage", (), {})
        ResultMessage = type("ResultMessage", (), {})
        ClaudeAgentOptions = SimpleNamespace

        @staticmethod
        def ClaudeSDKClient(options: Any = None) -> _FakeClient:  # noqa: N802
            c = _FakeClient(options, messages=[object(), object(), object()])
            created.append(c)
            return c

    monkeypatch.setattr(dispatch, "_import_sdk", lambda: _SDK)
    async for _ in stream.stream_events():
        pass

    client = created[0]
    # It processed the first message, saw the agent had settled, interrupted and stopped —
    # it did not drain all three messages.
    assert client.interrupted == 1
    assert client.consumed == 1


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
