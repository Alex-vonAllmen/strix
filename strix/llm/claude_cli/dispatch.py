"""``ClaudeCliStream`` — the ``claude-cli/`` lane's per-cycle executor (AD-1, AD-7).

This is the only module that imports ``claude_agent_sdk`` (lazily, in
:func:`_import_sdk`). It drives one ``ClaudeSDKClient`` agent loop for one Strix
run cycle and translates the SDK's message stream into the same
``openai``/Agents-SDK stream events the default ``Runner.run_streamed`` emits, so
``_run_cycle``, the event sink and the TUI consume it unchanged. It also feeds
token usage through the existing run hooks so cost accounting and budget
pause/stop fire exactly as on the default lane.

The class exposes exactly the streamed-run interface subset ``_run_cycle``
consumes (the inter-milestone seam): ``stream_events()``, ``run_loop_exception``,
``final_output``, ``to_input_list()`` (plus ``new_items`` for the refusal probe).

M1 scope: the SDK client is created fresh per cycle and seeded from ``input``
(re-feeding); sandbox shell/fs tools, multi-agent spawn semantics and full
session-store continuity are M2. On an SDK error ``stream_events()`` raises and
``_run_cycle``'s existing classification marks the agent failed/crashed.

The ``agents``/``openai`` imports below are core Strix dependencies (always
installed); only ``claude_agent_sdk`` is optional-at-runtime and stays lazy so the
default lane never loads it (AD-6). This module itself is imported lazily from
``strix.core.execution`` only when the lane is selected.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from agents.items import (
    MessageOutputItem,
    ModelResponse,
    ToolCallItem,
    ToolCallOutputItem,
)
from agents.run_context import RunContextWrapper
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from strix.llm.claude_cli import ClaudeCliLaneError, claude_cli_lane_model
from strix.llm.claude_cli.sandbox_bridge import build_sandbox_bridge, sandbox_tool_names
from strix.llm.claude_cli.server import build_tool_server, exposed_tool_names


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from agents.lifecycle import RunHooks
    from agents.memory import Session


logger = logging.getLogger(__name__)


def _import_sdk() -> Any:
    """Import ``claude_agent_sdk`` lazily; wrap failure as a terminal lane error."""
    try:
        import claude_agent_sdk  # noqa: PLC0415 - lazy by design (AD-6).
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ClaudeCliLaneError(
            "The claude-cli/ lane needs the `claude-agent-sdk` package, which is not "
            f"importable: {exc}. Reinstall Strix so the dependency is present."
        ) from exc
    return claude_agent_sdk


def _text_from_content(content: Any) -> str:
    """Best-effort plain text from a message ``content`` (str or list of items)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in (None, "text", "input_text", "output_text"):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p)
    return ""


def _input_to_prompt(input_data: Any) -> str:
    """Turn Strix's run-cycle ``input`` into a single prompt string for the SDK."""
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, list):
        parts: list[str] = []
        for item in input_data:
            if isinstance(item, dict):
                text = _text_from_content(item.get("content"))
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n\n".join(parts)
    return ""


# Text markers the `claude` CLI uses when the selected model is unknown/inaccessible.
_MODEL_ERROR_MARKERS = (
    "issue with the selected model",
    "may not exist or you may not have access",
)


def _looks_like_model_error(assistant_error: str | None, text: str) -> bool:
    if assistant_error and "model" in assistant_error.lower():
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in _MODEL_ERROR_MARKERS)


def _model_error_message(
    result_text: Any, assistant_error: str | None, model_slug: str | None
) -> str:
    """Build the terminal error text from the CLI's own message.

    Adds a hyphen-vs-dot hint when the failure is a model-not-found near-miss and the
    configured slug contains a ``.`` (the CLI's model ids use hyphens, e.g.
    ``claude-opus-4-8``, not ``claude-opus-4.8``).
    """
    base = (
        result_text.strip()
        if isinstance(result_text, str) and result_text.strip()
        else "the claude CLI returned an error"
    )
    message = f"claude-cli lane: {base}"
    if model_slug and "." in model_slug and _looks_like_model_error(assistant_error, base):
        suggestion = model_slug.replace(".", "-")
        message += (
            f" Did you mean 'claude-cli/{suggestion}'? "
            "The claude CLI's model ids use hyphens (e.g. claude-opus-4-8), not dots."
        )
    return message


class ClaudeCliStream:
    """Duck-typed replacement for a streamed ``RunResult`` on the ``claude-cli/`` lane.

    Consumes exactly the interface ``_run_cycle`` uses; every SDK shape stays
    inside this class (AD-7).
    """

    def __init__(
        self,
        agent: Any,
        *,
        input: Any,  # noqa: A002 - mirrors Runner.run_streamed's keyword name.
        run_config: Any,
        context: dict[str, Any],
        max_turns: int,
        session: Session | None,
        hooks: RunHooks[dict[str, Any]] | None,
    ) -> None:
        self._agent = agent
        self._input = input
        self._run_config = run_config
        self._context = context if isinstance(context, dict) else {}
        self._max_turns = max_turns
        self._session = session
        self._hooks = hooks

        model = getattr(run_config, "model", None)
        self._model_slug = claude_cli_lane_model(model if isinstance(model, str) else None)

        self._final_output: str | None = None
        self._transcript: list[dict[str, Any]] = []
        self._new_items: list[Any] = []
        self._run_loop_exception: BaseException | None = None
        self._sequence = 0
        # An error code the CLI attached to an assistant message this cycle (e.g.
        # "model_not_found"); remembered until the ResultMessage so the lane can fail
        # fast with a clear message instead of returning a tool-call-free turn that the
        # recovery loop would burn through (issue #5).
        self._assistant_error: str | None = None

    # -- Inter-milestone seam (duck-type contract consumed by _run_cycle) --------

    @property
    def run_loop_exception(self) -> BaseException | None:
        return self._run_loop_exception

    @property
    def final_output(self) -> str | None:
        return self._final_output

    @property
    def new_items(self) -> list[Any]:
        return self._new_items

    def to_input_list(self) -> list[Any]:
        """The SDK transcript captured so far, as response-input items (salvage)."""
        return list(self._transcript)

    def _sandbox_session(self) -> Any:
        """The run's sandbox session, for the M2 bridge.

        Prefer ``run_config.sandbox.session`` (present for root and children — they
        share one ``run_config``); fall back to ``context["sandbox_session"]``.
        Returns None when no sandbox is wired (e.g. a lane unit test), in which case
        the bridge is simply not attached.
        """
        sandbox = getattr(self._run_config, "sandbox", None)
        session = getattr(sandbox, "session", None)
        if session is not None:
            return session
        return self._context.get("sandbox_session")

    async def stream_events(self) -> AsyncIterator[Any]:
        sdk = _import_sdk()

        prompt = _input_to_prompt(self._input)
        if not prompt:
            prompt = "Continue."

        tools = list(getattr(self._agent, "tools", []) or [])
        mcp_servers: dict[str, Any] = {
            "strix": build_tool_server(tools, self._context, run_config=self._run_config)
        }
        allowed = exposed_tool_names(tools)

        # Sandbox bridge (M2, AD-3): shell/fs tools over the run's sandbox session,
        # replacing the OpenAI-SDK Shell/Filesystem capabilities that Claude Code
        # lacks. Every agent (root + children) shares the run's session, so the
        # bridge routes into the same Docker sandbox with the Caido proxy active.
        sandbox_session = self._sandbox_session()
        if sandbox_session is not None:
            mcp_servers["strix-sandbox"] = build_sandbox_bridge(sandbox_session)
            allowed = [*allowed, *sandbox_tool_names()]

        options = sdk.ClaudeAgentOptions(
            system_prompt=getattr(self._agent, "instructions", None),
            model=self._model_slug,
            mcp_servers=mcp_servers,
            allowed_tools=allowed,
            # Invariant I-1/I-2: no built-in host tools (Bash, Read, Web*, Task, ...);
            # only the mcp__strix__* host tools and mcp__strix-sandbox__* sandbox tools
            # are reachable, so the sandbox and proxy guarantees hold.
            tools=[],
            strict_mcp_config=True,
            permission_mode="bypassPermissions",
            max_turns=self._max_turns,
        )

        self._transcript.append({"role": "user", "content": prompt})

        client = sdk.ClaudeSDKClient(options=options)
        await client.connect()
        try:
            await client.query(prompt)
            async for message in client.receive_response():
                async for event in self._translate(sdk, message):
                    yield event
        finally:
            # Deterministic teardown: never leave an orphaned `claude` subprocess.
            try:
                await client.disconnect()
            except Exception:  # noqa: BLE001 - teardown must not mask a real error.
                logger.debug("claude-cli client disconnect failed", exc_info=True)

    # -- Translation -------------------------------------------------------------

    async def _translate(self, sdk: Any, message: Any) -> AsyncIterator[Any]:
        if isinstance(message, sdk.StreamEvent):
            event = self._raw_text_delta(message)
            if event is not None:
                yield event
            return

        if isinstance(message, sdk.AssistantMessage):
            error = getattr(message, "error", None)
            if error:
                self._assistant_error = str(error)
            for item in self._translate_assistant(sdk, message):
                yield item
            return

        if isinstance(message, sdk.UserMessage):
            for item in self._translate_tool_results(sdk, message):
                yield item
            return

        if isinstance(message, sdk.ResultMessage):
            # Fail fast on an errored CLI result (e.g. an unknown/inaccessible model):
            # raise a clear terminal error instead of settling a tool-call-free turn
            # that the lifecycle-recovery loop would burn through into a misleading
            # MaxTurnsExceeded (issue #5). `is_error` is authoritative — `subtype` is
            # "success" even on error — with the assistant error code as a backstop.
            if getattr(message, "is_error", False) or self._assistant_error:
                raise ClaudeCliLaneError(
                    _model_error_message(
                        getattr(message, "result", None),
                        self._assistant_error,
                        self._model_slug,
                    )
                )
            await self._record_usage(message)
            result_text = getattr(message, "result", None)
            if isinstance(result_text, str) and result_text:
                self._final_output = result_text
            return

    def _next_seq(self) -> int:
        self._sequence += 1
        return self._sequence

    def _raw_text_delta(self, message: Any) -> Any | None:
        raw = getattr(message, "event", None)
        if not isinstance(raw, dict) or raw.get("type") != "content_block_delta":
            return None
        delta = raw.get("delta")
        if not isinstance(delta, dict) or delta.get("type") != "text_delta":
            return None
        text = delta.get("text")
        if not isinstance(text, str) or not text:
            return None
        data = ResponseTextDeltaEvent(
            content_index=int(raw.get("index", 0) or 0),
            delta=text,
            item_id=str(getattr(message, "uuid", "") or ""),
            logprobs=[],
            output_index=0,
            sequence_number=self._next_seq(),
            type="response.output_text.delta",
        )
        return RawResponsesStreamEvent(data=data)

    def _translate_assistant(self, sdk: Any, message: Any) -> list[Any]:
        events: list[Any] = []
        item: Any
        for block in getattr(message, "content", []) or []:
            if isinstance(block, sdk.TextBlock):
                text = block.text or ""
                if not text:
                    continue
                raw_item = ResponseOutputMessage(
                    id=str(getattr(message, "message_id", "") or f"msg_{self._next_seq()}"),
                    content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
                    role="assistant",
                    status="completed",
                    type="message",
                )
                item = MessageOutputItem(agent=self._agent, raw_item=raw_item)
                self._new_items.append(item)
                self._transcript.append({"role": "assistant", "content": text})
                events.append(RunItemStreamEvent(name="message_output_created", item=item))
            elif isinstance(block, sdk.ToolUseBlock):
                arguments = json.dumps(block.input or {})
                raw_call = ResponseFunctionToolCall(
                    arguments=arguments,
                    call_id=block.id,
                    name=block.name,
                    type="function_call",
                    id=block.id,
                    status="completed",
                )
                item = ToolCallItem(agent=self._agent, raw_item=raw_call)
                self._new_items.append(item)
                self._transcript.append(
                    {
                        "type": "function_call",
                        "call_id": block.id,
                        "name": block.name,
                        "arguments": arguments,
                    }
                )
                events.append(RunItemStreamEvent(name="tool_called", item=item))
        return events

    def _translate_tool_results(self, sdk: Any, message: Any) -> list[Any]:
        events: list[Any] = []
        for block in getattr(message, "content", []) or []:
            if not isinstance(block, sdk.ToolResultBlock):
                continue
            output_text = _text_from_content(block.content)
            raw_item: dict[str, Any] = {
                "call_id": block.tool_use_id,
                "output": output_text,
                "type": "function_call_output",
            }
            item = ToolCallOutputItem(agent=self._agent, raw_item=raw_item, output=output_text)
            self._new_items.append(item)
            self._transcript.append(raw_item)
            events.append(RunItemStreamEvent(name="tool_output", item=item))
        return events

    # -- Usage / budget ----------------------------------------------------------

    async def _record_usage(self, message: Any) -> None:
        """Feed the run's token usage through the existing run hooks (SC-7).

        Reuses ``ReportUsageHooks.on_llm_end``, so cost accrual and the budget
        pause/stop exceptions are raised from the same code path as the default
        lane — no lane-specific budget branch. Budget exceptions raised here
        propagate out of ``stream_events()`` and are caught by ``_run_cycle``'s
        existing budget handlers.
        """
        if self._hooks is None:
            return
        usage = self._usage_from_sdk(getattr(message, "usage", None), message)
        if usage is None:
            return
        on_llm_end = getattr(self._hooks, "on_llm_end", None)
        if on_llm_end is None:
            return

        response = ModelResponse(output=[], usage=usage, response_id=None)
        wrapper: RunContextWrapper[dict[str, Any]] = RunContextWrapper(context=self._context)
        await on_llm_end(wrapper, self._agent, response)

    def _usage_from_sdk(self, usage_dict: Any, message: Any) -> Usage | None:
        if not isinstance(usage_dict, dict):
            return None
        input_tokens = int(usage_dict.get("input_tokens", 0) or 0)
        output_tokens = int(usage_dict.get("output_tokens", 0) or 0)
        cache_read = int(usage_dict.get("cache_read_input_tokens", 0) or 0)
        cache_write = int(usage_dict.get("cache_creation_input_tokens", 0) or 0)
        if input_tokens == 0 and output_tokens == 0 and cache_read == 0 and cache_write == 0:
            return None
        requests = int(getattr(message, "num_turns", 0) or 0) or 1
        return Usage(
            requests=requests,
            input_tokens=input_tokens + cache_read + cache_write,
            input_tokens_details=InputTokensDetails(
                cached_tokens=cache_read, cache_write_tokens=cache_write
            ),
            output_tokens=output_tokens,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=input_tokens + cache_read + cache_write + output_tokens,
        )


__all__ = ["ClaudeCliStream"]
