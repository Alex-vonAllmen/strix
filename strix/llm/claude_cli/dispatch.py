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

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
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


# Bounds on the transcript re-fed for cross-cycle continuity (issue #9, OQ-4).
_SEED_MAX_ITEMS = 40
_SEED_MAX_CHARS = 12_000

# Errored ResultMessages that are benign cycle ends, not fatal lane errors: the CLI
# hitting its per-run turn budget is the lane's analog of the default lane's
# MaxTurnsExceeded, so it must not fail fast (issue #9).
_BENIGN_TERMINAL_REASONS = frozenset({"max_turns"})
_BENIGN_RESULT_SUBTYPES = frozenset({"error_max_turns"})

# Coordinator statuses that mean the agent is DONE and its SDK cycle should end
# (finish_scan/agent_finish). Deliberately excludes "waiting"/"budget_paused",
# which are transient: a blocking parking tool resumes the same cycle (#24).
_TERMINAL_STATUSES = frozenset({"completed", "stopped", "crashed", "failed"})


def _is_benign_result(message: Any) -> bool:
    """Whether an errored ``ResultMessage`` is a normal turn-limit end, not a fatal error."""
    return (
        getattr(message, "subtype", None) in _BENIGN_RESULT_SUBTYPES
        or getattr(message, "terminal_reason", None) in _BENIGN_TERMINAL_REASONS
    )


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


# Tools that move a strix agent out of "running" — the only ones that end a
# turn the way the runner's recovery loop expects (finish_scan/agent_finish) or
# park it (wait_for_agents/respond_to_user). Used only for the stall diagnostic
# (#24); matched against the trailing segment of an MCP tool name.
_LIFECYCLE_TOOLS = frozenset({"finish_scan", "agent_finish", "wait_for_agents", "respond_to_user"})


def _is_lifecycle_tool(name: str) -> bool:
    """Whether ``name`` is a lifecycle/parking tool (e.g. ``mcp__strix__finish_scan``)."""
    return name.rsplit("__", 1)[-1] in _LIFECYCLE_TOOLS


def _system_prompt_option(instructions: Any) -> tuple[Any, str | None]:
    """Resolve the SDK ``system_prompt`` for an agent's instructions (#16).

    A ``str`` system prompt is rendered by the SDK transport into an inline
    ``--system-prompt <text>`` argv element. A child agent that loads skills
    produces a prompt larger than Linux's per-argument limit
    (``MAX_ARG_STRLEN``, 128 KiB), so ``execve`` of the bundled ``claude`` fails
    with ``E2BIG`` and the child crashes before it starts — the whole point of
    the multi-agent graph on this lane.

    Writing the prompt to a temp file and passing a ``SystemPromptFile``
    (``{"type": "file", "path": ...}``) makes the SDK use ``--system-prompt-file``
    instead, so the prompt never touches argv and the size ceiling disappears.

    Returns ``(system_prompt, temp_path)``; ``temp_path`` is non-None only when a
    file was written and must be unlinked by the caller after the run. ``None``
    and empty prompts pass through unchanged (the SDK handles those inline).
    """
    if isinstance(instructions, str) and instructions:
        fd, path = tempfile.mkstemp(prefix="strix-sysprompt-", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(instructions)
        except Exception:
            with contextlib.suppress(OSError):
                Path(path).unlink()
            raise
        return {"type": "file", "path": path}, path
    return instructions, None


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
        # Stall diagnostic (#24): tool names Claude Code called this cycle, so the
        # cycle-end log can report whether a lifecycle tool was ever issued.
        self._tools_seen: list[str] = []
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

    def _lifecycle_settled(self) -> bool:
        """Whether a *terminal* lifecycle tool has ended this agent.

        Ends the SDK cycle only for a terminal status — ``finish_scan`` /
        ``agent_finish`` (``completed``/``stopped``/``crashed``/``failed``).

        NOT ``waiting`` (#24): ``wait_for_agents`` is a *blocking* tool — it parks
        the agent (status ``waiting``), ``await``s ``wait_for_message`` inside the
        MCP call, then ``mark_running``s on wake and returns, so the SDK cycle
        must stay alive to resume it. Issue #9 originally ended the cycle on any
        non-``running`` status; that interrupted the blocking wait mid-flight,
        ``run_agent_loop`` returned on ``waiting``, and the root was abandoned
        with its children torn down. The default lane blocks-and-resumes
        ``wait_for_agents`` within one ``Runner`` run; matching that means
        leaving the parked cycle running until the tool itself resumes it.
        """
        coordinator = self._context.get("coordinator")
        agent_id = self._context.get("agent_id")
        if coordinator is None or not isinstance(agent_id, str):
            return False
        status = getattr(coordinator, "statuses", {}).get(agent_id)
        return status in _TERMINAL_STATUSES

    def _cycle_end_status(self) -> str:
        """The agent's coordinator status, for the #24 stall diagnostic."""
        coordinator = self._context.get("coordinator")
        agent_id = self._context.get("agent_id")
        if coordinator is None or not isinstance(agent_id, str):
            return "unknown"
        return str(getattr(coordinator, "statuses", {}).get(agent_id, "unknown"))

    async def _seed_prompt(self) -> str:
        """The cycle prompt: the cycle ``input`` if present, else a bounded transcript
        re-fed from the run's session, else ``"Continue."``.

        The lane builds a fresh SDK client per cycle, so an empty ``input`` (a recovery
        or continuation cycle) would otherwise reach the CLI as a bare ``"Continue."``
        with no task — which children hit first ("this session starts with just
        'Continue,'"). Re-feeding the session's recent turns restores context (issue #9,
        OQ-4).
        """
        prompt = _input_to_prompt(self._input)
        if prompt:
            return prompt
        if self._session is not None:
            seeded = await self._session_prompt()
            if seeded:
                return seeded
        return "Continue."

    async def _session_prompt(self) -> str:
        try:
            items = list(await self._session.get_items())  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 - continuity is best-effort; never block the run.
            logger.debug("could not read session items for claude-cli continuity", exc_info=True)
            return ""
        parts: list[str] = []
        for item in items[-_SEED_MAX_ITEMS:]:
            if not isinstance(item, dict):
                continue
            text = _text_from_content(item.get("content"))
            if not text:
                continue
            role = item.get("role")
            parts.append(f"{role}: {text}" if isinstance(role, str) else text)
        if not parts:
            return ""
        transcript = "\n\n".join(parts)
        if len(transcript) > _SEED_MAX_CHARS:
            transcript = transcript[-_SEED_MAX_CHARS:]
        return (
            "Continue your task using the tools. Here is the recent context of this "
            f"run:\n\n{transcript}"
        )

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

        prompt = await self._seed_prompt()

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

        # #16: pass the system prompt as a file, never inline argv — see
        # _system_prompt_option. prompt_file_path is unlinked in the finally.
        system_prompt, prompt_file_path = _system_prompt_option(
            getattr(self._agent, "instructions", None)
        )

        options = sdk.ClaudeAgentOptions(
            system_prompt=system_prompt,
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
            interrupted = False
            async for message in client.receive_response():
                async for event in self._translate(sdk, message):
                    yield event
                # The ResultMessage is the last message of a turn and carries the
                # cumulative usage + final_output (recorded in _translate). Stop on
                # it — whether the turn ended naturally, on max_turns, or via the
                # interrupt below. This is the ONLY exit, so usage/final_output are
                # never lost (#28).
                if isinstance(message, sdk.ResultMessage):
                    break
                # The lane's tool_use_behavior (issue #9): after a terminal lifecycle
                # tool (finish_scan/agent_finish) Claude Code would otherwise run on
                # to the CLI's max_turns. Interrupt to stop it — but do NOT break
                # here: keep draining until the ResultMessage the interrupt produces
                # (terminal_reason=aborted_streaming), so its usage is recorded
                # instead of thrown away (#28). Interrupt once.
                if not interrupted and self._lifecycle_settled():
                    interrupted = True
                    logger.info(
                        "[#24-diag] lane settled status=%s agent=%s; interrupting, "
                        "draining to ResultMessage",
                        self._cycle_end_status(),
                        self._context.get("agent_id"),
                    )
                    with contextlib.suppress(Exception):
                        await client.interrupt()
        finally:
            # Deterministic teardown: never leave an orphaned `claude` subprocess.
            try:
                await client.disconnect()
            except Exception:  # noqa: BLE001 - teardown must not mask a real error.
                logger.debug("claude-cli client disconnect failed", exc_info=True)
            # The claude process read --system-prompt-file at startup; the temp
            # file is safe to remove now (#16).
            if prompt_file_path is not None:
                with contextlib.suppress(OSError):
                    Path(prompt_file_path).unlink()

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
            # Fail fast on a genuinely fatal CLI result (unknown/inaccessible model,
            # auth) — a clear terminal error, not a tool-call-free turn the recovery
            # loop would burn into a misleading MaxTurnsExceeded (issue #5). But a
            # benign turn-limit result (`error_max_turns`) is the lane's analog of the
            # default lane's MaxTurnsExceeded, not a fatal error, so it must end the
            # cycle normally and let the recovery/lifecycle loop proceed (issue #9).
            is_error = getattr(message, "is_error", False)
            if (is_error or self._assistant_error) and not _is_benign_result(message):
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
            # Stall diagnostic (#24): why did this cycle end, and did the model
            # ever call a lifecycle tool? A cycle that ends with no lifecycle
            # tool and status still "running" is what the runner's recovery loop
            # burns through into "ended without finish_scan".
            lifecycle_seen = [t for t in self._tools_seen if _is_lifecycle_tool(t)]
            logger.info(
                "claude-cli cycle end: subtype=%s terminal_reason=%s num_turns=%s "
                "is_error=%s tools=%d lifecycle_tools=%s final_output=%s end_status=%s",
                getattr(message, "subtype", None),
                getattr(message, "terminal_reason", None),
                getattr(message, "num_turns", None),
                getattr(message, "is_error", None),
                len(self._tools_seen),
                lifecycle_seen or "none",
                "present" if self._final_output else "empty",
                self._cycle_end_status(),
            )
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
                self._tools_seen.append(block.name)
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
