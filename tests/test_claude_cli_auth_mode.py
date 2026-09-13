"""claude-cli/ lane counts as a subscription for cost accounting (issue #14).

A ``claude-cli/`` run authenticates with a Pro/Max subscription, not a metered
key, so it must be zero-cost: otherwise the litellm estimate accrues and a USD
``--max-budget`` stops a subscription scan early. ``codex.auth_mode`` is the one
function every call site uses to decide subscription-vs-metered, and
``report.state`` turns ``auth_mode == "subscription"`` into
``LLMUsageLedger.zero_cost``.
"""

from __future__ import annotations

from agents.usage import Usage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from strix.config import codex
from strix.report.usage import LLMUsageLedger


def test_auth_mode_subscription_for_claude_cli() -> None:
    assert codex.auth_mode("claude-cli/claude-opus-4-8") == "subscription"
    # Case-insensitive on the prefix, like the lane's own parsing.
    assert codex.auth_mode("Claude-CLI/claude-opus-4-8") == "subscription"


def test_auth_mode_metered_for_openrouter() -> None:
    assert codex.auth_mode("openrouter/z-ai/glm-5.3") == "api_key"
    assert codex.auth_mode(None) == "api_key"


def test_auth_mode_subscription_for_chatgpt_unchanged() -> None:
    # The existing Codex/ChatGPT subscription lane must keep working.
    assert codex.auth_mode("chatgpt/gpt-5") == "subscription"


def _usage(inp: int, out: int) -> Usage:
    return Usage(
        requests=1,
        input_tokens=inp,
        input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
        output_tokens=out,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=inp + out,
    )


def test_zero_cost_ledger_tracks_tokens_but_bills_nothing() -> None:
    # What state.py does for a subscription run: zero_cost = True. Tokens are
    # still recorded, but cost stays $0, so the budget hook never trips (C3).
    ledger = LLMUsageLedger()
    ledger.zero_cost = True
    ledger.record(agent_id="root", usage=_usage(1_000_000, 10_000), model="claude-opus-4-8")
    ledger.record_observed_cost(5.0)

    assert ledger.total_cost == 0.0
    record = ledger.to_record()
    assert record["cost"] == 0.0
    assert record["total_tokens"] == 1_010_000
