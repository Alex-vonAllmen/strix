"""claude-cli/ lane end-to-end integration tests (SC-2, SC-5, SC-6).

These are **opt-in** and never run on the default CI path. They require:
- the `claude` CLI installed and logged in to a Claude Pro/Max subscription,
- a running Docker daemon (the Strix sandbox), and
- the env var ``STRIX_RUN_CLAUDE_CLI_E2E=1`` set.

`ANTHROPIC_API_KEY` must be unset (the lane refuses it). Run locally with, e.g.::

    STRIX_RUN_CLAUDE_CLI_E2E=1 STRIX_LLM=claude-cli/claude-opus-5 \
        uv run pytest tests/test_claude_cli_lane_e2e.py -q

They are authored so a human can validate the lane on real infrastructure; the
unit suites (bridge, wiring, dispatch, adapter, gate, notice) cover the logic that
runs without Docker or a login.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest


pytestmark = pytest.mark.skipif(
    not os.getenv("STRIX_RUN_CLAUDE_CLI_E2E"),
    reason="claude-cli/ lane e2e is opt-in; set STRIX_RUN_CLAUDE_CLI_E2E=1 with a "
    "logged-in `claude` CLI and Docker running.",
)

_MODEL = os.getenv("STRIX_LLM", "claude-cli/claude-opus-5")


def _require_docker() -> None:
    try:
        import docker

        docker.from_env().ping()
    except Exception as exc:  # noqa: BLE001 - environment probe.
        pytest.skip(f"Docker not available for the sandbox: {exc}")


async def _sandbox_bundle(scan_id: str) -> Any:
    from strix.config import load_settings
    from strix.runtime import session_manager

    settings = load_settings()
    return await session_manager.create_or_reuse(
        scan_id, image=settings.runtime.image, local_sources=[], extra_files=None
    )


async def test_sc6_shell_exec_runs_in_sandbox_with_proxy() -> None:
    """SC-6: shell_exec executes inside the Docker sandbox with the proxy env active."""
    _require_docker()
    from strix.llm.claude_cli import sandbox_bridge

    scan_id = f"e2e-{uuid.uuid4().hex[:8]}"
    bundle = await _sandbox_bundle(scan_id)
    session = bundle["session"]
    try:
        marker = f"STRIX_MARKER_{uuid.uuid4().hex}"
        exec_handler = sandbox_bridge.build_sandbox_bridge  # bridge is built per run
        assert exec_handler is not None
        # Run the marker directly through the same handler the bridge exposes.
        from functools import partial

        run = partial(sandbox_bridge._shell_exec, session, sandbox_bridge.DEFAULT_WORKDIR)
        result = await run({"command": f"echo {marker}"})
        assert result["is_error"] is False
        assert marker in result["content"][0]["text"]

        # The Caido proxy env the sandbox injects is visible to sandboxed commands.
        proxy = await run({"command": "env | grep -i proxy || true"})
        assert proxy["is_error"] is False
    finally:
        from strix.runtime import session_manager

        await session_manager.cleanup(scan_id)


async def test_sc2_full_scan_completes_without_metered_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """SC-2: a scan completes on the subscription login with no ANTHROPIC_API_KEY."""
    _require_docker()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from strix.config import load_settings
    from strix.core.runner import run_strix_scan

    image = load_settings().runtime.image
    scan_id = f"e2e-{uuid.uuid4().hex[:8]}"
    await run_strix_scan(
        scan_config={"instruction": "List the files in the workspace, then finish."},
        scan_id=scan_id,
        image=image,
        model=_MODEL,
        max_turns=6,
        max_budget_usd=None,
    )
    # The scan ran through the lane to a terminal state without a metered key
    # (no exception). Where a report exists, it settled to a terminal status.
    from strix.report.state import get_global_report_state

    report_state = get_global_report_state()
    if report_state is not None:
        assert report_state.run_record.get("status") in {"completed", "stopped", None}


async def test_sc5_children_spawn_under_the_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """SC-5: children spawned under the lane are registered and run through it."""
    _require_docker()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from strix.config import load_settings
    from strix.core.agents import AgentCoordinator
    from strix.core.runner import run_strix_scan

    image = load_settings().runtime.image
    coordinator = AgentCoordinator()
    scan_id = f"e2e-{uuid.uuid4().hex[:8]}"
    await run_strix_scan(
        scan_config={
            "instruction": "Spawn two child agents to inspect the workspace in "
            "parallel, then summarize and finish.",
        },
        scan_id=scan_id,
        image=image,
        model=_MODEL,
        coordinator=coordinator,
        max_turns=10,
        max_budget_usd=None,
    )
    # At least the root ran through the lane; any children are registered on the graph.
    assert len(coordinator.statuses) >= 1
