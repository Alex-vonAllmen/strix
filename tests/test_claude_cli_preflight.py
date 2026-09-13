"""Pre-scan preflight is claude-cli/ lane-aware (issue #12).

``preflight_model_connection`` runs before a scan starts. For a ``claude-cli/``
model it must NOT probe LiteLLM (StrixProvider cannot resolve the lane, so the
probe would abort every lane scan with "LLM Provider NOT provided"); it must
instead enforce the lane gate and return. Metered models still get the probe.
"""

from __future__ import annotations

import shutil
from typing import Any

import pytest

from strix.interface import scan_setup
from strix.llm import claude_cli


class _ExplodingProvider:
    """Any use of StrixProvider in the lane path is a bug — blow up loudly."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("StrixProvider must not be constructed for the claude-cli/ lane")


async def test_preflight_skips_litellm_for_claude_cli_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # C1: lane admissible (claude on PATH, no metered key) → returns without
    # ever touching StrixProvider / LiteLLM.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr("strix.config.models.StrixProvider", _ExplodingProvider)

    await scan_setup.preflight_model_connection("claude-cli/claude-opus-4-8")


async def test_preflight_enforces_lane_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    # C2: the lane gate still fires — a metered ANTHROPIC key alongside the
    # lane is refused, before (and instead of) any LiteLLM probe.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-live-xxxx")
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr("strix.config.models.StrixProvider", _ExplodingProvider)

    with pytest.raises(claude_cli.ClaudeCliLaneError) as excinfo:
        await scan_setup.preflight_model_connection("claude-cli/claude-opus-4-8")

    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


async def test_preflight_probes_litellm_for_metered_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # C3: a normal metered model still goes through the StrixProvider/LiteLLM
    # probe, unchanged.
    called: dict[str, Any] = {}

    class _FakeModel:
        async def get_response(self, **kwargs: Any) -> None:
            called["get_response"] = kwargs

    class _FakeProvider:
        def get_model(self, model_name: str | None) -> _FakeModel:
            called["get_model"] = model_name
            return _FakeModel()

    monkeypatch.setattr("strix.config.models.StrixProvider", _FakeProvider)
    monkeypatch.setattr("strix.config.models.configure_sdk_model_defaults", lambda _s: None)
    monkeypatch.setattr("strix.core.inputs.make_model_settings", lambda *_a, **_k: None)

    await scan_setup.preflight_model_connection("openrouter/z-ai/glm-5.3")

    assert called.get("get_model") == "openrouter/z-ai/glm-5.3"
    assert "get_response" in called
