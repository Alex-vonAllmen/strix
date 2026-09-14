"""The system prompt tells agents source lives under target.workspace_path (#18).

Agents were reading `/workspace/apps/web/...` (repo-at-root) and hitting
`WorkspaceReadNotFoundError`, though a local target is mounted at
`/workspace/<subdir>`. The prompt now states, forcefully, that source reads must
be resolved under each target's workspace path, not /workspace root.
"""

from __future__ import annotations

from strix.agents.prompt import render_system_prompt


def _ctx() -> dict[str, object]:
    return {
        "scope_source": "test",
        "authorization_source": "test",
        "authorized_targets": [
            {
                "type": "local_code",
                "value": "/x/bench-2ea29ee0",
                "workspace_path": "/workspace/bench-2ea29ee0",
            }
        ],
    }


def test_prompt_directs_reads_under_workspace_path() -> None:
    prompt = render_system_prompt(is_whitebox=True, system_prompt_context=_ctx())
    assert prompt, "prompt render returned empty (template failure)"
    # The target's workspace path is shown...
    assert "/workspace/bench-2ea29ee0" in prompt
    # ...and the new directive forbids the repo-at-root assumption.
    assert "NEVER at /workspace root" in prompt
    assert "Resolve every source read relative to that workspace path" in prompt
    assert "source files are NOT at /workspace root" in prompt


def test_prompt_renders_without_targets() -> None:
    # No authorized_targets → the targets block (and the new directive) is simply
    # absent; rendering must still succeed.
    prompt = render_system_prompt(system_prompt_context={})
    assert prompt
    assert "NEVER at /workspace root" not in prompt
