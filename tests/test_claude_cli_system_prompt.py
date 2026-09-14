"""The claude-cli/ lane passes the system prompt as a file, not inline argv (#16).

A ``str`` system prompt is rendered by the SDK into ``--system-prompt <text>``;
a child agent's skill-loaded prompt exceeds Linux's per-arg limit
(``MAX_ARG_STRLEN``, 128 KiB) and ``execve`` fails with ``E2BIG``, crashing the
child before it starts. ``_system_prompt_option`` writes a non-empty prompt to a
temp file and returns a ``SystemPromptFile`` so the SDK uses
``--system-prompt-file`` instead.
"""

from __future__ import annotations

from pathlib import Path

from strix.llm.claude_cli.dispatch import _system_prompt_option


def test_none_and_empty_pass_through() -> None:
    # The SDK handles these inline (empty --system-prompt); no file needed.
    assert _system_prompt_option(None) == (None, None)
    assert _system_prompt_option("") == ("", None)


def test_string_prompt_is_written_to_a_file() -> None:
    # A prompt larger than MAX_ARG_STRLEN is exactly the crash case; it must not
    # be returned as a str (which the SDK would inline onto argv).
    prompt = "SYSTEM PROMPT\n" + ("x" * 200_000)
    option, path = _system_prompt_option(prompt)
    try:
        assert path is not None
        # SystemPromptFile shape the SDK dispatches to --system-prompt-file.
        assert option == {"type": "file", "path": path}
        assert not isinstance(option, str)
        assert Path(path).read_text(encoding="utf-8") == prompt
    finally:
        if path:
            Path(path).unlink()


def test_caller_can_unlink_and_helper_leaves_no_handle() -> None:
    _, path = _system_prompt_option("small prompt")
    assert path is not None and Path(path).exists()
    Path(path).unlink()  # caller owns cleanup; nothing else should hold the file
    assert not Path(path).exists()
