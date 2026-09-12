"""claude-cli/ lane one-time subscription/ToS notice (SC-8, AD-5).

The notice shows once and records acknowledgment in a flag file; a present flag
suppresses it; a corrupt flag is treated as not acknowledged and rewritten.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

import pytest

from strix.llm.claude_cli import notice


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def ack_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / ".strix" / "claude-cli-ack.json"
    monkeypatch.setattr(notice, "ACK_PATH", path)
    return path


class _RecordingConsole:
    instances: ClassVar[list[_RecordingConsole]] = []

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.prints = 0
        _RecordingConsole.instances.append(self)

    def print(self, *_args: object, **_kwargs: object) -> None:
        self.prints += 1


@pytest.fixture
def recording_console(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingConsole]:
    _RecordingConsole.instances = []
    monkeypatch.setattr(notice, "Console", _RecordingConsole)
    return _RecordingConsole


def _total_prints(console_cls: type[_RecordingConsole]) -> int:
    return sum(c.prints for c in console_cls.instances)


def test_first_use_shows_notice_and_writes_flag(
    ack_path: Path, recording_console: type[_RecordingConsole]
) -> None:
    assert notice.is_acknowledged() is False

    notice.show_notice_and_acknowledge()

    assert _total_prints(recording_console) == 1
    assert ack_path.exists()
    data = json.loads(ack_path.read_text(encoding="utf-8"))
    assert data["acknowledged"] is True
    assert data["lane"] == "claude-cli"
    assert "acknowledged_at" in data
    assert "strix_version" in data
    assert notice.is_acknowledged() is True


def test_second_use_shows_nothing(
    ack_path: Path, recording_console: type[_RecordingConsole]
) -> None:
    notice.show_notice_and_acknowledge()
    first_prints = _total_prints(recording_console)
    assert first_prints == 1
    assert ack_path.exists()

    notice.show_notice_and_acknowledge()
    # No additional print after acknowledgment.
    assert _total_prints(recording_console) == first_prints


def test_corrupt_flag_is_treated_as_unacknowledged_and_rewritten(
    ack_path: Path, recording_console: type[_RecordingConsole]
) -> None:
    ack_path.parent.mkdir(parents=True, exist_ok=True)
    ack_path.write_text("{ this is not valid json", encoding="utf-8")
    assert notice.is_acknowledged() is False

    notice.show_notice_and_acknowledge()

    assert _total_prints(recording_console) == 1
    assert notice.is_acknowledged() is True
    data = json.loads(ack_path.read_text(encoding="utf-8"))
    assert data["acknowledged"] is True


def test_wrong_shape_flag_is_not_acknowledged(ack_path: Path) -> None:
    ack_path.parent.mkdir(parents=True, exist_ok=True)
    # Valid JSON, wrong shape / value.
    ack_path.write_text(json.dumps({"acknowledged": "yes"}), encoding="utf-8")
    assert notice.is_acknowledged() is False
    ack_path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    assert notice.is_acknowledged() is False
