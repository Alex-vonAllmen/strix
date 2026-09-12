"""One-time subscription/ToS notice for the ``claude-cli/`` lane (AD-5).

On first lane use Strix shows a short notice about the subscription/ToS
implications of driving a Claude Pro/Max login through the ``claude`` CLI, then
records the acknowledgment in a flag file so the notice is not shown again.

The flag is not a secret (no credentials, unlike ``~/.strix/subscription-auth.json``),
so an ordinary file write is fine. A missing or corrupt flag is treated as *not
acknowledged*: the notice re-shows and the file is rewritten — a broken flag never
blocks a run.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from strix.telemetry._common import get_version


logger = logging.getLogger(__name__)

ACK_PATH = Path.home() / ".strix" / "claude-cli-ack.json"

_NOTICE_TEXT = (
    "You are running the claude-cli/ lane: Strix drives your Claude Pro/Max "
    "subscription login through the `claude` CLI instead of a metered API key. "
    "Automated, unattended use of a personal subscription through the CLI is "
    "outside Anthropic's official API surface — you are choosing this path "
    "knowingly. Only scan targets you are authorized to test.\n"
    f"This notice is shown once; acknowledgment is recorded in {ACK_PATH}."
)


def is_acknowledged() -> bool:
    """Whether the one-time notice has already been acknowledged.

    A missing, unreadable, corrupt, or wrongly-shaped flag file counts as not
    acknowledged.
    """
    try:
        data: Any = json.loads(ACK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and data.get("acknowledged") is True


def _write_ack() -> None:
    record = {
        "acknowledged": True,
        "lane": "claude-cli",
        "acknowledged_at": _dt.datetime.now().astimezone().isoformat(),
        "strix_version": get_version(),
    }
    try:
        ACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        ACK_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    except OSError:
        # A run must not die because the ack file could not be written; the
        # notice simply shows again next time.
        logger.warning("could not write claude-cli ack flag at %s", ACK_PATH, exc_info=True)


def show_notice_and_acknowledge() -> None:
    """Print the one-time notice (once) and record acknowledgment.

    Idempotent: if already acknowledged, does nothing.
    """
    if is_acknowledged():
        return

    try:
        Console(stderr=True).print(
            Panel(_NOTICE_TEXT, title="Claude subscription lane", border_style="yellow")
        )
    except Exception:  # noqa: BLE001 - a rendering problem must never block the scan.
        logger.info("claude-cli lane notice: %s", _NOTICE_TEXT)

    _write_ack()


__all__ = ["ACK_PATH", "is_acknowledged", "show_notice_and_acknowledge"]
