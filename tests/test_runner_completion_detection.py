"""Scan-completion detection is authoritative, not keyed on final_output (#29).

After a successful claude-cli scan the run's ``result.final_output`` is ``None``
even though ``finish_scan`` ran and persisted a report. The runner must decide
"did this scan finish?" from the persisted report state, so a clean finish is not
logged as ``ended without calling finish_scan`` — while a run that genuinely never
finishes still is.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from strix.core.runner import _scan_reported_complete


if TYPE_CHECKING:
    import pytest


def _install(monkeypatch: pytest.MonkeyPatch, report_state: Any) -> None:
    # runner.py imports the symbol directly, so patch it in that namespace.
    monkeypatch.setattr("strix.core.runner.get_global_report_state", lambda: report_state)


def _report_state(*, scan_results: Any = None, status: str | None = None) -> Any:
    return SimpleNamespace(scan_results=scan_results, run_record={"status": status})


def test_complete_when_finish_scan_persisted_results(monkeypatch: pytest.MonkeyPatch) -> None:
    # AC-1: finish_scan wrote scan_results.scan_completed=True → complete.
    _install(monkeypatch, _report_state(scan_results={"scan_completed": True}))
    assert _scan_reported_complete() is True


def test_complete_when_run_record_marked_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    # AC-3 fallback: run record marked completed even if scan_results is bare.
    _install(monkeypatch, _report_state(scan_results=None, status="completed"))
    assert _scan_reported_complete() is True


def test_incomplete_when_never_finished(monkeypatch: pytest.MonkeyPatch) -> None:
    # AC-2: a real text-only end — no persisted completion → still reported False,
    # so the runner keeps logging the ERROR for genuine failures.
    _install(monkeypatch, _report_state(scan_results=None, status="running"))
    assert _scan_reported_complete() is False


def test_incomplete_when_scan_results_not_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _report_state(scan_results={"scan_completed": False}, status="running"))
    assert _scan_reported_complete() is False


def test_incomplete_without_report_state(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, None)
    assert _scan_reported_complete() is False
