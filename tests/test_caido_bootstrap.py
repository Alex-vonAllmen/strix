"""A bootstrap that dies mid-setup must not leave its transport behind.

The bootstrap now runs concurrently with the scan start, so teardown can
cancel it at any await — including inside ``Client.connect()``, where the
client exists but no caller will ever see it to close it.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest

from strix.runtime import caido_bootstrap
from strix.runtime.caido_bootstrap import bootstrap_caido


class _FakeExecResult:
    stderr = b""
    exit_code = 0

    def __init__(self, stdout: str) -> None:
        self.stdout = stdout

    def ok(self) -> bool:
        return True


class _FakeSession:
    async def exec(self, *_args: Any, **_kwargs: Any) -> _FakeExecResult:
        return _FakeExecResult('{"data":{"loginAsGuest":{"token":{"accessToken":"t"}}}}')


class _FakeClient:
    def __init__(self, connect_error: BaseException) -> None:
        self.connect_error = connect_error
        self.closed = False

    async def connect(self) -> None:
        raise self.connect_error

    async def aclose(self) -> None:
        self.closed = True


async def _bootstrap_expecting(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> _FakeClient:
    """Run a bootstrap whose ``connect()`` fails with ``error``."""
    client = _FakeClient(error)
    # The SDK is imported inside bootstrap_caido (it is slow to import), so the
    # fakes are injected as the modules it imports.
    sdk = types.ModuleType("caido_sdk_client")
    sdk.Client = lambda *_a, **_k: client  # type: ignore[attr-defined]
    sdk.TokenAuthOptions = lambda token: token  # type: ignore[attr-defined]
    sdk_types = types.ModuleType("caido_sdk_client.types")
    sdk_types.CreateProjectOptions = lambda **_k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "caido_sdk_client", sdk)
    monkeypatch.setitem(sys.modules, "caido_sdk_client.types", sdk_types)

    with pytest.raises(type(error)):
        await bootstrap_caido(
            _FakeSession(),  # type: ignore[arg-type]
            host_url="http://host",
            container_url="http://container",
        )
    return client


async def test_cancellation_during_connect_closes_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = await _bootstrap_expecting(monkeypatch, asyncio.CancelledError())
    assert client.closed


async def test_failed_connect_closes_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = await _bootstrap_expecting(monkeypatch, RuntimeError("no listener"))
    assert client.closed


# --- #17: a login failure surfaces container-side diagnostics -----------------


class _FailingExecResult:
    """A failed curl (login) or a diagnostics dump, depending on the command."""

    def __init__(self, *, stdout: str = "", stderr: bytes = b"", exit_code: int = 7) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code

    def ok(self) -> bool:
        return self.exit_code == 0


class _DiagnosingSession:
    """Login always fails (connection refused); the diagnostics probe returns a log."""

    DIAG = (
        "--- caido process ---\nno caido-cli process running\n"
        "--- /tmp/caido_startup.log (tail) ---\nboom"
    )

    async def exec(self, *args: Any, **_kwargs: Any) -> _FailingExecResult:
        if any("graphql" in str(a) for a in args):
            return _FailingExecResult(stderr=b"curl: (7) Failed to connect", exit_code=7)
        return _FailingExecResult(stdout=self.DIAG, exit_code=0)


async def test_login_failure_includes_container_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    # Don't actually sleep through the backoff.
    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(caido_bootstrap.asyncio, "sleep", _no_sleep)

    with pytest.raises(RuntimeError) as excinfo:
        await caido_bootstrap._login_as_guest(
            _DiagnosingSession(),  # type: ignore[arg-type]
            container_url="http://container",
            attempts=2,
        )

    message = str(excinfo.value)
    assert "loginAsGuest failed after 2 attempts" in message
    # The opaque failure now carries WHY: process liveness + the startup log.
    assert "no caido-cli process running" in message
    assert "caido_startup.log" in message


async def test_diagnostics_never_raise() -> None:

    class _BrokenSession:
        async def exec(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("exec died")

    out = await caido_bootstrap._caido_failure_diagnostics(_BrokenSession())  # type: ignore[arg-type]
    assert "could not collect Caido diagnostics" in out
