"""Signal-safe lifecycle guard for isolated OCR subprocess groups.

OCR providers and probes run in private sessions so timeout cleanup can target
their entire process groups. That isolation also means a SIGTERM, SIGHUP, or
SIGINT sent only to the CLI would otherwise leave those groups behind. The
guard records signals passively, lets explicit checkpoints unwind into an
unconditional cleanup, and only then restores and replays prior behavior.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType
from typing import Any


class IsolatedSubprocessLifecycleError(RuntimeError):
    """Raised when an isolated subprocess cannot be guarded safely."""


class IsolatedSubprocessInterrupted(BaseException):
    """A prior returning signal handler interrupted a subprocess operation."""

    def __init__(self, signum: int) -> None:
        super().__init__(
            "isolated subprocess operation interrupted by "
            f"{signal.Signals(signum).name}"
        )
        self.signum = signum


class _PendingSignal(BaseException):
    """Internal checkpoint unwind for one passively recorded signal."""

    def __init__(
        self,
        signum: int,
        frame: FrameType | None,
        owner: object,
    ) -> None:
        super().__init__(signum)
        self.signum = signum
        self.frame = frame
        self.owner = owner


def _managed_signals() -> tuple[int, ...]:
    values = [signal.SIGTERM]
    sighup = getattr(signal, "SIGHUP", None)
    if isinstance(sighup, int):
        values.append(sighup)
    values.append(signal.SIGINT)
    return tuple(values)


def _restore_handlers(
    installed: list[int],
    previous: dict[int, Any],
) -> None:
    first_error: BaseException | None = None
    for signum in reversed(installed):
        try:
            signal.signal(signum, previous[signum])
        except BaseException as exc:  # pragma: no cover - defensive restore
            if first_error is None:
                first_error = exc
    if first_error is not None:  # pragma: no cover - signal API invariant
        raise first_error


def _replay_previous_handler(
    interrupted: _PendingSignal,
    previous: Any,
) -> None:
    if previous == signal.SIG_DFL:
        try:
            os.kill(os.getpid(), interrupted.signum)
        except OSError as exc:  # pragma: no cover - self-signal invariant
            raise SystemExit(128 + interrupted.signum) from exc
        # The default action should terminate the process synchronously.  Keep
        # a conventional signal-derived exit status as a defensive fallback.
        raise SystemExit(128 + interrupted.signum)  # pragma: no cover
    if callable(previous):
        previous(interrupted.signum, interrupted.frame)
        # A returning user handler cannot resume the operation after its stack
        # was unwound for cleanup. Surface a documented public interruption
        # instead of leaking the private checkpoint exception.
        raise IsolatedSubprocessInterrupted(interrupted.signum) from None
    raise IsolatedSubprocessLifecycleError(
        "unsupported previous termination-signal handler"
    )


class _IsolatedSubprocessSignalGuard:
    _MAX_WAIT_POLL_SECONDS = 0.1
    _active_guard: _IsolatedSubprocessSignalGuard | None = None

    def __init__(self) -> None:
        self._managed = _managed_signals()
        self._owner = object()
        self._pending: _PendingSignal | None = None
        self._previous: dict[int, Any] = {}
        self._installed: list[int] = []
        self._entered = False
        self._exiting = False
        self._spawning = False
        self._cleaning = False

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        # Never raise asynchronously. The first signal wins; later
        # TERM/HUP/INT events are coalesced so cleanup is not interrupted.
        if self._pending is None:
            self._pending = _PendingSignal(
                signum,
                frame,
                self._owner,
            )

    def _restore(self) -> None:
        self._exiting = True
        try:
            _restore_handlers(self._installed, self._previous)
        finally:
            self._installed.clear()
            self._entered = False
            if type(self)._active_guard is self:
                type(self)._active_guard = None

    def __enter__(self) -> _IsolatedSubprocessSignalGuard:
        if threading.current_thread() is not threading.main_thread():
            raise IsolatedSubprocessLifecycleError(
                "isolated subprocesses must be launched from the main thread"
            )
        if type(self)._active_guard is not None:
            raise IsolatedSubprocessLifecycleError(
                "isolated subprocess signal guards cannot be nested"
            )
        try:
            for signum in self._managed:
                self._previous[signum] = signal.getsignal(signum)
                if self._previous[signum] == signal.SIG_IGN:
                    continue
                signal.signal(signum, self._handle)
                self._installed.append(signum)
        except BaseException:
            self._restore()
            if self._pending is not None:
                _replay_previous_handler(
                    self._pending,
                    self._previous[self._pending.signum],
                )
            raise
        type(self)._active_guard = self
        self._entered = True
        return self

    def checkpoint(self) -> None:
        """Raise a recorded signal only at a caller-declared safe point."""

        if not self._entered or self._exiting:
            raise IsolatedSubprocessLifecycleError(
                "signal checkpoint requires an active safe lifecycle point"
            )
        if self._spawning or self._cleaning:
            return
        if self._pending is not None:
            raise self._pending

    def wait(
        self,
        process: subprocess.Popen,
        *,
        timeout_seconds: float,
    ) -> int:
        """Wait with signal checkpoints no more than 0.1 seconds apart."""

        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or timeout_seconds < 0
        ):
            raise IsolatedSubprocessLifecycleError(
                "timeout_seconds must be a finite nonnegative number"
            )
        try:
            timeout_is_finite = math.isfinite(timeout_seconds)
        except OverflowError:
            timeout_is_finite = False
        if not timeout_is_finite:
            raise IsolatedSubprocessLifecycleError(
                "timeout_seconds must be a finite nonnegative number"
            )
        deadline = time.monotonic() + timeout_seconds
        while True:
            self.checkpoint()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                returncode = process.poll()
                if returncode is not None:
                    self.checkpoint()
                    return returncode
                raise subprocess.TimeoutExpired(process.args, timeout_seconds)
            try:
                returncode = process.wait(
                    timeout=min(self._MAX_WAIT_POLL_SECONDS, remaining)
                )
            except subprocess.TimeoutExpired:
                continue
            self.checkpoint()
            return returncode

    @contextmanager
    def spawn_window(self) -> Iterator[None]:
        """Record signals across ``Popen`` and checkpoint after assignment."""

        if not self._entered or self._exiting:
            raise IsolatedSubprocessLifecycleError(
                "subprocess spawn window requires an active signal guard"
            )
        if self._spawning:
            raise IsolatedSubprocessLifecycleError(
                "subprocess spawn windows cannot be nested"
            )
        self._spawning = True
        try:
            yield
        finally:
            self._spawning = False
        self.checkpoint()

    @contextmanager
    def cleanup_window(self) -> Iterator[None]:
        """Mark teardown where recorded signals must remain passive."""

        if not self._entered or self._exiting:
            raise IsolatedSubprocessLifecycleError(
                "subprocess cleanup requires an active signal guard"
            )
        if self._cleaning:
            raise IsolatedSubprocessLifecycleError(
                "subprocess cleanup windows cannot be nested"
            )
        self._cleaning = True
        try:
            yield
        finally:
            self._cleaning = False

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._exiting = True
        self._restore()
        if self._pending is not None:
            _replay_previous_handler(
                self._pending,
                self._previous[self._pending.signum],
            )
        return False


def isolated_subprocess_signal_guard() -> _IsolatedSubprocessSignalGuard:
    """Return a TERM/HUP/INT guard for one isolated subprocess lifecycle.

    Python only permits signal-handler changes on its main thread. Failing
    before the guarded body runs prevents an unguarded process from being
    spawned by a worker thread. ``spawn_window`` closes the race between
    ``Popen`` returning and the caller recording the child PID without
    changing the signal mask inherited by the child. Handlers only record
    state; callers use ``checkpoint`` or ``wait`` outside ``cleanup_window``
    to begin a safe unwind.
    """

    return _IsolatedSubprocessSignalGuard()
