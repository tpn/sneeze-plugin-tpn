from __future__ import annotations

import io
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest


def _source_environment() -> dict[str, str]:
    source_root = Path(__file__).resolve().parents[1] / "src"
    return {
        "PATH": os.defpath,
        "PYTHONPATH": str(source_root),
    }


def _wait_for_path(path: Path, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    pytest.fail(f"timed out waiting for {path}")


def _wait_for_process_exit(pid: int, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    pytest.fail(f"process {pid} survived lifecycle cleanup")


def test_guard_restores_callable_and_preserves_ignored_signal():
    from sneeze.tpn.ocr_process_lifecycle import (
        isolated_subprocess_signal_guard,
    )

    previous_term = signal.getsignal(signal.SIGTERM)
    previous_hup = signal.getsignal(signal.SIGHUP)

    def prior_term(signum, frame):
        del signum, frame

    signal.signal(signal.SIGTERM, prior_term)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        with pytest.raises(RuntimeError, match="body failed"):
            with isolated_subprocess_signal_guard():
                assert signal.getsignal(signal.SIGTERM) is not prior_term
                assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
                raise RuntimeError("body failed")
        assert signal.getsignal(signal.SIGTERM) is prior_term
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGHUP, previous_hup)


def test_guard_rejects_nested_process_global_handler_ownership():
    from sneeze.tpn.ocr_process_lifecycle import (
        IsolatedSubprocessLifecycleError,
        isolated_subprocess_signal_guard,
    )

    with isolated_subprocess_signal_guard():
        with pytest.raises(
            IsolatedSubprocessLifecycleError,
            match="cannot be nested",
        ):
            with isolated_subprocess_signal_guard():
                pytest.fail("nested guard body must not run")


def test_checkpoint_stays_passive_inside_cleanup_window():
    from sneeze.tpn.ocr_process_lifecycle import (
        IsolatedSubprocessInterrupted,
        isolated_subprocess_signal_guard,
    )

    events: list[str] = []
    original = signal.getsignal(signal.SIGTERM)

    def prior_handler(signum, frame):
        del signum, frame
        events.append("prior-handler")

    signal.signal(signal.SIGTERM, prior_handler)
    try:
        with pytest.raises(IsolatedSubprocessInterrupted):
            with isolated_subprocess_signal_guard() as lifecycle:
                os.kill(os.getpid(), signal.SIGTERM)
                with lifecycle.cleanup_window():
                    lifecycle.checkpoint()
                    events.append("cleanup-finished")
                lifecycle.checkpoint()
    finally:
        signal.signal(signal.SIGTERM, original)

    assert events == ["cleanup-finished", "prior-handler"]


@pytest.mark.parametrize("signum", (signal.SIGTERM, signal.SIGINT))
def test_spawn_window_defers_unwind_until_process_handle_is_recorded(signum):
    from sneeze.tpn.ocr_process_lifecycle import (
        IsolatedSubprocessInterrupted,
        isolated_subprocess_signal_guard,
    )

    events: list[str] = []
    original = signal.getsignal(signum)

    def prior_handler(signum, frame):
        del signum, frame
        assert signal.getsignal(original_signum) is prior_handler
        events.append("prior-handler")

    original_signum = signum
    signal.signal(signum, prior_handler)
    try:
        with pytest.raises(IsolatedSubprocessInterrupted) as caught:
            with isolated_subprocess_signal_guard() as lifecycle:
                try:
                    with lifecycle.spawn_window():
                        os.kill(os.getpid(), signum)
                        events.append("process-recorded")
                finally:
                    events.append("process-cleanup")
    finally:
        signal.signal(signum, original)

    assert caught.value.signum == signum
    assert events == [
        "process-recorded",
        "process-cleanup",
        "prior-handler",
    ]


@pytest.mark.parametrize("signum", (signal.SIGTERM, signal.SIGHUP))
def test_guard_replays_default_signal_after_cleanup(tmp_path, signum):
    ready = tmp_path / "ready"
    cleaned = tmp_path / "cleaned"
    script = textwrap.dedent(
        f"""
        import signal
        import time
        from pathlib import Path

        from sneeze.tpn.ocr_process_lifecycle import (
            isolated_subprocess_signal_guard,
        )

        signum = {int(signum)}
        signal.signal(signum, signal.SIG_DFL)
        with isolated_subprocess_signal_guard() as lifecycle:
            try:
                Path({str(ready)!r}).write_text("ready")
                while True:
                    time.sleep(0.05)
                    lifecycle.checkpoint()
            finally:
                Path({str(cleaned)!r}).write_text("cleaned")
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        env=_source_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_path(ready)
        os.kill(process.pid, signum)
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert (stdout, stderr) == ("", "")
    assert process.returncode == -signum
    assert cleaned.read_text(encoding="utf-8") == "cleaned"


def test_guard_restores_python_sigint_handler_after_cleanup(tmp_path):
    ready = tmp_path / "ready"
    cleaned = tmp_path / "cleaned"
    script = textwrap.dedent(
        f"""
        import time
        from pathlib import Path

        from sneeze.tpn.ocr_process_lifecycle import (
            isolated_subprocess_signal_guard,
        )

        with isolated_subprocess_signal_guard() as lifecycle:
            try:
                Path({str(ready)!r}).write_text("ready")
                while True:
                    time.sleep(0.05)
                    lifecycle.checkpoint()
            finally:
                Path({str(cleaned)!r}).write_text("cleaned")
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        env=_source_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_path(ready)
        os.kill(process.pid, signal.SIGINT)
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert stdout == ""
    assert stderr.rstrip().endswith("KeyboardInterrupt")
    assert process.returncode == -signal.SIGINT
    assert cleaned.read_text(encoding="utf-8") == "cleaned"


@pytest.mark.parametrize(
    "signum",
    (signal.SIGTERM, signal.SIGHUP, signal.SIGINT),
)
def test_spawned_child_keeps_default_unblocked_signal(tmp_path, signum):
    from sneeze.tpn.ocr_process_lifecycle import (
        isolated_subprocess_signal_guard,
    )

    ready = tmp_path / "child-ready"
    child_code = textwrap.dedent(
        f"""
        import time
        from pathlib import Path

        Path({str(ready)!r}).write_text("ready")
        while True:
            time.sleep(1)
        """
    )
    original = signal.getsignal(signum)
    signal.signal(signum, signal.SIG_DFL)
    child: subprocess.Popen[bytes] | None = None
    try:
        with isolated_subprocess_signal_guard() as lifecycle:
            with lifecycle.spawn_window():
                child = subprocess.Popen(
                    [sys.executable, "-c", child_code],
                    env={"PATH": os.defpath},
                    start_new_session=True,
                )
            _wait_for_path(ready)
            os.kill(child.pid, signum)
            assert child.wait(timeout=5) == -signum
    finally:
        signal.signal(signum, original)
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()


def test_wait_uses_bounded_polling_checkpoints():
    from sneeze.tpn.ocr_process_lifecycle import (
        isolated_subprocess_signal_guard,
    )

    class FakeProcess:
        args = ["fake-process"]

        def __init__(self):
            self.timeouts: list[float] = []

        def poll(self):
            return None

        def wait(self, timeout):
            self.timeouts.append(timeout)
            if len(self.timeouts) < 3:
                raise subprocess.TimeoutExpired(self.args, timeout)
            return 0

    process = FakeProcess()
    with isolated_subprocess_signal_guard() as lifecycle:
        assert lifecycle.wait(process, timeout_seconds=1) == 0

    assert len(process.timeouts) == 3
    assert all(0 < value <= 0.1 for value in process.timeouts)


@pytest.mark.parametrize(
    "timeout",
    (
        pytest.param(True, id="boolean"),
        pytest.param("30", id="string"),
        pytest.param(-1, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ),
)
def test_wait_rejects_invalid_timeout_before_polling(timeout):
    from sneeze.tpn.ocr_process_lifecycle import (
        IsolatedSubprocessLifecycleError,
        isolated_subprocess_signal_guard,
    )

    class FakeProcess:
        args = ["fake-process"]

        def poll(self):
            raise AssertionError("invalid timeout reached process polling")

        def wait(self, timeout):
            raise AssertionError("invalid timeout reached process wait")

    with isolated_subprocess_signal_guard() as lifecycle:
        with pytest.raises(
            IsolatedSubprocessLifecycleError,
            match="finite nonnegative",
        ):
            lifecycle.wait(FakeProcess(), timeout_seconds=timeout)


def test_wait_allows_zero_for_an_immediate_completed_process_poll():
    from sneeze.tpn.ocr_process_lifecycle import (
        isolated_subprocess_signal_guard,
    )

    class CompleteProcess:
        args = ["complete-process"]

        def poll(self):
            return 0

        def wait(self, timeout):
            raise AssertionError("zero timeout should poll before waiting")

    with isolated_subprocess_signal_guard() as lifecycle:
        assert lifecycle.wait(CompleteProcess(), timeout_seconds=0) == 0


def test_all_isolated_popen_call_sites_fail_closed_off_main_thread(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn import (
        ocr_backend_setup,
        ocr_benchmark,
        ocr_execution,
        ocr_hardware,
        ocr_provider,
    )
    from sneeze.tpn.ocr_process_lifecycle import (
        IsolatedSubprocessLifecycleError,
    )

    popen_called = False

    def forbidden_popen(*args, **kwargs):
        nonlocal popen_called
        del args, kwargs
        popen_called = True
        raise AssertionError("Popen must not run off the main thread")

    monkeypatch.setattr(subprocess, "Popen", forbidden_popen)
    output = io.BytesIO()
    calls = (
        (
            "provider",
            lambda: ocr_provider._run_provider_process(
                argv=[sys.executable, "-c", "pass"],
                cwd=None,
                environment={"PATH": os.defpath},
                stdout_path=tmp_path / "provider.stdout",
                stderr_path=tmp_path / "provider.stderr",
                timeout_seconds=1,
                terminate_grace_seconds=0,
            ),
        ),
        (
            "benchmark",
            lambda: ocr_benchmark._run_provider_process(
                [sys.executable, "-c", "pass"],
                cwd=None,
                environment={"PATH": os.defpath},
                stdout_handle=output,
                stderr_handle=output,
                timeout_seconds=1,
            ),
        ),
        (
            "runtime",
            lambda: ocr_execution._run_bounded_runtime_probe(
                [sys.executable, "-c", "pass"],
                environment={"PATH": os.defpath},
                timeout_seconds=1,
                runner=ocr_execution._SUBPROCESS_RUNNER,
            ),
        ),
        (
            "hardware",
            lambda: ocr_hardware._run_bounded(
                [sys.executable, "-c", "pass"],
                environment={"PATH": os.defpath},
                timeout_seconds=1,
                runner=ocr_hardware._SUBPROCESS_RUNNER,
            ),
        ),
        (
            "doctor",
            lambda: ocr_backend_setup._run_bounded_probe(
                [sys.executable, "-c", "pass"],
                cwd=None,
                environment={"PATH": os.defpath},
                timeout_seconds=1,
            ),
        ),
    )

    for name, call in calls:
        errors: list[BaseException] = []

        def invoke(call=call, errors=errors) -> None:
            try:
                call()
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=invoke, name=f"test-{name}")
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive(), name
        assert len(errors) == 1, name
        assert isinstance(errors[0], IsolatedSubprocessLifecycleError), name

    assert popen_called is False


def test_term_reaps_stubborn_provider_group_before_parent_exits(tmp_path):
    process_ids = tmp_path / "provider-pids"
    provider = tmp_path / "stubborn-provider.py"
    provider.write_text(
        textwrap.dedent(
            f"""
            import os
            import signal
            import subprocess
            import sys
            import time
            from pathlib import Path

            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            child_code = (
                "import signal, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "time.sleep(60)"
            )
            child = subprocess.Popen([sys.executable, "-c", child_code])
            Path({str(process_ids)!r}).write_text(
                f"{{os.getpid()}} {{child.pid}}"
            )
            time.sleep(60)
            """
        ),
        encoding="utf-8",
    )
    wrapper = textwrap.dedent(
        f"""
        import os
        import sys

        from sneeze.tpn.ocr_benchmark import _run_provider_process

        with open(os.devnull, "wb") as output:
            _run_provider_process(
                [sys.executable, {str(provider)!r}],
                cwd=None,
                environment={{"PATH": os.defpath}},
                stdout_handle=output,
                stderr_handle=output,
                timeout_seconds=60,
            )
        """
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", wrapper],
        env=_source_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_path(process_ids)
        provider_pid, child_pid = (
            int(value)
            for value in process_ids.read_text(encoding="utf-8").split()
        )
        os.kill(parent.pid, signal.SIGTERM)
        stdout, stderr = parent.communicate(timeout=10)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()

    assert (stdout, stderr) == ("", "")
    assert parent.returncode == -signal.SIGTERM
    _wait_for_process_exit(provider_pid)
    _wait_for_process_exit(child_pid)


def test_term_during_slow_normal_cleanup_waits_for_descendant_reap(tmp_path):
    process_ids = tmp_path / "normal-provider-pids"
    child_ready = tmp_path / "normal-child-ready"
    cleanup_started = tmp_path / "normal-cleanup-started"
    provider = tmp_path / "normal-provider.py"
    provider.write_text(
        textwrap.dedent(
            f"""
            import os
            import subprocess
            import sys
            import time
            from pathlib import Path

            child_code = '''
            import signal
            import sys
            import time
            from pathlib import Path

            def ignore_term(signum, frame):
                del signum, frame
                Path(sys.argv[2]).write_text("cleanup-started")

            signal.signal(signal.SIGTERM, ignore_term)
            Path(sys.argv[1]).write_text("ready")
            while True:
                time.sleep(1)
            '''
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    child_code,
                    {str(child_ready)!r},
                    {str(cleanup_started)!r},
                ]
            )
            deadline = time.monotonic() + 5
            while not Path({str(child_ready)!r}).exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("child did not become ready")
                time.sleep(0.01)
            Path({str(process_ids)!r}).write_text(
                f"{{os.getpid()}} {{child.pid}}"
            )
            """
        ),
        encoding="utf-8",
    )
    wrapper = textwrap.dedent(
        f"""
        import os
        import sys

        from sneeze.tpn.ocr_benchmark import _run_provider_process

        with open(os.devnull, "wb") as output:
            _run_provider_process(
                [sys.executable, {str(provider)!r}],
                cwd=None,
                environment={{"PATH": os.defpath}},
                stdout_handle=output,
                stderr_handle=output,
                timeout_seconds=60,
            )
        """
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", wrapper],
        env=_source_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_path(process_ids)
        provider_pid, child_pid = (
            int(value)
            for value in process_ids.read_text(encoding="utf-8").split()
        )
        _wait_for_path(cleanup_started)
        assert parent.poll() is None
        os.kill(parent.pid, signal.SIGTERM)
        stdout, stderr = parent.communicate(timeout=10)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()

    assert (stdout, stderr) == ("", "")
    assert parent.returncode == -signal.SIGTERM
    _wait_for_process_exit(provider_pid)
    _wait_for_process_exit(child_pid)
