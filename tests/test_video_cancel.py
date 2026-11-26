# Copyright (c) 2026 Trent Nelson
# SPDX-License-Identifier: Apache-2.0

import io
import signal
import subprocess

import pytest

from sneeze.tpn.video_upscale import runner

CONTAINER_ID = "a" * 64


def test_interrupt_during_create_cleans_before_any_attach(monkeypatch):
    operations = []

    def control(args, timeout):
        operations.append(args[0])
        if args[0] == "create":
            signal.raise_signal(signal.SIGINT)
            return subprocess.CompletedProcess(args, 0, CONTAINER_ID, "")
        assert args == ["rm", "--force", CONTAINER_ID]
        signal.raise_signal(signal.SIGTERM)
        signal.raise_signal(signal.SIGINT)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(runner, "docker_control", control)
    command = [*runner.DOCKER, "run", "--name", "example"]
    with pytest.raises(KeyboardInterrupt), runner.cleanup_on_termination():
        with runner.worker_client(command, "example"):
            pytest.fail("Started a cancelled container")
    assert operations == ["create", "rm"]


@pytest.mark.parametrize("owned_container_exists", [True, False])
def test_lost_create_reply_checks_name_and_unique_owner(
    monkeypatch, owned_container_exists
):
    calls = []

    def control(args, timeout):
        calls.append(args)
        if args[0] == "create":
            raise TimeoutError("Docker create timed out")
        if args[0] == "ps":
            assert "name=^/example$" in args
            assert "label=" + calls[0][2] in args
            assert calls[0][2].startswith("sneeze.upscale.owner=")
            found = CONTAINER_ID if owned_container_exists else ""
            return subprocess.CompletedProcess(args, 0, found, "")
        assert args == ["rm", "--force", CONTAINER_ID]
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(runner, "docker_control", control)
    with pytest.raises(
        RuntimeError, match="Could not create example:.*timed"
    ):
        runner.create_container(["--name", "example"], "example")
    expected = ["create", "ps"] + (["rm"] if owned_container_exists else [])
    assert [call[0] for call in calls] == expected


@pytest.mark.parametrize("removal_fails", [True, False])
def test_attach_failure_cleans_and_surfaces_daemon_errors(
    monkeypatch, removal_fails
):
    calls = []

    class Client:
        stdout = io.StringIO("worker refused input\n")

        def wait(self):
            return 37

        def poll(self):
            return 37

    def popen(args, **kwargs):
        assert args == [*runner.DOCKER, "start", "--attach", CONTAINER_ID]
        assert kwargs["start_new_session"] is True
        return Client()

    def control(args, timeout):
        calls.append(args[0])
        if args[0] == "create":
            return subprocess.CompletedProcess(args, 0, CONTAINER_ID, "")
        assert args == ["rm", "--force", CONTAINER_ID]
        if removal_fails:
            raise TimeoutError("Docker rm timed out")
        return subprocess.CompletedProcess(args, 1, "", "No such container")

    monkeypatch.setattr(runner, "docker_control", control)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner.time, "sleep", lambda delay: None)
    error = (
        "Could not remove Docker container" if removal_fails else "exited 37"
    )
    with pytest.raises(RuntimeError, match=error):
        runner.monitor([*runner.DOCKER, "run"], "example", print, False)
    assert calls == ["create", *(["rm"] * (3 if removal_fails else 1))]


def test_control_times_out_without_extra_privileged_commands(monkeypatch):
    def timeout(args, **kwargs):
        assert args[0] == "timeout"
        assert args[3:] == [
            "sudo",
            "-n",
            "docker",
            "rm",
            "--force",
            "example",
        ]
        assert kwargs["start_new_session"] is True
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(runner.subprocess, "run", timeout)
    with pytest.raises(TimeoutError, match="Docker rm timed out"):
        runner.docker_control(["rm", "--force", "example"], 5)
