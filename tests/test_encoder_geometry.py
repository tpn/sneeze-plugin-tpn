# Copyright (c) 2026 Trent Nelson
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from types import SimpleNamespace

import pytest

from sneeze.tpn.video_upscale import encoder_geometry, runner


@pytest.mark.parametrize("requested", ["auto", "cpu", "nvenc"])
def test_encoder_request_survives_auto_resolution(
    tmp_path, monkeypatch, requested
):
    source = tmp_path / "input.mp4"
    source.touch()
    for name in ("models", "cache"):
        (tmp_path / "turbo" / name).mkdir(parents=True)
    monkeypatch.setattr(runner, "INSTALLATION", {"root": str(tmp_path)})
    monkeypatch.setattr(runner, "GB300", "example-inference-device")
    (job,) = runner.plan([source], {"encoder": requested})
    cmd = runner.container_command(job, tmp_path, "test-worker")
    assert f"VSR_ENCODER_REQUEST={requested}" in cmd
    selected = "cpu" if requested == "cpu" else "nvenc"
    assert f"VSR_ENCODER={selected}" in cmd


@pytest.mark.parametrize("requested", ["auto", "nvenc"])
def test_incompatible_target_grants_fallback_only_in_auto(
    monkeypatch, requested, capsys
):
    import_adapter = SimpleNamespace(RTX_UUID="example-codec-device")
    monkeypatch.setitem(sys.modules, "encode_rtx", import_adapter)
    stderr = (
        "[h264_nvenc @ address] Width 4098 exceeds 4096\n"
        "[h264_nvenc @ address] No capable devices found\n"
    )

    def run(cmd, *, env, **kwargs):
        assert "nullsrc=size=4098x720:rate=30000/1001" in cmd
        assert cmd[cmd.index("-frames:v") + 1] == "0"
        assert env["CUDA_VISIBLE_DEVICES"] == "example-codec-device"
        return subprocess.CompletedProcess(cmd, 1, stderr=stderr)

    monkeypatch.setattr(encoder_geometry.subprocess, "run", run)
    args = ("nvenc", requested, 4098, 720, "30000/1001")
    if requested == "nvenc":
        with pytest.raises(ValueError, match=r"4098x720.*--encoder cpu"):
            encoder_geometry.resolve_encoder(*args)
    else:
        assert encoder_geometry.resolve_encoder(*args) == "cpu"
        message = capsys.readouterr().out
        assert "Width 4098 exceeds 4096" in message and "CPU x264" in message


def test_auto_keeps_supported_nvenc_and_skips_probe_for_cpu(monkeypatch):
    monkeypatch.setattr(encoder_geometry, "nvenc_error", lambda *a: None)
    assert encoder_geometry.resolve_encoder(
        "nvenc", "auto", 320, 200, 24
    ) == ("nvenc")
    monkeypatch.setattr(
        encoder_geometry, "nvenc_error", lambda *a: pytest.fail("NVENC probe")
    )
    assert encoder_geometry.resolve_encoder("cpu", "auto", 320, 200, 24) == (
        "cpu"
    )
