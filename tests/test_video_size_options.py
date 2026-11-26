# Copyright (c) 2026 Trent Nelson
# SPDX-License-Identifier: Apache-2.0

import io

import pytest

from sneeze.tpn.video_upscale import runner


@pytest.mark.parametrize(
    ("options", "label", "expected_setting"),
    [
        ({"scale": "2.0"}, "2x", {"scale": "2"}),
        ({"scale": "1.5"}, "1.5x", {"scale": "3/2"}),
        ({"scale": "3/2"}, "1.5x", {"scale": "3/2"}),
        ({"scale": "4/3"}, "4over3x", {"scale": "4/3"}),
        (
            {"target_size": "1920X1080"},
            "1920x1080",
            {"target_size": "1920x1080"},
        ),
        (
            {"target_size": "AUTO×2160"},
            "autox2160",
            {"target_size": "autox2160"},
        ),
    ],
)
def test_custom_size_reaches_every_model(
    options, label, expected_setting, tmp_path, monkeypatch
):
    source = tmp_path / "input.mp4"
    source.touch()
    monkeypatch.setattr(runner, "INSTALLATION", {"root": str(tmp_path)})
    monkeypatch.setattr(runner, "GB300", "example-device")
    jobs = runner.plan([source], {"model": "all", "compare": True, **options})
    for job in jobs:
        for name in ("cache", "models"):
            (tmp_path / job.model / name).mkdir(parents=True)
        assert job.output.name == f"input-{job.model}-{label}.mp4"
        command = runner.container_command(job, tmp_path, "example")
        for key, value in expected_setting.items():
            assert f"VSR_{key.upper()}={value}" in command
            assert f"{key.replace('_', '-')}={value}" in runner.describe(
                job, 1, 3
            )
    size_suffix = "" if label == "2x" else f"-{label}"
    assert jobs[-1].preview.name == f"input-all{size_suffix}-compare.mp4"


def test_target_size_preserves_explicit_output(tmp_path):
    source = tmp_path / "input.mp4"
    source.touch()
    dest = tmp_path / "result.mp4"
    options = {"target_size": "1920xauto", "output": dest, "dry_run": True}
    out = io.StringIO()
    runner.execute([source], options, out, io.StringIO())
    assert str(dest) in out.getvalue()
    assert "target-size=1920xauto" in out.getvalue()
    assert not dest.exists()


def test_size_options_are_mutually_exclusive(tmp_path):
    source = tmp_path / "input.mp4"
    source.touch()
    with pytest.raises(ValueError, match="either --scale or --target-size"):
        runner.plan([source], {"target_size": "1920x1080", "scale": "2"})
