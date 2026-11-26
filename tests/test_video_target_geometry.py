# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Trent Nelson
"""Verify consistent display-space sizing across the video pipeline."""

from fractions import Fraction
from pathlib import Path

import pytest

from sneeze.tpn.video_upscale import media
from sneeze.tpn.video_upscale.sizing import resolve_target


@pytest.mark.parametrize(
    "source,options,expected",
    [
        ((181, 319), {}, (362, 638)),
        ((1280, 720), {"scale": "3/2"}, (1920, 1080)),
        ((1280, 720), {"scale": "0.5"}, (640, 360)),
        ((319, 181), {"scale": "1"}, (320, 182)),
        ((720, 1280), {"target_size": "autox2160"}, (1216, 2160)),
        ((720, 1280), {"target_size": "1216x2160"}, (1216, 2160)),
        ((1920, 800), {"target_size": "1280xAUTO"}, (1280, 534)),
        ((319, 181), {"target_size": "320×182"}, (320, 182)),
        ((320, 180), {"scale": "1/1000"}, (2, 2)),
    ],
)
def test_resolve_target(source, options, expected):
    assert resolve_target(*source, **options) == expected


def test_target_must_preserve_aspect_ratio():
    with pytest.raises(ValueError, match="1280xauto or autox720"):
        resolve_target(640, 480, target_size="1280x720")


def test_target_rejected_before_timestamp_scan(monkeypatch):
    tracks = [
        dict(codec_type="video", codec_name="av1", width=640, height=480)
    ]
    monkeypatch.setattr(media, "streams", lambda _: tracks)
    monkeypatch.setattr(
        media, "timestamps", lambda *_: pytest.fail("scanned frames")
    )
    monkeypatch.setenv("VSR_TARGET_SIZE", "1280x720")
    monkeypatch.delenv("VSR_SCALE", raising=False)
    with pytest.raises(ValueError, match="aspect ratio"):
        media.main("seedvr2", Path("source.mp4"))


def test_rotated_anamorphic_target_reaches_encoder_worker_and_verifier(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "source.mp4"
    src = dict(
        codec_type="video",
        codec_name="h264",
        width=340,
        height=180,
        sample_aspect_ratio="16:17",
        side_data_list=[{"rotation": 90}],
        avg_frame_rate="24/1",
    )
    out = dict(
        media.OUTPUT_FORMAT,
        codec_type="video",
        width=270,
        height=480,
        avg_frame_rate="24/1",
    )
    monkeypatch.setattr(media, "WORK", tmp_path)
    monkeypatch.setattr(
        media, "streams", lambda path: [src if path == source else out]
    )
    monkeypatch.setattr(media, "timestamps", lambda *args: (9, Fraction(0)))
    monkeypatch.delenv("VSR_SCALE", raising=False)
    monkeypatch.setenv("VSR_TARGET_SIZE", "autox480")
    monkeypatch.setenv("VSR_ENCODER", "nvenc")
    monkeypatch.setenv("VSR_ENCODER_REQUEST", "auto")

    def check_encoder(encoder, requested, width, height, rate):
        assert (encoder, requested, width, height, rate) == (
            "nvenc",
            "auto",
            270,
            480,
            24,
        )
        return "cpu"

    def model(command, *, env, check):
        assert (env["JOB_WIDTH"], env["JOB_HEIGHT"]) == ("180", "320")
        assert (env["JOB_TARGET_WIDTH"], env["JOB_TARGET_HEIGHT"]) == (
            "270",
            "480",
        )
        assert env["VSR_ENCODER"] == "cpu"
        command[-1].touch()

    monkeypatch.setattr(media, "resolve_encoder", check_encoder)
    monkeypatch.setattr(media.subprocess, "run", model)
    media.main("seedvr2", source)
    assert (tmp_path / "verified.mp4").exists()
    assert "270x480" in capsys.readouterr().out
