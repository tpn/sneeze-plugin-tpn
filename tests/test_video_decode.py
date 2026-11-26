# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Trent Nelson
"""Exercise video geometry and FFmpeg decoding with synthetic frames."""

import shutil
import subprocess
from fractions import Fraction

import pytest

from sneeze.tpn.video_upscale import media


@pytest.fixture
def ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required")

    def run(*args, data=None):
        return subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", *args],
            input=data,
            capture_output=True,
            check=True,
        )

    return run


def test_reader_corrects_sar_and_rotation_and_keeps_frame_order(
    ffmpeg, tmp_path
):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from sneeze.tpn.video_upscale import worker_io

    if b"libx264rgb" not in ffmpeg("-encoders").stdout:
        pytest.skip("Fixture requires the libx264rgb encoder")

    frames = np.zeros((3, 35, 51, 3), dtype=np.uint8)
    # Distinct BGR corners reveal channel swaps and rotation direction.
    frames[:, :17, :25] = (20, 40, 100)
    frames[:, :17, 25:] = (80, 40, 120)
    frames[:, 17:, :25] = (30, 160, 50)
    frames[:, 17:, 25:] = (120, 90, 60)
    frames[1] += 9
    frames[2] += 21
    source = tmp_path / "source.mp4"
    rotated = tmp_path / "rotated.mp4"
    ffmpeg(
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        "51x35",
        "-framerate",
        "30000/1001",
        "-i",
        "pipe:0",
        "-vf",
        "setsar=16/17",
        "-c:v",
        "libx264rgb",
        "-qp",
        "0",
        source,
        data=frames.tobytes(),
    )
    if b"-display_rotation" in ffmpeg("-h", "full").stdout:
        ffmpeg(
            "-display_rotation", "90", "-i", source, "-c:v", "copy", rotated
        )
    else:
        ffmpeg(
            "-i",
            source,
            "-c:v",
            "copy",
            "-metadata:s:v:0",
            "rotate=90",
            rotated,
        )
    info = media.video(media.streams(rotated))
    assert info["sample_aspect_ratio"] == "16:17"
    assert info["side_data_list"][0]["rotation"] == 90
    assert Fraction(info["avg_frame_rate"]) == Fraction(30000, 1001)
    assert media.timestamps(rotated, info, Fraction(30000, 1001)) == (3, 0)
    width, height = media.display_geometry(info)
    assert (width, height) == (35, 48)
    with worker_io.ffmpeg_capture(rotated, width, height) as reader:
        for original in frames:
            ok, bgr = reader.read()
            assert ok and bgr.shape == (48, 35, 3)
            # The display matrix rotates counterclockwise. Interior samples
            # stay constant under the modest resampling for square pixels.
            expected = np.rot90(original)
            for y in (0.25, 0.75):
                for x in (0.25, 0.75):
                    np.testing.assert_array_equal(
                        bgr[int(height * y), int(width * x)],
                        expected[
                            int(expected.shape[0] * y),
                            int(expected.shape[1] * x),
                        ],
                    )
        assert reader.read() == (False, None)


def test_reader_decodes_av1_with_odd_dimensions(
    ffmpeg, tmp_path, monkeypatch
):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from sneeze.tpn.video_upscale import worker_io

    if b"libaom-av1" not in ffmpeg("-encoders").stdout:
        pytest.skip("Fixture requires the libaom-av1 encoder")

    source = tmp_path / "source.mkv"
    frames = np.empty((3, 37, 55, 3), dtype=np.uint8)
    frames[:] = np.array([40, 85, 130], dtype=np.uint8)[:, None, None, None]
    ffmpeg(
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        "55x37",
        "-framerate",
        "24",
        "-i",
        "pipe:0",
        "-c:v",
        "libaom-av1",
        "-crf",
        "0",
        "-cpu-used",
        "8",
        "-pix_fmt",
        "yuv444p",
        source,
        data=frames.tobytes(),
    )
    info = media.video(media.streams(source))
    assert info["codec_name"] == "av1"
    assert media.display_geometry(info) == (55, 37)
    monkeypatch.setenv("JOB_FFMPEG_INPUT", "1")
    monkeypatch.setenv("JOB_WIDTH", "55")
    monkeypatch.setenv("JOB_HEIGHT", "37")
    monkeypatch.setattr(
        worker_io.cv2, "VideoCapture", lambda *_: pytest.fail("used OpenCV")
    )
    with worker_io.open_reader(source, "cpu", len(frames)) as reader:
        for original in frames:
            ok, bgr = reader.read()
            assert ok and bgr.shape == original.shape
            np.testing.assert_allclose(bgr, original, rtol=0, atol=2)
        assert reader.read() == (False, None)
