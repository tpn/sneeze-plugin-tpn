# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Trent Nelson
"""Check AV1 NVDEC against the software decoder on the same encoded frames."""

import os
import shutil
import subprocess

import pytest

from sneeze.tpn.video_upscale import media


@pytest.fixture(scope="module")
def codec_gpu():
    pytest.importorskip("cv2")
    adapter = pytest.importorskip("decode_rtx")
    if not all(
        shutil.which(cmd) for cmd in ("ffmpeg", "ffprobe", "nvidia-smi")
    ):
        pytest.skip("Requires the configured RTX codec runtime")
    status = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            adapter.RTX_UUID,
            "--query-gpu=name",
            "--format=csv,noheader",
        ],
        capture_output=True,
    )
    if status.returncode:
        pytest.skip("Configured codec GPU unavailable")
    return adapter.RTX_UUID


@pytest.mark.parametrize(
    ("pixel_format", "range_tag", "variant"),
    [
        ("yuv420p", None, "plain"),
        ("yuv420p", "tv", "plain"),
        ("yuv420p", "pc", "plain"),
        ("yuv420p10le", "tv", "plain"),
        ("yuv420p10le", "pc", "plain"),
        ("yuv420p", "tv", "grain"),
        ("yuv420p", "tv", "rotated"),
    ],
)
def test_av1_nvdec_matches_cpu_frames(
    tmp_path, monkeypatch, codec_gpu, pixel_format, range_tag, variant
):
    from sneeze.tpn.video_upscale import worker_io

    source = tmp_path / "source.mp4"
    frame_count = 6
    encoder = "libsvtav1" if variant == "grain" else "libaom-av1"
    builds = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"])
    if encoder.encode() not in builds:
        pytest.skip(f"Creating this AV1 fixture requires {encoder}")

    filters = [f"format={pixel_format}", "noise=alls=12:all_seed=42:allf=t"]
    if variant == "rotated":
        filters.append("setsar=16/17")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-v",
        "error",
        "-nostdin",
        "-filter_threads",
        "2",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=258x146:rate=24000/1001",
        "-vf",
        ",".join(filters),
        "-frames:v",
        str(frame_count),
        "-pix_fmt",
        pixel_format,
        "-c:v",
        encoder,
        "-threads:v",
        "2",
    ]
    if variant == "grain":
        command += [
            "-svtav1-params",
            "film-grain=12:film-grain-denoise=0:lp=2",
        ]
    else:
        command += ["-cpu-used", "8", "-crf", "28"]
    if range_tag:
        command += [
            "-colorspace",
            "bt709",
            "-color_range",
            range_tag,
            "-color_trc",
            "bt709",
            "-color_primaries",
            "bt709",
        ]
    subprocess.run([*command, source], check=True, capture_output=True)
    if variant == "grain":
        parameters = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-export_side_data",
                "+film_grain",
                "-read_intervals",
                "%+#1",
                "-show_frames",
                "-show_entries",
                "frame=side_data_list",
                source,
            ]
        )
        assert b"Film grain parameters" in parameters
    if variant == "rotated":
        rotated = tmp_path / "rotated.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-display_rotation",
                "90",
                "-i",
                source,
                "-c:v",
                "copy",
                rotated,
            ],
            check=True,
            capture_output=True,
        )
        source = rotated
    info = media.video(media.streams(source))
    width, height = media.display_geometry(info)
    assert (width, height) == (
        (146, 243) if variant == "rotated" else (258, 146)
    )
    assert info["pix_fmt"] == pixel_format
    if range_tag:
        assert info["color_range"] == range_tag
    else:
        assert info.get("color_space", "unknown") == "unknown"
    media.validate_source(info, "nvdec")

    monkeypatch.setenv("JOB_INPUT_CODEC", "av1")
    monkeypatch.setenv("JOB_INPUT_PIX_FMT", info["pix_fmt"])
    monkeypatch.setenv("JOB_INPUT_COLOR_RANGE", info.get("color_range", ""))
    monkeypatch.setenv("JOB_FFMPEG_INPUT", "1")
    monkeypatch.setenv("JOB_WIDTH", str(width))
    monkeypatch.setenv("JOB_HEIGHT", str(height))

    # Record the command that really decoded the frames. A silent fallback
    # to CPU would otherwise look like perfect agreement in this test.
    launched = []
    popen = subprocess.Popen
    inference_device = os.getenv("CUDA_VISIBLE_DEVICES")

    def capture(args, **kwargs):
        launched.append((args, kwargs.get("env")))
        return popen(args, **kwargs)

    monkeypatch.setattr(worker_io.subprocess, "Popen", capture)
    with (
        worker_io.open_reader(source, "cpu", frame_count) as software,
        worker_io.open_reader(source, "nvdec", frame_count) as hardware,
    ):
        for _ in range(frame_count):
            ok, reference = software.read()
            got, actual = hardware.read()
            assert ok and got
            assert reference.shape == actual.shape == (height, width, 3)
            assert (reference == actual).all()
        assert software.read() == hardware.read() == (False, None)

    assert len(launched) == 2
    cmd, env = launched[1]
    assert cmd[cmd.index("-c:v") + 1] == "av1_cuvid"
    assert env["CUDA_VISIBLE_DEVICES"] == codec_gpu
    assert os.getenv("CUDA_VISIBLE_DEVICES") == inference_device


@pytest.mark.parametrize(
    ("width", "height"), [(320, 179), (321, 180), (321, 179)]
)
def test_odd_stored_dimensions_require_software_av1_decoder(width, height):
    info = {
        "codec_name": "av1",
        "pix_fmt": "yuv420p",
        "width": width,
        "height": height,
    }
    with pytest.raises(ValueError, match=r"even stored.*--decoder cpu"):
        media.validate_source(info, "nvdec")
    media.validate_source(info, "cpu")


@pytest.mark.parametrize("pixel_format", ["yuv420p", "yuv420p10le"])
def test_untagged_av1_is_valid_for_nvdec(pixel_format):
    media.validate_source(
        {
            "codec_name": "av1",
            "width": 1280,
            "height": 720,
            "pix_fmt": pixel_format,
        },
        "nvdec",
    )


@pytest.mark.parametrize(
    ("attributes", "message"),
    [
        ({"pix_fmt": "yuv444p"}, "4:2:0"),
        ({"pix_fmt": "yuv420p12le"}, "8- or 10-bit"),
        ({"color_primaries": "bt2020"}, "Convert HDR"),
        ({"color_transfer": "smpte2084"}, "Convert HDR"),
    ],
)
def test_incompatible_av1_inputs_rejected_before_nvdec(attributes, message):
    info = {
        "codec_name": "av1",
        "width": 1280,
        "height": 720,
        "pix_fmt": "yuv420p",
    }
    with pytest.raises(ValueError, match=message):
        media.validate_source({**info, **attributes}, "nvdec")


def test_h264_nvdec_retains_color_requirements():
    info = {
        "codec_name": "h264",
        "width": 1280,
        "height": 720,
        "pix_fmt": "yuv420p",
        "color_range": "tv",
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
        "chroma_location": "left",
    }
    media.validate_source(info, "nvdec")
    del info["color_space"]
    with pytest.raises(ValueError, match="H.264 NVDEC needs color_space"):
        media.validate_source(info, "nvdec")
