# Copyright (c) 2026 Trent Nelson
# SPDX-License-Identifier: Apache-2.0
"""Build a labeled side-by-side viewing copy with CPU FFmpeg."""

import json
import os
import subprocess
import sys
import tempfile
import uuid
from fractions import Fraction
from pathlib import Path

LABELS = {
    "turbo": "FlashVSR+ Turbo",
    "seedvr2": "SeedVR2-7B",
    "ltx": "LTX-2.5 Refine Details",
}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def render_preview(
    source: Path, renders: list[tuple[str, Path]], staged: Path, report
):
    from .runner import DOCKER, IMAGE, monitor, mount

    if not staged.is_absolute() or os.path.lexists(staged):
        raise ValueError(f"Comparison requires a new absolute path: {staged}")
    models = [model for model, _ in renders]
    if (
        not 1 <= len(models) <= len(LABELS)
        or len(set(models)) != len(models)
        or any(model not in LABELS for model in models)
    ):
        raise ValueError("Supply a verified render for each selected model")
    paths = [source, *(path for _, path in renders)]
    name = f"sne-preview-{uuid.uuid4().hex}"
    cmd = [
        *DOCKER,
        "run",
        "--rm",
        "--pull=never",
        "--init",
        "--network=none",
        "--name",
        name,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
    ]
    cmd += mount(Path(__file__).resolve(), "/preview.py")
    for i, path in enumerate(paths):
        cmd += mount(path.resolve(strict=True), f"/input/{i}")
    report("Rendering the labeled side-by-side comparison…")
    with tempfile.TemporaryDirectory(dir=staged.parent) as work:
        cmd += mount(work, "/output", True)
        cmd += [
            "--entrypoint",
            "python3",
            IMAGE,
            "-u",
            "/preview.py",
            *models,
        ]
        monitor(cmd, name, report, False)
        os.link(Path(work) / "preview.mp4", staged)


def streams(path):
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                path,
            ]
        )
    )["streams"]


def audio(tracks):
    return [
        (t["codec_name"], t["sample_rate"], t["channels"])
        for t in tracks
        if t["codec_type"] == "audio"
    ]


def video(tracks):
    return next(t for t in tracks if t["codec_type"] == "video")


def encode(models):
    tracks = streams("/input/1")
    ref = video(tracks)
    rate = Fraction(ref["avg_frame_rate"])
    frames = int(ref["nb_frames"])
    # Fit the render into 960 x 960. Allow room for the label on narrow
    # portrait footage, without stretching either dimension.
    width, height = ref["width"], ref["height"]
    scale = min(1, 960 / max(width, height))
    width = max(2, 2 * round(width * scale / 2))
    height = max(2, 2 * round(height * scale / 2))
    panel_width = max(256, width)
    header = 40
    labels = ["Input", *(LABELS[model] for model in models)]
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-n",
        "-progress",
        "pipe:1",
        "-nostats",
        "-filter_complex_threads",
        "4",
    ]
    filters = []
    for i, label in enumerate(labels):
        cmd.extend(["-threads", "3", "-i", f"/input/{i}"])
        # All frames were verified on the same CFR grid before rendering.
        # Index by frame to align clocks with different container time bases.
        filters.append(
            f"[{i}:v:0]settb={rate.denominator}/{rate.numerator},setpts=N,"
            f"scale={width}:{height}:flags=lanczos:out_color_matrix=bt709:"
            "out_range=tv,format=yuv420p,setsar=1,"
            f"pad={panel_width}:{height + header}:(ow-iw)/2:{header},"
            f"drawtext=fontfile={FONT}:text='{label}':fontsize=23:"
            f"fontcolor=white:x=(w-tw)/2:y=({header}-th)/2[p{i}]"
        )
    count = len(labels)
    if count == 4:
        filters.append(
            "[p0][p1][p2][p3]xstack=inputs=4:"
            "layout=0_0|w0_0|0_h0|w0_h0:shortest=1[v]"
        )
        columns, rows = 2, 2
    else:
        inputs = "".join(f"[p{i}]" for i in range(count))
        filters.append(f"{inputs}hstack=inputs={count}:shortest=1[v]")
        columns, rows = count, 1
    destination = "/output/preview.mp4"
    cmd += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[v]",
        "-map",
        "1:a?",
        "-fps_mode",
        "passthrough",
        "-c:v",
        "libx264",
        "-threads",
        "6",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_primaries",
        "bt709",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        destination,
    ]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as process:
        for line in process.stdout:
            if line.startswith("frame="):
                print(f"VSR_PROGRESS {int(line[6:])} {frames}", flush=True)
        status = process.wait()
    if status:
        raise RuntimeError(f"Comparison FFmpeg exited with status {status}")
    out = streams(destination)
    result = video(out)
    if not (
        result["width"] == panel_width * columns
        and result["height"] == (height + header) * rows
        and int(result["nb_frames"]) == frames
        and Fraction(result["avg_frame_rate"]) == rate
        and audio(out) == audio(tracks)
    ):
        raise RuntimeError(
            "Comparison frame count, size, rate or audio changed"
        )
    print(
        f"Verified comparison {result['width']}x{result['height']}, "
        f"{frames} frames at {rate} fps, {len(audio(out))} audio track(s).",
        flush=True,
    )


if __name__ == "__main__":
    encode(sys.argv[1:])
