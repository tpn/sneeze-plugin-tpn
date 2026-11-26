# Copyright (c) 2026 Trent Nelson
# SPDX-License-Identifier: Apache-2.0
"""Check lossless H.264 encoding at the final display dimensions."""

import os
import subprocess


def nvenc_error(width, height, rate):
    # The codec adapter selects its GPU independently of model inference.
    from encode_rtx import RTX_UUID

    probe = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "verbose",
            "-f",
            "lavfi",
            "-i",
            f"nullsrc=size={width}x{height}:rate={rate}",
            "-frames:v",
            "0",
            "-c:v",
            "h264_nvenc",
            "-gpu",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "p1",
            "-tune",
            "lossless",
            "-rc",
            "constqp",
            "-qp",
            "0",
            "-bf",
            "0",
            "-f",
            "null",
            "-",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": RTX_UUID},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if probe.returncode == 0:
        return None
    # FFmpeg reports capability details at verbose level, followed by a
    # less specific "No capable devices found" for oversized frames.
    for line in probe.stderr.splitlines():
        if line.startswith("[h264_nvenc @") and any(
            word in line.lower()
            for word in ("exceeds", "failed", "cannot", "no capable")
        ):
            return line.split("] ", 1)[-1]
    return f"FFmpeg could not initialize h264_nvenc (exit {probe.returncode})"


def resolve_encoder(encoder, requested, width, height, rate):
    if encoder != "nvenc":
        return encoder
    try:
        error = nvenc_error(width, height, rate)
    except (ImportError, OSError) as exc:
        error = str(exc)
    if error is None:
        return encoder
    reason = (
        f"NVENC could not open {width}x{height} at {rate} fps: "
        f"{error.rstrip('.')}"
    )
    if requested == "nvenc":
        raise ValueError(f"{reason}. Use --encoder cpu for this output.")
    print(f"Input checked. {reason}. Using CPU x264.", flush=True)
    return "cpu"
