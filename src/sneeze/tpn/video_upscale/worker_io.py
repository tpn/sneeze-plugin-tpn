# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Trent Nelson
"""Video reader and lossless encoder for the container media workers."""

import subprocess
from contextlib import contextmanager

import cv2


@contextmanager
def open_reader(source, decoder, frame_count):
    if decoder == "nvdec":
        from decode_rtx import hardware_frames

        class Capture:
            # The upstream reader uses OpenCV's (success, BGR) interface.
            def __init__(self, frames):
                self.frames = frames

            def read(self):
                rgb = next(self.frames, None)
                if rgb is None:
                    return False, None
                return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        with hardware_frames(source, frame_count) as frames:
            yield Capture(frames)
        return

    reader = cv2.VideoCapture(str(source))
    try:
        if not reader.isOpened():
            raise ValueError(f"Could not open {source}")
        yield reader
    finally:
        reader.release()


@contextmanager
def open_encoder(output, encoder, width, height, rate):
    if encoder == "nvenc":
        from encode_rtx import RtxEncoder

        writer = RtxEncoder(output, rate)
        try:
            yield writer.append_data
        finally:
            writer.close()
        return

    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        rate,
        "-i",
        "pipe:0",
        "-vf",
        "scale=out_color_matrix=bt709:out_range=tv",
        "-c:v",
        "libx264",
        "-threads",
        "8",
        "-preset",
        "veryfast",
        "-qp",
        "0",
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
        "-movflags",
        "+faststart",
        str(output),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        yield lambda rgb: process.stdin.write(memoryview(rgb).cast("B"))
        process.stdin.close()
        if process.wait():
            raise RuntimeError(
                f"FFmpeg encoder exited with {process.returncode}"
            )
