# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Trent Nelson
"""Video reader and lossless encoder for the container media workers."""

import os
import subprocess
from contextlib import contextmanager

import cv2
import numpy as np


@contextmanager
def ffmpeg_capture(source, width, height, *, nvdec=False):
    """Read square-pixel BGR frames in display order and orientation."""
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-threads",
        "4",
    ]
    filters = [f"scale={width}:{height}:flags=lanczos", "setsar=1"]
    env = None
    if nvdec:
        from decode_rtx import RTX_UUID

        # CUVID's only implementation is hardware-backed. Keep device zero
        # local to the codec subprocess; the model keeps its inference GPU.
        command.extend(["-c:v", "av1_cuvid", "-gpu", "0"])
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": RTX_UUID}
        # Unpack NV12/P010 to the same planar format as software decoding
        # before swscale performs the final color conversion and resize.
        pixel_format = os.environ["JOB_INPUT_PIX_FMT"]
        if pixel_format not in ("yuv420p", "yuv420p10le"):
            raise ValueError(f"Unsupported AV1 NVDEC format {pixel_format}")
        filters.insert(0, f"format={pixel_format}")
        if os.getenv("JOB_INPUT_COLOR_RANGE") == "pc":
            # CUVID can label full-range AV1 surfaces as limited range.
            filters.insert(1, "setparams=range=full")
    command += [
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-filter_threads",
        "2",
        "-vf",
        ",".join(filters),
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "bgr24",
        "-c:v",
        "rawvideo",
        "-threads:v",
        "1",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    frame_bytes = width * height * 3
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, env=env
    ) as process:

        class Capture:
            def read(self):
                data = process.stdout.read(frame_bytes)
                if not data:
                    if process.wait():
                        raise RuntimeError(
                            f"FFmpeg decoder exited {process.returncode}"
                        )
                    return False, None
                if len(data) != frame_bytes:
                    raise RuntimeError("FFmpeg returned an incomplete frame")
                return True, np.frombuffer(data, np.uint8).reshape(
                    height, width, 3
                )

        try:
            yield Capture()
        finally:
            # The consumer can stop before EOF (failed or cancelled render).
            if process.poll() is None:
                process.terminate()
            process.stdout.close()


@contextmanager
def open_reader(source, decoder, frame_count):
    av1_nvdec = decoder == "nvdec" and os.getenv("JOB_INPUT_CODEC") == "av1"
    if av1_nvdec or (
        decoder != "nvdec" and os.getenv("JOB_FFMPEG_INPUT") == "1"
    ):
        with ffmpeg_capture(
            source,
            int(os.environ["JOB_WIDTH"]),
            int(os.environ["JOB_HEIGHT"]),
            nvdec=av1_nvdec,
        ) as reader:
            yield reader
        return
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
