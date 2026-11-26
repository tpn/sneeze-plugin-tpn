# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Trent Nelson
"""Validate the source, run one selected model, and verify the output MP4."""

import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
WORK = Path("/output")
NVDEC_FORMAT = {
    "codec_name": "h264",
    "pix_fmt": "yuv420p",
    "color_range": "tv",
    "color_space": "bt709",
    "color_transfer": "bt709",
    "color_primaries": "bt709",
    "chroma_location": "left",
}
OUTPUT_FORMAT = {
    k: v for k, v in NVDEC_FORMAT.items() if k != "chroma_location"
}


def streams(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", path],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"]


def video(track_list):
    for stream in track_list:
        if stream["codec_type"] == "video":
            return stream
    raise ValueError("File contains no video stream")


def timestamps(path, info, rate):
    # MKV and other containers can quantize CFR timestamps to a coarser clock.
    scale = Fraction(info["time_base"])
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=pts",
            "-of",
            "csv=p=0",
            path,
        ],
        text=True,
    )
    fields = [line.partition(",")[0] for line in raw.splitlines() if line]
    try:
        pts = sorted(int(field) for field in fields if field)
    except ValueError as error:
        raise ValueError(
            f"Missing video presentation timestamp in {path}"
        ) from error
    if not pts or any(a == b for a, b in zip(pts, pts[1:], strict=False)):
        raise ValueError("Video frames need distinct presentation timestamps")
    num, den = (rate * scale).as_integer_ratio()
    jitter = min(2 * num, den)
    for i, t in enumerate(pts):
        if 2 * abs((t - pts[0]) * num - i * den) > jitter:
            raise ValueError(
                f"{path}: variable or discontinuous frame times at {rate} fps"
            )
    return len(pts), pts[0] * scale


def validate_source(v, decoder):
    if v.get("disposition", {}).get("attached_pic") or v.get(
        "sample_aspect_ratio"
    ) not in (None, "N/A", "0:1", "1:1"):
        raise ValueError(
            "Use square pixels and put the main video stream first"
        )
    if any(
        float(info.get("rotation", 0)) % 360
        for info in v.get("side_data_list", [])
    ):
        raise ValueError(
            "Apply the source's display rotation before upscaling"
        )
    if (
        v.get("color_transfer") in ("smpte2084", "arib-std-b67")
        or v.get("color_primaries") == "bt2020"
    ):
        raise ValueError(
            "Convert HDR or wide-gamut input to SDR before upscaling"
        )
    if decoder == "nvdec":
        # The RTX decoder writes these exact tags into its raw pipe.
        for key, expected in NVDEC_FORMAT.items():
            if v.get(key) != expected:
                actual = v.get(key, "unspecified")
                raise ValueError(
                    f"NVDEC needs {key}={expected}, input is {actual}. "
                    "Use --decoder cpu for this source."
                )


def audio_plan(source, audio, mode):
    if mode not in ("auto", "copy", "aac", "drop"):
        raise ValueError("Choose audio mode auto, copy, aac or drop")
    if mode == "drop":
        return []
    codecs = []
    for track in audio:
        if mode == "aac":
            codecs.append("aac")
            continue
        # Ask this FFmpeg build whether MP4 can carry the actual input codec.
        # A one-packet probe avoids learning that after a long GPU render.
        index = track["index"]
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                source,
                "-map",
                f"0:{index}",
                "-c:a",
                "copy",
                "-frames:a",
                "1",
                "-f",
                "mp4",
                os.devnull,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            codecs.append("copy")
            continue
        reason = result.stderr.strip() or f"FFmpeg exited {result.returncode}"
        if mode == "copy":
            raise ValueError(
                f"Cannot copy audio stream {index} ({track['codec_name']}) "
                f"to MP4. Choose --audio auto, aac or drop.\n{reason}"
            )
        print(
            f"Audio stream copy failed for stream {index} "
            f"({track['codec_name']}); encoding this track as AAC.",
            flush=True,
        )
        codecs.append("aac")
    return codecs


def finalize_audio(silent, source, dest, audio, codecs, start, duration):
    if not codecs:
        silent.rename(dest)
        return
    # Normalize the audio timeline to the first displayed input video frame.
    # Keep short audio short; -t limits tracks extending beyond the video.
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-n",
        "-copyts",
        "-i",
        silent,
        "-itsoffset",
        f"{-float(start):.9f}",
        "-i",
        source,
        "-map",
        "0:v:0",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-t",
        f"{float(duration):.9f}",
        "-movflags",
        "+faststart",
    ]
    for i, codec in enumerate(codecs):
        cmd.extend([f"-c:a:{i}", codec])
        if codec == "aac":
            cmd.extend([f"-b:a:{i}", "192k"])
            track = audio[i]
            # AAC decoding can yield padded samples after the last edit.
            # Bound the filter by the track's end on the new video clock.
            end = duration
            if "start_pts" in track and "duration_ts" in track:
                last = track["start_pts"] + track["duration_ts"]
                end = min(end, last * Fraction(track["time_base"]) - start)
            cmd.extend([f"-filter:a:{i}", f"atrim=end={float(end):.9f}"])
    subprocess.run(
        [*cmd, dest], check=True, stderr=subprocess.PIPE, text=True
    )


def main(model, source):
    tracks = streams(source)
    v = video(tracks)
    validate_source(v, os.environ.get("VSR_DECODER", "auto"))
    width, height = v["width"], v["height"]
    raw_rate = v.get("avg_frame_rate", "0/0")
    rate = Fraction(v["r_frame_rate"] if raw_rate == "0/0" else raw_rate)
    if rate <= 0:
        raise ValueError("Video must have a positive, constant frame rate")
    count, first_pts = timestamps(source, v, rate)
    print(
        f"Input {width}x{height}, {count} frames at {rate} fps; "
        f"target {2 * width}x{2 * height}",
        flush=True,
    )
    audio = [t for t in tracks if t["codec_type"] == "audio"]
    codecs = audio_plan(source, audio, os.environ.get("VSR_AUDIO", "auto"))
    env = dict(
        os.environ,
        JOB_WIDTH=str(width),
        JOB_HEIGHT=str(height),
        JOB_RATE=str(rate),
        JOB_COUNT=str(count),
    )
    silent = WORK / "video.mp4"
    dest = WORK / "verified.mp4"
    print(f"Loading {model} on the GB300…", flush=True)
    subprocess.run(
        [sys.executable, "-u", SCRIPTS / f"{model}.py", source, silent],
        env=env,
        check=True,
    )
    print(
        "Finalizing MP4 and checking video timing and audio tracks…",
        flush=True,
    )
    finalize_audio(
        silent, source, dest, audio, codecs, first_pts, count / rate
    )
    out = streams(dest)
    w = video(out)
    frames, first = timestamps(dest, w, rate)
    if (
        (w["width"], w["height"], frames) != (2 * width, 2 * height, count)
        or first != 0
        or Fraction(w["avg_frame_rate"]) != rate
        or any(w.get(k) != expected for k, expected in OUTPUT_FORMAT.items())
    ):
        raise ValueError(
            f"Output video verification failed: {w['width']}x{w['height']}, "
            f"{frames} frames at {w.get('avg_frame_rate')}, first PTS {first}"
        )
    got = [t["codec_name"] for t in out if t["codec_type"] == "audio"]
    expected = [
        codec if codec == "aac" else track["codec_name"]
        for track, codec in zip(audio, codecs, strict=False)
    ]
    if expected != got:
        raise ValueError(
            f"Output audio verification expected {expected}, got {got}"
        )
    print(
        f"Verified {w['width']}x{w['height']}, {frames} frames "
        f"at {rate} fps, "
        f"{len(got)} audio track(s).",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main(sys.argv[1], Path(sys.argv[2]))
    except subprocess.CalledProcessError as error:
        reason = (error.stderr or "").strip()
        raise SystemExit(
            f"{Path(error.cmd[0]).name} exited {error.returncode}. {reason}"
        ) from None
    except (ValueError, KeyError, ZeroDivisionError) as error:
        raise SystemExit(str(error)) from None
