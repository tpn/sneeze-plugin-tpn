# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Trent Nelson
"""Validate the source, run one selected model, and verify the output MP4."""

import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

if __package__:
    from .encoder_geometry import resolve_encoder
    from .sizing import resolve_target
else:
    from encoder_geometry import resolve_encoder
    from sizing import resolve_target

SCRIPTS = Path(__file__).resolve().parent
WORK = Path("/output")
H264_NVDEC_FORMAT = {
    "codec_name": "h264",
    "pix_fmt": "yuv420p",
    "color_range": "tv",
    "color_space": "bt709",
    "color_transfer": "bt709",
    "color_primaries": "bt709",
    "chroma_location": "left",
}
OUTPUT_FORMAT = {
    k: v for k, v in H264_NVDEC_FORMAT.items() if k != "chroma_location"
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


def pixel_aspect_ratio(video_info):
    value = video_info.get("sample_aspect_ratio")
    if value in (None, "N/A", "0:1"):
        return Fraction(1)
    try:
        ratio = Fraction(value.replace(":", "/"))
        if ratio > 0:
            return ratio
    except (ValueError, ZeroDivisionError):
        pass
    raise ValueError(f"Invalid sample aspect ratio {value}") from None


def display_geometry(v):
    """Square-pixel raster in display orientation, rounded to whole pixels."""
    width = v["width"] * pixel_aspect_ratio(v)
    width = max(1, int(width + Fraction(1, 2)))
    height = v["height"]
    for side in v.get("side_data_list", []):
        rotation = float(side.get("rotation", 0)) % 360
        if rotation % 90:
            raise ValueError(
                f"Apply the {rotation:g}° display rotation before upscaling. "
                "Automatic orientation supports multiples of 90°."
            )
        if rotation in (90, 270):
            return height, width
    return width, height


def needs_display_transform(v):
    return pixel_aspect_ratio(v) != 1 or any(
        side.get("side_data_type") == "Display Matrix" or side.get("rotation")
        for side in v.get("side_data_list", [])
    )


def validate_source(v, decoder):
    if v.get("disposition", {}).get("attached_pic"):
        raise ValueError("Put the main video stream before the cover image")
    if (
        v.get("color_transfer") in ("smpte2084", "arib-std-b67")
        or v.get("color_primaries") == "bt2020"
    ):
        raise ValueError(
            "Convert HDR or wide-gamut input to SDR before upscaling"
        )
    if decoder != "nvdec":
        return
    if v["codec_name"] == "av1":
        if v["width"] % 2 or v["height"] % 2:
            # CUVID can resample into an even-sized surface. Cropping it
            # back does not reliably recover the original chroma samples.
            raise ValueError(
                "AV1 NVDEC requires even stored width and height. "
                "Use --decoder cpu for this source."
            )
        if v.get("pix_fmt") not in ("yuv420p", "yuv420p10le"):
            raise ValueError(
                f"AV1 NVDEC requires 8- or 10-bit 4:2:0, "
                f"input is {v.get('pix_fmt', 'unspecified')}. "
                "Use --decoder cpu for this source."
            )
        return
    if v["codec_name"] != "h264":
        raise ValueError(
            f"NVDEC input supports H.264 and AV1, got {v['codec_name']}. "
            "Use --decoder cpu for this source."
        )
    if needs_display_transform(v):
        raise ValueError(
            "Use --decoder cpu to normalize H.264 pixel aspect or orientation"
        )
    # The H.264 adapter writes these exact tags into its raw pipe.
    for key, expected in H264_NVDEC_FORMAT.items():
        if v.get(key) != expected:
            actual = v.get(key, "unspecified")
            raise ValueError(
                f"H.264 NVDEC needs {key}={expected}, input is {actual}. "
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
    decoder = os.environ.get("VSR_DECODER", "cpu")
    validate_source(v, decoder)
    width, height = display_geometry(v)
    target_width, target_height = resolve_target(
        width,
        height,
        scale=os.getenv("VSR_SCALE"),
        target_size=os.getenv("VSR_TARGET_SIZE"),
    )
    # OpenCV's bundled FFmpeg lacks a working software AV1 decoder here.
    # Keep the measured H.264/HEVC path for normal square-pixel material.
    if decoder == "nvdec":
        ffmpeg_input = v["codec_name"] == "av1"
        input_label = "RTX NVDEC"
    else:
        ffmpeg_input = needs_display_transform(v) or v["codec_name"] not in (
            "h264",
            "hevc",
        )
        input_label = "FFmpeg CPU" if ffmpeg_input else "OpenCV CPU"
    raw_rate = v.get("avg_frame_rate", "0/0")
    rate = Fraction(v["r_frame_rate"] if raw_rate == "0/0" else raw_rate)
    if rate <= 0:
        raise ValueError("Video must have a positive, constant frame rate")
    count, first_pts = timestamps(source, v, rate)
    print(
        f"Input {v['codec_name']} {v['width']}x{v['height']} stored "
        f"(SAR {v.get('sample_aspect_ratio') or 'unspecified'}), "
        f"{width}x{height} displayed, {count} frames at {rate} fps; "
        f"target {target_width}x{target_height} ({input_label} input)",
        flush=True,
    )
    audio = [t for t in tracks if t["codec_type"] == "audio"]
    codecs = audio_plan(source, audio, os.environ.get("VSR_AUDIO", "auto"))
    env = dict(
        os.environ,
        VSR_ENCODER=resolve_encoder(
            os.environ.get("VSR_ENCODER", "cpu"),
            os.environ.get("VSR_ENCODER_REQUEST", "auto"),
            target_width,
            target_height,
            rate,
        ),
        JOB_WIDTH=str(width),
        JOB_HEIGHT=str(height),
        JOB_TARGET_WIDTH=str(target_width),
        JOB_TARGET_HEIGHT=str(target_height),
        JOB_RATE=str(rate),
        JOB_COUNT=str(count),
        JOB_INPUT_CODEC=v["codec_name"],
        JOB_INPUT_PIX_FMT=v.get("pix_fmt", ""),
        JOB_INPUT_COLOR_RANGE=v.get("color_range", ""),
        JOB_FFMPEG_INPUT="1" if ffmpeg_input else "0",
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
        (w["width"], w["height"], frames)
        != (target_width, target_height, count)
        or first != 0
        or Fraction(w["avg_frame_rate"]) != rate
        or pixel_aspect_ratio(w) != 1
        or display_geometry(w) != (w["width"], w["height"])
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
