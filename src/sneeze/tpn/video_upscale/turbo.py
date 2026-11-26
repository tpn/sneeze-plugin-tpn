# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Trent Nelson
"""Render a silent, lossless 2× FlashVSR+ Turbo video."""

import argparse
import os
from fractions import Fraction
from pathlib import Path

import decoder_speed
import gpu_writer
import inference
import numpy as np
import sparse_fa4
import torch
import turbo_input
from inference import InferenceRequest, build_inference_pipeline, pipeline
from inference import output as video_output
from inference.config import INPUT_MULTIPLE
from inference.output import IncrementalVideoWriter
from inference.preprocessing import input as preprocessing
from inference.presets import model_paths_for_mode, stream_config_for_mode
from inference.runners import stream


def install_cpu_encoder(rate):
    get_writer = video_output.imageio.get_writer

    def lossless_x264(*args, **kwargs):
        # imageio-ffmpeg rounds its input frame rate to two decimal places.
        # FFmpeg uses our later, exact input rate for e.g. 24000/1001.
        kwargs["quality"] = None
        kwargs["input_params"] = ["-r", str(rate)]
        kwargs["ffmpeg_log_level"] = "error"
        kwargs["output_params"] += [
            "-vf",
            "scale=out_color_matrix=bt709:out_range=tv",
            "-qp",
            "0",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_primaries",
            "bt709",
            "-threads",
            "8",
        ]
        return get_writer(*args, **kwargs)

    video_output.imageio.get_writer = lossless_x264


def install_io(height, width, rate, frames, decoder, encoder, lookahead):
    # The DiT takes 128-aligned images. Pad before the 2× resize so the PIL
    # filter and all original pixels keep their scale and center.
    grid = INPUT_MULTIPLE // 2
    extra_h, extra_w = -height % grid, -width % grid
    top, left = extra_h // 2, extra_w // 2
    resize = preprocessing.pil_center_crop_resize

    def pad_resize(frames, target_height, target_width, **kwargs):
        if frames.shape[1:3] != (height, width):
            raise ValueError(
                f"Decoded dimensions changed to {frames.shape[1:3]}"
            )
        if extra_h or extra_w:
            frames = np.pad(
                frames,
                (
                    (0, 0),
                    (top, extra_h - top),
                    (left, extra_w - left),
                    (0, 0),
                ),
                mode="edge",
            )
        return resize(frames, target_height, target_width, **kwargs)

    preprocessing.pil_center_crop_resize = pad_resize
    if decoder == "nvdec":
        import decode_rtx

        decode_rtx.install()
    if lookahead:
        turbo_input.install_input_buffers(preprocessing, lookahead)
        read_chunks = pipeline.iter_lq_video

        def read_ahead(request, **kwargs):
            if request.lq_video is not None:
                return read_chunks(request, **kwargs)
            return turbo_input.prefetch(
                lambda: read_chunks(request, **kwargs), lookahead
            )

        pipeline.iter_lq_video = read_ahead
    if encoder == "cpu":
        install_cpu_encoder(rate)
    else:
        import encode_rtx

        encode_rtx.install()
    gpu_writer.install(IncrementalVideoWriter)
    append = IncrementalVideoWriter.append

    def crop_and_encode(writer, video, **kwargs):
        if encoder == "nvenc" and writer.frame_count == 0:
            # Give FFmpeg the exact CFR fraction, including NTSC rates.
            writer.writer.fps = rate
        crop = video[
            ..., 2 * top : 2 * (top + height), 2 * left : 2 * (left + width)
        ]
        append(writer, crop, **kwargs)
        count = writer.frame_count

        # The async writer's future resolves after all RGB frames in this
        # segment reach FFmpeg. Encoding overlaps the next segment.
        def report(future):
            if future.exception() is None:
                print(f"VSR_PROGRESS {count} {frames}", flush=True)

        writer._pending.add_done_callback(report)

    IncrementalVideoWriter.append = crop_and_encode

    # media.py owns audio and final validation for both models.
    def finish_silent(encoded_video, source, destination):
        os.replace(encoded_video, destination)

    pipeline.mux_source_audio = finish_silent
    return 2 * (height + extra_h), 2 * (width + extra_w)


def keep_dit_resident(bundle, state, device):
    if state.attention_cache_offload_layers or device.type != "cuda":
        raise RuntimeError(
            "Turbo requires GPU-resident transformer and attention caches"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    height, width = (
        int(os.environ["JOB_HEIGHT"]),
        int(os.environ["JOB_WIDTH"]),
    )
    rate = Fraction(os.environ["JOB_RATE"])
    expected = int(os.environ["JOB_COUNT"])
    decoder = os.getenv("VSR_DECODER", "auto")
    encoder = os.getenv("VSR_ENCODER", "auto")
    mode = os.getenv("VSR_TURBO_DECODER", "fast")
    microbatch = int(os.getenv("VSR_TURBO_MICROBATCH", "2"))
    seed = int(os.getenv("VSR_SEED", "1"))
    if decoder == "auto":
        decoder = "cpu"
    if encoder == "auto":
        encoder = "nvenc"
    if decoder not in ("cpu", "nvdec") or encoder not in ("cpu", "nvenc"):
        parser.error("Choose cpu/nvdec for input and cpu/nvenc for output")
    try:
        lookahead = turbo_input.resolve_lookahead(
            os.getenv("VSR_TURBO_INPUT"), os.getenv("VSR_TURBO_LOOKAHEAD")
        )
    except ValueError as error:
        parser.error(str(error))
    if mode not in ("fast", "exact") or microbatch not in (1, 2, 4):
        parser.error(
            "Turbo decoder must be fast or exact, with microbatch 1, 2 or 4"
        )
    if not 0 <= seed <= 2**32 - 1:
        parser.error("The seed must be between 0 and 4294967295")
    if (
        torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "NVIDIA GB300"
    ):
        raise RuntimeError(
            "Expose only the GB300 to Turbo using CUDA_VISIBLE_DEVICES"
        )
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(
        220 * 2**30 / torch.cuda.get_device_properties(0).total_memory
    )
    sparse_fa4.install()
    stream._offload_dit = stream._restore_dit = keep_dit_resident
    canvas_height, canvas_width = install_io(
        height, width, rate, expected, decoder, encoder, lookahead
    )
    source_dir = Path(inference.__file__).resolve().parents[1]
    weights = Path(decoder_speed.__file__).resolve().parent / "models/turbo"
    config = stream_config_for_mode(
        "turbo",
        seed=seed,
        decode_every=4,
        tiny_decoder_mode="parallel",
        cache_offload_mode="never",
        attn_kernel="triton_block_sparse",
    )
    model = build_inference_pipeline(
        config, model_paths_for_mode(source_dir, "turbo", weights)
    )
    variant = "packed-poly" if mode == "fast" else "packed-swap"
    decoder_speed.install(
        model.bundle.decoder, variant, chunk=microbatch, gpu_output=True
    )
    result = model.infer(
        InferenceRequest(
            input_path=args.input,
            output_path=args.output,
            height=canvas_height,
            width=canvas_width,
            fps=float(rate),
        ),
        progress=print,
    )
    if result.frames != expected or not sparse_fa4.dispatches:
        raise RuntimeError(
            f"Turbo returned {result.frames}/{expected} frames with "
            f"{sum(sparse_fa4.dispatches.values())} FA4 sparse calls"
        )
    print(
        f"Turbo rendered {result.frames} frames at "
        f"{2 * width}×{2 * height}, {rate} fps",
        flush=True,
    )


if __name__ == "__main__":
    main()
