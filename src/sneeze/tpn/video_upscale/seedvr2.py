# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Trent Nelson
"""Run one SeedVR2-7B upscale in the pinned inference container."""

import os
import sys
from functools import partial
from pathlib import Path

import attention
import inference_cli as engine
import norm_kernel
import numpy as np
import seed_input
import torch
import tuning
from worker_canvas import target_dimensions
from worker_io import open_encoder, open_reader


def main():
    source, output = map(Path, sys.argv[1:])
    source_width = int(os.environ["JOB_WIDTH"])
    source_height = int(os.environ["JOB_HEIGHT"])
    width, height = target_dimensions(source_width, source_height)
    rate = os.environ["JOB_RATE"]
    frame_count = int(os.environ["JOB_COUNT"])
    seed = int(os.environ.get("VSR_SEED", "42"))
    batch = int(os.environ.get("VSR_SEED_BATCH", "17"))
    chunk_size = int(os.environ.get("VSR_SEED_CHUNK", "53"))
    overlap = int(os.environ.get("VSR_SEED_OVERLAP", "4"))
    lookahead = seed_input.resolve_lookahead(os.getenv("VSR_SEED_LOOKAHEAD"))
    encoder = os.environ.get("VSR_ENCODER", "auto")
    decoder = os.environ.get("VSR_DECODER", "auto")
    if not 0 <= seed <= 2**32 - 1 - 1_000_000:
        raise ValueError(
            "SeedVR2 seed must be in 0..4293967295 (the VAE adds 1000000)"
        )
    if batch < 1 or batch % 4 != 1:
        raise ValueError("SeedVR2 batch must be 1, 5, 9, 13, 17, ... (4n+1)")
    if chunk_size < 1 or not 0 <= overlap <= min(batch - 1, chunk_size):
        raise ValueError(
            "Require chunk > 0 and 0 <= overlap <= min(batch - 1, chunk)"
        )
    if encoder not in ("auto", "cpu", "nvenc") or decoder not in (
        "auto",
        "cpu",
        "nvdec",
    ):
        raise ValueError(
            "Choose encoder auto/cpu/nvenc and decoder auto/cpu/nvdec"
        )

    if (
        torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "NVIDIA GB300"
    ):
        raise RuntimeError(
            "SeedVR2 requires the GB300 as the sole visible CUDA device"
        )
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(
        238 * 2**30 / torch.cuda.get_device_properties(0).total_memory
    )
    model_dir = Path(tuning.__file__).parent / "models"
    dit = "seedvr2_ema_7b_fp16.safetensors"
    for weight in (dit, engine.DEFAULT_VAE):
        if not (model_dir / weight).is_file():
            raise FileNotFoundError(model_dir / weight)

    sys.argv = [
        engine.__file__,
        str(source),
        "--model_dir",
        str(model_dir),
        "--dit_model",
        dit,
        "--resolution",
        str(min(width, height)),
        "--max_resolution",
        str(max(width, height)),
        "--batch_size",
        str(batch),
        "--chunk_size",
        str(chunk_size),
        "--temporal_overlap",
        str(overlap),
        "--cache_dit",
        "--cache_vae",
        "--dit_offload_device",
        "cuda:0",
        "--vae_offload_device",
        "cuda:0",
        "--tensor_offload_device",
        "cpu",
        "--seed",
        str(seed),
        "--attention_mode",
        "sdpa",
        "--output_format",
        "mp4",
        "--video_backend",
        "ffmpeg",
    ]
    if (width, height) != (2 * source_width, 2 * source_height):
        from src.core import generation_utils
        from torchvision.transforms import InterpolationMode, Resize

        # The stock shortest-edge transform rounds the longer side down,
        # which can miss the requested even raster by a pixel or two.
        # Both its shape probe and inference pipeline use this factory.
        resize = Resize(
            (height, width), InterpolationMode.BICUBIC, antialias=True
        )

        resize_input = partial(seed_input.resize_frames, resize)
        generation_utils.NaResize = lambda **kwargs: resize_input

    args = engine.parse_arguments()
    engine.debug.enabled = False
    attention.install()
    tuning.install_production("triton-fused")

    written = 0
    print(f"VSR_PROGRESS 0 {frame_count}", flush=True)
    with (
        seed_input.open_input(
            engine,
            lambda: open_reader(source, decoder, frame_count),
            frame_count,
            args.chunk_size,
            lookahead,
        ) as reader,
        open_encoder(output, encoder, width, height, rate) as append,
    ):
        chunks = engine._stream_video_chunks(
            cap=reader,
            frames_to_process=frame_count,
            chunk_size=args.chunk_size,
            overlap=args.temporal_overlap,
            args=args,
            device_id="0",
            debug=engine.debug,
            runner_cache={},
            log_progress=True,
            total_chunks=(frame_count + args.chunk_size - 1)
            // args.chunk_size,
            cleanup_timer_name="chunk_cleanup",
        )
        for chunk in chunks:
            if tuple(chunk.shape[1:]) != (height, width, 3):
                raise RuntimeError(
                    f"Expected {width}x{height} RGB, got {tuple(chunk.shape)}"
                )
            for frame in chunk:
                # Truncate RGB values exactly as in the pinned CLI.
                rgb = np.ascontiguousarray(
                    frame.cpu().numpy() * 255.0, dtype=np.uint8
                )
                append(rgb)
            written += len(chunk)
            print(f"VSR_PROGRESS {written} {frame_count}", flush=True)
    if written != frame_count:
        raise RuntimeError(
            f"SeedVR2 wrote {written} of {frame_count} expected frames"
        )
    fa4_calls = sum(
        n for key, n in attention.counts.items() if key[0] == "FA4_packed"
    )
    fused_calls = norm_kernel.call_counts["triton_groupnorm_silu"]
    if not fa4_calls or not fused_calls:
        raise RuntimeError(
            "The expected SeedVR2 FA4 and fused VAE kernels did not run"
        )
    print(
        f"SeedVR2 wrote {written} frames, {fa4_calls} FA4 calls, "
        f"{fused_calls} fused VAE calls",
        flush=True,
    )


if __name__ == "__main__":
    main()
