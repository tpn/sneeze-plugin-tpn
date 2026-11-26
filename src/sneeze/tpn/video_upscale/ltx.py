# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Trent Nelson
"""Upscale CFR video with LTX-2.5's tiled Refine Details workflow."""

import gc
import logging
import os
import sys
import time
import types
from collections import deque
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

if __package__:
    from .worker_canvas import ModelCanvas, target_dimensions
else:
    from worker_canvas import ModelCanvas, target_dimensions

# Distilled LTX-2.5's published eight-step Euler schedule.
SIGMAS = (
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
)
TILE_FRAMES = 97


def padded_frames(count):
    return 1 + 8 * ((count + 6) // 8)


def overlap_frames(chunk):
    if chunk < 9 or chunk > 241 or chunk % 8 != 1:
        raise ValueError("LTX chunk must be 8n+1 frames from 9 through 241")
    return min(33, 1 + 8 * ((chunk - 1) // 16))


def input_chunks(reader, count, chunk):
    overlap = overlap_frames(chunk)
    frames = []
    start = 0
    while start < count:
        end = min(start + chunk, count)
        while len(frames) < end - start:
            ok, frame = reader.read()
            if not ok:
                raise RuntimeError(
                    f"Missing input frame {start + len(frames)}"
                )
            frames.append(frame)
        yield start, frames
        if end == count:
            return
        frames = frames[-overlap:]
        start = end - overlap


def join_chunk(frames, previous_tail, overlap, last, append):
    """Blend shared frames once, holding only the next chunk's overlap."""
    import cv2

    tail = deque()
    for index, frame in enumerate(frames):
        if index < len(previous_tail):
            weight = (index + 1) / (len(previous_tail) + 1)
            frame = cv2.addWeighted(
                previous_tail[index], 1 - weight, frame, weight, 0
            )
        tail.append(frame)
        if last or len(tail) > overlap:
            append(tail.popleft())
    return list(tail)


def init_comfy(root):
    # Comfy's import namespace must start inside its checkout, ahead of the
    # sibling directory of the same name that holds top-level utilities.
    sys.path.insert(0, str(root / "comfy"))
    import comfy.options

    sys.argv = [
        "ltx",
        "--bf16-unet",
        "--bf16-text-enc",
        "--bf16-vae",
        "--use-pytorch-cross-attention",
    ]
    comfy.options.enable_args_parsing()
    # Load only the official node modules used by the headless workflow.
    package = types.ModuleType("ltxnodes")
    package.__path__ = [str(root / "ltx-nodes")]
    sys.modules[package.__name__] = package


def load_models(root, prompt, strength, rate):
    import folder_paths
    import nodes
    from comfy_extras.nodes_lt import LTXVConditioning
    from ltxnodes.iclora import LTXICLoRALoaderModelOnly
    from weights import GEMMA, REFINER, TRANSFORMER, VIDEO_VAE

    for kind in ("diffusion_models", "text_encoders", "vae", "loras"):
        folder = root / "models"
        if kind != "loras":
            folder = folder / kind
        folder_paths.add_model_folder_path(kind, str(folder), is_default=True)
    (model,) = nodes.UNETLoader().load_unet(Path(TRANSFORMER).name, "default")
    model, factor = LTXICLoRALoaderModelOnly.execute(
        model, REFINER, strength
    ).result
    if factor != 1:
        raise ValueError(
            f"Unexpected Refine Details reference downscale factor {factor}"
        )
    (vae,) = nodes.VAELoader().load_vae(Path(VIDEO_VAE).name)
    (clip,) = nodes.CLIPLoader().load_clip(
        Path(GEMMA).name, "ltxv", "default"
    )
    (positive,) = nodes.CLIPTextEncode().encode(clip, prompt)
    (negative,) = nodes.CLIPTextEncode().encode(clip, "")
    positive, negative = LTXVConditioning.execute(
        positive, negative, float(rate)
    ).result
    return model, vae, positive, negative


def guide_latents(frames, models, canvas):
    import cv2
    import nodes
    import torch
    from comfy_extras.nodes_lt import EmptyLTXVLatentVideo
    from ltxnodes.iclora import LTXAddVideoICLoRAGuide

    _, vae, positive, negative = models
    w, h = canvas.padded_width, canvas.padded_height
    count = padded_frames(len(frames))
    guide = torch.empty((count, h, w, 3), dtype=torch.float32, device="cpu")
    scaler = nodes.ImageScale()

    def lanczos(frames, width, height):
        image = torch.from_numpy(frames).float().div_(255)
        (scaled,) = scaler.upscale(
            image, "lanczos", width, height, "disabled"
        )
        return scaled.numpy()

    for i, bgr in enumerate(frames):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        guide[i] = torch.from_numpy(canvas.prepare(rgb[None], lanczos)[0])
    if len(frames) < count:
        guide[len(frames) :] = guide[len(frames) - 1]
    (empty,) = EmptyLTXVLatentVideo.execute(w, h, count, 1).result
    print("LTX Phase encoding the Refine Details video guide", flush=True)
    return LTXAddVideoICLoRAGuide.execute(
        positive,
        negative,
        vae,
        empty,
        guide,
        0,
        1.0,
        1.0,
        "disabled",
        False,
        256,
        64,
        True,
        TILE_FRAMES,
    ).result


@contextmanager
def denoise_progress(total):
    from ltxnodes import tiled_fusion_sampler as fusion

    original = fusion._one_sampler_step
    completed = 0
    last = time.monotonic()

    def step(*args, **kwargs):
        nonlocal completed, last
        result = original(*args, **kwargs)
        completed += 1
        now = time.monotonic()
        if completed == total or now - last >= 15:
            print(f"LTX Phase denoising tile {completed}/{total}", flush=True)
            last = now
        return result

    fusion._one_sampler_step = step
    try:
        yield
    finally:
        fusion._one_sampler_step = original


def denoise(frames, models, canvas, seed):
    import comfy.samplers
    import torch
    from attention_backend import calls, install
    from ltxnodes.tiled_fusion_sampler import LTXVTiledFusionSampler
    from ltxnodes.tiled_fusion_sizes import (
        resolve_tiling_plan,
        spatial_tile_grid,
        temporal_chunk_count,
    )

    model, vae, _, _ = models
    positive, negative, guide = guide_latents(frames, models, canvas)
    count = padded_frames(len(frames))
    plan = resolve_tiling_plan(
        canvas.source_width,
        canvas.source_height,
        count,
        tile_size="HD",
        output_size="4K",
    )
    nx, ny = spatial_tile_grid(
        canvas.padded_width,
        canvas.padded_height,
        plan.tile_width,
        plan.tile_height,
        plan.overlap_frac,
    )
    total = (
        nx * ny * temporal_chunk_count(count, TILE_FRAMES) * (len(SIGMAS) - 1)
    )
    print(
        f"LTX Phase denoising {total} tiles across eight Euler steps",
        flush=True,
    )
    calls.clear()
    install("fa4")
    try:
        with denoise_progress(total):
            (sampled,) = LTXVTiledFusionSampler().sample(
                model,
                positive,
                negative,
                guide,
                torch.tensor(SIGMAS),
                comfy.samplers.sampler_object("euler"),
                seed,
                1.0,
                tile_width=plan.tile_width,
                tile_height=plan.tile_height,
                overlap_frac=plan.overlap_frac,
                blend_var=0.05,
                grid_cycle=1,
                tile_frames=TILE_FRAMES,
                vae=vae,
                canvas_device="auto",
            )
    finally:
        install("native")
    fa4_calls = sum(n for (_, selected, _, _), n in calls.items() if selected)
    print(f"LTX Phase denoising complete, {fa4_calls} FA4 calls", flush=True)
    return sampled


def decode_vae(sampled, vae, vae_tile):
    import comfy.model_management
    import nodes

    decoder = nodes.VAEDecodeTiled()
    print(f"LTX Phase decoding with {vae_tile}-pixel VAE tiles", flush=True)
    try:
        return decoder.decode(vae, sampled, vae_tile, 64, 128, 32)[0]
    except Exception as error:
        if vae_tile <= 1536 or not comfy.model_management.is_oom(error):
            raise
    # Release the failed allocation and traceback before the second decode.
    print("LTX Phase retrying VAE decode with 1536-pixel tiles", flush=True)
    gc.collect()
    comfy.model_management.soft_empty_cache(force=True)
    return decoder.decode(vae, sampled, 1536, 64, 128, 32)[0]


def decoded_rgb(frames, models, canvas, seed, vae_tile):
    import torch

    sampled = denoise(frames, models, canvas, seed)
    gc.collect()
    decoded = decode_vae(sampled, models[1], vae_tile)
    w, h = canvas.padded_width, canvas.padded_height
    if tuple(decoded.shape) != (padded_frames(len(frames)), h, w, 3):
        raise RuntimeError(
            f"Unexpected LTX output shape {tuple(decoded.shape)}"
        )
    rows, columns = canvas.content
    for frame in decoded[: len(frames)]:
        crop = frame[rows, columns]
        yield (
            crop.cpu()
            .float()
            .clamp(0, 1)
            .mul_(255)
            .round_()
            .to(torch.uint8)
            .contiguous()
            .numpy()
        )


def main(source, output):
    import torch
    from worker_io import open_encoder, open_reader

    root = Path.cwd()
    width = int(os.environ["JOB_WIDTH"])
    height = int(os.environ["JOB_HEIGHT"])
    canvas = ModelCanvas(width, height, *target_dimensions(width, height), 32)
    count = int(os.environ["JOB_COUNT"])
    rate = Fraction(os.environ["JOB_RATE"])
    chunk = int(os.environ["VSR_LTX_CHUNK"])
    overlap = overlap_frames(chunk)
    vae_tile = int(os.environ["VSR_LTX_VAE_TILE"])
    if vae_tile not in (512, 1024, 1536, 2048):
        raise ValueError("LTX VAE tile must be 512, 1024, 1536 or 2048")
    if (
        torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "NVIDIA GB300"
    ):
        raise RuntimeError(
            "LTX requires the GB300 as the sole visible CUDA device"
        )
    init_comfy(root)
    seed = int(os.environ["VSR_SEED"])
    prompt = os.environ["VSR_LTX_PROMPT"]
    strength = float(os.environ["VSR_LTX_STRENGTH"])
    written = 0
    print(f"VSR_PROGRESS 0 {count}", flush=True)
    with torch.inference_mode():
        models = load_models(root, prompt, strength, rate)
        with (
            open_reader(source, "cpu", count) as reader,
            open_encoder(
                output, "cpu", canvas.width, canvas.height, str(rate)
            ) as append,
        ):

            def write(frame):
                nonlocal written
                append(frame)
                written += 1
                if written % 16 == 0:
                    print(f"VSR_PROGRESS {written} {count}", flush=True)

            tail = []
            for start, frames in input_chunks(reader, count, chunk):
                end = start + len(frames)
                print(
                    f"LTX Chunk frames {start + 1}–{end} of {count}",
                    flush=True,
                )
                rgb = decoded_rgb(frames, models, canvas, seed, vae_tile)
                tail = join_chunk(rgb, tail, overlap, end == count, write)
                print(f"VSR_PROGRESS {written} {count}", flush=True)
    if written != count:
        raise RuntimeError(f"LTX wrote {written} of {count} expected frames")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(*map(Path, sys.argv[1:]))
