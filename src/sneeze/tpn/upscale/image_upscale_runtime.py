# Copyright (c) 2026 Trent Nelson
# SPDX-License-Identifier: Apache-2.0

"""Heavyweight runtime entry point for the local HAT-L image upscaler."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe() -> dict[str, Any]:
    import torch

    cuda_available = torch.cuda.is_available()
    payload: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "spandrel": importlib.metadata.version("spandrel"),
        "pillow": importlib.metadata.version("Pillow"),
        "numpy": importlib.metadata.version("numpy"),
        "cuda_available": cuda_available,
        "torch_cuda": torch.version.cuda,
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        payload["gpu"] = {
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "bfloat16_supported": torch.cuda.is_bf16_supported(),
        }
        test = torch.ones((16, 16), device="cuda")
        payload["cuda_probe_sum"] = float((test @ test).sum().item())
    return payload


def _image_to_tensor(image, *, torch, np, device, dtype):
    array = np.asarray(image, dtype=np.uint8)
    tensor = (
        torch.from_numpy(array.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(dtype=torch.float32)
        .div_(255.0)
    )
    return tensor.to(device=device, dtype=dtype)


def _tensor_to_image(tensor, *, Image, np):
    array = (
        tensor.squeeze(0)
        .permute(1, 2, 0)
        .float()
        .mul(255.0)
        .round()
        .clamp_(0, 255)
        .to("cpu")
        .numpy()
        .astype(np.uint8)
    )
    return Image.fromarray(array)


def _run_tiled(model, tensor, *, torch, tile_size: int, overlap: int):
    _, _, height, width = tensor.shape
    scale = model.scale
    output = torch.empty(
        (1, model.output_channels, height * scale, width * scale),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    for top in range(0, height, tile_size):
        bottom = min(height, top + tile_size)
        input_top = max(0, top - overlap)
        input_bottom = min(height, bottom + overlap)
        for left in range(0, width, tile_size):
            right = min(width, left + tile_size)
            input_left = max(0, left - overlap)
            input_right = min(width, right + overlap)
            tile = tensor[
                ...,
                input_top:input_bottom,
                input_left:input_right,
            ]
            tile_output = model(tile)
            crop_top = (top - input_top) * scale
            crop_left = (left - input_left) * scale
            crop_bottom = crop_top + (bottom - top) * scale
            crop_right = crop_left + (right - left) * scale
            output[
                ...,
                top * scale : bottom * scale,
                left * scale : right * scale,
            ] = tile_output[
                ...,
                crop_top:crop_bottom,
                crop_left:crop_right,
            ]
    return output


def _publish_png(temporary_path: str, output: Path, *, force: bool) -> None:
    """Publish one complete PNG without a no-clobber race."""

    os.chmod(temporary_path, 0o644)
    if force:
        os.replace(temporary_path, output)
        return
    try:
        os.link(temporary_path, output)
    except FileExistsError as exc:
        raise ValueError(f"Output appeared during upscale: {output}") from exc
    os.unlink(temporary_path)


def upscale(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image
    from spandrel import ImageModelDescriptor, ModelLoader

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Source image does not exist: {source}")
    if not model_path.is_file():
        raise ValueError(f"Model does not exist: {model_path}")
    if source == output:
        raise ValueError("Source and output paths must be different")
    if output.exists() and not args.force:
        raise ValueError(f"Output already exists: {output}")
    if args.target_width <= 0:
        raise ValueError("Target width must be greater than zero")
    if args.tile_size < 0 or args.tile_overlap < 0:
        raise ValueError("Tile size and overlap cannot be negative")
    if args.tile_size and args.tile_overlap >= args.tile_size:
        raise ValueError("Tile overlap must be smaller than tile size")

    expected_model_hash = args.model_sha256.lower()
    actual_model_hash = file_sha256(model_path)
    if actual_model_hash != expected_model_hash:
        raise ValueError(
            "Model SHA-256 mismatch: "
            f"expected {expected_model_hash}, got {actual_model_hash}"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available in the upscale runtime")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    descriptor = ModelLoader().load_from_file(model_path)
    if not isinstance(descriptor, ImageModelDescriptor):
        raise ValueError("Checkpoint is not an image-to-image model")
    if descriptor.purpose != "SR" or descriptor.scale != 2:
        raise ValueError(
            "Checkpoint must be a native x2 super-resolution model"
        )
    if descriptor.input_channels != 3 or descriptor.output_channels != 3:
        raise ValueError("Checkpoint must accept and produce RGB images")
    descriptor.to(device=device, dtype=dtype).eval()

    with Image.open(source) as opened:
        opened.load()
        source_info = dict(opened.info)
        source_mode = opened.mode
        alpha = None
        if "A" in opened.getbands():
            alpha = opened.getchannel("A")
        rgb = opened.convert("RGB")
    source_width, source_height = rgb.size
    if args.target_width <= source_width:
        raise ValueError(
            f"Target width {args.target_width} does not upscale "
            f"source width {source_width}"
        )
    target_height = round(source_height * args.target_width / source_width)
    tensor = _image_to_tensor(
        rgb,
        torch=torch,
        np=np,
        device=device,
        dtype=dtype,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        if args.tile_size:
            result_tensor = _run_tiled(
                descriptor,
                tensor,
                torch=torch,
                tile_size=args.tile_size,
                overlap=args.tile_overlap,
            )
        else:
            result_tensor = descriptor(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    peak_memory_allocated = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None
    )
    peak_memory_reserved = (
        torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None
    )
    result = _tensor_to_image(result_tensor, Image=Image, np=np)
    native_dimensions = list(result.size)
    if result.size != (args.target_width, target_height):
        result = result.resize(
            (args.target_width, target_height),
            Image.Resampling.LANCZOS,
        )
    if alpha is not None:
        alpha = alpha.resize(result.size, Image.Resampling.LANCZOS)
        result.putalpha(alpha)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".png",
            delete=False,
        ) as handle:
            temporary_path = handle.name
        save_options: dict[str, Any] = {
            "format": "PNG",
            "compress_level": 9,
        }
        if source_info.get("icc_profile"):
            save_options["icc_profile"] = source_info["icc_profile"]
        result.save(temporary_path, **save_options)
        _publish_png(temporary_path, output, force=args.force)
        temporary_path = None
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass

    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": properties.name,
            "compute_capability": list(
                torch.cuda.get_device_capability(device)
            ),
            "total_memory_bytes": properties.total_memory,
            "peak_memory_allocated_bytes": peak_memory_allocated,
            "peak_memory_reserved_bytes": peak_memory_reserved,
        }
    return {
        "schema_version": 1,
        "kind": "sneeze_tpn_image_upscale",
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "path": str(source),
            "sha256": file_sha256(source),
            "size_bytes": source.stat().st_size,
            "mode": source_mode,
            "dimensions": [source_width, source_height],
        },
        "output": {
            "path": str(output),
            "sha256": file_sha256(output),
            "size_bytes": output.stat().st_size,
            "mode": result.mode,
            "dimensions": list(result.size),
        },
        "model": {
            "path": str(model_path),
            "sha256": actual_model_hash,
            "architecture": str(descriptor.architecture.id),
            "tags": descriptor.tags,
            "native_scale": descriptor.scale,
            "native_output_dimensions": native_dimensions,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torchvision": importlib.metadata.version("torchvision"),
            "spandrel": importlib.metadata.version("spandrel"),
            "device": str(device),
            "dtype": args.dtype,
            "gpu": gpu,
        },
        "parameters": {
            "target_width": args.target_width,
            "target_height": target_height,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "post_resize": native_dimensions
            != [args.target_width, target_height],
        },
        "timing": {"inference_seconds": inference_seconds},
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--model")
    parser.add_argument("--model-sha256")
    parser.add_argument("--target-width", type=int, default=3840)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not args.probe:
        for name in ("source", "output", "model", "model_sha256"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = probe() if args.probe else upscale(args)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
