# Copyright (c) 2026 Trent Nelson
# SPDX-License-Identifier: Apache-2.0

"""Sneeze commands for the isolated local image upscaler."""

from __future__ import annotations

import json

from sneeze.command import CommandError
from sneeze.commandinvariant import InvariantAwareCommand
from sneeze.invariant import (
    BoolInvariant,
    NonNegativeIntegerInvariant,
    PathInvariant,
    PositiveIntegerInvariant,
    StringInvariant,
)


def _option_value(
    command,
    name: str,
    default=None,
    *,
    private_first: bool = False,
):
    names = (f"_{name}", name) if private_first else (name, f"_{name}")
    for attr_name in names:
        value = getattr(command, attr_name, None)
        if value not in (None, ""):
            return value
    return default


class Setup(InvariantAwareCommand):
    """Install the pinned Blackwell runtime and HAT-L x2 checkpoint.

    This creates an isolated Python runtime under user data and downloads a
    hash-pinned model under the user cache. It does not add Torch or model
    dependencies to the lightweight Sneeze control environment. Use
    `--dry-run` to inspect the exact environment plan without changing files.

    Next: run `sne-tpn-upscale run --source IMAGE.png`.
    """

    runtime_dir = None
    _runtime_dir = None

    class RuntimeDirArg(StringInvariant):
        _help = "Isolated runtime directory; defaults under user data."
        _mandatory = False

    model_cache = None
    _model_cache = None

    class ModelCacheArg(StringInvariant):
        _help = "Model directory; defaults under the Sneeze user cache."
        _mandatory = False

    uv_executable = None
    _uv_executable = None

    class UvExecutableArg(StringInvariant):
        _help = "uv executable name or path. [default: %default]"
        _mandatory = False
        _default = "uv"

    cuda_visible_devices = None
    _cuda_visible_devices = None

    class CudaVisibleDevicesArg(StringInvariant):
        _help = (
            "Physical GPU selection for the setup probe. "
            "[default: %default]"
        )
        _mandatory = False
        _default = "0"

    dry_run = None

    class DryRunArg(BoolInvariant):
        _help = (
            "Print the setup receipt without changing files. "
            "[default: %default]"
        )
        _mandatory = False
        _default = False

    def run(self):
        from .image_upscale import ImageUpscaleError, setup_upscaler

        try:
            receipt = setup_upscaler(
                runtime_dir=_option_value(self, "runtime_dir"),
                model_cache=_option_value(self, "model_cache"),
                uv_executable=_option_value(self, "uv_executable", "uv"),
                cuda_visible_devices=_option_value(
                    self,
                    "cuda_visible_devices",
                    "0",
                ),
                dry_run=bool(self.dry_run),
            )
        except (ImageUpscaleError, OSError) as exc:
            raise CommandError(str(exc)) from exc
        self._out(json.dumps(receipt, indent=2, sort_keys=True))


class Run(InvariantAwareCommand):
    """Upscale one raster image to 4K with local HAT-L x2 on CUDA.

    The default 3840-pixel target preserves the source aspect ratio. A
    1920x1080 input therefore becomes exactly 3840x2160 with the model's
    native x2 path and no post-resize. The command refuses to replace output
    unless `--force` is explicit. Run `sne-tpn-upscale setup` once first.

    Classical HAT-L is intentionally used instead of a diffusion or GAN
    upscaler to reduce hallucination risk. It can still sharpen noise or add
    ringing, so native rerendering remains preferable when editable source
    exists.
    """

    _verbose_ = True

    source = None
    _source = None

    class SourceArg(PathInvariant):
        _help = "Existing source raster image."

    output = None
    _output = None

    class OutputArg(StringInvariant):
        _help = (
            "Output PNG; defaults to <source>-4k.png at 3840 pixels "
            "or <source>-<width>w.png otherwise."
        )
        _mandatory = False

    target_width = None
    _target_width = None

    class TargetWidthArg(PositiveIntegerInvariant):
        _help = (
            "Target width in pixels, preserving aspect; no greater than "
            "the model's native x2 width. "
            "[default: %default]"
        )
        _mandatory = False
        _default = 3840

    runtime_dir = None
    _runtime_dir = None

    class RuntimeDirArg(StringInvariant):
        _help = "Isolated runtime directory; defaults under user data."
        _mandatory = False

    model_cache = None
    _model_cache = None

    class ModelCacheArg(StringInvariant):
        _help = "Model directory; defaults under the Sneeze user cache."
        _mandatory = False

    cuda_visible_devices = None
    _cuda_visible_devices = None

    class CudaVisibleDevicesArg(StringInvariant):
        _help = "Physical GPU selection. [default: %default]"
        _mandatory = False
        _default = "0"

    device = None
    _device = None

    class DeviceArg(StringInvariant):
        _help = (
            "Runtime Torch device after CUDA filtering. [default: %default]"
        )
        _mandatory = False
        _default = "cuda:0"

    dtype = None
    _dtype = None

    class DtypeArg(StringInvariant):
        _help = "Inference dtype: bfloat16 or float32. [default: %default]"
        _mandatory = False
        _default = "bfloat16"

    tile_size = None
    _tile_size = None

    class TileSizeArg(NonNegativeIntegerInvariant):
        _help = "Core tile width; 0 uses the full image. [default: %default]"
        _mandatory = False
        _default = 0

    tile_overlap = None
    _tile_overlap = None

    class TileOverlapArg(NonNegativeIntegerInvariant):
        _help = "Tile context overlap in pixels. [default: %default]"
        _mandatory = False
        _default = 64

    receipt = None
    _receipt = None

    class ReceiptArg(StringInvariant):
        _help = "Optional JSON receipt output path."
        _mandatory = False

    force = None

    class ForceArg(BoolInvariant):
        _help = "Replace an existing output and receipt. [default: %default]"
        _mandatory = False
        _default = False

    def run(self):
        from .image_upscale import ImageUpscaleError, run_upscale

        try:
            receipt = run_upscale(
                source=_option_value(self, "source", private_first=True),
                output=_option_value(self, "output", private_first=True),
                target_width=int(_option_value(self, "target_width", 3840)),
                runtime_dir=_option_value(self, "runtime_dir"),
                model_cache=_option_value(self, "model_cache"),
                cuda_visible_devices=_option_value(
                    self,
                    "cuda_visible_devices",
                    "0",
                ),
                device=_option_value(self, "device", "cuda:0"),
                dtype=_option_value(self, "dtype", "bfloat16"),
                tile_size=int(_option_value(self, "tile_size", 0)),
                tile_overlap=int(_option_value(self, "tile_overlap", 64)),
                force=bool(self.force),
                receipt_path=_option_value(self, "receipt"),
            )
        except (ImageUpscaleError, OSError) as exc:
            raise CommandError(str(exc)) from exc
        self._out(json.dumps(receipt, indent=2, sort_keys=True))
