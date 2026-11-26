# Copyright (c) 2026 Trent Nelson
# SPDX-License-Identifier: Apache-2.0

"""Sneeze commands for local GPU video upscaling."""

from sneeze.command import CommandError
from sneeze.commandinvariant import InvariantAwareCommand
from sneeze.invariant import (
    BoolInvariant,
    NonNegativeIntegerInvariant,
    PositiveIntegerInvariant,
    StringInvariant,
)


class UpscaleVideo(InvariantAwareCommand):
    """Upscale local videos with FlashVSR+ Turbo, SeedVR2 or LTX."""

    _usage_ = "%prog [options] INPUT [INPUT ...]"
    _description_ = (
        "Upscale each input 2× with live frame progress and save a lossless "
        "H.264 MP4 at the original frame rate. To compare the input with "
        "all three models, run 'sne uv -m all --compare input.mp4'."
    )
    _vargc_ = True
    _conf_ = False
    _log_level_ = False
    _verbose_ = True
    _quiet_ = True

    class ModelArg(StringInvariant):
        _arg = "-m/--model"
        _metavar = "MODEL"
        _help = (
            "turbo (default), seedvr2 (standard 7B), ltx (LTX-2.5 22B "
            "Refine Details), both (Turbo and SeedVR2), or all."
        )
        _mandatory = False

    class OutputArg(StringInvariant):
        _arg = "-o/--output"
        _metavar = "FILE"
        _help = "Set the output MP4 path for a single render."
        _mandatory = False

    class OutputDirArg(StringInvariant):
        _arg = "--output-dir"
        _metavar = "DIR"
        _help = "Write all outputs here. Defaults to each input's directory."
        _mandatory = False

    class CompareArg(BoolInvariant):
        _arg = "--compare"
        _help = (
            "Also save a labeled, downscaled side-by-side preview of each "
            "input and its selected model outputs."
        )
        _mandatory = False
        _default = None

    class EncoderArg(StringInvariant):
        _arg = "--encoder"
        _metavar = "MODE"
        _help = (
            "auto (default), cpu, or nvenc. Auto uses NVENC for Turbo "
            "and CPU x264 for SeedVR2 and LTX. LTX requires CPU."
        )
        _mandatory = False

    class DecoderArg(StringInvariant):
        _arg = "--decoder"
        _metavar = "MODE"
        _help = (
            "Input video decoder. Choose auto (default), cpu, or nvdec. "
            "Auto uses CPU. LTX requires CPU. NVDEC accepts tagged "
            "limited-range BT.709 H.264 8-bit 4:2:0 with left chroma."
        )
        _mandatory = False

    class LookaheadArg(NonNegativeIntegerInvariant):
        _arg = "--lookahead"
        _metavar = "N"
        _help = (
            "Read 0–4 Turbo or SeedVR2 windows ahead. 0 reads on demand. "
            "One window is 16 frames for Turbo or --seed-chunk frames for "
            "SeedVR2. The defaults are 2 for Turbo and 0 for SeedVR2."
        )
        _mandatory = False

    class TurboLookaheadArg(NonNegativeIntegerInvariant):
        _arg = "--turbo-lookahead"
        _metavar = "N"
        _help = (
            "Override --lookahead for Turbo only (0–4 windows, default 2)."
        )
        _mandatory = False

    class SeedLookaheadArg(NonNegativeIntegerInvariant):
        _arg = "--seed-lookahead"
        _metavar = "N"
        _help = "Override --lookahead for SeedVR2 only (0–4 input chunks)."
        _mandatory = False

    class TurboInputArg(StringInvariant):
        _arg = "--turbo-input"
        _metavar = "MODE"
        _help = (
            "Prepare the next Turbo frame window during inference "
            "(prefetch, default), or read on demand (serial)."
        )
        _mandatory = False

    class TurboDecoderArg(StringInvariant):
        _arg = "--turbo-decoder"
        _metavar = "MODE"
        _help = "Turbo neural decoder arithmetic, fast (default) or exact."
        _mandatory = False

    class TurboMicrobatchArg(PositiveIntegerInvariant):
        _arg = "--turbo-microbatch"
        _metavar = "N"
        _help = "Turbo decoder microbatch size, 1, 2 (default), or 4."
        _mandatory = False

    class SeedBatchArg(PositiveIntegerInvariant):
        _arg = "--seed-batch"
        _metavar = "FRAMES"
        _help = "SeedVR2 temporal batch, 4n+1 frames (default 17)."
        _mandatory = False

    class SeedChunkArg(PositiveIntegerInvariant):
        _arg = "--seed-chunk"
        _metavar = "FRAMES"
        _help = "SeedVR2 outer chunk length (default 53)."
        _mandatory = False

    class SeedOverlapArg(NonNegativeIntegerInvariant):
        _arg = "--seed-overlap"
        _metavar = "FRAMES"
        _help = (
            "SeedVR2 overlap (default 4). Must be smaller than the batch "
            "and at most the chunk length."
        )
        _mandatory = False

    class SeedArg(NonNegativeIntegerInvariant):
        _arg = "--seed"
        _metavar = "N"
        _help = (
            "Model random seed (Turbo default 1, SeedVR2 and LTX default 42)."
        )
        _mandatory = False

    class LtxPromptArg(StringInvariant):
        _arg = "--ltx-prompt"
        _metavar = "TEXT"
        _help = (
            "Describe the scene for LTX. The default asks it to preserve "
            "the source and refine natural detail."
        )
        _mandatory = False

    class LtxStrengthArg(StringInvariant):
        _arg = "--ltx-strength"
        _metavar = "WEIGHT"
        _help = "LTX Refine Details adapter weight from 0 to 1 (default 1)."
        _mandatory = False

    class LtxVaeTileArg(PositiveIntegerInvariant):
        _arg = "--ltx-vae-tile"
        _metavar = "PIXELS"
        _help = "LTX VAE decode tile, 512, 1024, 1536 or 2048 px (default)."
        _mandatory = False

    class LtxChunkArg(PositiveIntegerInvariant):
        _arg = "--ltx-chunk"
        _metavar = "FRAMES"
        _help = "LTX chunk length, 8n+1 frames from 9 to 241 (default 241)."
        _mandatory = False

    class AudioArg(StringInvariant):
        _arg = "--audio"
        _metavar = "MODE"
        _help = "auto (copy with AAC fallback, default), copy, aac, or drop."
        _mandatory = False

    class DryRunArg(BoolInvariant):
        _arg = "--dry-run"
        _help = "Print the resolved jobs and settings without rendering."
        _mandatory = False
        _default = None

    class ForceArg(BoolInvariant):
        _arg = "-f/--force"
        _help = "Replace existing outputs after successful verification."
        _mandatory = False
        _default = None

    def run(self):
        try:
            from .runner import execute

            execute(self.args, self.options, self.ostream, self.estream)
        except (ValueError, OSError, RuntimeError) as exc:
            raise CommandError(str(exc)) from exc
