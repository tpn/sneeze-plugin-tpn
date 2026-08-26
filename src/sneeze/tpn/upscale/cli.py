# Copyright (c) 2026 Trent Nelson
# SPDX-License-Identifier: Apache-2.0

"""Branded Sneeze entry point for tpn's local image upscaler."""

from __future__ import annotations

from sneeze.cli import branded_main


def main(argv: list[str] | None = None) -> None:
    """Run only the tpn upscaler commands without loading OCR commands."""

    branded_main(
        "sne-tpn-upscale",
        ["-sneeze", "sneeze.tpn.upscale"],
        argv,
        auto_plugins=False,
    )
