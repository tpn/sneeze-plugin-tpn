# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Trent Nelson
"""Fit the requested display raster inside a model's aligned canvas."""

import os
from dataclasses import dataclass


def target_dimensions(source_width, source_height):
    return (
        int(os.getenv("JOB_TARGET_WIDTH", 2 * source_width)),
        int(os.getenv("JOB_TARGET_HEIGHT", 2 * source_height)),
    )


@dataclass(frozen=True)
class ModelCanvas:
    source_width: int
    source_height: int
    width: int
    height: int
    alignment: int

    @property
    def padded_width(self):
        return self.width + (-self.width % self.alignment)

    @property
    def padded_height(self):
        return self.height + (-self.height % self.alignment)

    @property
    def double(self):
        return (self.width, self.height) == (
            2 * self.source_width,
            2 * self.source_height,
        )

    @property
    def padding(self):
        # At 2× the original workflow pads whole source pixels before
        # resizing. Preserve its subpixel alignment and border filtering.
        unit = 2 if self.double else 1
        extra_w = self.padded_width - self.width
        extra_h = self.padded_height - self.height
        left = unit * (extra_w // (2 * unit))
        top = unit * (extra_h // (2 * unit))
        return left, top, extra_w - left, extra_h - top

    @property
    def content(self):
        left, top, _, _ = self.padding
        return slice(top, top + self.height), slice(left, left + self.width)

    def prepare(self, frames, resize):
        """Resize full frames and extend their edges to model alignment."""
        import numpy as np

        if frames.shape[1:] != (self.source_height, self.source_width, 3):
            raise ValueError(
                f"Unexpected decoded frame shape {frames.shape[1:]}"
            )

        left, top, right, bottom = self.padding
        if not any((left, top, right, bottom)):
            return resize(frames, self.width, self.height)
        if self.double:
            frames = np.pad(
                frames,
                (
                    (0, 0),
                    (top // 2, bottom // 2),
                    (left // 2, right // 2),
                    (0, 0),
                ),
                mode="edge",
            )
            return resize(frames, self.padded_width, self.padded_height)
        frames = resize(frames, self.width, self.height)
        return np.pad(
            frames,
            ((0, 0), (top, bottom), (left, right), (0, 0)),
            mode="edge",
        )
