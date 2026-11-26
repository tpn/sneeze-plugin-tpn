# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Trent Nelson
"""Resolve a scale or target raster against square-pixel display size."""

import re
from fractions import Fraction


def parse_scale(value):
    try:
        scale = Fraction(value)
    except (ValueError, TypeError, ZeroDivisionError):
        scale = 0
    if scale <= 0:
        raise ValueError(
            "--scale needs a positive decimal or fraction, e.g. 1.5 or 3/2"
        )
    return scale


def parse_target_size(value):
    match = re.fullmatch(
        r"([0-9]+|auto)[x×]([0-9]+|auto)", value.strip().lower()
    )
    if not match or match.groups() == ("auto", "auto"):
        raise ValueError(
            "--target-size expects WIDTHxHEIGHT, WIDTHxauto or autoxHEIGHT"
        )
    width, height = (
        None if axis == "auto" else int(axis) for axis in match.groups()
    )
    for axis in (width, height):
        if axis is not None and (axis <= 0 or axis % 2):
            raise ValueError(
                "--target-size dimensions must be positive even integers "
                "for H.264 4:2:0"
            )
    return width, height


def nearest_even(pixels):
    """Round to the nearest encodable dimension, breaking ties upward."""
    return max(2, 2 * ((pixels + 1) // 2))


def resolve_target(width, height, *, scale=None, target_size=None):
    if scale is not None and target_size is not None:
        raise ValueError("Choose either --scale or --target-size")
    if target_size is None:
        factor = Fraction(2) if scale is None else parse_scale(scale)
        return nearest_even(width * factor), nearest_even(height * factor)
    target_width, target_height = parse_target_size(target_size)
    if target_width is None:
        return nearest_even(
            Fraction(target_height * width, height)
        ), target_height
    if target_height is None:
        return target_width, nearest_even(
            Fraction(target_width * height, width)
        )
    # Each scaled axis can round by one pixel in either direction. Both
    # intervals must contain a common uniform scale. At the two-pixel minimum
    # even smaller scales round to two as well.
    low = max(
        Fraction(target_width - 1, width) if target_width > 2 else 0,
        Fraction(target_height - 1, height) if target_height > 2 else 0,
    )
    high = min(
        Fraction(target_width + 1, width), Fraction(target_height + 1, height)
    )
    if low >= high:
        raise ValueError(
            f"--target-size {target_size} changes the aspect ratio of the "
            f"{width}x{height} displayed input. Use {target_width}xauto or "
            f"autox{target_height} to preserve it."
        )
    return target_width, target_height
