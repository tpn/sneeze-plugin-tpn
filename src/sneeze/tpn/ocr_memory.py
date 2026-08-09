"""Shared, standard-library-only OCR memory-tier arithmetic."""

from __future__ import annotations


def canonical_memory_budget(
    total_memory_bytes: int,
    fraction: float,
    reserve_bytes: int,
) -> int:
    """Return the canonical tier budget after the fixed safety reserve."""

    if (
        not isinstance(total_memory_bytes, int)
        or isinstance(total_memory_bytes, bool)
        or total_memory_bytes <= 0
    ):
        raise ValueError("total memory must be a positive integer")
    if (
        not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or not 0 < float(fraction) < 1
    ):
        raise ValueError("memory fraction must be between zero and one")
    if (
        not isinstance(reserve_bytes, int)
        or isinstance(reserve_bytes, bool)
        or reserve_bytes < 0
    ):
        raise ValueError("memory reserve must be a nonnegative integer")
    return min(
        int(total_memory_bytes * float(fraction)),
        max(0, total_memory_bytes - reserve_bytes),
    )


__all__ = ["canonical_memory_budget"]
