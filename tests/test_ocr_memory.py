from __future__ import annotations

import pytest

from sneeze.tpn.ocr_memory import canonical_memory_budget


def test_canonical_memory_budget_uses_fraction_when_smaller():
    assert canonical_memory_budget(1000, 0.25, 100) == 250


def test_canonical_memory_budget_uses_positive_reserve_limit_when_smaller():
    assert canonical_memory_budget(1000, 0.9, 300) == 700


def test_canonical_memory_budget_clamps_to_zero_after_reserve():
    assert canonical_memory_budget(1000, 0.5, 1000) == 0


@pytest.mark.parametrize(
    ("total", "fraction", "reserve", "message"),
    [
        (True, 0.5, 0, "total memory"),
        (1000, True, 0, "memory fraction"),
        (1000, 0.0, 0, "memory fraction"),
        (1000, 1.0, 0, "memory fraction"),
        (1000, 0.5, True, "memory reserve"),
        (1000, 0.5, -1, "memory reserve"),
    ],
)
def test_canonical_memory_budget_rejects_invalid_inputs(
    total,
    fraction,
    reserve,
    message,
):
    with pytest.raises(ValueError, match=message):
        canonical_memory_budget(total, fraction, reserve)
