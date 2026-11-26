# Copyright (c) 2026 Trent Nelson
# SPDX-License-Identifier: Apache-2.0

from contextlib import closing
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

from sneeze.tpn.video_upscale.turbo_input import (
    install_input_buffers,
    prefetch,
    resolve_lookahead,
)


@pytest.mark.parametrize("lookahead", [1, 2, 4])
def test_reader_prepares_exactly_the_selected_windows_ahead(lookahead):
    count = 2 * lookahead + 3
    ready = [Event() for _ in range(count)]
    reads = []
    closes = []
    consumed = 0

    def chunks():
        try:
            for i in range(count):
                reads.append((i, consumed, get_ident()))
                ready[i].set()
                yield i
        finally:
            closes.append(get_ident())

    with closing(prefetch(chunks, lookahead)) as stream:
        for consumed in range(count):
            assert next(stream) == consumed
            assert ready[min(consumed + lookahead, count - 1)].wait(2)
        with pytest.raises(StopIteration):
            next(stream)

    assert len(reads) == count
    assert all(index - current <= lookahead for index, current, _ in reads)
    assert {thread for _, _, thread in reads} == set(closes)
    assert len(closes) == 1 and closes[0] != get_ident()


@pytest.mark.parametrize("count", [0, 1, 3])
def test_deeper_queue_preserves_short_inputs(count):
    def chunks():
        yield from range(count)

    assert list(prefetch(chunks, 4)) == list(range(count))


@pytest.mark.parametrize("close_early", [False, True])
def test_reader_failure_and_early_close_release_on_the_producer(close_early):
    opened, closed = [], []
    last_window_ready = Event()

    def chunks():
        opened.append(get_ident())
        try:
            for i in range(5):
                if i == 4:
                    last_window_ready.set()
                yield i
            raise ValueError("decode error")
        finally:
            closed.append(get_ident())

    with closing(prefetch(chunks, 4)) as stream:
        assert next(stream) == 0
        assert last_window_ready.wait(2)
        if not close_early:
            assert [next(stream) for _ in range(4)] == [1, 2, 3, 4]
            with pytest.raises(ValueError, match="decode error"):
                next(stream)

    assert closed == opened and len(closed) == 1
    assert closed[0] != get_ident()


@pytest.mark.parametrize(
    ("mode", "depth", "expected"),
    [
        (None, None, 1),
        ("prefetch", None, 1),
        ("serial", None, 0),
        (None, "4", 4),
        ("prefetch", "0", 0),
        ("serial", "2", 2),
    ],
)
def test_worker_lookahead_preserves_legacy_modes(mode, depth, expected):
    assert resolve_lookahead(mode, depth) == expected


@pytest.mark.parametrize(
    ("mode", "depth"),
    [(None, "-1"), (None, "5"), (None, "1.5"), ("unknown", "1")],
)
def test_worker_rejects_invalid_input_settings(mode, depth):
    with pytest.raises(ValueError):
        resolve_lookahead(mode, depth)


@pytest.mark.parametrize("lookahead", [1, 2, 4])
def test_buffers_preserve_partial_windows_while_reader_fills_queue(lookahead):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    preprocessing = SimpleNamespace()
    request = SimpleNamespace(height=None, width=None)
    install_input_buffers(preprocessing, lookahead)
    slots = lookahead + 1
    sizes = [17] + [16] * (2 * slots) + [3, 1]
    rng = np.random.default_rng(53)
    arrays = [
        rng.integers(256, size=(n, 7, 5, 3), dtype=np.uint8) for n in sizes
    ]
    ready = [Event() for _ in arrays]

    def chunks():
        for i, array in enumerate(arrays):
            window = preprocessing._frames_to_tensor(
                list(array), request, stream=True
            )
            ready[i].set()
            yield window

    pointers = []
    with closing(prefetch(chunks, lookahead)) as stream:
        for i, raw in enumerate(arrays):
            actual = next(stream)
            assert ready[min(i + lookahead, len(arrays) - 1)].wait(2)
            expected = (
                torch.from_numpy(raw)
                .permute(0, 3, 1, 2)
                .float()
                .div_(127.5)
                .sub_(1.0)
                .permute(1, 0, 2, 3)
                .unsqueeze(0)
            )
            assert actual.stride() == expected.stride()
            assert torch.equal(actual, expected)
            pointers.append(actual.data_ptr())
    assert len(set(pointers)) == slots
    assert pointers[slots:] == pointers[:-slots]
