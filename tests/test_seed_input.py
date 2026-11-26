# Copyright (c) 2026 Trent Nelson
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace
from contextlib import contextmanager
from threading import Event, get_ident
from unittest.mock import Mock

import pytest

from sneeze.tpn.video_upscale.seed_input import (
    open_input,
    resize_frames,
    resolve_lookahead,
)


class Source:
    """A BGR capture exposing the same read() interface as NVDEC."""

    def __init__(self, frames, fail_at=None):
        self.frames = frames
        self.fail_at = fail_at
        self.ready = [Event() for _ in frames]
        self.reads = []
        self.closed = []
        self.opened = []

    @contextmanager
    def open(self):
        self.opened.append(get_ident())
        try:
            yield self
        finally:
            self.closed.append(get_ident())

    def read(self):
        i = len(self.reads)
        if i == self.fail_at:
            raise ValueError("decode failed")
        if i == len(self.frames):
            return False, None
        self.reads.append(get_ident())
        self.ready[i].set()
        return True, self.frames[i]


@pytest.fixture
def frames():
    np = pytest.importorskip("numpy")
    return np.random.default_rng(42).integers(
        256, size=(25, 3, 7, 3), dtype=np.uint8
    )


def test_legacy_seed_worker_reads_on_demand():
    assert resolve_lookahead(None) == 0
    for depth in range(5):
        assert resolve_lookahead(str(depth)) == depth
    for depth in ("-1", "5", "1.5"):
        with pytest.raises(ValueError):
            resolve_lookahead(depth)


@pytest.mark.parametrize("depth", [0, 1, 2, 4])
def test_full_input_queue_preserves_order_and_partial_tail(frames, depth):
    engine = pytest.importorskip("inference_cli")
    torch = pytest.importorskip("torch")
    read = engine._read_frames_from_cap
    count = len(frames)
    chunk_size = 3
    expected = read(Source(frames), count)
    source = Source(frames)
    pointers = []
    with open_input(engine, source.open, count, chunk_size, depth) as cap:
        for first in range(0, count, chunk_size):
            size = min(chunk_size, count - first)
            result = engine._read_frames_from_cap(cap, size)
            last_queued = min(count, first + (depth + 1) * chunk_size) - 1
            assert source.ready[last_queued].wait(2)
            assert torch.equal(result, expected[first : first + size])
            assert result.dtype == torch.float32
            pointers.append(result.data_ptr())

    assert engine._read_frames_from_cap is read
    assert source.closed == source.opened and len(source.closed) == 1
    assert set(source.reads) == set(source.closed)
    if depth:
        assert source.opened[0] != get_ident()
        slots = depth + 1
        assert len(set(pointers)) == slots
        assert pointers[slots:] == pointers[:-slots]
    else:
        assert source.opened[0] == get_ident()


@pytest.mark.parametrize("stop_early", [False, True])
def test_input_reader_cleanup_and_hook_restoration(frames, stop_early):
    engine = pytest.importorskip("inference_cli")
    read = engine._read_frames_from_cap
    source = Source(frames, fail_at=6)
    with open_input(engine, source.open, len(frames), 3, 4) as cap:
        assert engine._read_frames_from_cap(cap, 3).shape[0] == 3
        if not stop_early:
            assert engine._read_frames_from_cap(cap, 3).shape[0] == 3
            with pytest.raises(ValueError, match="decode failed"):
                engine._read_frames_from_cap(cap, 3)
    assert engine._read_frames_from_cap is read
    assert source.closed == source.opened and len(source.closed) == 1
    assert source.closed[0] != get_ident()


@pytest.mark.parametrize("depth", [0, 4])
def test_upstream_chunk_processor_preserves_context_and_prepend(
    frames, monkeypatch, depth
):
    engine = pytest.importorskip("inference_cli")
    torch = pytest.importorskip("torch")
    frames = frames[:13]
    expected = engine._read_frames_from_cap(
        Source(frames), len(frames)
    ).half()
    calls = []

    def identity(frames_tensor, args, **kwargs):
        calls.append((frames_tensor.clone(), args.prepend_frames))
        return frames_tensor

    monkeypatch.setattr(engine, "_process_frames_core", identity)
    with open_input(
        engine, Source(frames).open, len(frames), 5, depth
    ) as cap:
        outputs = list(
            engine._stream_video_chunks(
                cap=cap,
                frames_to_process=len(frames),
                chunk_size=5,
                overlap=2,
                args=Namespace(prepend_frames=3),
                device_id="cpu",
                debug=engine.debug,
                runner_cache={},
            )
        )

    assert [len(part) for part in outputs] == [5, 5, 3]
    assert torch.equal(torch.cat(outputs), expected)
    assert [prepend for _, prepend in calls] == [3, 0, 0]
    for (tensor, _), (start, stop) in zip(
        calls, ((0, 5), (3, 10), (8, 13)), strict=True
    ):
        assert torch.equal(tensor, expected[start:stop])


def test_cuda_resize_retries_oversized_antialias_filter_on_cpu():
    frames = Mock(is_cuda=True)
    scaled = Mock()
    resize = Mock(
        side_effect=[
            RuntimeError("Too much shared memory required: 518020 vs 49152"),
            scaled,
        ]
    )

    assert resize_frames(resize, frames) is scaled.to.return_value
    resize.assert_any_call(frames)
    resize.assert_called_with(frames.cpu.return_value)
    scaled.to.assert_called_once_with(frames.device)


@pytest.mark.parametrize(
    ("on_cuda", "message"),
    [
        (True, "unrelated CUDA failure"),
        (False, "Too much shared memory required"),
    ],
)
def test_resize_propagates_other_errors(on_cuda, message):
    frames = Mock(is_cuda=on_cuda)
    resize = Mock(side_effect=RuntimeError(message))

    with pytest.raises(RuntimeError, match=message):
        resize_frames(resize, frames)
    resize.assert_called_once_with(frames)
