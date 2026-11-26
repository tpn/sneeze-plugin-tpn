# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Trent Nelson
"""Prepare SeedVR2 outer input chunks concurrently with inference."""

from contextlib import closing, contextmanager

if __package__:
    from .input_prefetch import parse_lookahead, prefetch
else:
    from input_prefetch import parse_lookahead, prefetch


def resolve_lookahead(raw_depth):
    return parse_lookahead(raw_depth, model="SeedVR2", default=0)


def resize_frames(resize, frames):
    try:
        return resize(frames)
    except RuntimeError as error:
        # CUDA antialias needs shared memory proportional to the shrink
        # factor. Apply the same filter on CPU if the kernel cannot fit it.
        if not frames.is_cuda or "Too much shared memory required" not in str(
            error
        ):
            raise
    return resize(frames.cpu()).to(frames.device)


def read_windows(make_capture, frame_count, chunk_size, lookahead):
    import cv2
    import numpy as np
    import torch

    # _stream_video_chunks clones the previous raw tail before its next
    # read. Reserve one slot for its current tensor and one per future read.
    slots = [None] * (lookahead + 1)
    index = 0
    remaining = frame_count
    with make_capture() as cap:
        while remaining:
            count = min(remaining, chunk_size)
            slot = slots[index]
            n = 0
            for _ in range(count):
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if slot is None or len(slot) < count:
                    slot = np.empty((count, *rgb.shape), dtype=np.float32)
                    slots[index] = slot
                if slot.shape[1:] != rgb.shape:
                    raise ValueError("Decoded SeedVR2 frame size changed")
                # Match upstream's RGB.astype(float32) / 255.0 in place.
                np.divide(rgb, 255.0, dtype=np.float32, out=slot[n])
                n += 1
            if not n:
                return
            yield count, torch.from_numpy(slot[:n])
            remaining -= n
            index = (index + 1) % len(slots)


@contextmanager
def open_input(engine, make_capture, frame_count, chunk_size, lookahead):
    """Provide the pinned engine with serial or prefetched raw CPU frames."""
    if not lookahead:
        with make_capture() as cap:
            yield cap
        return

    original_read = engine._read_frames_from_cap
    token = object()
    with closing(
        prefetch(
            lambda: read_windows(
                make_capture, frame_count, chunk_size, lookahead
            ),
            lookahead,
        )
    ) as windows:

        def read(cap, max_frames):
            if cap is not token:
                return original_read(cap, max_frames)
            requested, frames = next(windows, (max_frames, None))
            if requested != max_frames:
                raise RuntimeError(
                    "SeedVR2 requested an unexpected chunk size"
                )
            return frames

        engine._read_frames_from_cap = read
        try:
            yield token
        finally:
            engine._read_frames_from_cap = original_read
