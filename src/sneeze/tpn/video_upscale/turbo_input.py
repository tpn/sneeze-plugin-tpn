# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Trent Nelson
"""Normalize bounded FlashVSR+ input windows ahead of the model."""

from itertools import islice
from threading import local

# In Docker these modules are mounted as standalone worker scripts.
if __package__:
    from .input_prefetch import parse_lookahead
    from .input_prefetch import prefetch as prefetch
else:
    from input_prefetch import parse_lookahead
    from input_prefetch import prefetch as prefetch


def resolve_lookahead(input_mode, raw_depth):
    """Resolve the worker environment, including the older input mode."""
    if input_mode not in (None, "serial", "prefetch"):
        raise ValueError("Turbo input must be serial or prefetch")
    return parse_lookahead(
        raw_depth, model="Turbo", default=0 if input_mode == "serial" else 1
    )


def capture_chunks(
    make_capture,
    preprocessing,
    request,
    *,
    chunk_frames,
    first_chunk_frames=None,
    stream=True,
):
    """Feed a normalized BGR capture to Turbo's usual RGB preprocessing."""
    request.validate()
    size = chunk_frames if first_chunk_frames is None else first_chunk_frames
    if min(size, chunk_frames) <= 0:
        raise ValueError("Frame window sizes must be positive")
    with make_capture() as capture:

        def rgb_frames():
            while True:
                ok, frame = capture.read()
                if not ok:
                    return
                # _frames_to_tensor stacks the frames into a contiguous array.
                yield frame[:, :, ::-1]

        source = islice(rgb_frames(), request.max_frames)
        while frames := list(islice(source, size)):
            window = preprocessing._frames_to_tensor(
                frames, request, stream=stream
            )
            del frames
            yield window
            del window
            size = chunk_frames


def install_input_buffers(preprocessing, lookahead=1):
    """Normalize windows into a bounded ring of reusable CPU allocations."""
    import numpy as np
    import torch
    from PIL import Image

    # TemporalVideoStream copies a window before requesting the next one.
    # Its current window and the reader's lookahead each need their own slot.
    state = local()

    def convert(frames, request, *, stream):
        array = np.stack(frames)
        if request.height is not None:
            if request.width is None:
                raise ValueError("height and width must be provided together")
            resample = (
                Image.Resampling.BICUBIC
                if stream
                else Image.Resampling.BILINEAR
            )
            array = preprocessing.pil_center_crop_resize(
                array, request.height, request.width, resample=resample
            )
        if not hasattr(state, "slots"):
            state.slots = [None] * (lookahead + 1)
            state.index = 0
        index = state.index
        state.index = (index + 1) % len(state.slots)
        slot = state.slots[index]
        if (
            slot is None
            or slot.shape[1:] != array.shape[1:]
            or slot.shape[0] < len(frames)
        ):
            slot = torch.empty(array.shape, device="cpu", dtype=torch.float32)
            state.slots[index] = slot
        window = slot[: len(frames)]
        window.copy_(torch.from_numpy(array)).div_(127.5).sub_(1.0)
        return window.permute(3, 0, 1, 2).unsqueeze(0)

    preprocessing._frames_to_tensor = convert
