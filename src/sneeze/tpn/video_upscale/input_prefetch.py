# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Trent Nelson
"""Read ordered CPU input windows ahead of a video model."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor


def parse_lookahead(raw_depth, *, model, default):
    if raw_depth is None:
        return default
    message = f"{model} lookahead must be an integer from 0 to 4"
    try:
        depth = int(raw_depth)
    except (ValueError, TypeError):
        raise ValueError(message) from None
    if not 0 <= depth <= 4:
        raise ValueError(message)
    return depth


def prefetch(make_chunks, lookahead=1):
    """Yield ordered windows while one reader prepares the next windows."""
    if lookahead < 1:
        raise ValueError("Prefetch needs a positive lookahead")
    end = object()

    def read_chunks():
        # Create, advance and close the source exclusively on the worker.
        # Once this generator ends or raises, queued reads just see EOF.
        chunks = make_chunks()
        try:
            # Explicit iteration leaves close() below as the single owner.
            for chunk in chunks:  # noqa: UP028
                yield chunk
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()

    chunks = read_chunks()
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="video-input"
    ) as worker:
        pending = deque()
        try:
            for _ in range(lookahead):
                pending.append(worker.submit(next, chunks, end))
            while (chunk := pending.popleft().result()) is not end:
                pending.append(worker.submit(next, chunks, end))
                yield chunk
                del chunk
        finally:
            for future in pending:
                future.cancel()
            # The current read finishes before the worker closes OpenCV/NVDEC.
            worker.submit(chunks.close).result()
