# Copyright (c) 2026 Trent Nelson
# SPDX-License-Identifier: Apache-2.0

import sys
from itertools import pairwise
from types import SimpleNamespace

import pytest

from sneeze.tpn.video_upscale import ltx


class FrameReader:
    """Forward-only capture with one distinct value per frame."""

    def __init__(self, frames):
        self.remaining = iter(frames)
        self.decoded = 0

    def read(self):
        try:
            frame = next(self.remaining)
        except StopIteration:
            return False, None
        self.decoded += 1
        return True, frame


def test_spatial_padding_preserves_exact_2x_crop():
    assert ltx.spatial_padding(1920, 1080) == (0, 4, 0, 4)
    for width, height in ((1280, 720), (354, 206), (197, 351), (1, 1)):
        left, top, right, bottom = ltx.spatial_padding(width, height)
        assert 0 <= left + right < 16
        assert 0 <= top + bottom < 16
        assert abs(left - right) <= 1
        assert abs(top - bottom) <= 1
        assert (width + left + right) % 16 == 0
        assert (height + top + bottom) % 16 == 0


def test_temporal_padding_rounds_up_without_dropping_a_tail():
    for frames in range(1, 250):
        padded = ltx.padded_frames(frames)
        assert frames <= padded < frames + 8
        assert (padded - 1) % 8 == 0


@pytest.mark.parametrize("chunk", [9, 17, 97, 241])
def test_input_chunks_carry_context_and_decode_each_frame_once(chunk):
    overlap = ltx.overlap_frames(chunk)
    assert 0 < overlap < chunk
    stride = chunk - overlap
    for total in (1, 2, chunk - 1, chunk, chunk + 1, chunk + 2 * stride + 1):
        source = FrameReader(range(total))
        parts = list(ltx.input_chunks(source, total, chunk))
        assert parts[0][0] == 0
        assert parts[-1][0] + len(parts[-1][1]) == total
        for start, frames in parts:
            assert len(frames) <= chunk
            assert list(frames) == list(range(start, start + len(frames)))
        for (start, previous), (next_start, _) in pairwise(parts):
            assert len(previous) == chunk
            assert next_start == start + stride
        assert source.decoded == total


def test_input_chunks_report_premature_eof():
    source = FrameReader(range(10))
    with pytest.raises((ValueError, RuntimeError)):
        list(ltx.input_chunks(source, 15, 9))


@pytest.mark.parametrize("total", [1, 2, 16, 17, 18, 61])
def test_stitched_renders_preserve_every_source_index_once(total):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    chunk = 17
    overlap = ltx.overlap_frames(chunk)
    frames = [np.full((2, 3, 3), i, dtype=np.uint8) for i in range(total)]
    output = []
    tail = []
    for start, part in ltx.input_chunks(FrameReader(frames), total, chunk):
        last = start + len(part) == total
        tail = ltx.join_chunk(part, tail, overlap, last, output.append)
    assert len(tail) == 0
    np.testing.assert_array_equal(np.stack(output), np.stack(frames))


def test_overlap_blends_different_renders_of_same_frames():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    chunk = 17
    overlap = ltx.overlap_frames(chunk)
    dark = np.full((2, 2, 3), 10, dtype=np.uint8)
    light = np.full_like(dark, 240)
    output = []
    tail = ltx.join_chunk([dark] * chunk, [], overlap, False, output.append)
    assert len(output) == chunk - overlap
    tail = ltx.join_chunk(
        [light] * (overlap + 1), tail, overlap, True, output.append
    )
    assert len(tail) == 0
    assert len(output) == chunk + 1
    levels = [int(frame[0, 0, 0]) for frame in output]
    assert levels == sorted(levels)
    assert levels[: chunk - overlap] == [10] * (chunk - overlap)
    assert levels[-1] == 240
    assert any(10 < level < 240 for level in levels)


def test_guide_and_decoder_pad_replicate_then_crop_exactly(monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    height, width = 19, 23
    frames = np.random.default_rng(42).integers(
        256, size=(2, height, width, 3), dtype=np.uint8
    )

    # Identity denoiser/VAE plus an exact 2x scaler makes each output pixel
    # checkable, including asymmetric borders and the last frame.
    def scale(image, method, w, h, crop):
        assert (method, w, h, crop) == ("lanczos", 64, 64, "disabled")
        return (image.repeat_interleave(2, 1).repeat_interleave(2, 2),)

    def encode(_pos, _neg, _vae, empty, guide, *_options):
        assert empty == (64, 64, 9, 1)
        assert torch.equal(guide[1:], guide[1].expand_as(guide[1:]))
        expected = (
            np.pad(
                frames[0, :, :, ::-1], ((6, 7), (4, 5), (0, 0)), mode="edge"
            )
            .repeat(2, axis=0)
            .repeat(2, axis=1)
        )
        np.testing.assert_array_equal(
            (guide[0] * 255).round().numpy(), expected
        )
        return SimpleNamespace(result=(guide,))

    nodes = SimpleNamespace(
        ImageScale=lambda: SimpleNamespace(upscale=scale),
        VAEDecodeTiled=lambda: SimpleNamespace(
            decode=lambda _vae, latent, *_options: (latent,)
        ),
    )
    latent_node = SimpleNamespace(
        execute=lambda *shape: SimpleNamespace(result=(shape,))
    )
    lt = SimpleNamespace(EmptyLTXVLatentVideo=latent_node)
    iclora = SimpleNamespace(
        LTXAddVideoICLoRAGuide=SimpleNamespace(execute=encode)
    )
    management = SimpleNamespace()
    for name, module in {
        "nodes": nodes,
        "comfy_extras.nodes_lt": lt,
        "ltxnodes.iclora": iclora,
        "comfy": SimpleNamespace(model_management=management),
        "comfy.model_management": management,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    def identity_denoise(part, models, width, height, _seed):
        return ltx.guide_latents(part, models, width, height)[0]

    monkeypatch.setattr(ltx, "denoise", identity_denoise)
    output = list(
        ltx.decoded_rgb(frames, (None,) * 4, width, height, 42, 2048)
    )
    expected = frames[..., ::-1].repeat(2, axis=1).repeat(2, axis=2)
    np.testing.assert_array_equal(np.stack(output), expected)
