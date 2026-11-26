# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Trent Nelson
"""Verify edge extension and cropping at non-integer model scales."""

import pytest

from sneeze.tpn.video_upscale.worker_canvas import ModelCanvas


@pytest.mark.parametrize("alignment", [32, 128])
@pytest.mark.parametrize("target", [(48, 28), (40, 24), (18, 10), (64, 32)])
def test_custom_scale_retains_the_full_resized_image(alignment, target):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    width, height = 33, 19
    frames = np.random.default_rng(42).integers(
        256, size=(2, height, width, 3), dtype=np.uint8
    )
    canvas = ModelCanvas(width, height, *target, alignment)

    def resize(images, w, h):
        assert (w, h) == target
        return np.stack(
            [
                np.asarray(
                    Image.fromarray(frame).resize(
                        (w, h), Image.Resampling.BICUBIC
                    )
                )
                for frame in images
            ]
        )

    reference = resize(frames, *target)
    actual = canvas.prepare(frames, resize)
    assert actual.shape == (2, canvas.padded_height, canvas.padded_width, 3)
    rows, cols = canvas.content
    np.testing.assert_array_equal(actual[:, rows, cols], reference)
    np.testing.assert_array_equal(actual[:, 0, 0], reference[:, 0, 0])
    np.testing.assert_array_equal(actual[:, -1, -1], reference[:, -1, -1])
