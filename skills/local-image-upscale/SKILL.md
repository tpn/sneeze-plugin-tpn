---
name: local-image-upscale
description: >-
  Upscale existing 8-bit RGB or RGBA raster images locally with
  sne-tpn-upscale and its pinned HAT-L x2 CUDA runtime, producing a PNG and
  provenance receipt and visually checking the result. Use for resolution-only
  4K or x2 enhancement on this NVIDIA workstation and for choosing between a
  model upscale and a native rerender. Prefer an editable HTML, SVG, or design
  source when available. Do not use for semantic or generative edits.
---

# Local Image Upscale

Create a deterministic HAT-L x2 output with `sne-tpn-upscale`, preserve the
source, and return enough evidence to distinguish a completed file from a
qualified result.

## Choose the source path

- Start from the exact file the user named. Record its absolute path, SHA-256,
  dimensions, color mode, and color space before changing anything.
- Look beside the image and in its repository history for an editable HTML,
  SVG, canvas, or design source. A native rerender is the fidelity ceiling and
  should be the production result when it exists.
- Use the repository's native renderer and preserve aspect ratio, fonts, and
  external assets. If "4K" does not determine an exact geometry, clarify it
  instead of silently cropping, padding, or changing layout.
- If the user explicitly requests a model upscale or comparison, still produce
  it, but label it separately from the native rerender.
- Use this workflow only for resolution enhancement. Route semantic edits,
  object changes, and generative restoration to an image-editing workflow.

## Resolve and prepare the command

1. Resolve `sne-tpn-upscale` and record the executable path. Do not assume a
   similarly named checkout or stale worktree is interchangeable.
2. If it is not installed, use a reviewed checkout of
   `/home/trentn/src/sneeze-plugin-tpn` and follow its control-environment
   installation instructions. Keep Torch and the model out of that lightweight
   environment; `setup` owns the isolated runtime.
3. When setup is needed, inspect the plan before mutation:

   ```bash
   sne-tpn-upscale setup --dry-run
   sne-tpn-upscale setup
   ```

4. Require the setup probe to report the pinned Python, Torch, CUDA,
   torchvision, Spandrel, Pillow, and NumPy tuple. Claim Blackwell
   qualification only when the GPU is the RTX PRO 6000 Blackwell Workstation
   Edition with compute capability 12.0 and BF16 support.

Setup may reuse its private runtime and cached checkpoint. Do not vendor,
upload, or substitute the checkpoint. Its expected SHA-256 is
`2818c7ca8d72ec4cc5f31c93203d55252a662dd35cda34ce1a69661f97dcd38f`.

## Run the upscale

- Require an 8-bit `RGB` or `RGBA` source. Do not silently convert another
  mode; report the rejection and ask for an explicit conversion decision.
- Require `target_width > source_width` and no greater than the model's native
  x2 width. A 1920x1080 source reaches 3840x2160 without post-resize.
- Use absolute, distinct source, output, and receipt paths. Make the output
  suffix `.png` and always request a receipt.
- Preserve the source. Never add `--force` unless replacement of the exact
  resolved output and receipt paths was authorized.
- On the qualified host, keep the default BF16 full-frame path. If physical GPU
  `N` is selected with `--cuda-visible-devices N`, the filtered child device is
  still `--device cuda:0`.
- Use `--dtype float32` only for a conservative numerical comparison or when
  explicitly requested. Try `--tile-size 512 --tile-overlap 64` only after a
  real full-frame memory failure or an explicit lower-memory request; do not
  present the tiled path as the full-frame qualification.

Example:

```bash
sne-tpn-upscale run \
  --source /absolute/path/source.png \
  --output /absolute/path/source-4k-hat-l.png \
  --receipt /absolute/path/source-4k-hat-l.json
```

## Verify and report

1. Require successful command completion and both output files.
2. Verify the PNG is a regular 8-bit RGB or RGBA file with the receipt's exact
   dimensions, byte count, and SHA-256. Verify the receipt's source,
   implementation, and checkpoint identities against the exact files.
3. Check the recorded GPU, compute capability, runtime tuple, dtype, BF16 model
   capability, tile parameters, alpha handling, post-resize flag, inference
   time, and peak allocated/reserved memory. Keep inference and wall time
   distinct.
4. Visually inspect the output at useful zoom levels. Pay particular attention
   to small text, thin edges, flat color fields, chromatic ringing, invented
   detail, seams, and RGBA boundaries. Compare with a native rerender when one
   exists.
5. Return clickable absolute paths for the PNG and receipt, show the image when
   the client supports it, and report the exact hashes, dimensions, timing,
   memory, model/runtime identity, and any qualification boundary.

The HAT source is Apache-2.0. Upstream does not separately state terms for the
externally hosted checkpoint; report that caveat rather than implying the
weight has an independently verified license.
