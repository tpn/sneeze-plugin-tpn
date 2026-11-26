# sneeze-plugin-tpn

Public Sneeze plugin package for `sneeze.tpn`.

## Local video upscaling

`sne upscale-video` (alias `sne uv`) restores a local constant-frame-rate
SDR video and saves a square-pixel H.264 MP4 at the source frame rate.
All three models default to 2× the **display width and height**.
For example, a 720×576 video with a 16:15 sample aspect ratio displays
at 768×576 and defaults to 1536×1152. Sizing accounts for quarter-turn
rotation and rounds fractional display widths to whole pixels first.

Choose one sizing option to override the default. Scaling applies to each
input independently. The model renders at the requested size, with temporary
padding as needed for its internal alignment.

| Option | Example | Result for a 1280×720 input |
| --- | --- | --- |
| `--scale FACTOR` | `--scale 1.5` or `--scale 3/2` | 1920×1080 |
| `--target-size WIDTHxHEIGHT` | `--target-size 3840x2160` | 3840×2160 |
| `--target-size WIDTHxauto` | `--target-size 1920xauto` | 1920×1080 |
| `--target-size autoxHEIGHT` | `--target-size autox1440` | 2560×1440 |

Use `--scale 1` for restoration at the display size. Inferred dimensions
round to the nearest even pixel for H.264. An explicit dimension must be
even. If you specify both dimensions, their proportions must match the
display aspect ratio within rounding. Use `auto` on one axis when processing
inputs with different aspect ratios.

```sh
sne uv input.mp4
sne uv --target-size autox1080 input.mp4
sne uv -m seedvr2 -o enhanced.mp4 input.mp4
sne uv -m seedvr2 --decoder nvdec --encoder nvenc input-av1.mp4
sne uv -m ltx --compare input.mp4
sne uv -m all --compare --output-dir results input.mp4
sne uv --dry-run --output-dir results "first input.mp4" "second input.mp4"
```

| Model | Default configuration |
| --- | --- |
| `turbo` (default) | FlashVSR+ Turbo, FlashAttention-4, fast decoder with microbatch 2, CPU input decode and NVENC output |
| `seedvr2` | SeedVR2-7B standard, FlashAttention-4, fused VAE, CPU decode and x264 output |
| `ltx` | LTX-2.5 22B with the Refine Details adapter at full strength, FlashAttention-4, 2048-pixel VAE tiles, CPU decode and x264 output |

`-m both` selects Turbo and SeedVR2. `-m all` selects all three models.
Each input and model gets its own file. The default name is
`<input-stem>-<model>-2x.mp4`. Custom sizes appear in the output names.
With `--compare`, the command also produces a labeled viewing copy. The
four-panel version shows the input and Turbo above SeedVR2 and LTX.

Use `--lookahead 0..4` to set how many windows the CPU can prepare ahead
of Turbo or SeedVR2. Zero reads on demand. Turbo defaults to 2 windows
of typically 16 frames. SeedVR2 defaults to 0 and reads outer chunks of
53 frames. For independent settings, use
`--turbo-lookahead 2 --seed-lookahead 1`. SeedVR2's `--seed-overlap`
controls how many context frames neighboring chunks share.

LTX uses a content-neutral prompt. Set `--ltx-prompt "scene description"`
to describe the input or `--ltx-strength` to adjust the adapter weight
from 0 to 1. `--ltx-chunk` controls the number of frames per outer chunk
(9–241 in steps of 8, default 241). Longer clips use overlapping chunks.
`--ltx-vae-tile` accepts 512, 1024, 1536 or 2048 pixels.

Jobs run sequentially. The command reports model progress and end-to-end
fps on stderr and prints completed output paths on stdout. Use `-q` for
paths alone or `-v` for full worker logs. Audio defaults to stream copy
with AAC fallback. Outputs are staged and verified before publication.
See `sne uv --help` for codec, audio and model-specific options.

Press Ctrl-C to cancel the command and its queued jobs. During a render
or comparison, Sneeze removes its worker container and partial output,
releases the GPU lock, and exits with status 130. Ctrl-C while waiting
for the lock cancels only the waiting command. Completed outputs from
earlier jobs in the batch stay available.

The default CPU input path uses FFmpeg software decoding for AV1 and
display normalization. Turbo and SeedVR2 also support `--decoder nvdec`
for 8- or 10-bit 4:2:0 SDR AV1 with even stored dimensions on the
configured codec GPU. Use `--decoder cpu` for an odd stored width or
height. An explicit hardware request fails if NVDEC cannot decode the input.
The decoded frames return to host memory for color conversion
and transfer to the inference GPU. The output remains 8-bit H.264.
The H.264 NVDEC path requires square pixels, upright orientation and
tagged limited-range BT.709 8-bit 4:2:0 with left chroma.
Encoder selection is independent. In auto encoder mode, the command checks
the requested output size with NVENC before inference and uses CPU x264 when
NVENC cannot open it. An explicit `--encoder nvenc` reports the error.
The available GPU memory still limits the model's working resolution.

### Video runtime configuration

Rendering requires a prepared Linux GPU runtime with Docker, the NVIDIA
Container Toolkit, the pinned images used by the workers, and the selected
model's weights and runtime adapters. The current workers target NVIDIA
GB300 inference. Turbo defaults to a separate NVENC-capable codec GPU.
The host launcher uses `sudo -n docker`. Workers run offline.

Create `sneeze/video-upscale.toml` under your user configuration directory
(`$XDG_CONFIG_HOME`, normally `~/.config`). Use your own values for
the runtime root and inference GPU. The GPU UUID is available from
`nvidia-smi -L`.

```toml
root = "/path/to/video-runtimes"
gpu = "GPU-<inference-device-uuid>"
```

The optional `[paths]` table selects directories relative to `root`.
The defaults below show the layout expected of a prepared runtime.

| Key | Default directory | Contents |
| --- | --- | --- |
| `turbo` | `turbo` | Turbo weights and the `decoder_speed`, `sparse_fa4` and `gpu_writer` adapters |
| `flashvsr` | `flashvsr/source` | FlashVSR+ source with its pinned dependencies and attention operator |
| `seedvr2` | `seedvr2` | SeedVR2 source and weights, `inference_cli`, `attention`, `norm_kernel` and `tuning` |
| `ltx` | `ltx` | ComfyUI, LTX nodes and the `attention_backend` and `weights` modules |
| `ltx_assets` | `ltx-assets` | Directory mounted alongside the LTX runtime for shared assets and symlink targets |
| `codecs` | `codecs` | The `decode_rtx` and `encode_rtx` adapters, each exposing `RTX_UUID` to select the codec GPU |

The launcher mounts runtime assets read-only and the selected model's
`cache/` directory writable. Model weights, adapter implementations and
container images are provisioned separately from this plugin.

## Local 4K image upscaling

Use `sne-tpn-upscale` when the only available input is a raster image. If an
editable HTML, SVG, or design source exists, rerendering that source at the
target resolution is the fidelity ceiling and should be preferred.

The local ML path uses the classical x2
[HAT-L super-resolution model](https://github.com/XPixelGroup/HAT) through
[Spandrel](https://github.com/chaiNNer-org/spandrel). HAT-L was selected over
GAN and diffusion upscalers to reduce hallucination risk on diagrams, charts,
and text-heavy images. It can still oversharpen edges or add ringing, so the
output must be inspected before replacing a production asset.

`sne-tpn-upscale` is a branded Sneeze entry point shipped by the existing
`sneeze-plugin-tpn` distribution. Keeping its command module separate leaves
the sealed OCR command implementation and its qualified profiles unchanged;
installing or removing the `tpn` plugin manages both command surfaces.

Preview and then create the isolated runtime and model cache:

```bash
sne-tpn-upscale setup --dry-run
sne-tpn-upscale setup
```

The defaults use Python 3.12.13, PyTorch 2.11.0 with CUDA 13.0, and Spandrel
0.4.2. The runtime is stored under
`~/.local/share/sneeze/tpn/upscale/runtime`; the checkpoint is stored under
`~/.cache/sneeze/tpn/upscale/models`. The setup downloads an immutable mirror
of the official `HAT-L_SRx2_ImageNet-pretrain.pth` checkpoint. Its SHA-256 was
independently confirmed to match both the official Google Drive artifact and
the pinned mirror revision:
`2818c7ca8d72ec4cc5f31c93203d55252a662dd35cda34ce1a69661f97dcd38f`.
The HAT source repository is Apache-2.0 licensed; the upstream project does
not state separate terms for the externally hosted checkpoint.

Setup disables ambient uv configuration, resolves Torch only from the pinned
PyTorch CUDA index, resolves the remaining packages only from PyPI, and strips
parent package-manager, Python, dynamic-loader, CUDA injection, Torch cache,
proxy, CA override, virtual-environment, and secret-looking variables from
every setup and runtime subprocess. Fresh environments use the exact
uv-managed Python 3.12.13 runtime. Before code is
executed, setup and run validate ownership, write permissions, the
interpreter's direct canonical target, and symlink confinement across the
runtime and Python base. The same checks cover the packaged implementation and
checkpoint. The parent also verifies the implementation hash reported by the
isolated child and checks it again after inference.

Upscale a 1920x1080 PNG to native x2 4K UHD without replacing the source:

```bash
sne-tpn-upscale run \
  --source ./image.png \
  --output ./image-4k-hat-l.png \
  --receipt ./image-4k-hat-l.json
```

Inputs must use an 8-bit `RGB` or `RGBA` mode; other modes are rejected rather
than silently converted. HAT-L processes RGB only. For `RGBA` input, the alpha
channel is resized separately with Lanczos and that handling is recorded in the
receipt. The target may be the model's native x2 width or a smaller upscale;
targets beyond native x2 are rejected rather than enlarged a second time.

The Blackwell-qualified defaults select physical GPU 0, expose it as
`cuda:0`, use BF16, and process the full frame without tiling. On the RTX PRO
6000 Blackwell Workstation Edition, three full-frame runs were byte-for-byte
deterministic with a 12.85-second median. That inference timing excludes the
intentional per-run trust scans, hashes, model load, and PNG publication. Peak
CUDA memory was 38.85 GiB
allocated and 54.45 GiB reserved. BF16 agreed with FP32 at 58.14 dB PSNR on
the qualification crop. Use `--dtype float32` for the conservative numerical
path. Use `--tile-size 512 --tile-overlap 64` only when the full frame cannot
fit; that lower-memory path is available but is not part of the full-frame
Blackwell qualification.

For the text-heavy 1920x1080 qualification image, comparison against a native
4K rerender showed the expected tradeoff:

| Candidate | PSNR | RGB SSIM | Edge correlation |
|---|---:|---:|---:|
| HAT-L x2 BF16 | 23.219 dB | 0.94175 | 0.6150 |
| Lanczos x2 | 23.969 dB | 0.94025 | 0.6499 |

HAT-L restored sharper type and slightly improved SSIM, but introduced visible
chromatic ringing around small text. Lanczos was softer and cleaner. These are
comparative metrics because the native render is not an exact downsample
ancestor of the source raster. For that asset, the native 4K rerender is the
production result and the HAT-L output is retained only as model evidence.

The command writes PNG output atomically, refuses existing output and receipt
paths unless `--force` is explicit, and rejects symlink destinations even with
`--force`. It verifies the checkpoint hash before model loading, honors the
caller's group/other file-creation restrictions while preserving owner
read/write access, keeps the checkpoint non-group-writable, and emits a JSON
receipt with input, output, model, implementation, runtime, GPU, memory, and
timing identities.

## Local OCR architecture

The supported structured pipeline is `local-mineru25`:

```text
PDF -> PNG + extracted text -> DeepSeek-OCR grounded OCR
                            -> MinerU2.5-Pro structured Markdown
                            -> .ocr + .md + .json bundle
```

DeepSeek-OCR remains the qualified grounded-OCR stage. MinerU2.5-Pro-2605 is
the qualified Markdown stage. The built-in `deepseek-vllm-full` definition is
visible for provenance, but its free-form DeepSeek Markdown stage is withheld
from structured execution and tuning because of severe quality regressions.

The local configuration has three deliberately separate layers:

- The **backend registry** ships immutable model revisions, adapters, runtime
  tuples, and capability-level qualification with the plugin.
- The **installation registry** is machine-local. It binds those definitions
  to exact Python executables, model snapshots, CUDA routing, and optional
  attention backends, then hashes the result. Its default location is
  `$XDG_CONFIG_HOME/sneeze/tpn/ocr-installations.json` (or the equivalent
  path under `~/.config`).
- An **execution profile** is portable across matching machines. It identifies
  a GPU class, exact model-snapshot hashes, a count and SHA-256 identity for
  each installed-distribution manifest, and a reviewed hash of the plugin code
  that qualifies and runs each stage. The full sorted manifest is hashed at
  runtime; persisted tuning evidence retains only its count and SHA-256. The
  profile then records the settings and observed peak memory for the `25`,
  `50`, and `max` tiers.
  Profiles contain no local executable paths, GPU UUIDs, credentials, or
  tokens. Automatic selection searches reviewed profiles bundled with the
  plugin, then
  `$XDG_CONFIG_HOME/sneeze/tpn/ocr-profiles` (or the equivalent path under
  `~/.config`).

Inspect the definitions without importing Torch, Transformers, or vLLM:

```bash
sne ocr-backend-list
sne ocr-pipeline-list
```

Both commands accept `--format json` for content-hashed machine-readable
receipts. `ocr-pipeline-list` reports `local-mineru25` as enabled and
`deepseek-vllm-full` as withheld.

New local models should enter as pinned backend definitions and pipeline
stages, then be qualified for document quality independently of hardware
tuning. Per-GPU profiles optimize an already identified pipeline; they do not
silently substitute a different model.

## Run PDFs in one step

After registering both local backends and installing a matching qualified
profile, invoke `sne` from the registered DeepSeek environment:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 sne ocr-pdf-pages \
  --pdf-to-pages \
  --source-dir ./pdfs \
  --pdf-recursive \
  --pdf-parallelism 32 \
  --pdf-dpi 400 \
  --pipeline local-mineru25 \
  --execution-profile auto \
  --memory-tier 50
```

`--pipeline local-mineru25` already implies `--execution-profile auto`; it is
shown explicitly above to make selection clear. `auto` requires one unique
qualified profile matching the observed GPU, both stage runtimes, and the
requested memory tier. When the tier is omitted, resolution tries `max`, then
`50`, then `25`, and selects the highest supported match. The bundled A6000
profile supports `25`, `50`, and `max`; the RTX PRO 6000 Blackwell profile
supports `25` and `50`. Select an exact profile by portable ID or absolute JSON
path when needed:

```bash
sne ocr-pdf-pages \
  --source-dir ./pages \
  --pipeline local-mineru25 \
  --execution-profile custom-local-mineru25-profile \
  --memory-tier 50 \
  --explain-profile
```

`--explain-profile` prints the resolved execution receipt without running OCR
or writing output. Use `--profile-dir` for a comma-separated list of absolute
profile paths and `--installation-registry` for a non-default machine registry.

If `--dest-dir` is omitted, `ocr-pdf-pages` writes to `--source-dir`. This is
also true with `--pdf-to-pages`: the PDF tree receives the page directories,
extracted `.txt`, grounded `.ocr`, structured `.md`, and final `.json` files.
Supply `--dest-dir` when the source tree should remain read-only.

With `--pdf-to-pages`, existing page files are reused. `--force` rerenders and
reruns OCR; `--pdf-parallelism` and `--pdf-dpi` affect only the Poppler stage.
Without `--pdf-to-pages`, the command consumes existing PNG/TXT page pairs.

Structured execution reuses an existing `.ocr` or `.md` stage only when its
page JSON proves an exact match for the selected pipeline, execution profile,
memory tier, live pipeline definition, input image hash, and persisted stage
bytes. The retained stage must also have passed validation, ended at EOS rather
than the token limit, and have internally consistent promotion hashes. Missing,
legacy, or mismatched receipts fail closed instead of silently mixing outputs
from another model or profile; use a fresh destination or deliberate `--force`
to regenerate them.

The pipeline and execution profile select the model snapshots and all engine
settings. Do not combine structured execution with manual `--model`, batching,
memory, attention, token, n-gram, or window flags; conflicting flags are
rejected rather than silently overriding a tuned profile.

Production must set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and set
`CUDA_VISIBLE_DEVICES` to the value recorded for both installations.
Structured execution fails closed before model loading if either ambient
binding differs.

## Install and register local backends

Prerequisites are Linux with a supported NVIDIA GPU and driver, Python 3.12
with `venv`, Git, Poppler's `pdftocairo` and `pdftotext`, and either mamba or
conda for the isolated model runtimes. Model setup also needs enough local
storage and Hugging Face access for both pinned snapshots.

Use a non-group-writable setup umask before creating environments or model
caches:

```bash
umask 022
```

Registration treats every local executable, model, source, and import path as
code. Every ancestor must be owned by root or the current user and must not be
writable by group or other users, apart from a root-owned sticky ancestor such
as `/tmp`. Environment and cache symlinks must remain inside their trusted
roots. Editable `.pth` entries must be path-only and point to an existing
trusted checkout; arbitrary executable import hooks are rejected. Registries
and tuning directories are created privately as `0600` files under `0700`
directories.

If registration rejects an older conda environment, do not repair it with
`chmod -R`: conda package files may be hardlinked to the package cache. The
current create-environment command is a reproducible bootstrap, but it does not
break those hardlinks. Use the package manager outside Sneeze to clone with
copy mode into a new owner-controlled location before removing group/other
write bits. Likewise, independently copy an unsafe model cache before
normalizing its permissions. Register the copied Python and cache paths again,
then run `ocr-installation-list`; paths and content identities are deliberately
machine-local and exact.

On a clean shell, install the checkout into a small control environment first.
This environment provides `sne` and creates the dedicated model runtimes.
Activate it so every later bare `sne` command resolves this checkout:

```bash
git clone https://github.com/tpn/sneeze-plugin-tpn.git
cd sneeze-plugin-tpn
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
source .venv/bin/activate
command -v sne
sne ocr-pipeline-list
```

An existing plugin-enabled `sne` installation may be used instead. The
checkout must remain at the reviewed source revision throughout registration,
tuning, and profile review.

Start by inspecting the pinned definitions and previewing the environment
plans:

```bash
sne ocr-backend-list --format json
sne ocr-backend-create-env \
  --backend deepseek-vllm \
  --editable-root "$PWD" \
  --dry-run
sne ocr-backend-create-env --backend mineru25-pro-2605 --dry-run

sne ocr-backend-create-env \
  --backend deepseek-vllm \
  --editable-root "$PWD"
sne ocr-backend-create-env --backend mineru25-pro-2605
```

The DeepSeek plan installs
`vllm==0.21.1rc1.dev438+g4ff865c38` from vLLM's immutable full-build-commit
CUDA 13.0 index. The mutable nightly alias no longer lists that build, so do
not replace the commit-specific index or silently substitute a nearby vLLM
version; runtime registration and profile matching are exact. Upstream wheel
retention remains an external availability dependency.

`--editable-root` ensures the in-process DeepSeek runtime executes this exact
checkout; its portable implementation hash detects later source drift.
MinerU's standalone adapter attests the same source tree without importing the
plugin into its Transformers environment. Keep MinerU's stack separate from
DeepSeek's vLLM stack. Fetch each exact model revision into an absolute cache
directory, previewing first if desired:

```bash
sne ocr-backend-fetch-model \
  --backend deepseek-vllm \
  --cache-dir /absolute/path/to/deepseek-hf-cache \
  --dry-run

sne ocr-backend-fetch-model \
  --backend mineru25-pro-2605 \
  --cache-dir /absolute/path/to/mineru-hf-cache \
  --dry-run
```

After removing `--dry-run` and completing both downloads, register the exact
environment/cache/GPU bindings:

```bash
sne ocr-installation-register \
  --backend deepseek-vllm \
  --python-executable /absolute/path/to/sneeze-ocr-vllm/bin/python \
  --cache-dir /absolute/path/to/deepseek-hf-cache \
  --cuda-visible-devices 0 \
  --device cuda:0 \
  --attention-backend auto

sne ocr-installation-register \
  --backend mineru25-pro-2605 \
  --python-executable /absolute/path/to/sneeze-ocr-mineru25/bin/python \
  --cache-dir /absolute/path/to/mineru-hf-cache \
  --cuda-visible-devices 0 \
  --device cuda:0 \
  --attention-backend sdpa

sne ocr-installation-list
```

Registration validates and hashes the executable, pinned snapshot, adapter,
reviewed implementation bundle, CUDA selection, and complete normalized Python
distribution manifest, including transitive packages. Before enrollment it
also probes every modeled runtime distribution through the selected Python and
requires its exact pinned version. This catches stale CUDA-wheel metadata (for
example, `torch==2.11.0` where the backend requires
`torch==2.11.0+cu130`) before an expensive GPU sweep. A tuning plan binds the
installation record SHA-256 plus the executable, cache, snapshot, weights,
adapter, and implementation identities. A resume accepts only that exact plan
and its matching artifacts; reusing the same path with changed bytes is
rejected. Re-listing also revalidates current files, so a changed source,
environment, or cache fails closed. Add `--attention-backend` only when an
explicit implementation is part of the installation being tuned. Use the same
optional `--installation-registry /absolute/path/to/registry.json` on register,
list, tune, and run commands when not using the XDG default.

## Tune a new GPU

Profiles are GPU-specific because usable batch size, attention implementation,
throughput, and memory behavior depend on the card and runtime tuple. First
probe exactly one visible NVIDIA GPU:

```bash
sne ocr-hardware-probe \
  --torch-python /absolute/path/to/sneeze-ocr-vllm/bin/python \
  --cuda-visible-devices 0 \
  --output /absolute/scratch/hardware-receipt.json
```

The isolated Torch probe is required to classify dedicated versus unified
memory. Without `--torch-python`, hardware identity is still reported but
memory topology and tier budgets remain unavailable.

The probe calculates each tier capacity as
`min(floor(total_bytes * fraction), total_bytes - reserve_bytes)`:

- `25`: at most 25% of reported GPU memory.
- `50`: at most 50% of reported GPU memory.
- `max`: fraction 90% for dedicated VRAM or 75% for unified memory.

The reserve is 2 GiB for dedicated VRAM and 16 GiB for unified memory and is
applied to every tier through the formula above. Live free memory is checked
separately. A tier is not usable merely because its theoretical budget fits
the card. For integrated GPUs, a finite current cgroup-v2 memory limit also
caps both the capacity and live-availability calculation; host-visible RAM
outside the workload's cgroup is not counted.

The checked-in cuPhoton manifest drives the onboarding sweep. Inputs are
read-only; all generated OCR, logs, telemetry, and candidate artifacts must go
to a new or matching resumable scratch destination:

```bash
sne ocr-profile-tune \
  --pipeline local-mineru25 \
  --manifest /absolute/path/to/sneeze-plugin-tpn/benchmarks/ocr/cuphoton-regressions.jsonl \
  --input-root /absolute/read-only/path/to/cuphoton-papers/papers \
  --dest-dir /absolute/scratch/local-mineru25-tuning \
  --memory-tiers 25,50,max \
  --batch-sizes 1,2,4,8 \
  --deepseek-moe-backend triton \
  --discovery-repeats 1 \
  --confirmation-repeats 5 \
  --timeout-seconds 3600 \
  --profile-output /absolute/profile-dir/local-mineru25-custom-gpu.json \
  --profile-id local-mineru25-custom-gpu \
  --profile-display-name "Custom GPU local MinerU 2.5" \
  --dry-run
```

Remove `--dry-run` to execute. The tuner runs each stage in its own process,
uses immutable per-candidate scratch directories, rejects quality/provenance,
output-limit, and memory failures, selects the fastest eligible batch per tier,
then confirms the selection with repeated cold runs. Interrupted sessions can
resume only when their exact installation-bound plan and artifacts still
match.

Qualified DeepSeek tuning explicitly uses vLLM's Triton fused-MoE backend and
records both the requested and configured value. `auto` is intentionally not
eligible for a profile because its kernel choice can change with hardware,
runtime, or cache state. During each DeepSeek engine lifetime the plugin also
fixes and records `MAX_JOBS=1` and `FLASHINFER_NVCC_THREADS=1`, then restores
the caller's prior environment during cleanup. These are implementation safety
controls, not tuning knobs: they bound cold extension compilation on
unified-memory systems such as GB10 while the selected Triton path avoids the
problematic FlashInfer MoE JIT entirely.

For DeepSeek, every tier/batch candidate is swept with 1, 2, and 4 GiB of
explicit vLLM engine-overhead headroom (omitting a headroom that cannot fit the
tier). The vLLM allocation denominator is
`max(portable GPU total, torch CUDA total)` and the requested engine target is
`tier budget - headroom`; the resulting utilization and both byte values are
recorded as effective settings. This headroom is distinct from the hardware
safety reserve above. It prevents a tier from assigning its entire budget to
vLLM while leaving no space for CUDA/runtime overhead.

Manifest assertions may name the lanes to which they apply. The cuPhoton
fixture scopes raw-text recall to Markdown `parity`/`best`, while grounded-OCR
recall and Acero table-completion checks target `grounded_ocr`. Assertions
without `lanes` still apply to every evaluated lane. This prevents a materially
complete grounded pass from being rejected by a Markdown-specific metric.

`batch_size` has stage-specific meaning. DeepSeek batches whole pages, so the
checked-in three-page manifest exercises effective page batches `1`, `2`, and
`3`; requested sizes `4` and `8` both coalesce to `3` and are never reported as
larger tested batches. MinerU batches recognized layout blocks and can exercise
all four sizes on those same pages. Use a larger manifest when qualifying
larger DeepSeek page batches.

`--profile-output`, `--profile-id`, and `--profile-display-name` must be supplied
together. A completed three-page onboarding run writes a sealed **provisional**
profile. It verifies the hardware/runtime/settings layer; general model quality
remains the separately reviewed backend qualification recorded by the backend
registry. After at least five cold confirmations per supported tier, manually
review the immutable summaries, results, run receipts, and representative OCR
artifacts, then create a qualification candidate:

- every planned discovery candidate has a terminal evidence receipt;
- both stages have an eligible selection at every declared tier;
- all five cold confirmations contain all three pages and pass their scoped
  quality assertions;
- every page has complete provenance, an acceptable termination, and no
  `max_tokens` stop;
- observed peak memory remains within the declared tier budget;
- the hardware, installation, model, adapter, implementation, manifest, and
  effective-setting hashes match the sealed plan; and
- the retained OCR/Markdown artifacts contain no regression that aggregate
  metrics missed.

```bash
sne ocr-profile-qualify \
  --profile /absolute/profile-dir/local-mineru25-custom-gpu.json \
  --output /absolute/profile-dir/local-mineru25-custom-gpu.qualified.json \
  --reviewed-evidence
```

This command does not grant itself production authority. Formal onboarding is
completed by reviewing and checking the qualified JSON into
`src/sneeze/tpn/ocr_profiles_data/`, adding its exact content hash to the
source-controlled trusted-profile set, and rerunning the full tests. Changing
`qualification.state` and recomputing the public JSON hash cannot make a
profile trusted. A qualified profile requires passing, complete evidence for
every declared suite and supported tier, the current canonical memory policy,
at least five repeats, exact effective settings, source identity, model bytes,
and the count/SHA-256 identity of the complete runtime distribution manifest.
Package names from that manifest are deliberately not persisted.

Production execution accepts qualified profiles by default. A provisional
profile requires the conspicuous `--allow-provisional-profile` flag and should
be used only for an explicitly approved trial. Resolve it first without model
loading:

```bash
sne ocr-pdf-pages \
  --source-dir ./pages \
  --pipeline local-mineru25 \
  --execution-profile /absolute/profile-dir/local-mineru25-custom-gpu.json \
  --memory-tier max \
  --allow-provisional-profile \
  --explain-profile
```

Then copy the three-page fixture to a new scratch input tree and run an actual
scratch smoke before considering corpus work:

```bash
sne ocr-pdf-pages \
  --source-dir /absolute/scratch/cuphoton-three-page-input \
  --dest-dir /absolute/scratch/cuphoton-three-page-output \
  --text-source-dir /absolute/scratch/cuphoton-three-page-input \
  --pipeline local-mineru25 \
  --execution-profile /absolute/profile-dir/local-mineru25-custom-gpu.json \
  --memory-tier max \
  --allow-provisional-profile
```

Do not use `--force`, a canonical corpus, or a destination containing prior
artifacts for this smoke.

Machines with canonical corpora should list them in the private policy file
`$XDG_CONFIG_HOME/sneeze/tpn/ocr-protected-roots` (falling back to
`$HOME/.config/sneeze/tpn/ocr-protected-roots`). The file must have mode
`0600` and contain one existing canonical absolute directory per line, for
example `/absolute/path/to/canonical-corpus`. Benchmark, tuning, provider
evidence, installation-registry, hardware-receipt, and generated-profile
preflights reject destinations that overlap a configured root before creating
files. Keep this policy machine-local; its contents are not copied
automatically into profiles or evidence receipts. Normal provenance still
records paths explicitly supplied as benchmark or tuning inputs.

Inspect available profiles and their qualification state:

```bash
sne ocr-profile-list --profile-dir /absolute/profile-dir
sne ocr-profile-show \
  --profile-dir /absolute/profile-dir \
  --profile-id local-mineru25-custom-gpu
```

Both installation and profile records contain hashes and provenance, but no
secret values. Keep model caches, tuning output, and experimental OCR in
scratch or machine-local storage; never aim a forced experimental run at a
canonical paper corpus.

Native environments are the formally modeled installation path. A container
may be used to run an experimental tuning sweep when Sneeze, both registered
Python runtimes, the exact caches, and the scratch tree all live inside that
container. The current portable profile does not itself attest an OCI image or
mount topology, so an image-backed result remains provisional until that
launcher identity is independently recorded and reviewed.

## Current qualification status

The plugin bundles two trusted execution profiles:

- `local-mineru25-nvidia-rtx-a6000-sm86-48gb`, profile SHA-256
  `e3e1df959942b96141d0dbbfa65b47ded5bedede446a141461d0d77ef15b3aef`,
  covers the NVIDIA RTX A6000 (SM86, 51,527,024,640 bytes) at the 25%, 50%,
  and maximum-safe tiers. DeepSeek uses Triton MoE and batch 3 with 2, 1, and
  1 GiB of engine headroom at 25%, 50%, and max respectively. MinerU uses
  batch 2, 2, and 4 at those tiers.
- The selected A6000 settings were reconfirmed against the final source with
  one focused discovery and five cold repeats of all three regression pages
  for each of the six stage/tier configurations. All 90 confirmation outputs
  passed with complete provenance and EOS termination. `--execution-profile
  auto` can select this profile only when the live GPU and both registered
  runtime identities match it.
- `local-mineru25-nvidia-rtx-pro-6000-blackwell-sm120-96gb`, profile SHA-256
  `368531b003418d2cb0749949e8e842a6db929d8bb7e71b62ddaac0e3d2af2c51`,
  covers the NVIDIA RTX PRO 6000 Blackwell (SM120, 102,641,958,912 bytes) at
  the 25% and 50% tiers. At 25%, DeepSeek uses Triton MoE, batch 1, and 4 GiB
  of headroom while MinerU uses batch 2. At 50%, DeepSeek uses batch 3 with
  the same headroom and MinerU uses batch 4. Every stage/tier configuration
  passed five repeats of all three pages. The maximum tier was not tuned in
  the final source-bound qualification run.

GB10 hardware discovery and unified-memory controls remain available, but no
GB10 profile is bundled or trusted. Two final source-bound qualification
attempts failed closed during the first DeepSeek cold start when the unchanged
PSI pressure watchdog tripped; neither attempt completed a provider call or
produced OCR-quality evidence. The 50% settings therefore remain unqualified
rather than being published from older runtime evidence.

Automatic profile selection always requires the live GPU and both registered
runtime identities to match the reviewed profile exactly.

Portable profiles bind the GPU class, model bytes, Python distributions,
Torch/CUDA runtime tuple, attention implementation, and reviewed plugin code;
they do not currently bind an NVIDIA driver version or ambient dynamic-loader
state. The final A6000 evidence used driver `580.95.05`; it did not independently
attest loader-path absence. Treat a driver or dynamic-loader change as an
unqualified runtime change: re-register the installations and run the
three-page scratch smoke before production, requalifying when runtime or output
evidence differs.

## Lower-level commands

Render PDFs without OCR when the stages need separate destinations:

```bash
sne pdf-to-pages --source-dir ./pdfs --dest-dir ./pages --recursive --dpi 400
```

The renderer uses Poppler's `pdftocairo` and `pdftotext`. Install Poppler with
your OS package manager or conda/mamba.

Calling `ocr-pdf-pages` without `--pipeline` preserves the legacy manual
DeepSeek/vLLM interface. It remains useful for grounded OCR and controlled
experiments. Raw mode accepts `--moe-backend auto|triton` and defaults to
`auto`; an explicit `triton` request fails fast when the installed vLLM does
not support that modeled engine argument. Raw engine flags remain outside the
structured, GPU-profiled path. The lower-level local provider-spec and
benchmark workflow is documented in
[`benchmarks/ocr/README.md`](benchmarks/ocr/README.md).
