# sneeze-plugin-tpn

Public Sneeze plugin package for `sneeze.tpn`.

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
  --memory-tier max
```

`--pipeline local-mineru25` already implies `--execution-profile auto`; it is
shown explicitly above to make selection clear. `auto` requires one unique
qualified profile matching the observed GPU, both stage runtimes, and the
requested memory tier; the tier defaults to `max`. Select an exact profile by
portable ID or absolute JSON path when needed:

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
distribution manifest, including transitive packages. A tuning plan binds the
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

The plugin bundles one trusted execution profile:

- `local-mineru25-nvidia-rtx-a6000-sm86-48gb`, profile SHA-256
  `0b15775d6931f451b4efa9bb77a38ca2c546a707932e4ddee33da11955ac7f58`,
  covers the NVIDIA RTX A6000 (SM86, 51,527,024,640 bytes) at the 25%, 50%,
  and maximum-safe tiers. DeepSeek uses batch 3 with 1, 2, and 1 GiB of engine
  headroom at 25%, 50%, and max respectively. MinerU uses batch 2 at every
  tier.
- Its formal onboarding sweep executed 39 discovery candidates. Thirty-seven
  were eligible; the two rejected candidates were over-constrained DeepSeek
  25% configurations. All six selected stage/tier configurations then passed
  five repeats of all three regression pages with complete provenance and no
  max-token termination. Manual review covered all 25 distinct output hashes
  across the 90 confirmation artifacts. `--execution-profile auto` can select
  this profile only when the live GPU and both registered runtime identities
  match it.

The retained GB10 run remains diagnostic rather than qualification:

- GB10: MinerU passed all three pages in the 25%, 50%, and max exploratory
  arms, but DeepSeek failed quality on all three tiers. That arm also used an
  OCI-derived environment with vLLM 0.20.0 rather than the pinned 0.21.1
  development build. It therefore cannot produce a qualified profile, even
  aside from the current lack of OCI-launcher attestation.

Portable profiles bind the GPU class, model bytes, Python distributions,
Torch/CUDA runtime tuple, attention implementation, and reviewed plugin code;
they do not currently bind an NVIDIA driver version or an ambient
`LD_LIBRARY_PATH`. The A6000 evidence used driver `580.95.05` and a clean
environment with no loader-path override. Treat a driver or dynamic-loader
change as an unqualified runtime change: re-register the installations and run
the three-page scratch smoke before production, requalifying when runtime or
output evidence differs.

## Lower-level commands

Render PDFs without OCR when the stages need separate destinations:

```bash
sne pdf-to-pages --source-dir ./pdfs --dest-dir ./pages --recursive --dpi 400
```

The renderer uses Poppler's `pdftocairo` and `pdftotext`. Install Poppler with
your OS package manager or conda/mamba.

Calling `ocr-pdf-pages` without `--pipeline` preserves the legacy manual
DeepSeek/vLLM interface. It remains useful for grounded OCR and controlled
experiments, but raw engine flags are intentionally outside the structured,
GPU-profiled path. The lower-level local provider-spec and benchmark workflow
is documented in [`benchmarks/ocr/README.md`](benchmarks/ocr/README.md).
