# sneeze-plugin-tpn

Public Sneeze plugin package for `sneeze.tpn`.

## PDF Page Extraction

Render PDF files into per-document page directories:

```bash
sne pdf-to-pages --source-dir ./pdfs --dest-dir ./pages --recursive
```

Outputs are named like:

```text
pages/
  example/
    page-1.png
    page-1.txt
```

The command uses Poppler's `pdftocairo` and `pdftotext`. Install Poppler with
your OS package manager or conda/mamba.

## DeepSeek OCR

OCR rendered page images with a local vLLM DeepSeek OCR environment:

```bash
sne ocr-pdf-pages --source-dir ./pages --dest-dir ./ocr --limit 4
```

For the full OCR bundle, one command can render each PDF and then OCR the
resulting page tree:

```bash
sne ocr-pdf-pages \
  --pdf-to-pages \
  --source-dir ./pdfs \
  --pdf-recursive \
  --pdf-parallelism 32 \
  --pdf-dpi 400 \
  --prompt-mode full
```

With `--pdf-to-pages`, `--source-dir` is the PDF root. If `--dest-dir` is
omitted, per-PDF page directories, extracted text, and OCR outputs are written
under that same root, matching the defaults of the separate commands. Supply
an explicit `--dest-dir` when the PDF tree should remain read-only or the
rendered/OCR files belong elsewhere. Existing page files are reused. `--force`
rerenders them and reruns OCR, while `--pdf-parallelism` and `--pdf-dpi`
control only the Poppler stage. Encrypted or unreadable PDFs are reported
without creating source-side `.encrypted` sentinels.

`ocr-pdf-pages` always uses the in-process DeepSeek-OCR/vLLM adapter for
grounded OCR. Select its checkpoint with `--model` and pair a non-default Hub
model with its own immutable `--model-revision`; omitting both uses the pinned
DeepSeek-OCR defaults. The batching, memory, attention, token, n-gram, and
window flags only tune that engine and are not needed merely to choose the
default model. For example, the following optional settings reproduce the
constrained single-GPU profile from the corpus investigation:

```bash
CUDA_VISIBLE_DEVICES=0 sne ocr-pdf-pages \
  --pdf-to-pages \
  --source-dir ./pdfs \
  --pdf-recursive \
  --prompt-mode full \
  --model deepseek-ai/DeepSeek-OCR \
  --model-revision 9f30c71f441d010e5429c532364a86705536c53a \
  --batch-size 1 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --max-tokens 8192 \
  --gpu-memory-utilization 0.8
```

MinerU is not selected through `--model`: in full mode it is the qualified
out-of-process Markdown provider, while DeepSeek remains the grounded-OCR
engine. Select it with `--markdown-provider-spec`,
`--markdown-provider-lane`, and `--markdown-provider-run-dir` as documented
below. Hosted OpenAI and Anthropic profiles remain benchmark-only.

Without `--pdf-to-pages`, the command consumes existing page PNGs as before.
Use the two separate commands when rendered pages and OCR outputs need distinct
destination trees. HTML OCR remains a separate single-output mode via
`--output-format html`.

Full mode normally uses DeepSeek-OCR for both grounded OCR and Markdown. It can
instead keep DeepSeek for the grounded `.ocr` pass and use a local provider for
Markdown:

```bash
sne ocr-pdf-pages \
  --source-dir ./pages \
  --dest-dir ./ocr \
  --text-source-dir ./pages \
  --prompt-mode full \
  --markdown-provider-spec /absolute/scratch/provider-spec.json \
  --markdown-provider-lane best \
  --markdown-provider-run-dir /absolute/scratch/provider-evidence
```

The evidence directory must be absolute and new or empty. It must not overlap
the source or OCR destination. The grounded vLLM engine is shut down before one
provider process handles the complete Markdown batch. A candidate is promoted
atomically only after response-provenance, max-token, runaway-axis,
long-repetition, and table-completion checks pass. Invalid candidates remain in
the evidence directory and do not replace an existing `.md`; the command exits
nonzero if any page or batch fails.

Production provider specifications are local-only (`egress: "none"`), cannot
inherit secrets, and run with ambient inference controls scrubbed. The benchmark
harness has a separate, explicit approval gate for API-backed providers; see
[`benchmarks/ocr/README.md`](benchmarks/ocr/README.md).

## Structured local OCR backends

The backend registry keeps model revisions, environment tuples, adapter
identity, and per-capability qualification status in one place. Listing the
registry is lightweight and does not import Torch, Transformers, or vLLM:

```bash
sne ocr-backend-list
sne ocr-backend-list --format json
```

Use the setup commands to create an isolated environment, fetch the immutable
model revision, write a local provider specification, and validate the complete
binding. This MinerU example keeps the model, provider specification, and
diagnostic evidence in scratch space:

```bash
BACKEND_SCRATCH=$(mktemp -d /tmp/sneeze-tpn-ocr-backend.XXXXXX)
MODEL_CACHE="$BACKEND_SCRATCH/hf-cache"
MINERU_PY=/absolute/path/to/sneeze-ocr-mineru25/bin/python

sne ocr-backend-create-env --backend mineru25 --dry-run
sne ocr-backend-create-env --backend mineru25

sne ocr-backend-fetch-model \
  --backend mineru25 \
  --cache-dir "$MODEL_CACHE" \
  --dry-run
sne ocr-backend-fetch-model \
  --backend mineru25 \
  --cache-dir "$MODEL_CACHE"

sne ocr-backend-write-spec \
  --backend mineru25 \
  --output "$BACKEND_SCRATCH/provider-spec.json" \
  --python-executable "$MINERU_PY" \
  --cache-dir "$MODEL_CACHE"

sne ocr-backend-doctor \
  --provider-spec "$BACKEND_SCRATCH/provider-spec.json"
sne ocr-backend-doctor \
  --provider-spec "$BACKEND_SCRATCH/provider-spec.json" \
  --load-model
```

The dry runs print the exact command plans. The fetch command uses the pinned
revision rather than a mutable branch or tag. `ocr-backend-write-spec` refuses
to replace an existing file. The doctor checks the model snapshot, adapter,
offline settings, and runtime package tuple; `--load-model` adds an actual GPU
initialization check. Keep the resulting specification and all benchmark or
experimental OCR output in scratch space, never in either canonical paper
corpus.

`ocr-backend-write-spec` applies to provider-process backends such as MinerU.
DeepSeek remains the in-process grounded-OCR backend. The registry reports its
Markdown capability separately from its grounded-OCR capability, so a model is
not treated as generally qualified because one capability passed.

### MinerU2.5-Pro Markdown provider

The shipped adapter at `sneeze.tpn.providers.mineru25` implements the native
two-stage MinerU layout/recognition pipeline that qualified on the cuPhoton
regression and stratified suites. The registry pins it to:

```text
opendatalab/MinerU2.5-Pro-2605-1.2B
bff20d4ae2bf202df9f45284b4d43681555a97ed
```

Keep MinerU in a dedicated environment; do not install its Transformers stack
into the DeepSeek/vLLM environment. The qualified environment used Python
3.12.13, torch 2.11.0+cu130, torchvision 0.26.0+cu130, Transformers 4.57.6,
mineru-vl-utils 1.0.5, accelerate 1.14.0, huggingface-hub 0.36.2, and Pillow
12.2.0. The model card and `mineru-vl-utils` declare Apache-2.0; weights are not
vendored by this plugin.

The adapter records the original PNG hash/dimensions, all rendered stage
prompts, native structured output, stage token/termination traces, complete
pipeline and generation settings, exact model/config/tokenizer hashes, runtime
package identity, GPU/backend details, and adapter hash.

Complete constrained MinerU run for a single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 sne ocr-pdf-pages \
  --source-dir ./pages \
  --dest-dir "$BACKEND_SCRATCH/ocr" \
  --text-source-dir ./pages \
  --prompt-mode full \
  --batch-size 1 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --max-tokens 8192 \
  --markdown-provider-spec "$BACKEND_SCRATCH/provider-spec.json" \
  --markdown-provider-lane best \
  --markdown-provider-run-dir "$BACKEND_SCRATCH/provider-evidence"
```

Hosted OpenAI and Anthropic profiles are benchmark-only. They cannot be used as
the production Markdown provider above. Their offline planning, explicit
approval, retention, credential, and benchmark workflow is documented in
[`benchmarks/ocr/README.md`](benchmarks/ocr/README.md).

## Environment Bootstrap

Create a conda/mamba environment for OCR:

```bash
sne deepseek-ocr-create-env --env-name sneeze-ocr-vllm
```

Preview the commands first:

```bash
sne deepseek-ocr-create-env --dry-run
```

Download/cache the OCR model:

```bash
sne deepseek-ocr-download-model \
  --env-name sneeze-ocr-vllm \
  --model deepseek-ai/DeepSeek-OCR
```

The environment helper installs PyTorch and vLLM separately because CUDA wheel
indexes vary by host. The default is CUDA 13.0 style wheels (`cu130`).
