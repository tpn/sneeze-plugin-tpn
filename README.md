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

For the full OCR bundle, render pages first and then run:

```bash
sne pdf-to-pages --source-dir ./pdfs --dest-dir ./pages --recursive
sne ocr-pdf-pages --source-dir ./pages --dest-dir ./ocr --prompt-mode full
```

`--prompt-mode full` consumes existing page PNGs and writes `.ocr`, `.md`, and
`.json` outputs. It does not render PDFs itself. HTML OCR remains a separate
single-output mode via `--output-format html`.

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

## MinerU2.5-Pro Markdown Provider

The shipped adapter at `sneeze.tpn.providers.mineru25` implements the native
two-stage MinerU layout/recognition pipeline that qualified on the cuPhoton
regression and stratified suites. It is pinned by default to:

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

Download the exact revision into one explicit Hugging Face hub cache and use
that same `HF_HUB_CACHE` in the provider specification. `HF_HOME` alone points
at its `hub/` child and is not equivalent to a custom `cache_dir`:

```bash
HF_HUB_CACHE=/absolute/model-cache \
  /absolute/mineru-env/bin/python -c \
  'from huggingface_hub import snapshot_download; print(snapshot_download(repo_id="opendatalab/MinerU2.5-Pro-2605-1.2B", revision="bff20d4ae2bf202df9f45284b4d43681555a97ed", cache_dir="/absolute/model-cache"))'
```

Resolve the installed adapter path from the Sneeze environment:

```bash
python -c 'from sneeze.tpn.providers import mineru25; print(mineru25.__file__)'
```

Then create a scratch provider specification with absolute paths:

```json
{
  "schema_version": 1,
  "provider_id": "mineru2.5-pro-2605-transformers",
  "argv": [
    "/absolute/mineru-env/bin/python",
    "/absolute/path/to/sneeze/tpn/providers/mineru25.py",
    "--model-id", "opendatalab/MinerU2.5-Pro-2605-1.2B",
    "--revision", "bff20d4ae2bf202df9f45284b4d43681555a97ed",
    "--device", "cuda:0",
    "--attention-backend", "sdpa"
  ],
  "egress": "none",
  "model": {
    "requested": "opendatalab/MinerU2.5-Pro-2605-1.2B",
    "revision": "bff20d4ae2bf202df9f45284b4d43681555a97ed"
  },
  "lanes": {
    "best": {
      "prompt": "Use the official MinerU two-stage layout and recognition pipeline and convert its structured result to Markdown.",
      "settings": {
        "backend": "official-two-step-transformers",
        "image_analysis": false,
        "batch_size": 1
      },
      "include_raw_text": false
    }
  },
  "timeout_seconds": 3600,
  "secret_env_names": [],
  "env": {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "0",
    "HF_HUB_CACHE": "/absolute/model-cache",
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1"
  }
}
```

The adapter records the original PNG hash/dimensions, all rendered stage
prompts, native structured output, stage token/termination traces, complete
pipeline and generation settings, exact model/config/tokenizer hashes, runtime
package identity, GPU/backend details, and adapter hash.

Complete constrained MinerU run for a single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 sne ocr-pdf-pages \
  --source-dir ./pages \
  --dest-dir ./ocr \
  --text-source-dir ./pages \
  --prompt-mode full \
  --batch-size 1 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --max-tokens 8192 \
  --markdown-provider-spec /absolute/scratch/provider-spec.json \
  --markdown-provider-lane best \
  --markdown-provider-run-dir /absolute/scratch/provider-evidence
```

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
