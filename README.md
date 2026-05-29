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

Useful constrained run for a single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 sne ocr-pdf-pages \
  --source-dir ./pages \
  --dest-dir ./ocr \
  --batch-size 1 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --max-tokens 4096
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
