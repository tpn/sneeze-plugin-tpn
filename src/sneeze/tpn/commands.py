from __future__ import annotations

import gc
import hashlib
import math
import os
import secrets
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from sneeze.command import CommandError
from sneeze.commandinvariant import InvariantAwareCommand
from sneeze.invariant import (
    BoolInvariant,
    DirectoryInvariant,
    NonNegativeIntegerInvariant,
    PositiveIntegerInvariant,
    StringInvariant,
)

from .ocr_vllm import (
    DEFAULT_DEEPSEEK_MOE_BACKEND,
    DEFAULT_MARKDOWN_WINDOW_SIZE,
    DEFAULT_OCR_MODEL,
    DEFAULT_OCR_MODEL_REVISION,
    DEFAULT_OCR_WINDOW_SIZE,
    QUALIFIED_DEEPSEEK_MOE_BACKEND,
    validate_deepseek_moe_backend,
)

DEFAULT_OCR_ENV_NAME = "sneeze-ocr-vllm"
DEFAULT_TORCH_BACKEND = "cu130"
DEFAULT_PYTHON_VERSION = "3.12"
DEFAULT_CUDA_130_VLLM_WHEEL_INDEX = "https://wheels.vllm.ai/nightly/cu130/"
_STRUCTURED_PROVIDER_STARTUP_ALLOWANCE_SECONDS = 600
_STRUCTURED_PROVIDER_RUNTIME_SAFETY_FACTOR = 3.0
_STRUCTURED_PROVIDER_MAX_TIMEOUT_SECONDS = 7 * 24 * 60 * 60


def _structured_markdown_timeout_seconds(
    provider: Mapping[str, object],
    page_count: int,
) -> int | None:
    execution = provider.get("execution_profile")
    if not isinstance(execution, Mapping):
        return None
    profile = execution.get("profile")
    receipt = execution.get("receipt")
    memory_tier = execution.get("memory_tier")
    if memory_tier is None and isinstance(receipt, Mapping):
        memory_tier = receipt.get("memory_tier")
    try:
        throughput = profile["memory_tiers"][memory_tier]["stages"][
            "markdown"
        ]["confirmed_pages_per_second"]
        base_timeout = provider["spec"]["timeout_seconds"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "structured provider profile has no timeout evidence"
        ) from exc
    if (
        not isinstance(page_count, int)
        or isinstance(page_count, bool)
        or page_count <= 0
        or not isinstance(throughput, (int, float))
        or isinstance(throughput, bool)
        or not math.isfinite(float(throughput))
        or throughput <= 0
        or not isinstance(base_timeout, (int, float))
        or isinstance(base_timeout, bool)
        or base_timeout <= 0
    ):
        raise RuntimeError("structured provider timeout evidence is invalid")
    projected = math.ceil(
        _STRUCTURED_PROVIDER_STARTUP_ALLOWANCE_SECONDS
        + _STRUCTURED_PROVIDER_RUNTIME_SAFETY_FACTOR
        * page_count
        / float(throughput)
    )
    timeout = max(math.ceil(float(base_timeout)), projected)
    if timeout > _STRUCTURED_PROVIDER_MAX_TIMEOUT_SECONDS:
        raise RuntimeError(
            "structured provider projected runtime exceeds the safe bound"
        )
    return timeout


@dataclass(frozen=True)
class PdfPagesRunResult:
    """Page identities produced or reused by one renderer pass."""

    image_paths: tuple[str, ...]
    rendered_image_paths: frozenset[str]
    rendered_text_image_paths: frozenset[str]


def _run_pdf_to_pages(
    *,
    source_dir: str,
    dest_dir: str,
    recursive: bool,
    parallelism: int | None,
    limit: int | None,
    dpi: int,
    force: bool,
    no_text: bool,
    record_encrypted_sentinels: bool,
    out,
) -> PdfPagesRunResult:
    """Render one PDF tree using the shared command implementation."""

    from .pdf_pages import (
        build_pdf_page_tasks,
        build_pdf_pages_tasks,
        extract_pdf_page_and_text,
        inspect_png_file,
        preferred_page_image_path,
        prepare_pdf_pages_tasks,
        run_tasks_in_threads,
        write_encrypted_sentinels,
    )

    dest_dir = os.path.abspath(dest_dir)
    extract_text = not no_text
    workers = _resolve_workers(parallelism)
    os.makedirs(dest_dir, exist_ok=True)

    out(f"Scanning for PDFs under {source_dir}...")
    tasks = build_pdf_pages_tasks(
        source_dir,
        dest_dir,
        recursive=recursive,
        limit=limit,
        text_root=dest_dir if extract_text else None,
    )
    if not tasks:
        out("No PDFs found.")
        return PdfPagesRunResult((), frozenset(), frozenset())

    out(f"Found {len(tasks)} PDF(s); preparing with {workers} worker(s).")
    tasks, skipped = prepare_pdf_pages_tasks(
        tasks,
        max_workers=workers,
    )
    _report_skipped_pdfs(skipped, out)
    if not tasks:
        out("No readable PDFs found.")
        return PdfPagesRunResult((), frozenset(), frozenset())

    page_tasks = build_pdf_page_tasks(
        tasks,
        extract_text=extract_text,
        force=force,
        dpi=dpi,
    )
    rendered_image_paths = frozenset(
        os.path.abspath(task.image_output_path)
        for task in page_tasks
        if task.render_image
    )
    rendered_text_image_paths = frozenset(
        os.path.abspath(task.image_output_path)
        for task in page_tasks
        if task.render_text
    )
    if not page_tasks:
        out("No PDF pages need extraction.")
    else:
        run_tasks_in_threads(
            page_tasks,
            extract_pdf_page_and_text,
            max_workers=workers,
            desc="Extracting PDF pages",
        )

    expected_images = []
    for task in tasks:
        page_count = int(task.page_count or 0)
        digits = max(1, len(str(page_count)))
        for page_num in range(1, page_count + 1):
            image_path = preferred_page_image_path(
                task.dest_dir,
                page_num,
                digits,
            )
            problem = inspect_png_file(image_path)
            if problem is not None:
                raise RuntimeError(
                    f"Rendered page image is invalid: {image_path} "
                    f"({problem})"
                )
            expected_images.append(os.path.abspath(image_path))
    if page_tasks and record_encrypted_sentinels:
        written = write_encrypted_sentinels(skipped)
        if written:
            label = "sentinel" if len(written) == 1 else "sentinels"
            out(f"Wrote {len(written)} .encrypted {label}.")
    return PdfPagesRunResult(
        image_paths=tuple(expected_images),
        rendered_image_paths=rendered_image_paths,
        rendered_text_image_paths=rendered_text_image_paths,
    )


class TpnPluginInfo(InvariantAwareCommand):
    """Show basic information about this plugin."""

    def run(self):
        self._out("sneeze plugin: tpn")


class PdfToPages(InvariantAwareCommand):
    """Render PDFs into page PNG/text files for later OCR.

    Requires Poppler: `pdftocairo` for PNG rendering and `pdftotext` unless
    `--no-text` is used. Use the generated page tree with
    `sne ocr-pdf-pages`.
    """

    _verbose_ = True

    source_dir = None
    _source_dir = None

    class SourceDirArg(DirectoryInvariant):
        _help = "Source directory containing PDF files."

    dest_dir = None
    _dest_dir = None

    class DestDirArg(StringInvariant):
        _help = (
            "Destination root for per-PDF page directories. Defaults to "
            "<source-dir>."
        )
        _mandatory = False

    recursive = None

    class RecursiveArg(BoolInvariant):
        _help = "Recursively process PDF files. [default: %default]"
        _mandatory = False
        _default = False

    parallelism = None
    _parallelism = None

    class ParallelismArg(NonNegativeIntegerInvariant):
        _help = (
            "Parallel worker count; 0 means CPU count. [default: %default]"
        )
        _mandatory = False
        _default = 0

    limit = None
    _limit = None

    class LimitArg(PositiveIntegerInvariant):
        _help = "Limit the number of PDFs to process."
        _mandatory = False

    dpi = None
    _dpi = None

    class DpiArg(PositiveIntegerInvariant):
        _help = "Render DPI for page images. [default: %default]"
        _mandatory = False
        _default = 400

    force = None

    class ForceArg(BoolInvariant):
        _help = "Re-render existing page images/text. [default: %default]"
        _mandatory = False
        _default = False

    no_text = None

    class NoTextArg(BoolInvariant):
        _help = "Skip per-page text extraction. [default: %default]"
        _mandatory = False
        _default = False

    def run(self):
        source_dir = _option_value(
            self,
            "source_dir",
            private_first=True,
        )
        dest_dir = os.path.abspath(
            _option_value(self, "dest_dir", source_dir)
        )
        _run_pdf_to_pages(
            source_dir=source_dir,
            dest_dir=dest_dir,
            recursive=bool(self.recursive),
            parallelism=self._parallelism,
            limit=self._limit,
            dpi=self._dpi or 400,
            force=bool(self.force),
            no_text=bool(self.no_text),
            record_encrypted_sentinels=True,
            out=self._out,
        )


class PdfsToPages(PdfToPages):
    """Compatibility alias for `sne pdf-to-pages`.

    See `sne pdf-to-pages --help` for Poppler requirements and workflow.
    """


class OcrPdfPages(InvariantAwareCommand):
    """Run structured local OCR, optionally rendering source PDFs first.

    Recommended: select `--pipeline local-mineru25` to run registered
    DeepSeek grounded OCR plus MinerU structured Markdown with an exact GPU
    execution profile. The pipeline implies `--execution-profile auto`; use
    `--memory-tier 25`, `50`, or `max` to select a qualified tier, or omit it
    to use the profile's highest supported tier. Inspect setup with
    `sne ocr-backend-list`, registrations with
    `sne ocr-installation-list`, and profiles with `sne ocr-profile-list`.

    Pass `--pdf-to-pages` to render PDFs from `--source-dir` before OCR. The
    effective destination defaults to `--source-dir` when `--dest-dir` is
    omitted; otherwise render separately with `sne pdf-to-pages`.

    Legacy manual mode is selected only when `--pipeline` and
    `--execution-profile` are omitted. It exposes the DeepSeek/vLLM model and
    engine flags for controlled experiments; use
    `sne deepseek-ocr-create-env` and `sne deepseek-ocr-download-model` for
    that lower-level path. HTML is also a legacy single-output mode.
    """

    _verbose_ = True

    source_dir = None
    _source_dir = None

    class SourceDirArg(DirectoryInvariant):
        _help = (
            "Directory containing page PNG images, or PDFs with "
            "--pdf-to-pages."
        )

    dest_dir = None
    _dest_dir = None

    class DestDirArg(StringInvariant):
        _help = (
            "Destination directory for OCR output. "
            "Defaults to <source-dir>, including with --pdf-to-pages."
        )
        _mandatory = False

    text_source_dir = None
    _text_source_dir = None

    class TextSourceDirArg(DirectoryInvariant):
        _help = (
            "Directory containing per-page extracted text. Defaults to "
            "<source-dir> if present."
        )
        _mandatory = False

    pdf_to_pages = None

    class PdfToPagesArg(BoolInvariant):
        _arg = "--pdf-to-pages"
        _help = (
            "Render PDFs from --source-dir into the effective destination "
            "before OCR; it defaults to --source-dir. OCR then reads page "
            "images and extracted text from that destination. "
            "[default: %default]"
        )
        _mandatory = False
        _default = False

    pdf_recursive = None

    class PdfRecursiveArg(BoolInvariant):
        _arg = "--pdf-recursive"
        _help = (
            "Recursively find PDFs when --pdf-to-pages is enabled. "
            "[default: %default]"
        )
        _mandatory = False
        _default = False

    pdf_parallelism = None
    _pdf_parallelism = None

    class PdfParallelismArg(NonNegativeIntegerInvariant):
        _arg = "--pdf-parallelism"
        _help = (
            "PDF-render worker count; 0 means CPU count. [default: %default]"
        )
        _mandatory = False
        _default = 0

    pdf_dpi = None
    _pdf_dpi = None

    class PdfDpiArg(PositiveIntegerInvariant):
        _arg = "--pdf-dpi"
        _help = "PDF render DPI. [default: %default]"
        _mandatory = False
        _default = 400

    pipeline = None
    _pipeline = None

    class PipelineArg(StringInvariant):
        _arg = "--pipeline"
        _help = (
            "Structured local OCR pipeline; currently local-mineru25. "
            "Implies --execution-profile auto."
        )
        _mandatory = False

    execution_profile = None
    _execution_profile = None

    class ExecutionProfileArg(StringInvariant):
        _arg = "--execution-profile"
        _help = "Portable execution profile: auto, exact ID, or JSON path."
        _mandatory = False

    profile_dir = None
    _profile_dir = None

    class ProfileDirArg(StringInvariant):
        _arg = "--profile-dir"
        _help = "Comma-separated absolute profile JSON files or directories."
        _mandatory = False

    installation_registry = None
    _installation_registry = None

    class InstallationRegistryArg(StringInvariant):
        _arg = "--installation-registry"
        _help = "Machine-local OCR installation registry path."
        _mandatory = False

    memory_tier = None
    _memory_tier = None

    class MemoryTierArg(StringInvariant):
        _arg = "--memory-tier"
        _help = (
            "Structured profile memory tier: 25, 50, or max. "
            "Defaults to the highest supported tier."
        )
        _mandatory = False

    allow_provisional_profile = None

    class AllowProvisionalProfileArg(BoolInvariant):
        _arg = "--allow-provisional-profile"
        _help = (
            "Allow a provisional profile for an explicitly approved trial. "
            "[default: %default]"
        )
        _mandatory = False
        _default = False

    explain_profile = None

    class ExplainProfileArg(BoolInvariant):
        _arg = "--explain-profile"
        _help = (
            "Resolve and print structured execution without OCR or writes. "
            "[default: %default]"
        )
        _mandatory = False
        _default = False

    output_format = None
    _output_format = None

    class OutputFormatArg(StringInvariant):
        _help = "Output format: md, html, or ocr. [default: %default]"
        _mandatory = False
        _default = "md"

    model = None
    _model = None

    class ModelArg(StringInvariant):
        _help = (
            "Legacy manual mode only: DeepSeek OCR model name or local "
            "path; structured pipelines pin the model. [default: %default]"
        )
        _mandatory = False
        _default = DEFAULT_OCR_MODEL

    model_revision = None
    _model_revision = None

    class ModelRevisionArg(StringInvariant):
        _help = (
            "Legacy manual mode only: immutable Hugging Face model revision; "
            "structured pipelines pin the revision. [default: %default]"
        )
        _mandatory = False
        _default = DEFAULT_OCR_MODEL_REVISION

    prompt = None
    _prompt = None

    class PromptArg(StringInvariant):
        _help = "Override the OCR prompt. `<image>` is added if omitted."
        _mandatory = False

    prompt_mode = None
    _prompt_mode = None

    class PromptModeArg(StringInvariant):
        _help = (
            "Prompt mode: plain_ocr, markdown, html, or full. "
            "[default: %default]"
        )
        _mandatory = False
        _default = "markdown"

    keep_grounding = None

    class KeepGroundingArg(BoolInvariant):
        _help = "Keep DeepSeek grounding tags in output. [default: %default]"
        _mandatory = False
        _default = False

    no_grounding = None

    class NoGroundingArg(BoolInvariant):
        _help = "Do not request grounding tags. [default: %default]"
        _mandatory = False
        _default = False

    limit = None
    _limit = None

    class LimitArg(PositiveIntegerInvariant):
        _help = "Limit the number of pages to OCR."
        _mandatory = False

    offset = None
    _offset = None

    class OffsetArg(NonNegativeIntegerInvariant):
        _help = "Skip the first N page images before --limit."
        _mandatory = False
        _default = 0

    scan_parallelism = None
    _scan_parallelism = None

    class ScanParallelismArg(NonNegativeIntegerInvariant):
        _help = (
            "Workers for startup page discovery; 0 means auto. "
            "[default: %default]"
        )
        _mandatory = False
        _default = 0

    batch_size = None
    _batch_size = None

    class BatchSizeArg(PositiveIntegerInvariant):
        _help = "Client-side OCR batch size."
        _mandatory = False

    enable_chunked_prefill = None

    class EnableChunkedPrefillArg(BoolInvariant):
        _help = "Enable vLLM chunked prefill. [default: %default]"
        _mandatory = False
        _default = True

    max_model_len = None
    _max_model_len = None

    class MaxModelLenArg(PositiveIntegerInvariant):
        _help = "vLLM max_model_len."
        _mandatory = False

    max_num_batched_tokens = None
    _max_num_batched_tokens = None

    class MaxNumBatchedTokensArg(PositiveIntegerInvariant):
        _help = "vLLM max_num_batched_tokens."
        _mandatory = False

    max_num_seqs = None
    _max_num_seqs = None

    class MaxNumSeqsArg(PositiveIntegerInvariant):
        _help = "vLLM max_num_seqs."
        _mandatory = False

    attention_backend = None
    _attention_backend = None

    class AttentionBackendArg(StringInvariant):
        _help = "vLLM attention_backend override."
        _mandatory = False

    mm_encoder_attn_backend = None
    _mm_encoder_attn_backend = None

    class MmEncoderAttnBackendArg(StringInvariant):
        _help = "vLLM multimodal encoder attention backend override."
        _mandatory = False

    moe_backend = None
    _moe_backend = None

    class MoeBackendArg(StringInvariant):
        _help = (
            "vLLM MoE backend: auto or triton. [default: %default]"
        )
        _mandatory = False
        _default = DEFAULT_DEEPSEEK_MOE_BACKEND

    scheduler_delay_factor = None
    _scheduler_delay_factor = None

    class SchedulerDelayFactorArg(StringInvariant):
        _help = "vLLM scheduler_delay_factor."
        _mandatory = False

    tensor_parallel_size = None
    _tensor_parallel_size = None

    class TensorParallelSizeArg(PositiveIntegerInvariant):
        _help = "vLLM tensor_parallel_size."
        _mandatory = False

    gpu_memory_utilization = None
    _gpu_memory_utilization = None

    class GpuMemoryUtilizationArg(StringInvariant):
        _help = "vLLM gpu_memory_utilization."
        _mandatory = False

    temperature = None
    _temperature = None

    class TemperatureArg(StringInvariant):
        _help = "Sampling temperature. [default: %default]"
        _mandatory = False
        _default = "0.0"

    max_tokens = None
    _max_tokens = None

    class MaxTokensArg(PositiveIntegerInvariant):
        _help = "Max output tokens per page. [default: %default]"
        _mandatory = False
        _default = 8192

    ngram_size = None
    _ngram_size = None

    class NgramSizeArg(PositiveIntegerInvariant):
        _help = "DeepSeek OCR n-gram size. [default: %default]"
        _mandatory = False
        _default = 10

    window_size = None
    _window_size = None

    class WindowSizeArg(PositiveIntegerInvariant):
        _help = (
            "DeepSeek OCR window size. Ungrounded Markdown uses at least "
            f"{DEFAULT_MARKDOWN_WINDOW_SIZE}. [default: %default]"
        )
        _mandatory = False
        _default = DEFAULT_OCR_WINDOW_SIZE

    force = None

    class ForceArg(BoolInvariant):
        _help = (
            "Re-run OCR when output files already exist; with "
            "--pdf-to-pages, also re-render page images/text. "
            "[default: %default]"
        )
        _mandatory = False
        _default = False

    markdown_provider_spec = None
    _markdown_provider_spec = None

    class MarkdownProviderSpecArg(StringInvariant):
        _help = (
            "Local Markdown provider specification for full mode. "
            "DeepSeek still produces grounded OCR."
        )
        _mandatory = False

    markdown_provider_lane = None
    _markdown_provider_lane = None

    class MarkdownProviderLaneArg(StringInvariant):
        _help = "Provider lane for full-mode Markdown. [default: best]"
        _mandatory = False

    markdown_provider_run_dir = None
    _markdown_provider_run_dir = None

    class MarkdownProviderRunDirArg(StringInvariant):
        _help = (
            "New or empty absolute scratch directory for provider evidence."
        )
        _mandatory = False

    def run(self):
        from .ocr_vllm import (
            DeepSeekOcrOfflineProcessor,
            StructuredOutputReuseError,
            collect_ocr_full_items,
            collect_ocr_tasks,
            resolve_scan_workers,
        )

        source_dir = _option_value(
            self,
            "source_dir",
            private_first=True,
        )
        dest_option = _option_value(self, "dest_dir", private_first=True)
        dest_dir = os.path.abspath(dest_option or source_dir)
        explicit_text_source_dir = _option_value(
            self,
            "text_source_dir",
            private_first=True,
        )
        if (
            self.pdf_to_pages
            and explicit_text_source_dir is not None
            and os.path.realpath(explicit_text_source_dir)
            != os.path.realpath(dest_dir)
        ):
            raise CommandError(
                "--text-source-dir must equal the effective destination "
                "when --pdf-to-pages is enabled"
            )
        output_format = _option_value(self, "output_format", "md")
        output_format = output_format.lower().strip()
        if output_format not in {"md", "html", "ocr"}:
            raise CommandError("--output-format must be md, html, or ocr")
        prompt_mode = _option_value(self, "prompt_mode", "markdown")
        prompt_mode = prompt_mode.lower().strip()
        if prompt_mode not in {"plain_ocr", "markdown", "html", "full"}:
            raise CommandError(
                "--prompt-mode must be plain_ocr, markdown, html, or full"
            )
        if prompt_mode == "full" and output_format != "md":
            raise CommandError(
                "--prompt-mode full writes .ocr, .md, and .json and "
                "requires --output-format md"
            )
        structured = self._resolve_structured_execution(
            prompt_mode=prompt_mode,
            output_format=output_format,
        )
        if structured is not None:
            prompt_mode = "full"
            output_format = "md"
            if self.explain_profile:
                from .ocr_backend_setup import json_bytes

                self._out(
                    json_bytes(structured["receipt"], pretty=True)
                    .decode("utf-8")
                    .rstrip()
                )
                return
        scheduler_delay_factor = _maybe_float(
            _option_value(self, "scheduler_delay_factor")
        )
        gpu_memory_utilization = _maybe_float(
            _option_value(self, "gpu_memory_utilization")
        )
        temperature = _maybe_float(
            _option_value(self, "temperature"),
            default=0.0,
        )
        page_source_dir = dest_dir if self.pdf_to_pages else source_dir
        if structured is not None:
            markdown_provider = self._prepare_profile_markdown_provider(
                structured,
                source_dir=page_source_dir,
                dest_dir=dest_dir,
            )
        else:
            markdown_provider = self._prepare_markdown_provider(
                prompt_mode=prompt_mode,
                source_dir=page_source_dir,
                dest_dir=dest_dir,
            )
        os.makedirs(dest_dir, exist_ok=True)
        pdf_pages_result = None
        if self.pdf_to_pages:
            try:
                pdf_pages_result = _run_pdf_to_pages(
                    source_dir=source_dir,
                    dest_dir=dest_dir,
                    recursive=bool(self.pdf_recursive),
                    parallelism=self._pdf_parallelism,
                    limit=None,
                    dpi=self._pdf_dpi or 400,
                    force=bool(self.force),
                    no_text=False,
                    record_encrypted_sentinels=False,
                    out=self._out,
                )
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                raise CommandError(
                    f"PDF rendering failed before OCR: {exc}"
                ) from exc
        output_ext = "ocr" if output_format == "ocr" else output_format
        text_source_dir = explicit_text_source_dir
        if text_source_dir is None and os.path.isdir(
            f"{page_source_dir}.txt"
        ):
            text_source_dir = f"{page_source_dir}.txt"
        elif text_source_dir is None:
            text_source_dir = page_source_dir

        scan_workers = resolve_scan_workers(self._scan_parallelism)
        self._out(f"Scanning page images under {page_source_dir}...")
        items = []
        if prompt_mode == "full":
            try:
                items, pdf_count = collect_ocr_full_items(
                    source_dir=page_source_dir,
                    dest_dir=dest_dir,
                    text_source_dir=text_source_dir,
                    allowed_image_paths=(
                        pdf_pages_result.image_paths
                        if pdf_pages_result is not None
                        else None
                    ),
                    refresh_image_paths=(
                        pdf_pages_result.rendered_image_paths
                        if pdf_pages_result is not None
                        else None
                    ),
                    refresh_metadata_image_paths=(
                        pdf_pages_result.rendered_text_image_paths
                        if pdf_pages_result is not None
                        else None
                    ),
                    verify_input_freshness=bool(self.pdf_to_pages),
                    force=bool(self.force),
                    scan_parallelism=scan_workers,
                    out=self._out,
                    expected_execution_receipt=(
                        structured["receipt"]
                        if structured is not None
                        else None
                    ),
                )
            except StructuredOutputReuseError as exc:
                raise CommandError(str(exc)) from exc
            if self._offset:
                items = items[self._offset :]
            if self._limit:
                items = items[: self._limit]
            if not items:
                self._out("No page images need full OCR.")
                return
            work_count = len(items)
        else:
            tasks, pdf_count = collect_ocr_tasks(
                source_dir=page_source_dir,
                dest_dir=dest_dir,
                text_source_dir=text_source_dir,
                allowed_image_paths=(
                    pdf_pages_result.image_paths
                    if pdf_pages_result is not None
                    else None
                ),
                refresh_image_paths=(
                    pdf_pages_result.rendered_image_paths
                    if pdf_pages_result is not None
                    else None
                ),
                verify_input_freshness=bool(self.pdf_to_pages),
                output_ext=output_ext,
                force=bool(self.force),
                scan_parallelism=scan_workers,
                out=self._out,
            )
            if self._offset:
                tasks = tasks[self._offset :]
            if self._limit:
                tasks = tasks[: self._limit]
            if not tasks:
                self._out("No page images found to OCR.")
                return
            work_count = len(tasks)

        self._out(f"Found {work_count} page(s) across {pdf_count} PDF(s).")
        self._out(f"Writing OCR output to {dest_dir}.")
        structured_environment = None
        try:
            grounded_settings = (
                structured["tier"]["stages"]["grounded_ocr"]["settings"]
                if structured is not None
                else {}
            )
            window_size = grounded_settings.get(
                "window_size",
                self._window_size or DEFAULT_OCR_WINDOW_SIZE,
            )
            if prompt_mode == "markdown":
                window_size = max(
                    window_size,
                    DEFAULT_MARKDOWN_WINDOW_SIZE,
                )
            moe_backend = validate_deepseek_moe_backend(
                grounded_settings.get(
                    "moe_backend",
                    _option_value(
                        self, "moe_backend", DEFAULT_DEEPSEEK_MOE_BACKEND
                    ),
                ).strip().lower()
            )
            if structured is not None:
                deepseek_installation = structured["installations"][
                    "deepseek-vllm"
                ]
                structured_environment = _set_process_environment(
                    {
                        "HF_HUB_CACHE": deepseek_installation["hf_cache"][
                            "path"
                        ],
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "HF_HUB_DISABLE_TELEMETRY": "1",
                    }
                )
                processor_model = deepseek_installation["snapshot"]["path"]
                processor_revision = None
            else:
                processor_model = _option_value(
                    self, "model", DEFAULT_OCR_MODEL
                )
                processor_revision = _option_value(
                    self,
                    "model_revision",
                    DEFAULT_OCR_MODEL_REVISION,
                )
            processor = DeepSeekOcrOfflineProcessor(
                model=processor_model,
                model_revision=processor_revision,
                output_format=output_format,
                prompt=_option_value(self, "prompt"),
                prompt_mode=(
                    "plain_ocr" if prompt_mode == "full" else prompt_mode
                ),
                keep_grounding=bool(self.keep_grounding),
                use_grounding=False if self.no_grounding else None,
                batch_size=grounded_settings.get(
                    "batch_size", self._batch_size
                ),
                enable_chunked_prefill=grounded_settings.get(
                    "enable_chunked_prefill",
                    bool(self.enable_chunked_prefill),
                ),
                max_model_len=grounded_settings.get(
                    "max_model_len", self._max_model_len
                ),
                max_num_batched_tokens=grounded_settings.get(
                    "max_num_batched_tokens",
                    self._max_num_batched_tokens,
                ),
                max_num_seqs=grounded_settings.get(
                    "max_num_seqs", self._max_num_seqs
                ),
                attention_backend=_profile_backend_value(
                    grounded_settings.get(
                        "attention_backend",
                        _option_value(self, "attention_backend"),
                    )
                ),
                mm_encoder_attn_backend=_profile_backend_value(
                    grounded_settings.get(
                        "mm_encoder_attn_backend",
                        _option_value(self, "mm_encoder_attn_backend"),
                    )
                ),
                moe_backend=moe_backend,
                scheduler_delay_factor=grounded_settings.get(
                    "scheduler_delay_factor", scheduler_delay_factor
                ),
                tensor_parallel_size=self._tensor_parallel_size,
                gpu_memory_utilization=grounded_settings.get(
                    "gpu_memory_utilization", gpu_memory_utilization
                ),
                temperature=grounded_settings.get("temperature", temperature),
                max_tokens=grounded_settings.get(
                    "max_tokens", self._max_tokens or 8192
                ),
                ngram_size=grounded_settings.get(
                    "ngram_size", self._ngram_size or 10
                ),
                window_size=window_size,
                out=self._out,
                verbose=self._verbose,
            )
            if prompt_mode == "full":
                self._run_full_mode(
                    processor,
                    items,
                    markdown_provider=markdown_provider,
                    execution_plan=structured,
                )
            else:
                processor.run(tasks)
        except (RuntimeError, ValueError) as exc:
            raise CommandError(str(exc)) from exc
        finally:
            if structured_environment is not None:
                _restore_process_environment(structured_environment)

    def _resolve_structured_execution(
        self,
        *,
        prompt_mode: str,
        output_format: str,
    ) -> dict | None:
        pipeline_name = _option_value(self, "pipeline")
        profile_selection = _option_value(self, "execution_profile")
        structured = bool(
            pipeline_name
            or profile_selection
            or _option_value(self, "memory_tier")
            or self.allow_provisional_profile
            or self.explain_profile
        )
        if not structured:
            return None
        if prompt_mode not in {"markdown", "full"} or output_format != "md":
            raise CommandError(
                "structured OCR owns full-mode prompts and requires Markdown "
                "output"
            )
        conflicts = self._structured_option_conflicts()
        if conflicts:
            raise CommandError(
                "structured OCR settings come from the execution profile; "
                "remove: " + ", ".join(conflicts)
            )
        ambient_vllm_controls = sorted(
            name for name in os.environ if name.startswith("VLLM_")
        )
        if ambient_vllm_controls:
            raise CommandError(
                "structured OCR rejects unmodeled ambient vLLM controls; "
                "unset: " + ", ".join(ambient_vllm_controls)
            )
        from .ocr_backend_setup import make_receipt
        from .ocr_execution import (
            OcrExecutionError,
            assert_in_process_deepseek_installation,
            build_execution_observation,
            profile_search_paths,
            resolve_execution_profile,
        )
        from .ocr_hardware import HardwareProbeError, probe_nvidia_hardware
        from .ocr_installations import (
            InstallationRegistryError,
            load_installation_registry,
            validate_single_gpu_cuda_binding,
        )
        from .ocr_profiles import OcrProfileError, load_execution_profile

        raw_memory_tier = _option_value(self, "memory_tier")
        memory_tier = (
            raw_memory_tier.lower() if raw_memory_tier is not None else None
        )
        if memory_tier is not None and memory_tier not in {"25", "50", "max"}:
            raise CommandError("--memory-tier must be 25, 50, or max")
        try:
            if profile_selection and profile_selection != "auto":
                candidate = Path(profile_selection).expanduser()
                if (
                    candidate.is_absolute()
                    or profile_selection.startswith((".", "~"))
                    or "/" in profile_selection
                    or profile_selection.endswith(".json")
                ):
                    inferred = load_execution_profile(candidate)[
                        "pipeline_id"
                    ]
                    pipeline_name = pipeline_name or inferred
            if pipeline_name is None:
                raise OcrExecutionError(
                    "--execution-profile by ID requires --pipeline"
                )
            from .ocr_profiles import get_pipeline_definition

            pipeline = get_pipeline_definition(pipeline_name)
            if pipeline["pipeline_id"] != "local-mineru25":
                raise OcrExecutionError(
                    "structured production currently supports only "
                    "local-mineru25; DeepSeek Markdown tuning is withheld"
                )
            profile_selection = profile_selection or "auto"
            registry = load_installation_registry(
                _option_value(self, "installation_registry"),
                validate_current=True,
                missing_ok=False,
            )
            installations = registry["installations"]
            deepseek = installations.get("deepseek-vllm")
            mineru = installations.get("mineru25-pro-2605")
            if not isinstance(deepseek, dict) or not isinstance(mineru, dict):
                raise OcrExecutionError(
                    "local-mineru25 requires registered DeepSeek and MinerU "
                    "installations"
                )
            visible, _ = validate_single_gpu_cuda_binding(
                deepseek.get("cuda"),
                label="deepseek-vllm.cuda",
            )
            mineru_visible, _ = validate_single_gpu_cuda_binding(
                mineru.get("cuda"),
                label="mineru25-pro-2605.cuda",
            )
            if mineru_visible != visible:
                raise OcrExecutionError(
                    "pipeline installations select different physical GPUs"
                )
            assert_in_process_deepseek_installation(deepseek)
            hardware = probe_nvidia_hardware(
                torch_python=deepseek["python_executable"]["path"],
                visible_devices=visible,
                cuda_device_order="PCI_BUS_ID",
            )
            observation = build_execution_observation(
                pipeline_id=pipeline["pipeline_id"],
                hardware=hardware,
                installations=installations,
            )
            resolved = resolve_execution_profile(
                profile_selection,
                pipeline_id=pipeline["pipeline_id"],
                memory_tier=memory_tier,
                observation=observation,
                search_paths=profile_search_paths(
                    _option_value(self, "profile_dir")
                ),
                allow_provisional=bool(self.allow_provisional_profile),
            )
            memory_tier = resolved["memory_tier"]
            profile = resolved["profile"]
            tier = profile["memory_tiers"][memory_tier]
            grounded_settings = tier["stages"]["grounded_ocr"]["settings"]
            torch_total = (
                hardware["memory"].get("total_sources", {}).get("torch_cuda")
            )
            if not isinstance(torch_total, int) or torch_total <= 0:
                torch_total = profile["hardware_selector"][
                    "total_memory_bytes"
                ]
            expected_denominator = max(
                profile["hardware_selector"]["total_memory_bytes"],
                torch_total,
            )
            if (
                grounded_settings.get(
                    "gpu_memory_utilization_denominator_bytes"
                )
                != expected_denominator
            ):
                raise OcrExecutionError(
                    "selected profile GPU-memory denominator differs from "
                    "the live CUDA capacity"
                )
            free_bytes = hardware["memory"].get("free_memory_bytes")
            reserve_bytes = tier["reserve_bytes"]
            required_bytes = tier["budget_bytes"] + reserve_bytes
            if not isinstance(free_bytes, int) or free_bytes < required_bytes:
                raise OcrExecutionError(
                    "selected profile tier is not currently available: "
                    f"needs budget+reserve {required_bytes} bytes, observed "
                    f"{free_bytes!r} bytes free"
                )
            receipt = make_receipt(
                "ocr_structured_execution",
                {
                    "pipeline_id": pipeline["pipeline_id"],
                    "pipeline_definition_sha256": pipeline[
                        "definition_sha256"
                    ],
                    "profile_id": profile["profile_id"],
                    "profile_sha256": profile["profile_sha256"],
                    "qualification_state": profile["qualification"]["state"],
                    "allow_provisional_profile": bool(
                        self.allow_provisional_profile
                    ),
                    "memory_tier": memory_tier,
                    "memory_budget_bytes": tier["budget_bytes"],
                    "reserve_bytes": reserve_bytes,
                    "hardware_selector": profile["hardware_selector"],
                    "match": resolved["match"],
                    "stage_settings": {
                        stage_id: stage_record["settings"]
                        for stage_id, stage_record in tier["stages"].items()
                    },
                    "attention_backend": {
                        "requested": grounded_settings.get(
                            "attention_backend"
                        ),
                        "qualified_selected": profile["stages"][
                            "grounded_ocr"
                        ]["runtime_selector"]["attention_backend"],
                        "selected_verification": (
                            "deferred_until_engine_initialization"
                        ),
                    },
                    "installation_record_sha256": {
                        key: installations[key]["record_sha256"]
                        for key in ("deepseek-vllm", "mineru25-pro-2605")
                    },
                    "deepseek_snapshot": {
                        "revision": deepseek["model"]["revision"],
                        "files_sha256": deepseek["snapshot"]["files_sha256"],
                    },
                },
            )
        except (
            OcrExecutionError,
            OcrProfileError,
            HardwareProbeError,
            InstallationRegistryError,
            OSError,
        ) as exc:
            raise CommandError(str(exc)) from exc
        return {
            **resolved,
            "pipeline": pipeline,
            "tier": tier,
            "memory_tier": memory_tier,
            "installations": installations,
            "receipt": receipt,
        }

    def _structured_option_conflicts(self) -> list[str]:
        conflicts = []
        exact_defaults = (
            ("model", DEFAULT_OCR_MODEL, "--model"),
            (
                "model_revision",
                DEFAULT_OCR_MODEL_REVISION,
                "--model-revision",
            ),
            ("temperature", "0.0", "--temperature"),
            ("max_tokens", 8192, "--max-tokens"),
            ("ngram_size", 10, "--ngram-size"),
            ("window_size", DEFAULT_OCR_WINDOW_SIZE, "--window-size"),
            (
                "moe_backend",
                DEFAULT_DEEPSEEK_MOE_BACKEND,
                "--moe-backend",
            ),
        )
        for name, expected, option in exact_defaults:
            value = _option_value(self, name, expected)
            if value != expected:
                conflicts.append(option)
        optional = (
            ("prompt", "--prompt"),
            ("batch_size", "--batch-size"),
            ("max_model_len", "--max-model-len"),
            ("max_num_batched_tokens", "--max-num-batched-tokens"),
            ("max_num_seqs", "--max-num-seqs"),
            ("attention_backend", "--attention-backend"),
            ("mm_encoder_attn_backend", "--mm-encoder-attn-backend"),
            ("scheduler_delay_factor", "--scheduler-delay-factor"),
            ("tensor_parallel_size", "--tensor-parallel-size"),
            ("gpu_memory_utilization", "--gpu-memory-utilization"),
            ("markdown_provider_spec", "--markdown-provider-spec"),
            ("markdown_provider_lane", "--markdown-provider-lane"),
        )
        for name, option in optional:
            if _option_value(self, name) not in (None, ""):
                conflicts.append(option)
        if self.keep_grounding:
            conflicts.append("--keep-grounding")
        if self.no_grounding:
            conflicts.append("--no-grounding")
        return conflicts

    def _prepare_profile_markdown_provider(
        self,
        execution: dict,
        *,
        source_dir: str,
        dest_dir: str,
    ) -> dict:
        from .ocr_backend_setup import _atomic_write_new_bytes, json_bytes
        from .ocr_execution import build_profile_provider_spec
        from .ocr_profiles import write_execution_profile

        requested_run_dir = _option_value(self, "markdown_provider_run_dir")
        if requested_run_dir:
            run_dir_path = _preflight_markdown_provider_run_dir(
                requested_run_dir,
                source_dir=source_dir,
                dest_dir=dest_dir,
            )
            parent = run_dir_path.parent
            run_dir = str(run_dir_path)
        else:
            parent = Path(tempfile.gettempdir()).resolve()
            run_dir = None
        runtime_root = _create_profile_runtime_root(
            parent,
            source_dir=source_dir,
            dest_dir=dest_dir,
        )
        if run_dir is None:
            run_dir_path = _preflight_markdown_provider_run_dir(
                runtime_root / "provider-evidence",
                source_dir=source_dir,
                dest_dir=dest_dir,
            )
            run_dir = str(run_dir_path)
            self._out(
                f"Profile provider evidence will be preserved at {run_dir}."
            )
        profile_path = runtime_root / "execution-profile.json"
        write_execution_profile(execution["profile"], profile_path)
        spec = build_profile_provider_spec(
            profile=execution["profile"],
            profile_path=profile_path,
            memory_tier=execution["memory_tier"],
            installation=execution["installations"]["mineru25-pro-2605"],
            allow_provisional=bool(self.allow_provisional_profile),
        )
        spec_path = runtime_root / "provider-spec.json"
        _atomic_write_new_bytes(spec_path, json_bytes(spec, pretty=True))
        provider = self._prepare_markdown_provider(
            prompt_mode="full",
            source_dir=source_dir,
            dest_dir=dest_dir,
            spec_path_override=str(spec_path),
            run_dir_override=run_dir,
            lane_override="best",
        )
        provider["execution_profile"] = execution
        provider["runtime_root"] = str(runtime_root)
        return provider

    def _prepare_markdown_provider(
        self,
        *,
        prompt_mode: str,
        source_dir: str,
        dest_dir: str,
        spec_path_override: str | None = None,
        run_dir_override: str | None = None,
        lane_override: str | None = None,
    ) -> dict | None:
        spec_path = spec_path_override or _option_value(
            self, "markdown_provider_spec"
        )
        run_dir = run_dir_override or _option_value(
            self, "markdown_provider_run_dir"
        )
        if not spec_path:
            if run_dir:
                raise CommandError(
                    "--markdown-provider-run-dir requires "
                    "--markdown-provider-spec"
                )
            return None
        if prompt_mode != "full":
            raise CommandError(
                "--markdown-provider-spec is valid only with "
                "--prompt-mode full"
            )
        if not run_dir:
            raise CommandError(
                "--markdown-provider-spec requires "
                "--markdown-provider-run-dir"
            )
        from .ocr_provider import (
            ProviderError,
            load_local_provider_spec,
            validate_provider_run_dir,
        )

        lane = lane_override or _option_value(
            self, "markdown_provider_lane", "best"
        )
        candidate_dir = _preflight_markdown_provider_run_dir(
            run_dir,
            source_dir=source_dir,
            dest_dir=dest_dir,
        )
        try:
            resolved_spec_path = (
                Path(spec_path).expanduser().resolve(strict=True)
            )
            spec = load_local_provider_spec(resolved_spec_path)
            if lane not in spec["lanes"]:
                raise ProviderError(f"provider does not define lane: {lane}")
            evidence_dir = validate_provider_run_dir(candidate_dir)
        except (OSError, ProviderError) as exc:
            raise CommandError(str(exc)) from exc
        return {
            "spec_path": str(resolved_spec_path),
            "spec": spec,
            "lane": lane,
            "run_dir": str(evidence_dir),
        }

    def _run_full_mode(
        self,
        processor,
        items,
        *,
        markdown_provider: dict | None = None,
        execution_plan: dict | None = None,
    ) -> None:
        from .ocr_vllm import (
            OcrPageTask,
            sampling_params_with_minimum_window,
            write_ocr_page_bundle,
        )

        ocr_tasks = [
            OcrPageTask(
                image_path=item.image_path,
                output_path=item.ocr_path,
                text_source_path=item.text_path,
            )
            for item in items
            if item.need_ocr
        ]
        markdown_tasks = [
            OcrPageTask(
                image_path=item.image_path,
                output_path=item.markdown_path,
                text_source_path=item.text_path,
            )
            for item in items
            if item.need_markdown
        ]
        total_actions = len(ocr_tasks) + len(markdown_tasks) + len(items)
        llm = None
        sampling_params = None
        try:
            processor.init_progressbar(total=total_actions)
            if ocr_tasks or (markdown_tasks and markdown_provider is None):
                llm, sampling_params = processor.prepare_engine()
                if execution_plan is not None:
                    from .ocr_execution import (
                        deepseek_engine_identity_mismatches,
                    )

                    deepseek_installation = execution_plan["installations"][
                        "deepseek-vllm"
                    ]
                    execution_settings = execution_plan["tier"]["stages"][
                        "grounded_ocr"
                    ]["settings"]
                    mismatches = deepseek_engine_identity_mismatches(
                        execution_plan["profile"],
                        installation=deepseek_installation,
                        engine_metadata=getattr(
                            processor,
                            "_engine_metadata",
                            {},
                        ),
                        execution_settings=execution_settings,
                    )
                    if mismatches:
                        raise RuntimeError(
                            "DeepSeek engine identity differs from the "
                            "selected execution profile: "
                            + "; ".join(mismatches)
                        )
            if ocr_tasks:
                processor.output_format = "ocr"
                processor.prompt_mode = "plain_ocr"
                processor.use_grounding = True
                processor.keep_grounding = True
                grounded_candidates = {}
                grounded_run_tasks = ocr_tasks
                if execution_plan is not None:
                    (
                        grounded_run_tasks,
                        grounded_candidates,
                    ) = _stage_grounded_ocr_tasks(ocr_tasks)
                try:
                    processor.run(
                        grounded_run_tasks,
                        llm=llm,
                        sampling_params=sampling_params,
                    )
                except BaseException as exc:
                    _discard_grounded_ocr_candidates(
                        grounded_candidates,
                        original_error=exc,
                    )
                    raise
                if grounded_candidates:
                    _finalize_grounded_ocr_candidates(
                        processor,
                        grounded_candidates,
                    )
            provider_failures = []
            if markdown_tasks and markdown_provider is not None:
                if llm is not None:
                    processor.shutdown_engine(llm)
                    llm = None
                    sampling_params = None
                    _release_cuda_memory()
                provider_failures = self._run_markdown_provider(
                    processor,
                    items,
                    markdown_provider,
                )
            elif markdown_tasks:
                markdown_sampling_params = (
                    sampling_params_with_minimum_window(
                        sampling_params,
                        DEFAULT_MARKDOWN_WINDOW_SIZE,
                    )
                )
                processor.output_format = "md"
                processor.prompt_mode = "markdown"
                processor.use_grounding = False
                processor.keep_grounding = False
                processor.run(
                    markdown_tasks,
                    llm=llm,
                    sampling_params=markdown_sampling_params,
                )
            all_metadata = getattr(
                processor,
                "generation_metadata",
                {},
            )
            page_failures: dict[str, list[str]] = {}
            batch_failures = []
            provider_run_dir = (
                markdown_provider["run_dir"] if markdown_provider else None
            )
            for path, flags in provider_failures:
                if path == provider_run_dir:
                    batch_failures.extend(flags)
                    continue
                page_failures.setdefault(path, []).extend(flags)
            for item in items:
                generation_metadata = {
                    key: all_metadata[path]
                    for key, path in (
                        ("ocr", item.ocr_path),
                        ("markdown", item.markdown_path),
                    )
                    if path in all_metadata
                }
                if execution_plan is not None:
                    for metadata in generation_metadata.values():
                        metadata["execution_profile"] = execution_plan[
                            "receipt"
                        ]
                payload = write_ocr_page_bundle(
                    item,
                    generation_metadata=generation_metadata,
                )
                quality = payload.get("markdown_quality", {})
                flags = []
                for label, path, required in (
                    ("grounded_ocr", item.ocr_path, item.need_ocr),
                    ("markdown", item.markdown_path, item.need_markdown),
                ):
                    if not required:
                        continue
                    metadata = all_metadata.get(path)
                    validation = (
                        metadata.get("validation")
                        if isinstance(metadata, Mapping)
                        else None
                    )
                    if isinstance(validation, Mapping):
                        metadata_flags = validation.get("flags")
                        if isinstance(metadata_flags, list):
                            flags.extend(
                                value
                                for value in metadata_flags
                                if isinstance(value, str)
                            )
                    if not metadata or metadata.get("status") != "completed":
                        flags.append(f"{label}_generation_failed")
                    elif execution_plan is not None:
                        termination = metadata.get("termination")
                        if (
                            not isinstance(termination, Mapping)
                            or termination.get("eos") is not True
                        ):
                            flags.append(f"{label}_non_eos_termination")
                        if (
                            isinstance(termination, Mapping)
                            and termination.get("max_tokens") is True
                        ):
                            flags.append(f"{label}_max_tokens_termination")
                    if not os.path.isfile(path):
                        flags.append(f"{label}_output_missing")
                if item.need_markdown and quality.get("status") != "pass":
                    flags.extend(
                        quality.get("flags")
                        or ["markdown_quality_unavailable"]
                    )
                elif quality.get("status") == "fail":
                    flags.extend(quality.get("flags", []))
                if flags:
                    page_failures.setdefault(item.markdown_path, []).extend(
                        flags
                    )
                processor.update_progressbar(1)
            page_failures = {
                path: list(dict.fromkeys(flags))
                for path, flags in page_failures.items()
            }
            batch_failures = list(dict.fromkeys(batch_failures))
            if page_failures or batch_failures:
                details = []
                if page_failures:
                    details.append(
                        f"{len(page_failures)} page(s): "
                        + "; ".join(
                            f"{path} ({', '.join(flags)})"
                            for path, flags in page_failures.items()
                        )
                    )
                if batch_failures:
                    details.append(
                        "provider batch (" + ", ".join(batch_failures) + ")"
                    )
                raise RuntimeError(
                    "OCR page validation failed. Outputs and request "
                    "metadata were preserved for diagnosis; rerun with "
                    "--force after correcting the settings. "
                    + "; ".join(details)
                )
        finally:
            processor.close_progressbar()
            processor.shutdown_engine(llm)

    def _run_markdown_provider(
        self,
        processor,
        items,
        provider: dict,
    ) -> list[tuple[str, list[str]]]:
        from .ocr_provider import ProviderError, run_local_provider_batch
        from .ocr_vllm import (
            analyze_markdown_quality,
            file_sha256,
            read_optional_text,
        )

        lane_spec = provider["spec"]["lanes"][provider["lane"]]
        item_by_id = {}
        pages = []
        for item in items:
            if not item.need_markdown:
                continue
            identity = f"{item.base}\0{item.image_sha256}".encode()
            page_id = hashlib.sha256(identity).hexdigest()
            item_by_id[page_id] = item
            pages.append(
                {
                    "page_id": page_id,
                    "image_path": os.path.abspath(item.image_path),
                    "image_sha256": item.image_sha256,
                    "image_dims": item.image_dims,
                    "instruction": lane_spec["prompt"],
                    "settings": lane_spec.get("settings", {}),
                }
            )
        try:
            timeout_seconds = _structured_markdown_timeout_seconds(
                provider,
                len(pages),
            )
            batch = run_local_provider_batch(
                spec_path=provider["spec_path"],
                lane=provider["lane"],
                run_dir=provider["run_dir"],
                pages=pages,
                out=self._out,
                timeout_seconds=timeout_seconds,
            )
        except (OSError, ProviderError) as exc:
            message = f"provider launch failed: {exc}"
            for item in item_by_id.values():
                processor.generation_metadata[item.markdown_path] = {
                    "status": "error",
                    "backend": "local_provider",
                    "error": [message],
                    "evidence": {"run_dir": provider["run_dir"]},
                    "promotion": {
                        "promoted": False,
                        "candidate_sha256": None,
                        "persisted_output_sha256": file_sha256(
                            item.markdown_path
                        ),
                    },
                }
                processor.update_progressbar(1)
            return [
                (item.markdown_path, [message])
                for item in item_by_id.values()
            ]

        batch_protocol_flags = list(
            dict.fromkeys(
                "provider_batch_protocol_error: " + error
                for error in batch.get("protocol_errors", [])
            )
        )
        failures = []
        returned_ids = set()
        for result in batch["results"]:
            page_id = result.get("page_id")
            item = item_by_id.get(page_id)
            if item is None:
                continue
            returned_ids.add(page_id)
            metadata = dict(result.get("normalized_metadata") or {})
            output_text = result.get("output_text")
            persisted_before = file_sha256(item.markdown_path)
            flags = list(result.get("errors") or [])
            flags.extend(batch_protocol_flags)
            execution = provider.get("execution_profile")
            if execution is not None:
                from .ocr_execution import (
                    stage_model_mismatches,
                    stage_runtime_mismatches,
                )

                runtime = metadata.get("runtime")
                if not isinstance(runtime, dict):
                    flags.append("provider_runtime_missing")
                else:
                    flags.extend(
                        "provider_runtime_mismatch: " + reason
                        for reason in stage_runtime_mismatches(
                            execution["profile"],
                            stage_id="markdown",
                            runtime=runtime,
                        )
                    )
                model = metadata.get("model")
                if not isinstance(model, dict):
                    flags.append("provider_model_missing")
                else:
                    flags.extend(
                        "provider_model_mismatch: " + reason
                        for reason in stage_model_mismatches(
                            execution["profile"],
                            stage_id="markdown",
                            model=model,
                        )
                    )
                observed_profile = (
                    runtime.get("execution_profile")
                    if isinstance(runtime, dict)
                    else None
                )
                expected_profile = execution["receipt"]
                if not isinstance(observed_profile, dict) or any(
                    observed_profile.get(key) != expected_profile[key]
                    for key in (
                        "profile_id",
                        "profile_sha256",
                        "qualification_state",
                        "memory_tier",
                    )
                ):
                    flags.append("provider_execution_profile_mismatch")
            if not result.get("valid"):
                flags.append("provider_response_invalid")
            if not isinstance(output_text, str) or not output_text.strip():
                flags.append("provider_markdown_empty")
                output_text = (
                    output_text if isinstance(output_text, str) else ""
                )
            termination = metadata.get("termination")
            if (
                not isinstance(termination, Mapping)
                or termination.get("eos") is not True
            ):
                flags.append("provider_non_eos_termination")
            if (
                isinstance(termination, Mapping)
                and termination.get("max_tokens") is True
            ):
                flags.append("provider_max_tokens_termination")
            quality = analyze_markdown_quality(
                output_text,
                ocr_text=read_optional_text(item.ocr_path),
                raw_text=read_optional_text(item.text_path),
            )
            if quality.get("status") != "pass":
                flags.extend(
                    quality.get("flags")
                    or ["provider_markdown_quality_unavailable"]
                )
            flags = list(dict.fromkeys(flags))
            candidate_sha256 = hashlib.sha256(
                output_text.encode("utf-8")
            ).hexdigest()
            promoted = False
            if not flags:
                try:
                    _atomic_write_text(item.markdown_path, output_text)
                    promoted = True
                except OSError as exc:
                    flags.append(
                        "provider_markdown_promotion_failed: " + str(exc)
                    )
            metadata["validation"] = {
                "status": "fail" if flags else "pass",
                "flags": flags,
                "markdown_quality": quality,
            }
            metadata["promotion"] = {
                "promoted": promoted,
                "candidate_sha256": candidate_sha256,
                "persisted_output_sha256_before": persisted_before,
                "persisted_output_sha256": file_sha256(item.markdown_path),
            }
            if flags:
                metadata["status"] = "error"
                failures.append((item.markdown_path, flags))
            processor.generation_metadata[item.markdown_path] = metadata
            processor.update_progressbar(1)

        for page_id, item in item_by_id.items():
            if page_id in returned_ids:
                continue
            flags = ["provider_result_missing", *batch_protocol_flags]
            processor.generation_metadata[item.markdown_path] = {
                "status": "error",
                "backend": "local_provider",
                "error": flags,
                "evidence": {"run_dir": provider["run_dir"]},
                "promotion": {
                    "promoted": False,
                    "candidate_sha256": None,
                    "persisted_output_sha256": file_sha256(
                        item.markdown_path
                    ),
                },
            }
            failures.append((item.markdown_path, flags))
            processor.update_progressbar(1)
        if batch_protocol_flags:
            failures.append(
                (
                    provider["run_dir"],
                    batch_protocol_flags,
                )
            )
        return failures


class OcrBenchmark(InvariantAwareCommand):
    """Benchmark an isolated OCR provider on a hash-pinned page suite.

    Provider processes receive scratch copies of page inputs through a JSONL
    file protocol. External providers require explicit --allow-external.
    """

    _verbose_ = True

    manifest = None
    _manifest = None

    class ManifestArg(StringInvariant):
        _help = "Path to the hash-pinned benchmark manifest JSONL."

    input_root = None
    _input_root = None

    class InputRootArg(DirectoryInvariant):
        _help = "Read-only root containing benchmark source artifacts."

    provider_spec = None
    _provider_spec = None

    class ProviderSpecArg(StringInvariant):
        _help = "Path to a provider specification JSON file."

    dest_dir = None
    _dest_dir = None

    class DestDirArg(StringInvariant):
        _help = "New or empty scratch destination for all benchmark evidence."

    lanes = None
    _lanes = None

    class LanesArg(StringInvariant):
        _help = "Comma-separated lanes; defaults to the spec or plan."
        _mandatory = False

    repeats = None
    _repeats = None

    class RepeatsArg(PositiveIntegerInvariant):
        _help = "Repeats per lane; local default 5, external from plan."
        _mandatory = False

    allow_external = None

    class AllowExternalArg(BoolInvariant):
        _help = (
            "Permit a separately approved external plan. [default: %default]"
        )
        _mandatory = False
        _default = False

    external_plan = None
    _external_plan = None

    class ExternalPlanArg(StringInvariant):
        _help = "Hash-bound external upload-plan JSON path."
        _mandatory = False

    external_approval = None
    _external_approval = None

    class ExternalApprovalArg(StringInvariant):
        _help = "Separate external upload-approval JSON path."
        _mandatory = False

    def run(self):
        import json

        from .ocr_benchmark import BenchmarkError, run_benchmark

        external_plan = _option_value(self, "external_plan")
        lanes_value = _option_value(self, "lanes")
        repeats_value = _option_value(self, "repeats")
        if external_plan and (lanes_value is None or repeats_value is None):
            try:
                from .ocr_external import validate_external_upload_plan

                with open(external_plan, encoding="utf-8") as handle:
                    external_plan_record = validate_external_upload_plan(
                        json.load(handle)
                    )
            except (OSError, ValueError) as exc:
                raise CommandError(
                    "could not read the external plan for lane/repeat "
                    "defaults"
                ) from exc
            if lanes_value is None:
                lanes_value = ",".join(
                    lane["lane_id"] for lane in external_plan_record["lanes"]
                )
            if repeats_value is None:
                repeats_value = external_plan_record["execution"]["repeats"]
        lanes = (
            [lane.strip() for lane in lanes_value.split(",")]
            if lanes_value is not None
            else None
        )
        try:
            summary = run_benchmark(
                manifest_path=_option_value(self, "manifest"),
                input_root=_option_value(self, "input_root"),
                provider_spec_path=_option_value(self, "provider_spec"),
                dest_dir=_option_value(self, "dest_dir"),
                lanes=lanes,
                repeats=int(repeats_value or 5),
                allow_external=bool(self.allow_external),
                external_plan_path=external_plan,
                external_approval_path=_option_value(
                    self,
                    "external_approval",
                ),
                out=self._out,
            )
        except BenchmarkError as exc:
            raise CommandError(str(exc)) from exc
        self._out(
            f"Benchmark completed: {summary['passed_attempt_count']}/"
            f"{summary['attempt_count']} attempts passed."
        )
        if not summary["all_pass"]:
            raise CommandError(
                "OCR benchmark failed; raw evidence and summary were "
                "preserved"
            )


class OcrExternalList(InvariantAwareCommand):
    """List dated hosted OCR policy profiles without reading credentials."""

    format = None
    _format = None

    class FormatArg(StringInvariant):
        _help = "Output format: text or json. [default: %default]"
        _mandatory = False
        _default = "text"

    def run(self):
        import json

        from .ocr_external import (
            get_external_provider_profile,
            list_external_provider_profiles,
        )

        profiles = [
            get_external_provider_profile(profile_id)
            for profile_id in list_external_provider_profiles()
        ]
        output_format = _option_value(self, "format", "text").lower()
        if output_format == "json":
            self._out(json.dumps(profiles, indent=2, sort_keys=True))
            return
        if output_format != "text":
            raise CommandError("--format must be text or json")
        for profile in profiles:
            self._out(
                f"{profile['profile_id']}: "
                f"{profile['model']['requested']}; "
                f"retention={profile['retention']['policy_id']}; "
                f"credential={profile['credential_env_name']}"
            )


class OcrExternalPlan(InvariantAwareCommand):
    """Create a byte-free hosted OCR plan, spec, and request previews.

    This command validates local input hashes but writes no source bytes,
    reads no credential, and performs no network access. The resulting plan
    is not permission to transmit data; `sne ocr-external-approve` must create
    a separate approval record before the benchmark can launch.
    """

    profile = None
    _profile = None

    class ProfileArg(StringInvariant):
        _help = "Dated hosted-provider profile ID."

    manifest = None
    _manifest = None

    class ManifestArg(StringInvariant):
        _help = "Hash-pinned benchmark manifest JSONL."

    input_root = None
    _input_root = None

    class InputRootArg(DirectoryInvariant):
        _help = "Read-only root containing benchmark page inputs."

    dest_dir = None
    _dest_dir = None

    class DestDirArg(StringInvariant):
        _help = "New or empty absolute scratch plan directory."

    python_executable = None
    _python_executable = None

    class PythonExecutableArg(StringInvariant):
        _help = "Absolute Python executable; defaults to the current one."
        _mandatory = False

    lane = None
    _lane = None

    class LaneArg(StringInvariant):
        _help = "Single benchmark lane ID. [default: %default]"
        _mandatory = False
        _default = "parity"

    prompt = None
    _prompt = None

    class PromptArg(StringInvariant):
        _help = "Exact provider instruction. [default: common OCR prompt]"
        _mandatory = False

    prompt_mode = None
    _prompt_mode = None

    class PromptModeArg(StringInvariant):
        _help = "Prompt-policy label. [default: %default]"
        _mandatory = False
        _default = "parity"

    include_raw_text = None

    class IncludeRawTextArg(BoolInvariant):
        _help = "Include extracted raw text in this upload lane."
        _mandatory = False
        _default = False

    repeats = None
    _repeats = None

    class RepeatsArg(PositiveIntegerInvariant):
        _help = "Approved repeats per case. [default: %default]"
        _mandatory = False
        _default = 1

    max_api_attempts = None
    _max_api_attempts = None

    class MaxApiAttemptsArg(PositiveIntegerInvariant):
        _help = "Maximum API delivery attempts per case. [default: %default]"
        _mandatory = False
        _default = 1

    max_input_tokens = None
    _max_input_tokens = None

    class MaxInputTokensArg(PositiveIntegerInvariant):
        _help = (
            "Post-response input-token scope limit; budget reserves the "
            "full context. [default: %default]"
        )
        _mandatory = False
        _default = 262144

    max_output_tokens = None
    _max_output_tokens = None

    class MaxOutputTokensArg(PositiveIntegerInvariant):
        _help = "Maximum generated tokens per request. [default: %default]"
        _mandatory = False
        _default = 8192

    budget_usd = None
    _budget_usd = None

    class BudgetUsdArg(StringInvariant):
        _help = "Maximum approved USD budget as an exact decimal string."

    valid_hours = None
    _valid_hours = None

    class ValidHoursArg(PositiveIntegerInvariant):
        _help = "Plan lifetime in hours. [default: %default]"
        _mandatory = False
        _default = 24

    timeout_seconds = None
    _timeout_seconds = None

    class TimeoutSecondsArg(PositiveIntegerInvariant):
        _help = "Per-provider-process timeout. [default: %default]"
        _mandatory = False
        _default = 120

    def run(self):
        import json
        import sys

        from .ocr_external_commands import (
            DEFAULT_EXTERNAL_PROMPT,
            ExternalCommandError,
            create_external_plan_bundle,
        )

        try:
            receipt = create_external_plan_bundle(
                profile_id=_option_value(self, "profile"),
                manifest_path=_option_value(self, "manifest"),
                input_root=_option_value(
                    self,
                    "input_root",
                    private_first=True,
                ),
                dest_dir=_option_value(self, "dest_dir"),
                python_executable=_option_value(
                    self,
                    "python_executable",
                    sys.executable,
                ),
                lane_id=_option_value(self, "lane", "parity"),
                prompt=_option_value(
                    self,
                    "prompt",
                    DEFAULT_EXTERNAL_PROMPT,
                ),
                prompt_mode=_option_value(
                    self,
                    "prompt_mode",
                    "parity",
                ),
                include_raw_text=bool(self.include_raw_text),
                repeats=int(_option_value(self, "repeats", 1)),
                max_api_attempts_per_case=int(
                    _option_value(self, "max_api_attempts", 1)
                ),
                max_input_tokens_per_case=int(
                    _option_value(self, "max_input_tokens", 262144)
                ),
                max_output_tokens=int(
                    _option_value(self, "max_output_tokens", 8192)
                ),
                budget_usd=_option_value(self, "budget_usd"),
                valid_hours=int(_option_value(self, "valid_hours", 24)),
                timeout_seconds=int(
                    _option_value(self, "timeout_seconds", 120)
                ),
            )
        except ExternalCommandError as exc:
            raise CommandError(str(exc)) from exc
        self._out(json.dumps(receipt, indent=2, sort_keys=True))


class OcrExternalApprove(InvariantAwareCommand):
    """Create a separate, expiring approval for one exact upload plan.

    This is still an offline command: it neither reads credentials nor calls
    a provider. Passing `--accept-retention` explicitly accepts the retention
    policy embedded in the hash-bound plan.
    """

    plan = None
    _plan = None

    class PlanArg(StringInvariant):
        _help = "External upload-plan JSON path."

    output = None
    _output = None

    class OutputArg(StringInvariant):
        _help = "New absolute approval-record JSON path."

    approved_by = None
    _approved_by = None

    class ApprovedByArg(StringInvariant):
        _help = "Approving principal identifier."

    accept_retention = None

    class AcceptRetentionArg(BoolInvariant):
        _help = "Explicitly accept the plan's provider retention policy."
        _mandatory = False
        _default = False

    valid_hours = None
    _valid_hours = None

    class ValidHoursArg(PositiveIntegerInvariant):
        _help = (
            "Approval lifetime, capped by plan expiry. [default: %default]"
        )
        _mandatory = False
        _default = 24

    def run(self):
        import json

        from .ocr_external_commands import (
            ExternalCommandError,
            create_external_approval,
        )

        try:
            receipt = create_external_approval(
                plan_path=_option_value(self, "plan"),
                output_path=_option_value(self, "output"),
                approved_by=_option_value(self, "approved_by"),
                accept_retention=bool(self.accept_retention),
                valid_hours=int(_option_value(self, "valid_hours", 24)),
            )
        except ExternalCommandError as exc:
            raise CommandError(str(exc)) from exc
        self._out(json.dumps(receipt, indent=2, sort_keys=True))


class OcrBackendList(InvariantAwareCommand):
    """List pinned OCR backends and per-capability qualification state.

    This command performs no network access and imports no model runtime.
    Use `--format json` for a content-hashed registry receipt.
    """

    format = None
    _format = None

    class FormatArg(StringInvariant):
        _help = "Output format: text or json. [default: %default]"
        _mandatory = False
        _default = "text"

    def run(self):
        from .ocr_backend_setup import make_receipt, receipt_text
        from .ocr_backends import list_backends

        output_format = _option_value(self, "format", "text")
        output_format = output_format.strip().lower()
        if output_format not in {"text", "json"}:
            raise CommandError("--format must be text or json")
        backends = [item.as_dict() for item in list_backends()]
        if output_format == "json":
            receipt = make_receipt(
                "ocr_backend_registry",
                {"backends": backends},
            )
            self._out(receipt_text(receipt))
            return
        for backend in backends:
            capabilities = ", ".join(
                f"{item['capability']}={item['status']}"
                for item in backend["capabilities"]
            )
            model = backend["model"]
            self._out(
                f"{backend['backend_id']}: {capabilities}; "
                f"{backend['execution_mode']}; "
                f"{model['requested']}@{model['revision']}"
            )


class OcrInstallationRegister(InvariantAwareCommand):
    """Register one pinned local OCR backend installation.

    The machine-local registry binds a portable backend definition to an
    exact Python executable, model snapshot, adapter, and CUDA selection.
    Registration validates and hashes every binding before writing it.
    """

    backend = None
    _backend = None

    class BackendArg(StringInvariant):
        _arg = "--backend"
        _help = "Built-in OCR backend ID or alias."

    python_executable = None
    _python_executable = None

    class PythonExecutableArg(StringInvariant):
        _arg = "--python-executable"
        _help = "Absolute Python executable in the backend environment."

    cache_dir = None
    _cache_dir = None

    class CacheDirArg(DirectoryInvariant):
        _arg = "--cache-dir"
        _help = "Absolute Hugging Face cache containing the pinned model."

    installation_registry = None
    _installation_registry = None

    class InstallationRegistryArg(StringInvariant):
        _arg = "--installation-registry"
        _help = "Registry path; defaults to the machine-local XDG path."
        _mandatory = False

    cuda_visible_devices = None
    _cuda_visible_devices = None

    class CudaVisibleDevicesArg(StringInvariant):
        _arg = "--cuda-visible-devices"
        _help = "Exact physical GPU selection. [default: %default]"
        _mandatory = False
        _default = "0"

    device = None
    _device = None

    class DeviceArg(StringInvariant):
        _arg = "--device"
        _help = (
            "Logical device inside the backend process. [default: %default]"
        )
        _mandatory = False
        _default = "cuda:0"

    attention_backend = None
    _attention_backend = None

    class AttentionBackendArg(StringInvariant):
        _arg = "--attention-backend"
        _help = "Optional backend-specific attention implementation."
        _mandatory = False

    def run(self):
        from .ocr_backend_setup import make_receipt, receipt_text
        from .ocr_backends import get_backend
        from .ocr_installations import (
            InstallationRegistryError,
            default_installation_registry_path,
            register_backend_installation,
        )

        registry_value = _option_value(self, "installation_registry")
        registry_path = (
            default_installation_registry_path()
            if registry_value is None
            else Path(registry_value).expanduser()
        )
        try:
            backend = get_backend(_option_value(self, "backend"))
            registry = register_backend_installation(
                backend.backend_id,
                python_executable=_option_value(
                    self, "python_executable", private_first=True
                ),
                hf_cache=_option_value(self, "cache_dir", private_first=True),
                cuda_visible_devices=_option_value(
                    self, "cuda_visible_devices", "0"
                ),
                device=_option_value(self, "device", "cuda:0"),
                attention_backend=_option_value(self, "attention_backend"),
                path=registry_value,
            )
            receipt = make_receipt(
                "ocr_backend_installation_registration",
                {
                    "registry_path": str(registry_path),
                    "registry_content_sha256": registry["content_sha256"],
                    "installation": registry["installations"][
                        backend.backend_id
                    ],
                },
            )
        except (OSError, ValueError, InstallationRegistryError) as exc:
            raise CommandError(str(exc)) from exc
        self._out(receipt_text(receipt))


class OcrInstallationList(InvariantAwareCommand):
    """List and revalidate machine-local OCR backend installations."""

    installation_registry = None
    _installation_registry = None

    class InstallationRegistryArg(StringInvariant):
        _arg = "--installation-registry"
        _help = "Registry path; defaults to the machine-local XDG path."
        _mandatory = False

    format = None
    _format = None

    class FormatArg(StringInvariant):
        _arg = "--format"
        _help = "Output format: text or json. [default: %default]"
        _mandatory = False
        _default = "text"

    def run(self):
        from .ocr_backend_setup import make_receipt, receipt_text
        from .ocr_installations import (
            InstallationRegistryError,
            default_installation_registry_path,
            load_installation_registry,
        )

        output_format = _option_value(self, "format", "text").strip().lower()
        if output_format not in {"text", "json"}:
            raise CommandError("--format must be text or json")
        registry_value = _option_value(self, "installation_registry")
        registry_path = (
            default_installation_registry_path()
            if registry_value is None
            else Path(registry_value).expanduser()
        )
        try:
            registry = load_installation_registry(
                registry_value,
                validate_current=True,
                missing_ok=True,
            )
        except (OSError, ValueError, InstallationRegistryError) as exc:
            raise CommandError(str(exc)) from exc
        installations = [
            registry["installations"][key]
            for key in sorted(registry["installations"])
        ]
        if output_format == "json":
            receipt = make_receipt(
                "ocr_backend_installation_registry",
                {
                    "registry_path": str(registry_path),
                    "registry_content_sha256": registry["content_sha256"],
                    "installations": installations,
                },
            )
            self._out(receipt_text(receipt))
            return
        if not installations:
            self._out("No OCR backend installations are registered.")
            return
        for record in installations:
            cuda = record["cuda"]
            self._out(
                f"{record['backend_id']}: "
                f"{record['model']['id']}@{record['model']['revision']}; "
                f"python={record['python_executable']['path']}; "
                f"cuda={cuda['visible_devices']}:{cuda['device']}; "
                f"record={record['record_sha256']}"
            )


class OcrPipelineList(InvariantAwareCommand):
    """List built-in OCR pipelines and structured-execution support."""

    format = None
    _format = None

    class FormatArg(StringInvariant):
        _arg = "--format"
        _help = "Output format: text or json. [default: %default]"
        _mandatory = False
        _default = "text"

    def run(self):
        from .ocr_backend_setup import make_receipt, receipt_text
        from .ocr_profiles import list_pipeline_definitions

        output_format = _option_value(self, "format", "text").strip().lower()
        if output_format not in {"text", "json"}:
            raise CommandError("--format must be text or json")
        pipelines = []
        for definition in list_pipeline_definitions():
            record = dict(definition)
            enabled = record["pipeline_id"] == "local-mineru25"
            record["structured_execution"] = {
                "enabled": enabled,
                "reason": (
                    "supported"
                    if enabled
                    else (
                        "withheld until the true ungrounded Markdown lane "
                        "is implemented and tuned"
                    )
                ),
            }
            pipelines.append(record)
        if output_format == "json":
            receipt = make_receipt(
                "ocr_pipeline_registry",
                {"pipelines": pipelines},
            )
            self._out(receipt_text(receipt))
            return
        for pipeline in pipelines:
            stages = " -> ".join(
                f"{stage['stage_id']}:{stage['backend_id']}"
                for stage in pipeline["stages"]
            )
            support = (
                "enabled"
                if pipeline["structured_execution"]["enabled"]
                else "withheld"
            )
            self._out(
                f"{pipeline['pipeline_id']}: {stages}; structured={support}"
            )


def _profile_plugin_status(
    profile,
    *,
    trusted_profile_sha256s,
):
    """Describe source-controlled trust without changing profile content."""

    digest = profile["profile_sha256"]
    qualification_state = profile["qualification"]["state"]
    trusted = digest in trusted_profile_sha256s
    production_eligible = qualification_state == "qualified" and trusted
    if qualification_state != "qualified":
        reason = "profile qualification is provisional"
    elif not trusted:
        reason = "profile hash is not trusted by this plugin build"
    else:
        reason = None
    return {
        "profile_id": profile["profile_id"],
        "profile_sha256": digest,
        "qualification_state": qualification_state,
        "trusted_by_current_plugin": trusted,
        "production_eligible": production_eligible,
        "production_ineligibility_reason": reason,
    }


class OcrProfileList(InvariantAwareCommand):
    """List validated profiles and their current-plugin trust status."""

    # OcrPipelineList owns the conventional ``opl`` abbreviation.
    _shortname_ = "oprl"

    profile_dir = None
    _profile_dir = None

    class ProfileDirArg(StringInvariant):
        _arg = "--profile-dir"
        _help = (
            "Comma-separated profile paths; defaults to bundled profiles "
            "first, then the XDG directory."
        )
        _mandatory = False

    format = None
    _format = None

    class FormatArg(StringInvariant):
        _arg = "--format"
        _help = "Output format: text or json. [default: %default]"
        _mandatory = False
        _default = "text"

    def run(self):
        from .ocr_backend_setup import make_receipt, receipt_text
        from .ocr_execution import (
            OcrExecutionError,
            profile_search_paths,
        )
        from .ocr_profiles import (
            TRUSTED_QUALIFIED_PROFILE_SHA256S,
            OcrProfileError,
            list_execution_profiles,
        )

        output_format = _option_value(self, "format", "text").strip().lower()
        if output_format not in {"text", "json"}:
            raise CommandError("--format must be text or json")
        configured = _option_value(self, "profile_dir")
        try:
            paths = profile_search_paths(configured)
            sources = tuple(path for path in paths if path.exists())
            profiles = list_execution_profiles(paths=sources)
        except (
            OSError,
            ValueError,
            OcrExecutionError,
            OcrProfileError,
        ) as exc:
            raise CommandError(str(exc)) from exc
        statuses = tuple(
            _profile_plugin_status(
                profile,
                trusted_profile_sha256s=(TRUSTED_QUALIFIED_PROFILE_SHA256S),
            )
            for profile in profiles
        )
        if output_format == "json":
            receipt = make_receipt(
                "ocr_execution_profile_registry",
                {
                    "search_paths": [str(path) for path in paths],
                    "profiles": list(profiles),
                    "profile_statuses": list(statuses),
                },
            )
            self._out(receipt_text(receipt))
            return
        if not profiles:
            self._out("No OCR execution profiles were found.")
            return
        for profile, status in zip(profiles, statuses, strict=True):
            supported = ",".join(
                tier
                for tier, value in profile["memory_tiers"].items()
                if value["supported"]
            )
            self._out(
                f"{profile['profile_id']}: "
                f"pipeline={profile['pipeline_id']}; "
                f"qualification={profile['qualification']['state']}; "
                f"tiers={supported or 'none'}; "
                "trusted_by_current_plugin="
                f"{'yes' if status['trusted_by_current_plugin'] else 'no'}; "
                "production_eligible="
                f"{'yes' if status['production_eligible'] else 'no'}; "
                f"sha256={profile['profile_sha256']}"
            )


class OcrProfileShow(InvariantAwareCommand):
    """Show one validated profile and its plugin trust status as JSON."""

    profile_id = None
    _profile_id = None

    class ProfileIdArg(StringInvariant):
        _arg = "--profile-id"
        _help = "Exact portable execution-profile ID."

    profile_dir = None
    _profile_dir = None

    class ProfileDirArg(StringInvariant):
        _arg = "--profile-dir"
        _help = (
            "Comma-separated profile paths; defaults to bundled profiles "
            "first, then the XDG directory."
        )
        _mandatory = False

    def run(self):
        from .ocr_backend_setup import make_receipt, receipt_text
        from .ocr_execution import OcrExecutionError, profile_search_paths
        from .ocr_profiles import (
            TRUSTED_QUALIFIED_PROFILE_SHA256S,
            OcrProfileError,
            show_execution_profile,
        )

        try:
            paths = profile_search_paths(_option_value(self, "profile_dir"))
            profile = show_execution_profile(
                _option_value(self, "profile_id"),
                paths=tuple(path for path in paths if path.exists()),
            )
        except (
            OSError,
            ValueError,
            OcrExecutionError,
            OcrProfileError,
        ) as exc:
            raise CommandError(str(exc)) from exc
        receipt = make_receipt(
            "ocr_execution_profile_detail",
            {
                "profile": profile,
                "profile_status": _profile_plugin_status(
                    profile,
                    trusted_profile_sha256s=(
                        TRUSTED_QUALIFIED_PROFILE_SHA256S
                    ),
                ),
            },
        )
        self._out(receipt_text(receipt))


class OcrProfileQualify(InvariantAwareCommand):
    """Create a source-review candidate from a provisional GPU profile.

    Qualification is deliberately two-step. This command verifies complete
    repeated evidence and writes a sealed qualified candidate; production
    trust still requires checking that profile and its exact hash into the
    plugin's reviewed profile registry.
    """

    profile = None
    _profile = None

    class ProfileArg(StringInvariant):
        _help = "Absolute path to a sealed provisional execution profile."

    output = None
    _output = None

    class OutputArg(StringInvariant):
        _help = "New absolute path for the qualified review candidate."

    reviewed_evidence = None

    class ReviewedEvidenceArg(BoolInvariant):
        _arg = "--reviewed-evidence"
        _help = (
            "Acknowledge manual review of the referenced immutable tuning "
            "evidence. [default: %default]"
        )
        _mandatory = False
        _default = False

    def run(self):
        from .ocr_backend_setup import make_receipt, receipt_text
        from .ocr_profiles import (
            TRUSTED_QUALIFIED_PROFILE_SHA256S,
            OcrProfileError,
            load_execution_profile,
            qualify_execution_profile,
            write_execution_profile,
        )

        if not self.reviewed_evidence:
            raise CommandError(
                "--reviewed-evidence is required after manually reviewing "
                "the immutable tuning summaries and artifacts"
            )
        source = Path(_option_value(self, "profile")).expanduser()
        destination = Path(_option_value(self, "output")).expanduser()
        if not source.is_absolute() or not destination.is_absolute():
            raise CommandError("profile qualification paths must be absolute")
        try:
            provisional = load_execution_profile(source)
            qualified = qualify_execution_profile(provisional)
            written = write_execution_profile(qualified, destination)
        except (OSError, ValueError, OcrProfileError) as exc:
            raise CommandError(str(exc)) from exc
        digest = written["profile_sha256"]
        receipt = make_receipt(
            "ocr_execution_profile_qualification_candidate",
            {
                "source_path": str(source.resolve(strict=True)),
                "source_profile_sha256": provisional["profile_sha256"],
                "output_path": str(destination.resolve(strict=True)),
                "qualified_profile_sha256": digest,
                "trusted_by_current_plugin": (
                    digest in TRUSTED_QUALIFIED_PROFILE_SHA256S
                ),
                "next_gate": (
                    "review and check in the profile, then add its exact "
                    "hash to TRUSTED_QUALIFIED_PROFILE_SHA256S"
                ),
            },
        )
        self._out(receipt_text(receipt))


class OcrHardwareProbe(InvariantAwareCommand):
    """Probe one NVIDIA GPU and calculate 25%, 50%, and max OCR budgets.

    The probe imports no model runtime in the CLI process. When
    `--torch-python` is supplied it launches that interpreter in an isolated,
    secret-scrubbed subprocess. Multi-GPU hosts must select exactly one
    device.
    """

    torch_python = None
    _torch_python = None

    class TorchPythonArg(StringInvariant):
        _help = (
            "Backend Python used for an isolated memory-topology probe; "
            "without it tier budgets remain unavailable."
        )
        _mandatory = False

    cuda_visible_devices = None
    _cuda_visible_devices = None

    class CudaVisibleDevicesArg(StringInvariant):
        _help = "Exact CUDA_VISIBLE_DEVICES selection; defaults to ambient."
        _mandatory = False

    output = None
    _output = None

    class OutputArg(StringInvariant):
        _help = "Optional new absolute path for the JSON receipt."
        _mandatory = False

    def run(self):
        from .ocr_backend_setup import _atomic_write_new_bytes, json_bytes
        from .ocr_hardware import (
            HardwareProbeError,
            calculate_memory_tiers,
            make_hardware_receipt,
            probe_nvidia_hardware,
        )
        from .ocr_path_security import (
            TrustedPathError,
            prepare_trusted_directory,
            reject_protected_corpus_overlap,
        )

        visible_devices = _option_value(
            self,
            "cuda_visible_devices",
        )
        try:
            output = _option_value(self, "output")
            output_path = Path(output).expanduser() if output else None
            if output_path is not None:
                reject_protected_corpus_overlap(
                    output_path,
                    label="hardware probe output",
                )
                output_parent = prepare_trusted_directory(
                    output_path.parent,
                    label="hardware probe output parent",
                )
                output_path = output_parent / output_path.name
            observation = probe_nvidia_hardware(
                torch_python=_option_value(self, "torch_python"),
                visible_devices=visible_devices,
                cuda_device_order=(
                    "PCI_BUS_ID" if visible_devices is not None else None
                ),
            )
            budgets = calculate_memory_tiers(observation)
            receipt = make_hardware_receipt(
                observation,
                memory_budgets=budgets,
            )
            if output_path is not None:
                _atomic_write_new_bytes(
                    output_path,
                    json_bytes(receipt, pretty=True),
                )
        except (
            OSError,
            ValueError,
            HardwareProbeError,
            TrustedPathError,
        ) as exc:
            raise CommandError(str(exc)) from exc
        self._out(json_bytes(receipt, pretty=True).decode("utf-8").rstrip())


class OcrProfileTune(InvariantAwareCommand):
    """Tune local OCR execution profiles on one previously unknown GPU.

    Candidate processes use immutable scratch directories and may be resumed.
    The checked-in three-page suite produces provisional hardware evidence;
    it does not by itself create a new model-quality qualification.
    """

    _verbose_ = True

    pipeline = None
    _pipeline = None

    class PipelineArg(StringInvariant):
        _help = "Pipeline to tune. [default: %default]"
        _mandatory = False
        _default = "local-mineru25"

    manifest = None
    _manifest = None

    class ManifestArg(StringInvariant):
        _help = "Hash-pinned benchmark manifest JSONL."

    input_root = None
    _input_root = None

    class InputRootArg(DirectoryInvariant):
        _help = "Read-only benchmark input root."

    dest_dir = None
    _dest_dir = None

    class DestDirArg(StringInvariant):
        _help = "Absolute empty or matching resumable scratch directory."

    installation_registry = None
    _installation_registry = None

    class InstallationRegistryArg(StringInvariant):
        _arg = "--installation-registry"
        _help = (
            "Validated machine-local OCR installation registry; defaults "
            "to the XDG registry."
        )
        _mandatory = False

    deepseek_python = None
    _deepseek_python = None

    class DeepseekPythonArg(StringInvariant):
        _help = "Explicit raw-fallback DeepSeek/vLLM Python executable."
        _mandatory = False

    deepseek_cache_dir = None
    _deepseek_cache_dir = None

    class DeepseekCacheDirArg(DirectoryInvariant):
        _help = "Explicit raw-fallback DeepSeek Hugging Face cache."
        _mandatory = False

    mineru_python = None
    _mineru_python = None

    class MineruPythonArg(StringInvariant):
        _help = "Explicit raw-fallback MinerU Python executable."
        _mandatory = False

    mineru_cache_dir = None
    _mineru_cache_dir = None

    class MineruCacheDirArg(DirectoryInvariant):
        _help = "Explicit raw-fallback MinerU Hugging Face cache."
        _mandatory = False

    cuda_visible_devices = None
    _cuda_visible_devices = None

    class CudaVisibleDevicesArg(StringInvariant):
        _help = (
            "Raw fallback only: single physical GPU selection. "
            "[raw-fallback default: 0]"
        )
        _mandatory = False

    device = None
    _device = None

    class DeviceArg(StringInvariant):
        _help = (
            "Raw fallback only: CUDA device within each isolated process. "
            "[raw-fallback default: cuda:0]"
        )
        _mandatory = False

    deepseek_attention_backend = None
    _deepseek_attention_backend = None

    class DeepseekAttentionBackendArg(StringInvariant):
        _help = (
            "Raw fallback only: DeepSeek attention backend or auto. "
            "[raw-fallback default: auto]"
        )
        _mandatory = False

    deepseek_moe_backend = None
    _deepseek_moe_backend = None

    class DeepseekMoeBackendArg(StringInvariant):
        _help = (
            "DeepSeek MoE backend to qualify; must be triton. "
            "[default: %default]"
        )
        _mandatory = False
        _default = QUALIFIED_DEEPSEEK_MOE_BACKEND

    mineru_attention_backend = None
    _mineru_attention_backend = None

    class MineruAttentionBackendArg(StringInvariant):
        _help = (
            "Raw fallback only: MinerU attention backend. "
            "[raw-fallback default: sdpa]"
        )
        _mandatory = False

    memory_tiers = None
    _memory_tiers = None

    class MemoryTiersArg(StringInvariant):
        _help = "Comma-separated memory tiers. [default: %default]"
        _mandatory = False
        _default = "25,50,max"

    batch_sizes = None
    _batch_sizes = None

    class BatchSizesArg(StringInvariant):
        _help = (
            "Ascending comma-separated candidate batches. [default: %default]"
        )
        _mandatory = False
        _default = "1,2,4,8"

    discovery_repeats = None
    _discovery_repeats = None

    class DiscoveryRepeatsArg(PositiveIntegerInvariant):
        _help = "Cold repeats per discovery candidate. [default: %default]"
        _mandatory = False
        _default = 1

    confirmation_repeats = None
    _confirmation_repeats = None

    class ConfirmationRepeatsArg(PositiveIntegerInvariant):
        _help = "Required repeats for later confirmation. [default: %default]"
        _mandatory = False
        _default = 5

    timeout_seconds = None
    _timeout_seconds = None

    class TimeoutSecondsArg(PositiveIntegerInvariant):
        _help = "Per-candidate process timeout. [default: %default]"
        _mandatory = False
        _default = 3600

    profile_output = None
    _profile_output = None

    class ProfileOutputArg(StringInvariant):
        _arg = "--profile-output"
        _help = "Absolute output path for the sealed provisional profile."
        _mandatory = False

    profile_id = None
    _profile_id = None

    class ProfileIdArg(StringInvariant):
        _arg = "--profile-id"
        _help = "Portable ID for the provisional execution profile."
        _mandatory = False

    profile_display_name = None
    _profile_display_name = None

    class ProfileDisplayNameArg(StringInvariant):
        _arg = "--profile-display-name"
        _help = "Human-readable provisional execution-profile name."
        _mandatory = False

    dry_run = None

    class DryRunArg(BoolInvariant):
        _help = (
            "Print the exact plan without GPU inference. [default: %default]"
        )
        _mandatory = False
        _default = False

    def run(self):
        from .ocr_backend_setup import json_bytes
        from .ocr_hardware import (
            HardwareProbeError,
            calculate_memory_tiers,
            probe_nvidia_hardware,
        )
        from .ocr_installations import (
            InstallationRegistryError,
            build_backend_installation,
            load_installation_registry,
        )
        from .ocr_profiles import get_pipeline_definition
        from .ocr_tuning import (
            TuningError,
            build_tuning_plan,
            run_tuning_session,
        )

        try:
            pipeline = get_pipeline_definition(
                _option_value(self, "pipeline", "local-mineru25")
            )["pipeline_id"]
            if pipeline == "deepseek-vllm-full":
                raise TuningError(
                    "deepseek-vllm-full tuning is withheld until its true "
                    "ungrounded Markdown lane is implemented"
                )
            profile_output = _option_value(self, "profile_output")
            profile_id = _option_value(self, "profile_id")
            profile_display_name = _option_value(self, "profile_display_name")
            profile_values = (
                profile_output,
                profile_id,
                profile_display_name,
            )
            if any(value is not None for value in profile_values) and not all(
                value is not None for value in profile_values
            ):
                raise TuningError(
                    "--profile-output, --profile-id, and "
                    "--profile-display-name must be supplied together"
                )
            if (
                profile_output is not None
                and not Path(profile_output).expanduser().is_absolute()
            ):
                raise TuningError("--profile-output must be an absolute path")

            raw_paths = {
                "deepseek_python": _option_value(
                    self, "deepseek_python", private_first=True
                ),
                "deepseek_cache_dir": _option_value(
                    self, "deepseek_cache_dir", private_first=True
                ),
                "mineru_python": _option_value(
                    self, "mineru_python", private_first=True
                ),
                "mineru_cache_dir": _option_value(
                    self, "mineru_cache_dir", private_first=True
                ),
            }
            supplied_raw = {
                name for name, value in raw_paths.items() if value is not None
            }
            registry_path = _option_value(self, "installation_registry")
            if registry_path is not None and supplied_raw:
                raise TuningError(
                    "--installation-registry cannot be combined with raw "
                    "Python/cache fallback options"
                )
            if supplied_raw and len(supplied_raw) != len(raw_paths):
                missing = sorted(set(raw_paths) - supplied_raw)
                raise TuningError(
                    "raw installation fallback requires all Python/cache "
                    "options; missing " + ", ".join(missing)
                )
            runtime_overrides = (
                ("cuda_visible_devices", "--cuda-visible-devices"),
                ("device", "--device"),
                (
                    "deepseek_attention_backend",
                    "--deepseek-attention-backend",
                ),
                (
                    "mineru_attention_backend",
                    "--mineru-attention-backend",
                ),
            )
            if not supplied_raw:
                rejected = [
                    option
                    for name, option in runtime_overrides
                    if _option_value(self, name) is not None
                ]
                if rejected:
                    raise TuningError(
                        "registry-backed tuning uses the registered CUDA "
                        "and attention settings; remove: "
                        + ", ".join(rejected)
                    )
            if supplied_raw:
                visible = _option_value(self, "cuda_visible_devices", "0")
                device = _option_value(self, "device", "cuda:0")
                records = {
                    "deepseek-vllm": build_backend_installation(
                        "deepseek-vllm",
                        python_executable=raw_paths["deepseek_python"],
                        hf_cache=raw_paths["deepseek_cache_dir"],
                        cuda_visible_devices=visible,
                        device=device,
                        attention_backend=_option_value(
                            self,
                            "deepseek_attention_backend",
                            "auto",
                        ),
                    ),
                    "mineru25-pro-2605": build_backend_installation(
                        "mineru25-pro-2605",
                        python_executable=raw_paths["mineru_python"],
                        hf_cache=raw_paths["mineru_cache_dir"],
                        cuda_visible_devices=visible,
                        device=device,
                        attention_backend=_option_value(
                            self,
                            "mineru_attention_backend",
                            "sdpa",
                        ),
                    ),
                }
            else:
                registry = load_installation_registry(
                    registry_path,
                    validate_current=True,
                    missing_ok=False,
                )
                records = registry["installations"]
                required = (
                    "deepseek-vllm",
                    "mineru25-pro-2605",
                )
                missing = [
                    backend_id
                    for backend_id in required
                    if backend_id not in records
                ]
                if missing:
                    raise TuningError(
                        "local-mineru25 tuning requires registered backends: "
                        + ", ".join(missing)
                    )
                deepseek_record = records["deepseek-vllm"]
                mineru_record = records["mineru25-pro-2605"]
                visible = deepseek_record["cuda"]["visible_devices"]
                if mineru_record["cuda"]["visible_devices"] != visible:
                    raise TuningError(
                        "registered pipeline backends select different GPUs"
                    )
            installations = {}
            for backend_id, attention_default in (
                ("deepseek-vllm", "auto"),
                ("mineru25-pro-2605", "sdpa"),
            ):
                record = records[backend_id]
                installations[backend_id] = {
                    "python_executable": record["python_executable"]["path"],
                    "cache_dir": record["hf_cache"]["path"],
                    "cuda_visible_devices": record["cuda"]["visible_devices"],
                    "device": record["cuda"]["device"],
                    "attention_backend": (
                        record["cuda"]["attention_backend"]
                        or attention_default
                    ),
                    "identity": {
                        "record_sha256": record["record_sha256"],
                        "python_executable_sha256": record[
                            "python_executable"
                        ]["sha256"],
                        "installed_distributions_count": record[
                            "python_executable"
                        ]["installed_distributions_count"],
                        "installed_distributions_sha256": record[
                            "python_executable"
                        ]["installed_distributions_sha256"],
                        "hf_cache_identity_sha256": record["hf_cache"][
                            "identity_sha256"
                        ],
                        "snapshot_files_sha256": record["snapshot"][
                            "files_sha256"
                        ],
                        "snapshot_weights_sha256": record["snapshot"][
                            "weights_sha256"
                        ],
                        "adapter_sha256": (
                            record["adapter"]["sha256"]
                            if record["adapter"] is not None
                            else None
                        ),
                        "implementation_sha256": record["implementation"][
                            "sha256"
                        ],
                    },
                }
            observation = probe_nvidia_hardware(
                torch_python=installations["deepseek-vllm"][
                    "python_executable"
                ],
                visible_devices=visible,
                cuda_device_order="PCI_BUS_ID",
            )
            budgets = calculate_memory_tiers(observation)
            hardware = {
                "selector": observation["selector"],
                "memory_budgets": budgets,
                "observation_sha256": observation["selector_sha256"],
                "torch_cuda_total_memory_bytes": observation["memory"][
                    "total_sources"
                ].get("torch_cuda"),
            }
            tiers = tuple(
                item.strip()
                for item in _option_value(
                    self, "memory_tiers", "25,50,max"
                ).split(",")
                if item.strip()
            )
            batches = tuple(
                int(item.strip())
                for item in _option_value(
                    self, "batch_sizes", "1,2,4,8"
                ).split(",")
                if item.strip()
            )
            plan = build_tuning_plan(
                pipeline_id=pipeline,
                hardware=hardware,
                installations=installations,
                manifest_path=_option_value(self, "manifest"),
                input_root=_option_value(
                    self,
                    "input_root",
                    private_first=True,
                ),
                memory_tiers=tiers,
                batch_sizes=batches,
                discovery_repeats=self._discovery_repeats or 1,
                confirmation_repeats=self._confirmation_repeats or 5,
                timeout_seconds=self._timeout_seconds or 3600,
                deepseek_moe_backend=_option_value(
                    self,
                    "deepseek_moe_backend",
                    QUALIFIED_DEEPSEEK_MOE_BACKEND,
                ),
            )
            if self.dry_run:
                self._out(
                    json_bytes(plan, pretty=True).decode("utf-8").rstrip()
                )
                return
            summary = run_tuning_session(
                plan=plan,
                dest_dir=_option_value(self, "dest_dir"),
                out=self._out,
                profile_path=profile_output,
                profile_id=profile_id,
                profile_display_name=profile_display_name,
            )
        except (
            OSError,
            ValueError,
            HardwareProbeError,
            InstallationRegistryError,
            TuningError,
        ) as exc:
            raise CommandError(str(exc)) from exc
        self._out(json_bytes(summary, pretty=True).decode("utf-8").rstrip())
        if not summary["complete"]:
            raise CommandError(
                "OCR profile tuning is incomplete; partial evidence was "
                "preserved for a safe resume"
            )


class OcrBackendCreateEnv(InvariantAwareCommand):
    """Create a dedicated environment for a pinned OCR backend.

    The environment and package command plan is backend-specific. Use
    `--dry-run` to inspect the exact plan without executing it.
    """

    _verbose_ = True

    backend = None
    _backend = None

    class BackendArg(StringInvariant):
        _help = "Built-in OCR backend ID or alias."

    env_name = None
    _env_name = None

    class EnvNameArg(StringInvariant):
        _help = "Environment name; defaults to the backend profile."
        _mandatory = False

    python = None
    _python = None

    class PythonArg(StringInvariant):
        _help = "Python version; defaults to the backend profile."
        _mandatory = False

    manager = None
    _manager = None

    class ManagerArg(StringInvariant):
        _help = "Environment manager executable; auto-detects mamba/conda."
        _mandatory = False

    torch_backend = None
    _torch_backend = None

    class TorchBackendArg(StringInvariant):
        _help = "PyTorch wheel backend. [default: %default]"
        _mandatory = False
        _default = DEFAULT_TORCH_BACKEND

    editable_root = None
    _editable_root = None

    class EditableRootArg(StringInvariant):
        _help = (
            "DeepSeek only: install this exact checkout editable; use this "
            "path for source-based GPU profile qualification."
        )
        _mandatory = False

    package_spec = None
    _package_spec = None

    class PackageSpecArg(StringInvariant):
        _help = (
            "DeepSeek only: bootstrap package spec when --editable-root is "
            "omitted; the mutable default is not a qualification source."
        )
        _mandatory = False
        _default = "sneeze-plugin-tpn[pdf,ocr]"

    dry_run = None

    class DryRunArg(BoolInvariant):
        _help = "Print commands without executing them. [default: %default]"
        _mandatory = False
        _default = False

    def run(self):
        from .ocr_backend_setup import (
            BackendSetupError,
            build_environment_commands,
            environment_plan_receipt,
            receipt_text,
            resolve_manager,
        )
        from .ocr_backends import BackendRegistryError, get_backend

        try:
            backend = get_backend(_option_value(self, "backend"))
            manager = resolve_manager(_option_value(self, "manager"))
            env_name = _option_value(
                self,
                "env_name",
                backend.default_env_name,
            )
            python_version = _option_value(
                self,
                "python",
                backend.default_python,
            )
            torch_backend = _option_value(
                self,
                "torch_backend",
                DEFAULT_TORCH_BACKEND,
            )
            commands = build_environment_commands(
                backend,
                manager=manager,
                env_name=env_name,
                python_version=python_version,
                torch_backend=torch_backend,
                editable_root=_option_value(self, "editable_root"),
                package_spec=_option_value(
                    self,
                    "package_spec",
                    "sneeze-plugin-tpn[pdf,ocr]",
                ),
            )
            _run_or_print(
                commands,
                dry_run=bool(self.dry_run),
                out=self._out,
            )
            receipt = environment_plan_receipt(
                backend,
                manager=manager,
                env_name=env_name,
                python_version=python_version,
                torch_backend=torch_backend,
                commands=commands,
                dry_run=bool(self.dry_run),
            )
        except (BackendRegistryError, BackendSetupError) as exc:
            raise CommandError(str(exc)) from exc
        self._out(receipt_text(receipt))


class OcrBackendFetchModel(InvariantAwareCommand):
    """Fetch the immutable model revision for a built-in OCR backend.

    The command never follows a mutable branch or tag. It emits a JSON receipt
    containing the exact revision, expected snapshot path, and command hash.
    """

    _verbose_ = True

    backend = None
    _backend = None

    class BackendArg(StringInvariant):
        _help = "Built-in OCR backend ID or alias."

    cache_dir = None
    _cache_dir = None

    class CacheDirArg(StringInvariant):
        _help = "Absolute Hugging Face hub cache directory."

    env_name = None
    _env_name = None

    class EnvNameArg(StringInvariant):
        _help = "Environment name; defaults to the backend profile."
        _mandatory = False

    manager = None
    _manager = None

    class ManagerArg(StringInvariant):
        _help = "Environment manager executable; auto-detects mamba/conda."
        _mandatory = False

    dry_run = None

    class DryRunArg(BoolInvariant):
        _help = "Print the command without executing it. [default: %default]"
        _mandatory = False
        _default = False

    def run(self):
        from .ocr_backend_setup import (
            BackendSetupError,
            build_model_fetch_command,
            model_fetch_receipt,
            receipt_text,
        )
        from .ocr_backends import BackendRegistryError, get_backend

        try:
            backend = get_backend(_option_value(self, "backend"))
            cache_dir = _option_value(self, "cache_dir")
            env_name = _option_value(
                self,
                "env_name",
                backend.default_env_name,
            )
            command = build_model_fetch_command(
                backend,
                manager=_option_value(self, "manager"),
                env_name=env_name,
                cache_dir=cache_dir,
            )
            _run_or_print(
                [command],
                dry_run=bool(self.dry_run),
                out=self._out,
            )
            receipt = model_fetch_receipt(
                backend,
                cache_dir=cache_dir,
                command=command,
                dry_run=bool(self.dry_run),
            )
        except (BackendRegistryError, BackendSetupError) as exc:
            raise CommandError(str(exc)) from exc
        self._out(receipt_text(receipt))


class OcrBackendWriteSpec(InvariantAwareCommand):
    """Write a validated local provider spec for a process backend.

    The output is created atomically and an existing path is never replaced.
    Model identity and revision come only from the pinned backend registry.
    """

    backend = None
    _backend = None

    class BackendArg(StringInvariant):
        _help = "Built-in provider-process OCR backend ID or alias."

    output = None
    _output = None

    class OutputArg(StringInvariant):
        _help = "New absolute output path for the provider spec JSON."

    python_executable = None
    _python_executable = None

    class PythonExecutableArg(StringInvariant):
        _help = "Absolute Python executable in the backend environment."

    cache_dir = None
    _cache_dir = None

    class CacheDirArg(DirectoryInvariant):
        _help = "Absolute Hugging Face hub cache containing the model."

    device = None
    _device = None

    class DeviceArg(StringInvariant):
        _help = "Provider device; defaults to the backend profile."
        _mandatory = False

    attention_backend = None
    _attention_backend = None

    class AttentionBackendArg(StringInvariant):
        _help = "Attention backend; defaults to the backend profile."
        _mandatory = False

    cuda_visible_devices = None
    _cuda_visible_devices = None

    class CudaVisibleDevicesArg(StringInvariant):
        _help = "CUDA_VISIBLE_DEVICES value. [default: %default]"
        _mandatory = False
        _default = "0"

    timeout_seconds = None
    _timeout_seconds = None

    class TimeoutSecondsArg(PositiveIntegerInvariant):
        _help = "Provider process timeout in seconds."
        _mandatory = False

    def run(self):
        from .ocr_backend_setup import (
            receipt_text,
            write_provider_spec,
        )
        from .ocr_backends import get_backend

        try:
            backend = get_backend(_option_value(self, "backend"))
            receipt = write_provider_spec(
                backend,
                output_path=_option_value(self, "output"),
                python_executable=_option_value(
                    self,
                    "python_executable",
                ),
                cache_dir=_option_value(
                    self,
                    "cache_dir",
                    private_first=True,
                ),
                device=_option_value(self, "device"),
                attention_backend=_option_value(
                    self,
                    "attention_backend",
                ),
                cuda_visible_devices=_option_value(
                    self,
                    "cuda_visible_devices",
                    "0",
                ),
                timeout_seconds=self._timeout_seconds,
            )
        except (OSError, ValueError) as exc:
            raise CommandError(str(exc)) from exc
        self._out(receipt_text(receipt))


class OcrBackendDoctor(InvariantAwareCommand):
    """Validate a built-in local provider spec and optionally load its model.

    Checks cover the exact model pin, adapter, offline mode, cached snapshot,
    and qualified runtime package tuple. Ambient secret values are scrubbed
    from the package probe and are never included in the JSON receipt.
    """

    provider_spec = None
    _provider_spec = None

    class ProviderSpecArg(StringInvariant):
        _help = "Path to a built-in local provider specification."

    load_model = None

    class LoadModelArg(BoolInvariant):
        _help = (
            "Initialize the qualified model/client on GPU. "
            "[default: %default]"
        )
        _mandatory = False
        _default = False

    def run(self):
        from .ocr_backend_setup import (
            doctor_provider_spec,
            receipt_text,
        )

        try:
            receipt = doctor_provider_spec(
                _option_value(self, "provider_spec"),
                load_model=bool(self.load_model),
            )
        except (OSError, ValueError) as exc:
            raise CommandError(str(exc)) from exc
        self._out(receipt_text(receipt))
        if not receipt["all_pass"]:
            raise CommandError(
                "OCR backend doctor failed; JSON findings were preserved"
            )


class DeepseekOcrCreateEnv(InvariantAwareCommand):
    """Create/update a conda environment for `sne ocr-pdf-pages`.

    Installs Python, Poppler, PyTorch, vLLM, and plugin dependencies needed to
    run DeepSeek OCR locally.

    Next: use `sne deepseek-ocr-download-model` to cache the OCR model.
    """

    env_name = None
    _env_name = None

    class EnvNameArg(StringInvariant):
        _help = "Conda/mamba environment name. [default: %default]"
        _mandatory = False
        _default = DEFAULT_OCR_ENV_NAME

    python = None
    _python = None

    class PythonArg(StringInvariant):
        _help = "Python version for the environment. [default: %default]"
        _mandatory = False
        _default = DEFAULT_PYTHON_VERSION

    manager = None
    _manager = None

    class ManagerArg(StringInvariant):
        _help = "Environment manager executable; auto-detects mamba/conda."
        _mandatory = False

    torch_backend = None
    _torch_backend = None

    class TorchBackendArg(StringInvariant):
        _help = "PyTorch wheel backend, for example cu130 or cu121."
        _mandatory = False
        _default = DEFAULT_TORCH_BACKEND

    vllm_package = None
    _vllm_package = None

    class VllmPackageArg(StringInvariant):
        _help = "vLLM package spec. [default: %default]"
        _mandatory = False
        _default = "vllm"

    editable_root = None
    _editable_root = None

    class EditableRootArg(StringInvariant):
        _help = (
            "Install this exact checkout editable; use this path for "
            "source-based GPU profile qualification."
        )
        _mandatory = False

    package_spec = None
    _package_spec = None

    class PackageSpecArg(StringInvariant):
        _help = (
            "Bootstrap package spec if --editable-root is not set; the "
            "mutable default is not a qualification source."
        )
        _mandatory = False
        _default = "sneeze-plugin-tpn[pdf,ocr]"

    dry_run = None

    class DryRunArg(BoolInvariant):
        _help = "Print commands without executing them. [default: %default]"
        _mandatory = False
        _default = False

    def run(self):
        manager = _resolve_manager(self.manager or self._manager)
        env_name = self.env_name or self._env_name or DEFAULT_OCR_ENV_NAME
        python_version = self.python or self._python or DEFAULT_PYTHON_VERSION
        torch_backend = (
            self.torch_backend or self._torch_backend or DEFAULT_TORCH_BACKEND
        )
        vllm_package = self.vllm_package or self._vllm_package or "vllm"
        editable_root = self._editable_root or self.editable_root
        package_spec = (
            self._package_spec
            or self.package_spec
            or "sneeze-plugin-tpn[pdf,ocr]"
        )
        commands = build_ocr_env_commands(
            manager=manager,
            env_name=env_name,
            python_version=python_version,
            torch_backend=torch_backend,
            vllm_package=vllm_package,
            editable_root=editable_root,
            package_spec=package_spec,
        )
        _run_or_print(commands, dry_run=bool(self.dry_run), out=self._out)


class DeepseekOcrDownloadModel(InvariantAwareCommand):
    """Download/cache the DeepSeek OCR model for `sne ocr-pdf-pages`.

    Usually run this after `sne deepseek-ocr-create-env`, optionally with
    `--env-name`, so the model is present before the first OCR run.
    """

    model = None
    _model = None

    class ModelArg(StringInvariant):
        _help = "Hugging Face model ID. [default: %default]"
        _mandatory = False
        _default = DEFAULT_OCR_MODEL

    local_dir = None
    _local_dir = None

    class LocalDirArg(StringInvariant):
        _help = "Optional destination directory."
        _mandatory = False

    revision = None
    _revision = None

    class RevisionArg(StringInvariant):
        _help = "Immutable model revision. [default: %default]"
        _mandatory = False
        _default = DEFAULT_OCR_MODEL_REVISION

    env_name = None
    _env_name = None

    class EnvNameArg(StringInvariant):
        _help = "Run the download inside this conda/mamba environment."
        _mandatory = False

    manager = None
    _manager = None

    class ManagerArg(StringInvariant):
        _help = "Environment manager executable; auto-detects mamba/conda."
        _mandatory = False

    dry_run = None

    class DryRunArg(BoolInvariant):
        _help = "Print the command without executing it. [default: %default]"
        _mandatory = False
        _default = False

    def run(self):
        command = build_model_download_command(
            model=self.model or self._model or DEFAULT_OCR_MODEL,
            local_dir=self._local_dir or self.local_dir,
            revision=(
                self._revision or self.revision or DEFAULT_OCR_MODEL_REVISION
            ),
            env_name=self._env_name or self.env_name,
            manager=self._manager or self.manager,
        )
        _run_or_print([command], dry_run=bool(self.dry_run), out=self._out)


def build_ocr_env_commands(
    *,
    manager: str,
    env_name: str,
    python_version: str,
    torch_backend: str,
    vllm_package: str,
    editable_root: str | None,
    package_spec: str,
) -> list[list[str]]:
    pytorch_index = f"https://download.pytorch.org/whl/{torch_backend}"
    commands = [
        [
            manager,
            "create",
            "-n",
            env_name,
            "-y",
            "-c",
            "conda-forge",
            f"python={python_version}",
            "pip",
            "uv",
            "poppler",
        ],
        [
            manager,
            "run",
            "-n",
            env_name,
            "python",
            "-m",
            "pip",
            "install",
            "-U",
            "pip",
        ],
        [
            manager,
            "run",
            "-n",
            env_name,
            "uv",
            "pip",
            "install",
            "-U",
            "huggingface_hub",
            "pdf2image",
            "pillow",
            "tqdm",
            "transformers",
        ],
        [
            manager,
            "run",
            "-n",
            env_name,
            "uv",
            "pip",
            "install",
            "-U",
            "--pre",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            pytorch_index,
        ],
    ]
    vllm_command = [
        manager,
        "run",
        "-n",
        env_name,
        "uv",
        "pip",
        "install",
        "-U",
        "--pre" if torch_backend == "cu130" else "",
        vllm_package,
    ]
    vllm_command = [part for part in vllm_command if part]
    if torch_backend == "cu130":
        vllm_command.extend(
            ["--extra-index-url", DEFAULT_CUDA_130_VLLM_WHEEL_INDEX]
        )
    commands.append(vllm_command)
    if editable_root:
        commands.append(
            [
                manager,
                "run",
                "-n",
                env_name,
                "python",
                "-m",
                "pip",
                "install",
                "-e",
                editable_root,
            ]
        )
    else:
        commands.append(
            [
                manager,
                "run",
                "-n",
                env_name,
                "python",
                "-m",
                "pip",
                "install",
                "-U",
                package_spec,
            ]
        )
    return commands


def build_model_download_command(
    *,
    model: str,
    local_dir: str | None,
    revision: str | None,
    env_name: str | None,
    manager: str | None,
) -> list[str]:
    code = (
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "model, local_dir, revision = sys.argv[1:4]\n"
        "kwargs = {'repo_id': model}\n"
        "if local_dir:\n"
        "    kwargs['local_dir'] = local_dir\n"
        "if revision:\n"
        "    kwargs['revision'] = revision\n"
        "print(snapshot_download(**kwargs))\n"
    )
    command = [
        "python",
        "-c",
        code,
        model,
        local_dir or "",
        revision or "",
    ]
    if env_name:
        manager_bin = _resolve_manager(manager)
        return [manager_bin, "run", "-n", env_name, *command]
    return command


def _report_skipped_pdfs(skipped, out) -> None:
    if not skipped:
        return
    encrypted = [item for item in skipped if item.is_encrypted]
    unreadable = [item for item in skipped if not item.is_encrypted]
    label = "PDF" if len(skipped) == 1 else "PDFs"
    out(f"Skipped {len(skipped)} {label}:")
    if encrypted:
        enc_label = "PDF" if len(encrypted) == 1 else "PDFs"
        out(f"  {len(encrypted)} password-protected {enc_label}")
    if unreadable:
        read_label = "PDF" if len(unreadable) == 1 else "PDFs"
        out(f"  {len(unreadable)} unreadable {read_label}")
    for item in skipped:
        out(f"  {item.source_path}")
        out(f"    {item.error}")


def _resolve_workers(value: int | None) -> int:
    workers = int(value or 0)
    if workers <= 0:
        return os.cpu_count() or 1
    return workers


def _option_value(
    command,
    name: str,
    default=None,
    *,
    private_first: bool = False,
):
    names = (f"_{name}", name) if private_first else (name, f"_{name}")
    for attr_name in names:
        value = getattr(command, attr_name, None)
        if value not in (None, ""):
            return value
    return default


def _set_process_environment(
    values: dict[str, str],
) -> dict[str, str | None]:
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    return previous


def _restore_process_environment(values: dict[str, str | None]) -> None:
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _maybe_float(value, default=None):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CommandError(f"Expected a float, got {value!r}") from exc


def _profile_backend_value(value):
    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    return value


def _paths_overlap(left: str, right: str) -> bool:
    left = os.path.realpath(left)
    right = os.path.realpath(right)
    try:
        common = os.path.commonpath((left, right))
    except ValueError:
        return False
    return common in {left, right}


def _preflight_markdown_provider_run_dir(
    run_dir: str | os.PathLike[str],
    *,
    source_dir: str,
    dest_dir: str,
) -> Path:
    """Resolve and reject unsafe evidence locations without writing."""

    candidate = Path(run_dir).expanduser()
    if not candidate.is_absolute():
        raise CommandError(
            "Markdown provider evidence directory must be absolute"
        )
    resolved = Path(os.path.realpath(candidate))
    for label, root in (
        ("source", source_dir),
        ("destination", dest_dir),
    ):
        if _paths_overlap(str(resolved), os.path.abspath(root)):
            raise CommandError(
                "Markdown provider evidence directory must be "
                f"separate from the OCR {label} directory"
            )
    from .ocr_path_security import (
        TrustedPathError,
        reject_protected_corpus_overlap,
    )

    try:
        reject_protected_corpus_overlap(
            resolved,
            label="Markdown provider evidence directory",
        )
    except TrustedPathError as exc:
        raise CommandError(str(exc)) from exc
    return resolved


def _create_profile_runtime_root(
    parent: Path,
    *,
    source_dir: str,
    dest_dir: str,
) -> Path:
    """Create a preflighted private directory for generated profile files."""

    canonical_parent = Path(os.path.realpath(parent.expanduser()))
    for _attempt in range(100):
        candidate = canonical_parent / (
            ".sneeze-ocr-profile-" + secrets.token_hex(12)
        )
        _preflight_markdown_provider_run_dir(
            candidate,
            source_dir=source_dir,
            dest_dir=dest_dir,
        )
        try:
            canonical_parent.mkdir(parents=True, exist_ok=True)
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        except OSError as exc:
            raise CommandError(
                f"Could not create OCR profile runtime directory: {exc}"
            ) from exc
        return candidate
    raise CommandError("Could not allocate OCR profile runtime directory")


def _stage_grounded_ocr_tasks(tasks):
    """Redirect structured grounded OCR into private same-filesystem files."""

    from .ocr_vllm import OcrPageTask

    staged = []
    candidates: dict[str, str] = {}
    try:
        for task in tasks:
            directory = os.path.dirname(task.output_path) or "."
            os.makedirs(directory, exist_ok=True)
            descriptor, candidate = tempfile.mkstemp(
                dir=directory,
                prefix=f".{os.path.basename(task.output_path)}.",
                suffix=".grounded-candidate",
            )
            os.close(descriptor)
            candidates[task.output_path] = candidate
            staged.append(
                OcrPageTask(
                    image_path=task.image_path,
                    output_path=candidate,
                    text_source_path=task.text_source_path,
                )
            )
    except BaseException as exc:
        _discard_grounded_ocr_candidates(
            candidates,
            original_error=exc,
        )
        raise
    return staged, candidates


def _discard_grounded_ocr_candidates(
    candidates,
    *,
    original_error: BaseException | None = None,
) -> None:
    """Remove unowned staged OCR files after an interrupted generation."""

    for candidate in candidates.values():
        try:
            os.unlink(candidate)
        except FileNotFoundError:
            pass
        except OSError as exc:
            if original_error is not None:
                original_error.add_note(
                    "could not remove staged OCR candidate "
                    f"{os.path.basename(candidate)}: {exc}"
                )


def _finalize_grounded_ocr_candidates(processor, candidates) -> None:
    """Validate and atomically promote staged structured grounded OCR."""

    from .ocr_vllm import file_sha256, read_optional_text

    for persisted_path, candidate_path in candidates.items():
        raw_metadata = processor.generation_metadata.pop(
            candidate_path,
            None,
        )
        metadata = (
            dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        )
        flags = []
        if metadata.get("status") != "completed":
            flags.append("grounded_ocr_generation_failed")
        termination = metadata.get("termination")
        if (
            not isinstance(termination, Mapping)
            or termination.get("eos") is not True
        ):
            flags.append("grounded_ocr_non_eos_termination")
        if (
            isinstance(termination, Mapping)
            and termination.get("max_tokens") is True
        ):
            flags.append("grounded_ocr_max_tokens_termination")
        candidate_text = read_optional_text(candidate_path)
        if not isinstance(candidate_text, str) or not candidate_text.strip():
            flags.append("grounded_ocr_candidate_empty")
        flags = list(dict.fromkeys(flags))
        candidate_sha256 = file_sha256(candidate_path)
        persisted_before = file_sha256(persisted_path)
        promoted = False
        if not flags:
            try:
                _atomic_write_text(persisted_path, candidate_text)
                promoted = True
            except OSError as exc:
                flags.append("grounded_ocr_promotion_failed: " + str(exc))
        candidate_retained = True
        if promoted:
            try:
                os.unlink(candidate_path)
                candidate_retained = False
            except FileNotFoundError:
                candidate_retained = False
            except OSError:
                # Promotion is already durable. A private diagnostic copy is
                # harmless and is truthfully recorded below.
                candidate_retained = True
        metadata["validation"] = {
            "status": "fail" if flags else "pass",
            "flags": flags,
        }
        metadata["promotion"] = {
            "promoted": promoted,
            "candidate_path": (
                candidate_path if candidate_retained else None
            ),
            "candidate_sha256": candidate_sha256,
            "persisted_output_sha256_before": persisted_before,
            "persisted_output_sha256": file_sha256(persisted_path),
        }
        if flags:
            metadata["status"] = "error"
        processor.generation_metadata[persisted_path] = metadata


def _atomic_write_text(path: str, text: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    try:
        target_mode = os.stat(path).st_mode & 0o7777
    except FileNotFoundError:
        current_umask = os.umask(0)
        os.umask(current_umask)
        target_mode = 0o666 & ~current_umask
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=directory,
            prefix=f".{os.path.basename(path)}.",
            suffix=".candidate",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            os.chmod(temporary_path, target_mode)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def _release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _resolve_manager(value: str | None) -> str:
    if value:
        return value
    for candidate in ("mamba", "conda"):
        path = shutil.which(candidate)
        if path:
            return path
    raise CommandError("Could not find mamba or conda on PATH")


def _run_or_print(
    commands: Iterable[list[str]],
    *,
    dry_run: bool,
    out,
) -> None:
    commands = list(commands)
    from .ocr_backend_setup import (
        BackendSetupError,
        assert_command_plan_safe,
    )

    try:
        assert_command_plan_safe(commands)
    except BackendSetupError as exc:
        raise CommandError(str(exc)) from exc
    for command in commands:
        out(" ".join(_quote(arg) for arg in command))
        if dry_run:
            continue
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise CommandError(
                f"Command failed with exit status {completed.returncode}"
            )


def _quote(value: str) -> str:
    import shlex

    return shlex.quote(str(value))


def package_root() -> str:
    return str(Path(__file__).resolve().parents[3])
