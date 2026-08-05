from __future__ import annotations

import gc
import hashlib
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
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
    DEFAULT_MARKDOWN_WINDOW_SIZE,
    DEFAULT_OCR_MODEL,
    DEFAULT_OCR_MODEL_REVISION,
    DEFAULT_OCR_WINDOW_SIZE,
)

DEFAULT_OCR_ENV_NAME = "sneeze-ocr-vllm"
DEFAULT_TORCH_BACKEND = "cu130"
DEFAULT_PYTHON_VERSION = "3.12"
DEFAULT_CUDA_130_VLLM_WHEEL_INDEX = "https://wheels.vllm.ai/nightly/cu130/"


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
            "Parallel worker count; 0 means CPU count. "
            "[default: %default]"
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
    """OCR page images, optionally rendering source PDFs first.

    Pass `--pdf-to-pages` to render PDFs from `--source-dir` into the
    effective destination before OCR; it defaults to `--source-dir` when
    `--dest-dir` is omitted. Otherwise render pages separately with
    `sne pdf-to-pages`. Create a vLLM env with
    `sne deepseek-ocr-create-env`, cache the model with
    `sne deepseek-ocr-download-model`, and run this command inside that env.
    Use `--prompt-mode full` to write `.ocr`, `.md`, and `.json` page
    bundles from existing page images. HTML output is a separate
    `--output-format html` mode, not part of `full`.
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
            "PDF-render worker count; 0 means CPU count. "
            "[default: %default]"
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

    output_format = None
    _output_format = None

    class OutputFormatArg(StringInvariant):
        _help = "Output format: md, html, or ocr. [default: %default]"
        _mandatory = False
        _default = "md"

    model = None
    _model = None

    class ModelArg(StringInvariant):
        _help = "DeepSeek OCR model name or local path. [default: %default]"
        _mandatory = False
        _default = DEFAULT_OCR_MODEL

    model_revision = None
    _model_revision = None

    class ModelRevisionArg(StringInvariant):
        _help = (
            "Immutable Hugging Face model revision. [default: %default]"
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
        _help = "Provider lane for full-mode Markdown. [default: %default]"
        _mandatory = False
        _default = "best"

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
        dest_dir = os.path.abspath(
            dest_option or source_dir
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
        text_source_dir = _option_value(
            self,
            "text_source_dir",
            private_first=True,
        )
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
            )
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
        try:
            window_size = self._window_size or DEFAULT_OCR_WINDOW_SIZE
            if prompt_mode == "markdown":
                window_size = max(
                    window_size,
                    DEFAULT_MARKDOWN_WINDOW_SIZE,
                )
            processor = DeepSeekOcrOfflineProcessor(
                model=_option_value(self, "model", DEFAULT_OCR_MODEL),
                model_revision=_option_value(
                    self,
                    "model_revision",
                    DEFAULT_OCR_MODEL_REVISION,
                ),
                output_format=output_format,
                prompt=_option_value(self, "prompt"),
                prompt_mode=(
                    "plain_ocr" if prompt_mode == "full" else prompt_mode
                ),
                keep_grounding=bool(self.keep_grounding),
                use_grounding=False if self.no_grounding else None,
                batch_size=self._batch_size,
                enable_chunked_prefill=bool(self.enable_chunked_prefill),
                max_model_len=self._max_model_len,
                max_num_batched_tokens=self._max_num_batched_tokens,
                max_num_seqs=self._max_num_seqs,
                attention_backend=_option_value(self, "attention_backend"),
                mm_encoder_attn_backend=_option_value(
                    self,
                    "mm_encoder_attn_backend",
                ),
                scheduler_delay_factor=scheduler_delay_factor,
                tensor_parallel_size=self._tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                temperature=temperature,
                max_tokens=self._max_tokens or 8192,
                ngram_size=self._ngram_size or 10,
                window_size=window_size,
                out=self._out,
                verbose=self._verbose,
            )
            if prompt_mode == "full":
                self._run_full_mode(
                    processor,
                    items,
                    markdown_provider=markdown_provider,
                )
            else:
                processor.run(tasks)
        except RuntimeError as exc:
            raise CommandError(str(exc)) from exc

    def _prepare_markdown_provider(
        self,
        *,
        prompt_mode: str,
        source_dir: str,
        dest_dir: str,
    ) -> dict | None:
        spec_path = _option_value(self, "markdown_provider_spec")
        run_dir = _option_value(self, "markdown_provider_run_dir")
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
        from .ocr_benchmark import PROTECTED_CORPUS_ROOTS
        from .ocr_provider import (
            ProviderError,
            load_local_provider_spec,
            validate_provider_run_dir,
        )

        lane = _option_value(self, "markdown_provider_lane", "best")
        if not os.path.isabs(run_dir):
            raise CommandError(
                "Markdown provider evidence directory must be absolute"
            )
        candidate_dir = os.path.realpath(run_dir)
        for label, root in (
            ("source", source_dir),
            ("destination", dest_dir),
        ):
            if _paths_overlap(candidate_dir, os.path.abspath(root)):
                raise CommandError(
                    "Markdown provider evidence directory must be "
                    f"separate from the OCR {label} directory"
                )
        for protected_root in PROTECTED_CORPUS_ROOTS:
            if _paths_overlap(candidate_dir, str(protected_root)):
                raise CommandError(
                    "Markdown provider evidence directory must be "
                    "separate from protected canonical corpora"
                )
        try:
            resolved_spec_path = (
                Path(spec_path).expanduser().resolve(strict=True)
            )
            spec = load_local_provider_spec(resolved_spec_path)
            if lane not in spec["lanes"]:
                raise ProviderError(
                    f"provider does not define lane: {lane}"
                )
            evidence_dir = validate_provider_run_dir(run_dir)
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
            if ocr_tasks:
                processor.output_format = "ocr"
                processor.prompt_mode = "plain_ocr"
                processor.use_grounding = True
                processor.keep_grounding = True
                processor.run(
                    ocr_tasks,
                    llm=llm,
                    sampling_params=sampling_params,
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
                    if not metadata or metadata.get("status") != "completed":
                        flags.append(f"{label}_generation_failed")
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
                    page_failures.setdefault(
                        item.markdown_path, []
                    ).extend(flags)
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
            batch = run_local_provider_batch(
                spec_path=provider["spec_path"],
                lane=provider["lane"],
                run_dir=provider["run_dir"],
                pages=pages,
                out=self._out,
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
            if not result.get("valid"):
                flags.append("provider_response_invalid")
            if not isinstance(output_text, str) or not output_text.strip():
                flags.append("provider_markdown_empty")
                output_text = (
                    output_text if isinstance(output_text, str) else ""
                )
            termination = metadata.get("termination") or {}
            if termination.get("max_tokens") is True:
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
                "persisted_output_sha256": file_sha256(
                    item.markdown_path
                ),
            }
            if flags:
                metadata["status"] = "error"
                failures.append((item.markdown_path, flags))
            processor.generation_metadata[item.markdown_path] = metadata
            processor.update_progressbar(1)

        for page_id, item in item_by_id.items():
            if page_id in returned_ids:
                continue
            flags = ["provider_result_missing"]
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
        if batch.get("protocol_errors"):
            failures.append(
                (
                    provider["run_dir"],
                    [
                        "provider_batch_protocol_error: " + error
                        for error in batch["protocol_errors"]
                    ],
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
            "Permit a separately approved external plan. "
            "[default: %default]"
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
        if external_plan and (
            lanes_value is None or repeats_value is None
        ):
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
                    lane["lane_id"]
                    for lane in external_plan_record["lanes"]
                )
            if repeats_value is None:
                repeats_value = external_plan_record["execution"][
                    "repeats"
                ]
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
            "Approval lifetime, capped by plan expiry. "
            "[default: %default]"
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
        _help = "DeepSeek only: install this plugin checkout editable."
        _mandatory = False

    package_spec = None
    _package_spec = None

    class PackageSpecArg(StringInvariant):
        _help = "DeepSeek only: plugin package spec to install."
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
            manager = resolve_manager(
                _option_value(self, "manager")
            )
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
        _help = "Install this plugin editable from the given repo root."
        _mandatory = False

    package_spec = None
    _package_spec = None

    class PackageSpecArg(StringInvariant):
        _help = "Package spec to install if --editable-root is not set."
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
                self._revision
                or self.revision
                or DEFAULT_OCR_MODEL_REVISION
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
    names = (
        (f"_{name}", name)
        if private_first
        else (name, f"_{name}")
    )
    for attr_name in names:
        value = getattr(command, attr_name, None)
        if value not in (None, ""):
            return value
    return default


def _maybe_float(value, default=None):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CommandError(f"Expected a float, got {value!r}") from exc


def _paths_overlap(left: str, right: str) -> bool:
    left = os.path.realpath(left)
    right = os.path.realpath(right)
    try:
        common = os.path.commonpath((left, right))
    except ValueError:
        return False
    return common in {left, right}


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
