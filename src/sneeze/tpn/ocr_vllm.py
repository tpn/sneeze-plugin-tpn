from __future__ import annotations

import inspect
import multiprocessing
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial

DEFAULT_OCR_MODEL = "deepseek-ai/DeepSeek-OCR"


@dataclass(frozen=True)
class OcrPageTask:
    image_path: str
    output_path: str
    text_source_path: str | None = None


@dataclass(frozen=True)
class OcrScanCandidate:
    image_path: str
    base: str
    text_path: str | None = None


def resolve_scan_workers(scan_parallelism: int | None) -> int:
    workers = int(scan_parallelism or 0)
    if workers <= 0:
        workers = min(os.cpu_count() or 1, 32)
    return workers


def collect_ocr_tasks(
    *,
    source_dir: str,
    dest_dir: str,
    text_source_dir: str | None,
    output_ext: str,
    force: bool,
    scan_parallelism: int,
    out: Callable[[str], None] | None = None,
) -> tuple[list[OcrPageTask], int]:
    candidates, pdf_count = collect_ocr_scan_candidates(
        source_dir=source_dir,
        dest_dir=dest_dir,
        text_source_dir=text_source_dir,
    )
    if not candidates:
        return [], pdf_count
    if out:
        label = "image" if len(candidates) == 1 else "images"
        out(
            f"Indexed {len(candidates)} page {label}; collecting OCR work "
            f"with {scan_parallelism} scan worker(s)..."
        )
    records = collect_ocr_scan_results(
        candidates,
        partial(
            ocr_task_record_from_candidate,
            dest_dir=os.path.abspath(dest_dir),
            output_ext=output_ext,
            force=force,
        ),
        max_workers=scan_parallelism,
    )
    tasks = []
    for record in records:
        if record is None:
            continue
        image_path, output_path, text_path = record
        tasks.append(
            OcrPageTask(
                image_path=image_path,
                output_path=output_path,
                text_source_path=text_path,
            )
        )
    return tasks, pdf_count


def collect_ocr_scan_candidates(
    *,
    source_dir: str,
    dest_dir: str,
    text_source_dir: str | None,
) -> tuple[list[OcrScanCandidate], int]:
    candidates = []
    pdf_dirs = set()
    dest_dir = os.path.abspath(dest_dir)
    source_dir = os.path.abspath(source_dir)
    for root, dirs, files in os.walk(source_dir):
        dirs.sort()
        root_abs = os.path.abspath(root)
        if dest_dir != source_dir and (
            root_abs == dest_dir or root_abs.startswith(dest_dir + os.sep)
        ):
            continue
        for filename in sorted(files):
            if not filename.lower().endswith(".png"):
                continue
            rel_root = os.path.relpath(root, source_dir)
            if rel_root == ".":
                rel_root = ""
            rel_path = (
                os.path.join(rel_root, filename)
                if rel_root
                else filename
            )
            base, _ = os.path.splitext(rel_path)
            text_path = None
            if text_source_dir:
                candidate = os.path.join(text_source_dir, base + ".txt")
                if os.path.isfile(candidate):
                    text_path = candidate
            candidates.append(
                OcrScanCandidate(
                    image_path=os.path.join(root, filename),
                    base=base,
                    text_path=text_path,
                )
            )
            pdf_dirs.add(rel_root if rel_root else ".")
    return candidates, max(len(pdf_dirs), 1 if candidates else 0)


def ocr_task_record_from_candidate(
    candidate: OcrScanCandidate,
    dest_dir: str,
    output_ext: str,
    force: bool,
) -> tuple[str, str, str | None] | None:
    output_path = os.path.join(dest_dir, f"{candidate.base}.{output_ext}")
    if not force and os.path.exists(output_path):
        return None
    return candidate.image_path, output_path, candidate.text_path


def collect_ocr_scan_results(
    candidates: list[OcrScanCandidate],
    worker_fn,
    *,
    max_workers: int,
    progress_desc: str = "Scanning page images",
):
    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover - optional dependency
        tqdm = None
    worker_count = int(max_workers or 1)
    if worker_count <= 0:
        worker_count = resolve_scan_workers(worker_count)
    if worker_count <= 1 or len(candidates) <= 1:
        results = []
        iterator = candidates
        progress = None
        if tqdm is not None:
            progress = tqdm(
                iterator,
                total=len(candidates),
                desc=progress_desc,
                unit="image",
            )
            iterator = progress
        try:
            for candidate in iterator:
                results.append(worker_fn(candidate))
        finally:
            if progress is not None:
                progress.close()
        return results
    chunksize = max(1, len(candidates) // (worker_count * 8))
    results = []
    progress = None
    if tqdm is not None:
        progress = tqdm(
            total=len(candidates),
            desc=progress_desc,
            unit="image",
        )
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            for result in executor.map(
                worker_fn,
                candidates,
                chunksize=chunksize,
            ):
                results.append(result)
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return results


def prepare_vllm_process_env(
    *,
    verbose: Callable[[str], None] | None = None,
) -> None:
    if "OPENBLAS_NUM_THREADS" not in os.environ:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        if verbose:
            verbose("Setting OPENBLAS_NUM_THREADS=1 for vLLM startup.")


def apply_vllm_compatibility_kwargs(
    llm_kwargs: dict,
    *,
    out: Callable[[str], None] | None = None,
    verbose: Callable[[str], None] | None = None,
) -> dict:
    llm_kwargs = dict(llm_kwargs)
    attention_config = llm_kwargs.get("attention_config")
    if isinstance(attention_config, dict):
        attention_config = dict(attention_config)
    else:
        attention_config = {}
    explicit_backend = (
        llm_kwargs.get("attention_backend") is not None
        or attention_config.get("backend") is not None
    )
    legacy_backend = os.environ.pop("VLLM_ATTENTION_BACKEND", None)
    if legacy_backend is not None and verbose:
        verbose(
            "Translating VLLM_ATTENTION_BACKEND into vLLM constructor kwargs."
        )
    if legacy_backend and not explicit_backend:
        llm_kwargs["attention_backend"] = legacy_backend
        explicit_backend = True
    legacy_disable_raw = os.environ.pop(
        "VLLM_DISABLE_FLASHINFER_PREFILL",
        None,
    )
    legacy_disable = coerce_env_bool(legacy_disable_raw)
    explicit_disable = (
        "disable_flashinfer_prefill" in attention_config
        or legacy_disable_raw is not None
    )
    if legacy_disable is not None:
        attention_config["disable_flashinfer_prefill"] = legacy_disable
        if verbose:
            verbose(
                "Translating VLLM_DISABLE_FLASHINFER_PREFILL into vLLM "
                "constructor kwargs."
            )
    capabilities = visible_gpu_compute_capabilities()
    if capabilities and min(capabilities) < 80:
        updated = []
        if not explicit_backend:
            llm_kwargs["attention_backend"] = "FLEX_ATTENTION"
            updated.append("attention_backend=FLEX_ATTENTION")
        if not explicit_disable:
            attention_config["disable_flashinfer_prefill"] = True
            updated.append("attention_config.disable_flashinfer_prefill=True")
        if updated and out:
            out(
                "Detected compute capability < 8.0; setting "
                f"{', '.join(updated)}."
            )
    if attention_config:
        llm_kwargs["attention_config"] = attention_config
    return llm_kwargs


def coerce_env_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def visible_gpu_compute_capabilities() -> list[int]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,uuid,pci.bus_id,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        capability = parse_compute_capability(parts[3])
        if capability is None:
            continue
        rows.append(
            {
                "index": parts[0],
                "uuid": parts[1].upper(),
                "capability": capability,
            }
        )
    if not rows:
        return []
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return [row["capability"] for row in rows]
    tokens = [token.strip() for token in visible_devices.split(",")]
    tokens = [token for token in tokens if token]
    if not tokens or tokens == ["-1"]:
        return []
    if all(token.isdigit() for token in tokens):
        wanted = set(tokens)
        capabilities = [
            row["capability"] for row in rows if row["index"] in wanted
        ]
        return capabilities or [row["capability"] for row in rows]
    if all(token.upper().startswith("GPU-") for token in tokens):
        wanted = {token.upper() for token in tokens}
        capabilities = [
            row["capability"] for row in rows if row["uuid"] in wanted
        ]
        return capabilities or [row["capability"] for row in rows]
    return [row["capability"] for row in rows]


def parse_compute_capability(value: str) -> int | None:
    match = re.match(r"^\s*(\d+)(?:\.(\d+))?\s*$", value or "")
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2) or "0")
    return (major * 10) + minor


class DeepSeekOcrOfflineProcessor:
    def __init__(
        self,
        *,
        model: str,
        output_format: str,
        prompt: str | None = None,
        prompt_mode: str | None = None,
        keep_grounding: bool = False,
        use_grounding: bool | None = None,
        batch_size: int | None = None,
        enable_prefix_caching: bool = False,
        enable_chunked_prefill: bool = False,
        max_model_len: int | None = None,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        attention_backend: str | None = None,
        mm_encoder_attn_backend: str | None = None,
        scheduler_delay_factor: float | None = None,
        tensor_parallel_size: int | None = None,
        gpu_memory_utilization: float | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        ngram_size: int = 10,
        window_size: int = 50,
        whitelist_token_ids: list[int] | None = None,
        no_tqdm: bool = False,
        out: Callable[[str], None] | None = None,
        verbose: Callable[[str], None] | None = None,
    ) -> None:
        self.model = model
        self.output_format = output_format
        self.prompt = prompt
        self.prompt_mode = prompt_mode
        self.keep_grounding = keep_grounding
        if use_grounding is None:
            mode = (prompt_mode or "").lower().strip()
            use_grounding = mode == "plain_ocr"
        self.use_grounding = use_grounding
        self.batch_size = batch_size
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.attention_backend = attention_backend
        self.mm_encoder_attn_backend = mm_encoder_attn_backend
        self.scheduler_delay_factor = scheduler_delay_factor
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or [128821, 128822]
        self.no_tqdm = no_tqdm
        self._out = out
        self._verbose = verbose
        self._tqdm = None

    def prepare_engine(self):
        return self._build_llm_and_params()

    def run(
        self,
        tasks: Iterable[OcrPageTask],
        *,
        llm=None,
        sampling_params=None,
    ) -> None:
        tasks = list(tasks)
        if not tasks:
            self._log("No OCR tasks to run.")
            return
        created_engine = llm is None or sampling_params is None
        if created_engine:
            llm, sampling_params = self._build_llm_and_params()
        batch_size = self._resolve_batch_size()
        self._log(
            f"Processing {len(tasks)} page(s), batch size {batch_size}."
        )
        created_progressbar = False
        if self._tqdm is None:
            self.init_progressbar(total=len(tasks))
            created_progressbar = True
        try:
            for batch in self._batch(tasks, batch_size):
                contents = self._generate_batch_results(
                    llm,
                    sampling_params,
                    batch,
                )
                for content, task in zip(contents, batch, strict=False):
                    if content is None:
                        continue
                    if not self.keep_grounding:
                        content = self._strip_grounding(content)
                    self._write_output(task.output_path, content)
                self.update_progressbar(len(batch))
        finally:
            if created_progressbar:
                self.close_progressbar()
            if created_engine:
                self.shutdown_engine(llm)

    def init_progressbar(self, total: int | None = None) -> None:
        if self.no_tqdm or self._tqdm is not None:
            return
        try:
            from tqdm import tqdm
        except ImportError:
            return
        self._tqdm = tqdm(total=total, desc="OCR pages", unit="page")

    def update_progressbar(self, amount: int = 1) -> None:
        if self._tqdm is not None:
            self._tqdm.update(amount)

    def close_progressbar(self) -> None:
        if self._tqdm is None:
            return
        self._tqdm.close()
        self._tqdm = None

    def _build_llm_and_params(self):
        prepare_vllm_process_env(verbose=self._verbose)
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "vLLM is not installed; install an OCR environment first."
            ) from exc
        try:
            from vllm.model_executor.models.deepseek_ocr import (
                NGramPerReqLogitsProcessor,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "vLLM does not expose DeepSeek OCR helpers; check the vLLM "
                "installation."
            ) from exc
        llm_kwargs = {
            "model": self.model,
            "enable_prefix_caching": self.enable_prefix_caching,
            "mm_processor_cache_gb": 0,
            "disable_mm_preprocessor_cache": True,
            "logits_processors": [NGramPerReqLogitsProcessor],
            "trust_remote_code": True,
        }
        self._maybe_add(
            llm_kwargs,
            "enable_chunked_prefill",
            self.enable_chunked_prefill,
        )
        self._maybe_add(llm_kwargs, "max_model_len", self.max_model_len)
        self._maybe_add(
            llm_kwargs,
            "max_num_batched_tokens",
            self.max_num_batched_tokens,
        )
        self._maybe_add(llm_kwargs, "max_num_seqs", self.max_num_seqs)
        self._maybe_add(
            llm_kwargs,
            "attention_backend",
            self.attention_backend,
        )
        self._maybe_add(
            llm_kwargs,
            "mm_encoder_attn_backend",
            self.mm_encoder_attn_backend,
        )
        self._maybe_add(
            llm_kwargs,
            "scheduler_delay_factor",
            self.scheduler_delay_factor,
        )
        self._maybe_add(
            llm_kwargs,
            "tensor_parallel_size",
            self.tensor_parallel_size,
        )
        self._maybe_add(
            llm_kwargs,
            "gpu_memory_utilization",
            self.gpu_memory_utilization,
        )
        llm_kwargs = apply_vllm_compatibility_kwargs(
            llm_kwargs,
            out=self._out,
            verbose=self._verbose,
        )
        llm_kwargs = self._filter_llm_kwargs(llm_kwargs)
        llm = LLM(**llm_kwargs)
        sampling_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        extra_args = {
            "ngram_size": self.ngram_size,
            "window_size": self.window_size,
            "whitelist_token_ids": self.whitelist_token_ids,
        }
        params = self._filter_kwargs(SamplingParams.__init__, sampling_kwargs)
        if self._accepts_kwargs(SamplingParams.__init__):
            params["extra_args"] = extra_args
        elif "extra_args" in inspect.signature(SamplingParams).parameters:
            params["extra_args"] = extra_args
        sampling_params = SamplingParams(**params)
        return llm, sampling_params

    def _resolve_batch_size(self) -> int:
        if self.batch_size and self.batch_size > 0:
            return self.batch_size
        if self.max_num_seqs and self.max_num_seqs > 0:
            return self.max_num_seqs
        return 8

    def _batch(self, tasks: list[OcrPageTask], batch_size: int):
        for index in range(0, len(tasks), batch_size):
            yield tasks[index : index + batch_size]

    def _generate_batch_results(
        self,
        llm,
        sampling_params,
        batch: list[OcrPageTask],
    ) -> list[str | None]:
        results: list[str | None] = [None] * len(batch)
        prepared_tasks: list[tuple[int, OcrPageTask]] = []
        inputs = []
        images = []
        for index, task in enumerate(batch):
            try:
                image = self._open_image(task.image_path)
            except Exception as exc:
                self._log_skip(task, exc, reason="unable to load image")
                continue
            prompt = self._build_prompt(task.text_source_path)
            inputs.append(
                {"prompt": prompt, "multi_modal_data": {"image": image}}
            )
            images.append(image)
            prepared_tasks.append((index, task))
        if not prepared_tasks:
            return results
        try:
            outputs = self._generate(
                llm,
                inputs,
                sampling_params=sampling_params,
            )
            for (index, _task), output in zip(
                prepared_tasks,
                outputs,
                strict=False,
            ):
                results[index] = output.outputs[0].text
            return results
        except Exception as exc:
            if len(prepared_tasks) == 1:
                _index, task = prepared_tasks[0]
                self._log_skip(task, exc, reason="OCR failed")
                return results
            self._log(
                "Batch OCR failed for "
                f"{len(prepared_tasks)} page(s); retrying individually: "
                f"{self._format_exception(exc)}"
            )
        finally:
            for image in images:
                try:
                    image.close()
                except Exception:
                    pass
        for index, task in prepared_tasks:
            results[index] = self._generate_batch_results(
                llm,
                sampling_params,
                [task],
            )[0]
        return results

    def _open_image(self, path: str):
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Pillow is required to load page images."
            ) from exc
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(path)
        image.load()
        if image.mode not in ("RGB", "RGBA", "LA"):
            converted = image.convert("RGB")
            image.close()
            return converted
        if image.mode == "RGB":
            return image
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image.close()
        return background

    def _build_prompt(self, text_source_path: str | None) -> str:
        prompt = self.prompt or self._default_prompt()
        return self._normalize_prompt(prompt)

    def _default_prompt(self) -> str:
        prompt_mode = (self.prompt_mode or "").lower().strip()
        if prompt_mode == "plain_ocr":
            instruction = "Free OCR."
        elif prompt_mode == "html" or self.output_format == "html":
            instruction = "Convert the document to HTML."
        else:
            instruction = "Convert the document to markdown."
        return f"<image>\n{instruction}"

    def _normalize_prompt(self, prompt: str) -> str:
        prompt = prompt.strip()
        if "<image>" not in prompt:
            prompt = f"<image>\n{prompt}"
        if self.use_grounding and "<|grounding|>" not in prompt:
            image_token = "<image>"
            if prompt.startswith(image_token):
                remainder = prompt[len(image_token) :].lstrip("\n")
                prompt = (
                    f"{image_token}\n<|grounding|>{remainder}"
                    if remainder
                    else f"{image_token}\n<|grounding|>"
                )
            else:
                prompt = f"<image>\n<|grounding|>{prompt}"
        return prompt

    def _write_output(self, path: str, content: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content.strip() + "\n")

    def _strip_grounding(self, content: str) -> str:
        content = content.replace("<|grounding|>", "")
        pattern = re.compile(
            r"<\|ref\|>(?P<label>.*?)<\|/ref\|>\s*"
            r"<\|det\|>\s*\[.*?\]\s*<\|/det\|>",
            re.DOTALL,
        )
        return pattern.sub(
            lambda match: match.group("label"),
            content,
        ).strip()

    def _filter_kwargs(self, fn, kwargs):
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return kwargs
        if self._accepts_kwargs(fn, signature=signature):
            return kwargs
        return {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }

    def _filter_llm_kwargs(self, kwargs):
        try:
            from vllm.engine.arg_utils import EngineArgs
        except Exception:  # pragma: no cover - depends on vLLM version
            try:
                from vllm import LLM
            except Exception:
                return kwargs
            return self._filter_kwargs(LLM.__init__, kwargs)
        return self._filter_kwargs(EngineArgs.__init__, kwargs)

    def _maybe_add(self, kwargs: dict, key: str, value) -> None:
        if value is not None:
            kwargs[key] = value

    def _accepts_kwargs(self, fn, signature=None) -> bool:
        if signature is None:
            try:
                signature = inspect.signature(fn)
            except (TypeError, ValueError):
                return False
        return any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )

    def _generate(self, llm, prompts, *, sampling_params):
        generate_kwargs = {}
        try:
            signature = inspect.signature(llm.generate)
            if "use_tqdm" in signature.parameters:
                generate_kwargs["use_tqdm"] = False
        except (TypeError, ValueError):
            generate_kwargs = {}
        return llm.generate(
            prompts,
            sampling_params=sampling_params,
            **generate_kwargs,
        )

    def _log_skip(
        self,
        task: OcrPageTask,
        exc: Exception,
        *,
        reason: str,
    ) -> None:
        self._log(
            f"Skipping {task.image_path}; {reason}: "
            f"{self._format_exception(exc)}"
        )

    def _format_exception(self, exc: Exception) -> str:
        message = str(exc).strip()
        return message or exc.__class__.__name__

    def shutdown_engine(self, llm) -> None:
        if llm is None:
            return
        llm_engine = getattr(llm, "llm_engine", None)
        candidates = (
            ("llm.shutdown", getattr(llm, "shutdown", None)),
            ("llm_engine.shutdown", getattr(llm_engine, "shutdown", None)),
            (
                "llm_engine.engine_core.shutdown",
                getattr(
                    getattr(llm_engine, "engine_core", None),
                    "shutdown",
                    None,
                ),
            ),
        )
        for name, fn in candidates:
            if not callable(fn):
                continue
            try:
                fn()
                self._vlog(f"vLLM cleanup via {name}.")
            except Exception as exc:
                self._vlog(
                    f"Best-effort vLLM cleanup failed via {name}: {exc}"
                )
            return

    def _log(self, message: str) -> None:
        if self._out:
            self._out(message)

    def _vlog(self, message: str) -> None:
        if self._verbose:
            self._verbose(message)
