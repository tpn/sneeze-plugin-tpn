from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
import json
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from functools import partial

DEFAULT_OCR_MODEL = "deepseek-ai/DeepSeek-OCR"
DEFAULT_OCR_MODEL_REVISION = "9f30c71f441d010e5429c532364a86705536c53a"
DEFAULT_OCR_WINDOW_SIZE = 50
DEFAULT_MARKDOWN_WINDOW_SIZE = 128
DEFAULT_DEEPSEEK_MOE_BACKEND = "auto"
QUALIFIED_DEEPSEEK_MOE_BACKEND = "triton"
DEEPSEEK_MOE_BACKENDS = frozenset({"auto", "triton"})
DEEPSEEK_MAX_JOBS = 1
DEEPSEEK_FLASHINFER_NVCC_THREADS = 1
DEEPSEEK_JIT_ENV_NAMES = ("MAX_JOBS", "FLASHINFER_NVCC_THREADS")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_STRUCTURED_EXECUTION_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "pipeline_id",
        "pipeline_definition_sha256",
        "profile_id",
        "profile_sha256",
        "qualification_state",
        "allow_provisional_profile",
        "memory_tier",
        "memory_budget_bytes",
        "reserve_bytes",
        "hardware_selector",
        "match",
        "stage_settings",
        "attention_backend",
        "installation_record_sha256",
        "deepseek_snapshot",
        "receipt_sha256",
    }
)


def validate_deepseek_moe_backend(moe_backend: object) -> str:
    """Validate the modeled vLLM MoE implementation selector."""

    if not isinstance(moe_backend, str) or (
        moe_backend not in DEEPSEEK_MOE_BACKENDS
    ):
        raise ValueError(
            "DeepSeek moe_backend must be one of: "
            + ", ".join(sorted(DEEPSEEK_MOE_BACKENDS))
        )
    return moe_backend


def deepseek_jit_environment() -> dict[str, str]:
    """Return the fixed implementation-bound JIT environment."""

    return {
        "MAX_JOBS": str(DEEPSEEK_MAX_JOBS),
        "FLASHINFER_NVCC_THREADS": str(
            DEEPSEEK_FLASHINFER_NVCC_THREADS
        ),
    }


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
    refresh_output: bool = False
    refresh_metadata: bool = False
    verify_input_freshness: bool = False


@dataclass(frozen=True)
class OcrFullItem:
    image_path: str
    base: str
    ocr_path: str
    markdown_path: str
    json_path: str
    text_path: str | None = None
    need_ocr: bool = False
    need_markdown: bool = False
    need_json: bool = False
    image_dims: dict[str, int] | None = None
    image_sha256: str | None = None


class StructuredOutputReuseError(RuntimeError):
    """Raised when an existing bundle cannot prove structured provenance."""


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
    allowed_image_paths: Iterable[str] | None = None,
    refresh_image_paths: Iterable[str] | None = None,
    verify_input_freshness: bool = False,
) -> tuple[list[OcrPageTask], int]:
    candidates, pdf_count = collect_ocr_scan_candidates(
        source_dir=source_dir,
        dest_dir=dest_dir,
        text_source_dir=text_source_dir,
        allowed_image_paths=allowed_image_paths,
        refresh_image_paths=refresh_image_paths,
        verify_input_freshness=verify_input_freshness,
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


def collect_ocr_full_items(
    *,
    source_dir: str,
    dest_dir: str,
    text_source_dir: str | None,
    force: bool,
    scan_parallelism: int,
    out: Callable[[str], None] | None = None,
    allowed_image_paths: Iterable[str] | None = None,
    refresh_image_paths: Iterable[str] | None = None,
    refresh_metadata_image_paths: Iterable[str] | None = None,
    verify_input_freshness: bool = False,
    expected_execution_receipt: Mapping | None = None,
) -> tuple[list[OcrFullItem], int]:
    candidates, pdf_count = collect_ocr_scan_candidates(
        source_dir=source_dir,
        dest_dir=dest_dir,
        text_source_dir=text_source_dir,
        allowed_image_paths=allowed_image_paths,
        refresh_image_paths=refresh_image_paths,
        refresh_metadata_image_paths=refresh_metadata_image_paths,
        verify_input_freshness=verify_input_freshness,
    )
    if not candidates:
        return [], pdf_count
    if out:
        label = "image" if len(candidates) == 1 else "images"
        out(
            f"Indexed {len(candidates)} page {label}; collecting full OCR "
            f"work with {scan_parallelism} scan worker(s)..."
        )
    records = collect_ocr_scan_results(
        candidates,
        partial(
            ocr_full_item_from_candidate,
            dest_dir=os.path.abspath(dest_dir),
            force=force,
            expected_execution_receipt=(
                dict(expected_execution_receipt)
                if expected_execution_receipt is not None
                else None
            ),
        ),
        max_workers=scan_parallelism,
    )
    return [record for record in records if record is not None], pdf_count


def collect_ocr_scan_candidates(
    *,
    source_dir: str,
    dest_dir: str,
    text_source_dir: str | None,
    allowed_image_paths: Iterable[str] | None = None,
    refresh_image_paths: Iterable[str] | None = None,
    refresh_metadata_image_paths: Iterable[str] | None = None,
    verify_input_freshness: bool = False,
) -> tuple[list[OcrScanCandidate], int]:
    candidates = []
    pdf_dirs = set()
    dest_dir = os.path.abspath(dest_dir)
    source_dir = os.path.abspath(source_dir)
    allowed_paths = (
        {os.path.abspath(path) for path in allowed_image_paths}
        if allowed_image_paths is not None
        else None
    )
    refresh_paths = {
        os.path.abspath(path) for path in (refresh_image_paths or ())
    }
    refresh_metadata_paths = {
        os.path.abspath(path)
        for path in (refresh_metadata_image_paths or ())
    }
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
            image_path = os.path.abspath(os.path.join(root, filename))
            if allowed_paths is not None and image_path not in allowed_paths:
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
                    image_path=image_path,
                    base=base,
                    text_path=text_path,
                    refresh_output=image_path in refresh_paths,
                    refresh_metadata=(
                        image_path in refresh_metadata_paths
                    ),
                    verify_input_freshness=verify_input_freshness,
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
    refresh_output = force or candidate.refresh_output
    if (
        candidate.verify_input_freshness
        and path_is_newer(candidate.image_path, output_path)
    ):
        refresh_output = True
    if not refresh_output and os.path.exists(output_path):
        return None
    return candidate.image_path, output_path, candidate.text_path


def ocr_full_item_from_candidate(
    candidate: OcrScanCandidate,
    dest_dir: str,
    force: bool,
    expected_execution_receipt: dict | None = None,
) -> OcrFullItem | None:
    ocr_path = os.path.join(dest_dir, f"{candidate.base}.ocr")
    markdown_path = os.path.join(dest_dir, f"{candidate.base}.md")
    json_path = os.path.join(dest_dir, f"{candidate.base}.json")
    refresh_output = force or candidate.refresh_output
    refresh_metadata = candidate.refresh_metadata
    if candidate.verify_input_freshness:
        if any(
            path_is_newer(candidate.image_path, output_path)
            for output_path in (ocr_path, markdown_path, json_path)
        ):
            refresh_output = True
        if candidate.text_path and path_is_newer(
            candidate.text_path,
            json_path,
        ):
            refresh_metadata = True
    need_ocr = refresh_output or not os.path.exists(ocr_path)
    need_markdown = refresh_output or not os.path.exists(markdown_path)
    need_json = (
        refresh_output
        or refresh_metadata
        or not os.path.exists(json_path)
    )
    image_sha256 = None
    if expected_execution_receipt is not None:
        image_sha256 = file_sha256(candidate.image_path)
        reused_stages = []
        if not need_ocr:
            reused_stages.append(("grounded_ocr", "ocr", ocr_path))
        if not need_markdown:
            reused_stages.append(("markdown", "markdown", markdown_path))
        if reused_stages:
            _validate_structured_output_reuse(
                json_path=json_path,
                image_sha256=image_sha256,
                expected_execution_receipt=expected_execution_receipt,
                stages=reused_stages,
            )
    if (
        not force
        and not (need_ocr or need_markdown or need_json)
        and persisted_markdown_quality_status(json_path) == "fail"
    ):
        need_json = True
    if not (need_ocr or need_markdown or need_json):
        return None
    if image_sha256 is None:
        image_sha256 = file_sha256(candidate.image_path)
    return OcrFullItem(
        image_path=candidate.image_path,
        base=candidate.base,
        ocr_path=ocr_path,
        markdown_path=markdown_path,
        json_path=json_path,
        text_path=candidate.text_path,
        need_ocr=need_ocr,
        need_markdown=need_markdown,
        need_json=need_json,
        image_dims=ocr_image_dims(candidate.image_path),
        image_sha256=image_sha256,
    )


def _canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _execution_receipt_is_valid(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    receipt = dict(value)
    if set(receipt) != _STRUCTURED_EXECUTION_RECEIPT_KEYS:
        return False
    digest = receipt.pop("receipt_sha256", None)
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        return False
    profile_id = receipt.get("profile_id")
    qualification_state = receipt.get("qualification_state")
    allow_provisional = receipt.get("allow_provisional_profile")
    memory_tier = receipt.get("memory_tier")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "ocr_structured_execution"
        or receipt.get("pipeline_id") != "local-mineru25"
        or not isinstance(profile_id, str)
        or not re.fullmatch(
            r"[a-z0-9]+(?:[a-z0-9._-]*[a-z0-9])?", profile_id
        )
        or not _SHA256_RE.fullmatch(str(receipt.get("profile_sha256", "")))
        or qualification_state not in {"provisional", "qualified"}
        or not isinstance(allow_provisional, bool)
        or (qualification_state == "provisional" and not allow_provisional)
        or memory_tier not in {"25", "50", "max"}
        or not _positive_receipt_integer(receipt.get("memory_budget_bytes"))
        or not _nonnegative_receipt_integer(receipt.get("reserve_bytes"))
    ):
        return False
    try:
        from .ocr_profiles import get_pipeline_definition

        pipeline = get_pipeline_definition("local-mineru25")
    except (ImportError, RuntimeError, ValueError):
        return False
    if (
        not _SHA256_RE.fullmatch(
            str(receipt.get("pipeline_definition_sha256", ""))
        )
        or receipt["pipeline_definition_sha256"]
        != pipeline["definition_sha256"]
        or not _structured_receipt_hardware_is_valid(
            receipt.get("hardware_selector")
        )
        or not _structured_receipt_match_is_valid(receipt)
        or not _structured_receipt_settings_are_valid(
            receipt.get("stage_settings")
        )
        or not _structured_receipt_attention_is_valid(
            receipt.get("attention_backend")
        )
        or not _structured_receipt_installations_are_valid(
            receipt.get("installation_record_sha256")
        )
        or not _structured_receipt_snapshot_is_valid(
            receipt.get("deepseek_snapshot")
        )
    ):
        return False
    try:
        return digest == _canonical_json_sha256(receipt)
    except (TypeError, ValueError):
        return False


def _positive_receipt_integer(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def _nonnegative_receipt_integer(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _structured_receipt_hardware_is_valid(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "vendor",
        "product_name",
        "pci_device_id",
        "compute_capability",
        "memory_model",
        "total_memory_bytes",
    }:
        return False
    capability = value.get("compute_capability")
    pci_device_id = value.get("pci_device_id")
    return (
        value.get("schema_version") == 1
        and value.get("vendor") == "nvidia"
        and isinstance(value.get("product_name"), str)
        and bool(value["product_name"].strip())
        and (pci_device_id is None or isinstance(pci_device_id, str))
        and isinstance(capability, list)
        and len(capability) == 2
        and all(_nonnegative_receipt_integer(item) for item in capability)
        and value.get("memory_model")
        in {"dedicated", "unified", "unknown"}
        and _positive_receipt_integer(value.get("total_memory_bytes"))
    )


def _structured_receipt_match_is_valid(receipt: Mapping) -> bool:
    value = receipt.get("match")
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "profile_id",
        "pipeline_id",
        "memory_tier",
        "matched",
        "reasons",
        "profile_sha256",
    }:
        return False
    return (
        value.get("schema_version") == 1
        and value.get("profile_id") == receipt.get("profile_id")
        and value.get("pipeline_id") == receipt.get("pipeline_id")
        and value.get("memory_tier") == receipt.get("memory_tier")
        and value.get("matched") is True
        and value.get("reasons") == []
        and value.get("profile_sha256") == receipt.get("profile_sha256")
    )


def _structured_receipt_settings_are_valid(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"grounded_ocr", "markdown"}
        and all(
            isinstance(value[stage], Mapping) and bool(value[stage])
            for stage in ("grounded_ocr", "markdown")
        )
    )


def _structured_receipt_attention_is_valid(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value)
        == {"requested", "qualified_selected", "selected_verification"}
        and isinstance(value.get("requested"), str)
        and bool(value["requested"].strip())
        and isinstance(value.get("qualified_selected"), str)
        and bool(value["qualified_selected"].strip())
        and value.get("selected_verification")
        == "deferred_until_engine_initialization"
    )


def _structured_receipt_installations_are_valid(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"deepseek-vllm", "mineru25-pro-2605"}
        and all(
            isinstance(digest, str) and bool(_SHA256_RE.fullmatch(digest))
            for digest in value.values()
        )
    )


def _structured_receipt_snapshot_is_valid(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"revision", "files_sha256"}
        and isinstance(value.get("revision"), str)
        and bool(re.fullmatch(r"[0-9a-f]{40}", value["revision"]))
        and isinstance(value.get("files_sha256"), str)
        and bool(_SHA256_RE.fullmatch(value["files_sha256"]))
    )


def _structured_reuse_error(
    json_path: str,
    stage_label: str,
    reason: str,
) -> StructuredOutputReuseError:
    return StructuredOutputReuseError(
        f"Refusing to reuse structured OCR bundle {json_path} stage "
        f"{stage_label}: {reason}. Use a fresh --dest-dir or deliberately "
        "rerun with --force."
    )


def _validate_structured_output_reuse(
    *,
    json_path: str,
    image_sha256: str | None,
    expected_execution_receipt: Mapping,
    stages: Iterable[tuple[str, str, str]],
) -> None:
    """Require exact, successful provenance before reusing stage outputs."""

    stage_records = list(stages)
    stage_label = stage_records[0][0] if stage_records else "bundle"
    if not _execution_receipt_is_valid(expected_execution_receipt):
        raise _structured_reuse_error(
            json_path,
            stage_label,
            "the selected execution receipt is invalid",
        )
    try:
        with open(json_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError) as exc:
        raise _structured_reuse_error(
            json_path,
            stage_label,
            "the existing JSON provenance is absent or malformed",
        ) from exc
    if not isinstance(payload, Mapping):
        raise _structured_reuse_error(
            json_path,
            stage_label,
            "the existing JSON provenance is not an object",
        )
    if not isinstance(image_sha256, str) or payload.get(
        "image_sha256"
    ) != image_sha256:
        raise _structured_reuse_error(
            json_path,
            stage_label,
            "the bundle image SHA-256 does not match the current image",
        )
    generation_metadata = payload.get("generation_metadata")
    if not isinstance(generation_metadata, Mapping):
        raise _structured_reuse_error(
            json_path,
            stage_label,
            "generation metadata is missing",
        )
    expected_receipt = dict(expected_execution_receipt)
    for stage_label, metadata_key, output_path in stage_records:
        metadata = generation_metadata.get(metadata_key)
        if not isinstance(metadata, Mapping):
            raise _structured_reuse_error(
                json_path,
                stage_label,
                "stage generation metadata is missing",
            )
        observed_receipt = metadata.get("execution_profile")
        if (
            not _execution_receipt_is_valid(observed_receipt)
            or dict(observed_receipt) != expected_receipt
        ):
            raise _structured_reuse_error(
                json_path,
                stage_label,
                "the recorded execution profile or memory tier differs",
            )
        stage_image = metadata.get("image")
        if not isinstance(stage_image, Mapping) or stage_image.get(
            "sha256"
        ) != image_sha256:
            raise _structured_reuse_error(
                json_path,
                stage_label,
                "the stage image SHA-256 does not match the current image",
            )
        if (
            metadata.get("status") != "completed"
            or metadata.get("finish_reason") != "stop"
        ):
            raise _structured_reuse_error(
                json_path,
                stage_label,
                "the prior stage did not finish successfully",
            )
        validation = metadata.get("validation")
        if (
            not isinstance(validation, Mapping)
            or validation.get("status") != "pass"
            or validation.get("flags") != []
        ):
            raise _structured_reuse_error(
                json_path,
                stage_label,
                "the prior stage did not complete validation successfully",
            )
        termination = metadata.get("termination")
        if (
            not isinstance(termination, Mapping)
            or termination.get("eos") is not True
            or termination.get("length") is not False
            or termination.get("max_tokens") is not False
        ):
            raise _structured_reuse_error(
                json_path,
                stage_label,
                "the prior stage was not EOS-terminated below max_tokens",
            )
        promotion = metadata.get("promotion")
        candidate_sha256 = (
            promotion.get("candidate_sha256")
            if isinstance(promotion, Mapping)
            else None
        )
        persisted_sha256 = (
            promotion.get("persisted_output_sha256")
            if isinstance(promotion, Mapping)
            else None
        )
        try:
            with open(output_path, "rb") as handle:
                output_bytes = handle.read()
            output_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise _structured_reuse_error(
                json_path,
                stage_label,
                "the persisted stage output is missing or not UTF-8",
            ) from exc
        payload_text_key = (
            "ocr_text" if metadata_key == "ocr" else "markdown_text"
        )
        payload_text = payload.get(payload_text_key)
        if (
            not isinstance(payload_text, str)
            or payload_text.encode("utf-8") != output_bytes
        ):
            raise _structured_reuse_error(
                json_path,
                stage_label,
                f"the bundle {payload_text_key} differs from the stage file",
            )
        output_sha256 = hashlib.sha256(output_bytes).hexdigest()
        if (
            not isinstance(promotion, Mapping)
            or promotion.get("promoted") is not True
            or not isinstance(candidate_sha256, str)
            or not _SHA256_RE.fullmatch(candidate_sha256)
            or not isinstance(persisted_sha256, str)
            or not _SHA256_RE.fullmatch(persisted_sha256)
            or candidate_sha256 != persisted_sha256
            or persisted_sha256 != output_sha256
        ):
            raise _structured_reuse_error(
                json_path,
                stage_label,
                "the promoted output hash does not match the current bytes",
            )


def path_is_newer(source_path: str, output_path: str) -> bool:
    """Return whether a source exists and is newer than an output."""

    try:
        source_mtime = os.stat(source_path).st_mtime_ns
    except OSError:
        return False
    try:
        output_mtime = os.stat(output_path).st_mtime_ns
    except OSError:
        return False
    return source_mtime > output_mtime


def persisted_markdown_quality_status(json_path: str) -> str | None:
    try:
        with open(json_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    quality = payload.get("markdown_quality")
    if not isinstance(quality, dict):
        return None
    status = quality.get("status")
    return status if isinstance(status, str) else None


def ocr_image_dims(image_path: str) -> dict[str, int] | None:
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(image_path) as image:
            return {"w": image.size[0], "h": image.size[1]}
    except Exception:
        return None


def file_sha256(path: str) -> str | None:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def write_ocr_page_bundle(
    item: OcrFullItem,
    *,
    generation_metadata: dict | None = None,
) -> dict:
    ocr_text = read_optional_text(item.ocr_path)
    markdown_text = read_optional_text(item.markdown_path)
    raw_text = read_optional_text(item.text_path)
    merged_metadata = read_existing_generation_metadata(item.json_path) or {}
    merged_metadata.update(generation_metadata or {})
    generation_metadata = merged_metadata
    payload = {
        "image_path": item.image_path,
        "ocr_path": item.ocr_path,
        "markdown_path": item.markdown_path,
        "text_path": item.text_path,
        "ocr_text": ocr_text,
        "markdown_text": markdown_text,
        "raw_text": raw_text,
        "image_dims": item.image_dims,
        "image_sha256": item.image_sha256,
    }
    if generation_metadata:
        payload["generation_metadata"] = generation_metadata
    if markdown_text is not None:
        payload["markdown_quality"] = analyze_markdown_quality(
            markdown_text,
            ocr_text=ocr_text,
            raw_text=raw_text,
        )
    payload = {
        key: value
        for key, value in payload.items()
        if value is not None
    }
    directory = os.path.dirname(item.json_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(item.json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return payload


def read_existing_generation_metadata(json_path: str) -> dict | None:
    try:
        with open(json_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("generation_metadata")
    return metadata if isinstance(metadata, dict) else None


def read_optional_text(path: str | None) -> str | None:
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return None


_INTEGER_RE = re.compile(r"(?<![\w.])[-+]?\d+(?![\w.])")
_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_TABLE_CAPTION_RE = re.compile(
    r"^\s*(?:(?:<[^>\n]+>|[*_]{1,2})\s*)*"
    r"table\s+(\d+[a-z]?)\b",
    re.IGNORECASE | re.MULTILINE,
)
_TABLE_LABEL_RE = re.compile(
    r"\btable\s+(\d+[a-z]?)\b",
    re.IGNORECASE,
)
_PIPE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_GROUNDING_BOX_LINE_RE = re.compile(
    r"^[ \t]*(text|sub_title|equation|image|image_caption|title|table|"
    r"table_caption|table_footnote)[ \t]*"
    r"\[\[[ \t]*[-+]?\d+"
    r"(?:[ \t]*,[ \t]*[-+]?\d+){3}[ \t]*\]\][ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


def _word_tokens(text: str | None) -> list[str]:
    return _WORD_RE.findall((text or "").casefold())


def _grounded_content_tokens(text: str | None) -> list[str]:
    """Tokenize OCR content while ignoring DeepSeek box coordinates."""

    return _word_tokens(_GROUNDING_BOX_LINE_RE.sub("", text or ""))


def _source_coverage(text: str, source: str | None) -> float | None:
    source_words = [word.lower() for word in _word_tokens(source)]
    if not source_words:
        return None
    output_counts = Counter(word.lower() for word in _word_tokens(text))
    source_counts = Counter(source_words)
    matched = sum(
        min(count, output_counts.get(word, 0))
        for word, count in source_counts.items()
    )
    return round(matched / len(source_words), 6)


def monotonic_integer_run(
    text: str,
    *,
    max_gap_chars: int = 64,
) -> dict:
    matches = list(_INTEGER_RE.finditer(text or ""))
    if not matches:
        return {
            "terms": 0,
            "step": None,
            "first": None,
            "last": None,
            "char_start": None,
            "char_end": None,
        }
    best_count = run_count = 1
    best_start = run_start = 0
    best_end = 0
    best_step = run_step = None
    for index in range(1, len(matches)):
        previous = matches[index - 1]
        current = matches[index]
        delta = int(current.group()) - int(previous.group())
        gap = current.start() - previous.end()
        if delta and gap <= max_gap_chars:
            if run_count >= 2 and delta == run_step:
                run_count += 1
            else:
                run_count = 2
                run_start = index - 1
                run_step = delta
        else:
            run_count = 1
            run_start = index
            run_step = None
        if run_count > best_count:
            best_count = run_count
            best_start = run_start
            best_end = index
            best_step = run_step
    return {
        "terms": best_count,
        "step": best_step,
        "first": int(matches[best_start].group()),
        "last": int(matches[best_end].group()),
        "char_start": matches[best_start].start(),
        "char_end": matches[best_end].end(),
    }


def repeated_long_block_metrics(
    text: str,
    *,
    source_text: str | None = None,
) -> dict | None:
    blocks = [
        re.sub(r"\s+", " ", block).strip()
        for block in re.split(r"\n\s*\n", text)
    ]
    blocks = [block for block in blocks if len(_word_tokens(block)) >= 40]
    if not blocks:
        return None
    counts = Counter(blocks)
    block, occurrences = max(
        counts.items(),
        key=lambda item: (item[1], len(item[0])),
    )
    if occurrences < 2:
        return None
    source_coverage = _source_coverage(block, source_text)
    return {
        "occurrences": occurrences,
        "characters": len(block),
        "words": len(_word_tokens(block)),
        "sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(),
        "source_coverage": source_coverage,
    }


def _ngrams(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    return {
        tuple(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    }


def _nonoverlapping(positions: list[int], span_words: int) -> list[int]:
    selected = []
    for position in positions:
        if not selected or position - selected[-1] >= span_words:
            selected.append(position)
    return selected


def repeated_word_window_metrics(
    markdown_text: str,
    raw_text: str | None,
    grounded_ocr_text: str | None,
    *,
    span_words: int = 50,
    support_ngram_words: int = 5,
) -> dict:
    tokens = _word_tokens(markdown_text)
    positions_by_window: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index in range(len(tokens) - span_words + 1):
        window = tuple(tokens[index : index + span_words])
        positions_by_window[window].append(index)
    raw_support = _ngrams(_word_tokens(raw_text), support_ngram_words)
    ocr_support = _ngrams(
        _word_tokens(grounded_ocr_text),
        support_ngram_words,
    )
    combined_support = raw_support | ocr_support
    candidates = []
    for window, positions in positions_by_window.items():
        selected = _nonoverlapping(positions, span_words)
        if len(selected) < 2:
            continue
        window_ngrams = [
            window[index : index + support_ngram_words]
            for index in range(span_words - support_ngram_words + 1)
        ]

        def coverage(support, ngrams=tuple(window_ngrams)):
            return sum(
                ngram in support for ngram in ngrams
            ) / len(ngrams)

        candidates.append(
            {
                "nonoverlap_occurrences": len(selected),
                "period_words": min(
                    right - left
                    for left, right in zip(
                        positions,
                        positions[1:],
                        strict=False,
                    )
                ),
                "first_word_offset": positions[0],
                "raw_5gram_coverage": coverage(raw_support),
                "grounded_ocr_5gram_coverage": coverage(ocr_support),
                "combined_5gram_coverage": coverage(combined_support),
                "span_sha256": hashlib.sha256(
                    " ".join(window).encode()
                ).hexdigest(),
            }
        )

    def score(candidate):
        return (
            candidate["nonoverlap_occurrences"],
            -candidate["combined_5gram_coverage"],
            -candidate["first_word_offset"],
        )

    best = max(candidates, key=score, default=None)
    unsupported = max(
        (
            candidate
            for candidate in candidates
            if candidate["combined_5gram_coverage"] < 0.2
        ),
        key=score,
        default=None,
    )
    return {
        "span_words": span_words,
        "support_ngram_words": support_ngram_words,
        "max_repeat": best,
        "max_unsupported_repeat": unsupported,
    }


def table_completion_metrics(
    markdown_text: str,
    raw_text: str | None,
    grounded_ocr_text: str | None,
) -> dict:
    def captions(text):
        return {
            label.casefold()
            for label in _TABLE_CAPTION_RE.findall(text or "")
        }

    expected = captions(raw_text) & captions(grounded_ocr_text)
    observed = {
        label.casefold()
        for label in _TABLE_LABEL_RE.findall(markdown_text or "")
    }
    missing = expected - observed
    raw_words = _word_tokens(raw_text)
    markdown_words = _word_tokens(markdown_text)
    grounded_markdown_words = _grounded_content_tokens(markdown_text)
    grounded_ocr_words = _grounded_content_tokens(grounded_ocr_text)
    raw_counts = Counter(raw_words)
    markdown_counts = Counter(markdown_words)
    recall = (
        sum((raw_counts & markdown_counts).values()) / len(raw_words)
        if raw_words
        else 1.0
    )
    grounded_ocr_counts = Counter(grounded_ocr_words)
    grounded_markdown_counts = Counter(grounded_markdown_words)
    grounded_ocr_recall = (
        sum((grounded_ocr_counts & grounded_markdown_counts).values())
        / len(grounded_ocr_words)
        if grounded_ocr_words
        else 1.0
    )
    word_ratio = (
        len(markdown_words) / len(raw_words) if raw_words else 1.0
    )
    lines = (markdown_text or "").splitlines()
    return {
        "expected_caption_labels": sorted(expected),
        "observed_table_labels": sorted(observed),
        "missing_caption_labels": sorted(missing),
        "missing_caption_fraction": (
            len(missing) / len(expected) if expected else 0.0
        ),
        "raw_token_multiset_recall": recall,
        "grounded_ocr_token_multiset_recall": grounded_ocr_recall,
        "markdown_raw_word_ratio": word_ratio,
        "nonblank_lines": sum(bool(line.strip()) for line in lines),
        "pipe_table_rows": sum(
            bool(_PIPE_ROW_RE.match(line)) for line in lines
        ),
        "ends_with_pipe_table_row": bool(
            re.search(r"\|\s*$", markdown_text or "")
        ),
    }


def token_repetition_metrics(token_ids: Iterable[int]) -> dict:
    ids = list(token_ids)
    metrics = {"token_count": len(ids), "unique_token_fraction": 0.0}
    if ids:
        metrics["unique_token_fraction"] = round(len(set(ids)) / len(ids), 6)
    ngrams = {}
    for size in (10, 25, 50, 75):
        windows = len(ids) - size + 1
        if windows <= 0:
            ngrams[str(size)] = {
                "max_occurrences": 0,
                "repeated_window_fraction": 0.0,
            }
            continue
        counts = Counter(
            tuple(ids[index : index + size])
            for index in range(windows)
        )
        repeated = sum(count for count in counts.values() if count > 1)
        ngrams[str(size)] = {
            "max_occurrences": max(counts.values()),
            "repeated_window_fraction": round(repeated / windows, 6),
        }
    metrics["ngrams"] = ngrams
    return metrics


def analyze_markdown_quality(
    markdown_text: str,
    *,
    ocr_text: str | None,
    raw_text: str | None,
) -> dict:
    axis = monotonic_integer_run(markdown_text)
    repetition = repeated_word_window_metrics(
        markdown_text,
        raw_text,
        ocr_text,
    )
    table = table_completion_metrics(
        markdown_text,
        raw_text,
        ocr_text,
    )
    flags = []
    if axis["terms"] >= 64:
        flags.append("runaway_monotonic_integer_axis")
    unsupported = repetition["max_unsupported_repeat"]
    if (
        unsupported
        and unsupported["nonoverlap_occurrences"] >= 3
    ):
        flags.append("unsupported_long_span_repetition")
    if (
        table["expected_caption_labels"]
        and table["missing_caption_fraction"] >= 0.5
        and table["raw_token_multiset_recall"] < 0.5
        and table["markdown_raw_word_ratio"] < 0.5
    ):
        flags.append("premature_table_truncation")
    return {
        "schema_version": 1,
        "status": "fail" if flags else "pass",
        "flags": flags,
        "thresholds": {
            "axis_min_terms": 64,
            "axis_max_gap_chars": 64,
            "repeat_min_occurrences": 3,
            "repeat_span_words": 50,
            "repeat_support_ngram_words": 5,
            "repeat_max_evidence_coverage": 0.2,
            "table_min_missing_caption_fraction": 0.5,
            "table_max_raw_token_recall": 0.5,
            "table_max_markdown_raw_word_ratio": 0.5,
        },
        "axis": axis,
        "repetition": repetition,
        "table_completion": table,
    }


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_safe(item) for item in value), key=repr)
    return repr(value)


def sampling_params_metadata(sampling_params) -> dict:
    field_names = getattr(sampling_params, "__struct_fields__", None)
    if field_names:
        return {
            name: _json_safe(getattr(sampling_params, name))
            for name in field_names
        }
    try:
        values = vars(sampling_params)
    except TypeError:
        return {"repr": repr(sampling_params)}
    return _json_safe(values)


def sampling_params_with_minimum_window(
    sampling_params,
    minimum_window_size: int,
):
    clone = getattr(sampling_params, "clone", None)
    if callable(clone):
        updated = clone()
    else:
        updated = copy.deepcopy(sampling_params)
    extra_args = dict(getattr(updated, "extra_args", None) or {})
    extra_args["window_size"] = max(
        int(extra_args.get("window_size") or 0),
        minimum_window_size,
    )
    updated.extra_args = extra_args
    return updated


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _poppler_version() -> str | None:
    executable = shutil.which("pdftoppm")
    if not executable:
        return None
    try:
        completed = subprocess.run(
            [executable, "-v"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    output = "\n".join((completed.stdout, completed.stderr))
    match = re.search(r"\bversion\s+([^\s]+)", output)
    return match.group(1) if match else None


def resolve_model_metadata(
    model: str,
    revision: str | None = None,
) -> dict:
    resolved_path = None
    if os.path.exists(model):
        resolved_path = os.path.realpath(model)
    else:
        try:
            from huggingface_hub import snapshot_download

            resolved_path = snapshot_download(
                repo_id=model,
                revision=revision,
                local_files_only=True,
            )
        except Exception:
            resolved_path = None
    snapshot = None
    if resolved_path:
        normalized = resolved_path.replace(os.sep, "/")
        match = re.search(r"/snapshots/([^/]+)(?:/|$)", normalized)
        if match:
            snapshot = match.group(1)
    return {
        "requested": model,
        "revision": revision,
        "resolved_path": resolved_path,
        "snapshot": snapshot,
    }


def installed_distributions_identity() -> dict[str, object]:
    """Return the complete normalized distribution identity."""

    distributions = sorted(
        f"{distribution.metadata.get('Name', distribution.name)}=="
        f"{distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    payload = json.dumps(
        distributions,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "count": len(distributions),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "distributions": distributions,
    }


def selected_attention_backend(llm) -> str | None:
    try:
        from vllm.config import set_current_vllm_config
        from vllm.v1.attention.selector import get_attn_backend

        vllm_config = llm.llm_engine.vllm_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        with set_current_vllm_config(vllm_config):
            backend = get_attn_backend(
                head_size=model_config.get_head_size(),
                dtype=model_config.dtype,
                kv_cache_dtype=(
                    cache_config.cache_dtype if cache_config else None
                ),
                use_mla=model_config.use_mla,
                num_heads=model_config.get_num_attention_heads(
                    vllm_config.parallel_config
                ),
            )
        return backend.get_name()
    except Exception:
        return None


def engine_metadata(
    llm,
    *,
    model: str,
    model_revision: str | None,
    requested_attention_backend: str | None,
    requested_mm_encoder_attn_backend: str | None,
    requested_moe_backend: str,
    startup_installed_distributions: dict[str, object],
    engine_init_kwargs: dict | None = None,
) -> dict:
    try:
        import torch
    except ImportError:  # pragma: no cover - vLLM requires torch
        torch = None
    try:
        import vllm
    except ImportError:  # pragma: no cover - caller already imported vLLM
        vllm = None
    configured_backend = None
    configured_dtype = None
    configured_moe_backend = None
    try:
        vllm_config = llm.llm_engine.vllm_config
        configured_backend = vllm_config.attention_config.backend
        configured_backend = getattr(
            configured_backend,
            "name",
            configured_backend,
        )
        configured_dtype = str(vllm_config.model_config.dtype)
    except Exception:
        configured_backend = None
        configured_dtype = None
    try:
        configured_moe_backend = (
            llm.llm_engine.vllm_config.kernel_config.moe_backend
        )
        configured_moe_backend = getattr(
            configured_moe_backend,
            "value",
            configured_moe_backend,
        )
    except Exception:
        configured_moe_backend = None
    return {
        "model": resolve_model_metadata(model, model_revision),
        "runtime": {
            "python": platform.python_version(),
            "torch": getattr(torch, "__version__", None),
            "torch_cuda": getattr(
                getattr(torch, "version", None),
                "cuda",
                None,
            ),
            "vllm": getattr(vllm, "__version__", None),
            "transformers": _package_version("transformers"),
            "tokenizers": _package_version("tokenizers"),
            "pillow": _package_version("pillow"),
            "poppler": _poppler_version(),
            "dtype": configured_dtype,
            "attention_backend_requested": (
                requested_attention_backend or configured_backend
            ),
            "attention_backend_selected": selected_attention_backend(llm),
            "mm_encoder_attention_backend_requested": (
                requested_mm_encoder_attn_backend
            ),
            "moe_backend_requested": requested_moe_backend,
            "moe_backend_configured": configured_moe_backend,
            "jit_environment": {
                name: os.environ.get(name) for name in DEEPSEEK_JIT_ENV_NAMES
            },
            "vllm_batch_invariant": os.environ.get(
                "VLLM_BATCH_INVARIANT",
                "0",
            ),
            "engine_init_kwargs": _json_safe(engine_init_kwargs),
            "installed_distributions_count": (
                startup_installed_distributions["count"]
            ),
            "installed_distributions_sha256": (
                startup_installed_distributions["sha256"]
            ),
        },
    }


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
        model_revision: str | None = DEFAULT_OCR_MODEL_REVISION,
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
        moe_backend: str = DEFAULT_DEEPSEEK_MOE_BACKEND,
        scheduler_delay_factor: float | None = None,
        tensor_parallel_size: int | None = None,
        gpu_memory_utilization: float | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        ngram_size: int = 10,
        window_size: int = DEFAULT_OCR_WINDOW_SIZE,
        whitelist_token_ids: list[int] | None = None,
        no_tqdm: bool = False,
        out: Callable[[str], None] | None = None,
        verbose: Callable[[str], None] | None = None,
    ) -> None:
        startup_identity = installed_distributions_identity()
        self._startup_installed_distributions = {
            "count": startup_identity["count"],
            "sha256": startup_identity["sha256"],
        }
        self.model = model
        self.model_revision = model_revision
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
        self.moe_backend = validate_deepseek_moe_backend(moe_backend)
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
        self.generation_metadata: dict[str, dict] = {}
        self._engine_metadata: dict = {}
        self._jit_environment_previous: dict[str, str | None] | None = None

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
        self._activate_jit_environment()
        try:
            return self._build_llm_and_params_with_jit_environment()
        except BaseException:
            self._restore_jit_environment()
            raise

    def _build_llm_and_params_with_jit_environment(self):
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
            "moe_backend": self.moe_backend,
        }
        self._maybe_add(llm_kwargs, "revision", self.model_revision)
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
        self._engine_metadata = engine_metadata(
            llm,
            model=self.model,
            model_revision=self.model_revision,
            requested_attention_backend=self.attention_backend,
            requested_mm_encoder_attn_backend=(
                self.mm_encoder_attn_backend
            ),
            requested_moe_backend=self.moe_backend,
            startup_installed_distributions=(
                self._startup_installed_distributions
            ),
            engine_init_kwargs=llm_kwargs,
        )
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
        prepared_tasks: list[tuple[int, OcrPageTask, str]] = []
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
            prepared_tasks.append((index, task, prompt))
        if not prepared_tasks:
            return results
        try:
            outputs = self._generate(
                llm,
                inputs,
                sampling_params=sampling_params,
            )
            for (index, task, prompt), output in zip(
                prepared_tasks,
                outputs,
                strict=False,
            ):
                if not output.outputs:
                    continue
                completion = output.outputs[0]
                results[index] = completion.text
                self.generation_metadata[task.output_path] = (
                    self._generation_metadata(
                        task=task,
                        prompt=prompt,
                        request_output=output,
                        completion=completion,
                        sampling_params=sampling_params,
                    )
                )
            return results
        except Exception as exc:
            if len(prepared_tasks) == 1:
                _index, task, prompt = prepared_tasks[0]
                self.generation_metadata[task.output_path] = (
                    self._generation_failure_metadata(
                        task=task,
                        prompt=prompt,
                        sampling_params=sampling_params,
                        exc=exc,
                    )
                )
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
        for index, task, _prompt in prepared_tasks:
            results[index] = self._generate_batch_results(
                llm,
                sampling_params,
                [task],
            )[0]
        return results

    def _generation_metadata(
        self,
        *,
        task: OcrPageTask,
        prompt: str,
        request_output,
        completion,
        sampling_params,
    ) -> dict:
        token_ids = list(completion.token_ids or [])
        prompt_token_ids = list(request_output.prompt_token_ids or [])
        finish_reason = getattr(completion, "finish_reason", None)
        stop_reason = getattr(completion, "stop_reason", None)
        max_tokens = getattr(sampling_params, "max_tokens", None)
        stopped = finish_reason == "stop"
        return {
            "status": "completed",
            "image": {
                "path": task.image_path,
                "sha256": file_sha256(task.image_path),
                "dimensions": ocr_image_dims(task.image_path),
            },
            "prompt": prompt,
            "prompt_mode": self.prompt_mode,
            "output_format": self.output_format,
            **self._engine_metadata,
            "sampling_params": sampling_params_metadata(sampling_params),
            "logits_processor_extra_args": _json_safe(
                getattr(sampling_params, "extra_args", None)
            ),
            "request_id": getattr(request_output, "request_id", None),
            "prompt_token_count": len(prompt_token_ids),
            "prompt_token_ids_sha256": hashlib.sha256(
                json.dumps(prompt_token_ids, separators=(",", ":")).encode(
                    "ascii"
                )
            ).hexdigest(),
            "generated_token_count": len(token_ids),
            "generated_token_ids_sha256": hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
            "finish_reason": finish_reason,
            "stop_reason": _json_safe(stop_reason),
            "termination": {
                "eos": stopped and stop_reason is None,
                "length": finish_reason == "length",
                "max_tokens": (
                    max_tokens is not None
                    and len(token_ids) >= max_tokens
                ),
            },
            "repetition": {
                "tokens": token_repetition_metrics(token_ids),
                "longest_monotonic_integer_run": monotonic_integer_run(
                    completion.text
                ),
                "repeated_long_block": repeated_long_block_metrics(
                    completion.text
                ),
            },
        }

    def _generation_failure_metadata(
        self,
        *,
        task: OcrPageTask,
        prompt: str,
        sampling_params,
        exc: Exception,
    ) -> dict:
        return {
            "status": "error",
            "image": {
                "path": task.image_path,
                "sha256": file_sha256(task.image_path),
                "dimensions": ocr_image_dims(task.image_path),
            },
            "prompt": prompt,
            "prompt_mode": self.prompt_mode,
            "output_format": self.output_format,
            **self._engine_metadata,
            "sampling_params": sampling_params_metadata(sampling_params),
            "logits_processor_extra_args": _json_safe(
                getattr(sampling_params, "extra_args", None)
            ),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }

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
        except Exception as exc:  # pragma: no cover - version fallback
            if self.moe_backend != DEFAULT_DEEPSEEK_MOE_BACKEND:
                raise RuntimeError(
                    "the installed vLLM cannot verify support for explicit "
                    f"moe_backend={self.moe_backend}"
                ) from exc
            try:
                from vllm import LLM
            except Exception:
                return kwargs
            filtered = self._filter_kwargs(LLM.__init__, kwargs)
        else:
            filtered = self._filter_kwargs(EngineArgs.__init__, kwargs)
        if (
            self.moe_backend != DEFAULT_DEEPSEEK_MOE_BACKEND
            and "moe_backend" not in filtered
        ):
            raise RuntimeError(
                "the installed vLLM does not support explicit "
                f"moe_backend={self.moe_backend}"
            )
        return filtered

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

    def _activate_jit_environment(self) -> None:
        if self._jit_environment_previous is not None:
            raise RuntimeError("DeepSeek engine lifecycle is already active")
        self._jit_environment_previous = {
            name: os.environ.get(name) for name in DEEPSEEK_JIT_ENV_NAMES
        }
        os.environ.update(deepseek_jit_environment())

    def _restore_jit_environment(self) -> None:
        previous = self._jit_environment_previous
        if previous is None:
            return
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._jit_environment_previous = None

    def shutdown_engine(self, llm) -> None:
        try:
            self._shutdown_engine(llm)
        finally:
            self._restore_jit_environment()

    def _shutdown_engine(self, llm) -> None:
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
