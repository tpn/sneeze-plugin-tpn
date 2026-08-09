"""Local JSONL adapter for the pinned MinerU2.5-Pro OCR pipeline.

This module deliberately imports only the standard library at startup. Run it
with the Python executable from a dedicated MinerU environment; the primary
DeepSeek/vLLM environment does not need MinerU's Transformers dependencies.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import runpy
import sys
import time
import traceback
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "opendatalab/MinerU2.5-Pro-2605-1.2B"
DEFAULT_REVISION = "bff20d4ae2bf202df9f45284b4d43681555a97ed"
QUALIFIED_REQUEST_SETTINGS = {
    "backend": "official-two-step-transformers",
    "batch_size": 1,
    "image_analysis": False,
}
TUNING_REQUEST_SETTING_KEYS = frozenset(
    {
        *QUALIFIED_REQUEST_SETTINGS,
        "memory_budget_bytes",
        "memory_tier",
    }
)
MAX_TUNING_BATCH_SIZE = 64
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PROFILE_MAX_BYTES = 8 * 1024 * 1024
_IMPLEMENTATION_COMPONENT_PATHS = (
    "ocr_backend_setup.py",
    "commands.py",
    "ocr_benchmark.py",
    "ocr_execution.py",
    "ocr_hardware.py",
    "ocr_implementation.py",
    "ocr_installations.py",
    "ocr_memory.py",
    "ocr_path_security.py",
    "ocr_process_lifecycle.py",
    "ocr_profiles.py",
    "ocr_provider.py",
    "ocr_tuning.py",
    "providers/mineru25.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-jsonl", required=True)
    parser.add_argument("--responses-jsonl", required=True)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--tuning", action="store_true")
    parser.add_argument("--execution-profile")
    parser.add_argument("--profile-sha256")
    parser.add_argument("--memory-tier", choices=("25", "50", "max"))
    parser.add_argument("--allow-provisional-profile", action="store_true")
    return parser.parse_args()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("request contains a duplicate JSON object key")
        value[key] = item
    return value


def read_jsonl(path: str) -> list[dict[str, Any]]:
    values = []
    attempt_ids: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(
                line,
                object_pairs_hook=_strict_json_object,
            )
            if not isinstance(value, dict):
                raise ValueError(
                    f"request at line {line_number} must be an object"
                )
            attempt_id = value.get("attempt_id")
            if isinstance(attempt_id, str):
                if attempt_id in attempt_ids:
                    raise ValueError("requests contain duplicate attempt IDs")
                attempt_ids.add(attempt_id)
            values.append(value)
    if not values:
        raise ValueError("requests JSONL is empty")
    return values


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        handle.write("\n")


def package_versions(names: list[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def installed_distributions_identity() -> dict[str, Any]:
    distributions = sorted(
        f"{distribution.metadata.get('Name', distribution.name)}=="
        f"{distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    return {
        "count": len(distributions),
        "sha256": sha256_json(distributions),
        "distributions": distributions,
    }


def startup_installed_distributions_identity() -> dict[str, Any]:
    """Capture the registered environment before runtime plugin imports."""

    identity = installed_distributions_identity()
    return {
        "count": identity["count"],
        "sha256": identity["sha256"],
    }


def implementation_record() -> dict[str, Any]:
    """Hash the standalone adapter and every in-scope plugin component."""

    root = Path(__file__).resolve().parents[1]
    components = {
        relative: sha256_path(root / relative)
        for relative in _IMPLEMENTATION_COMPONENT_PATHS
    }
    backend_module = runpy.run_path(
        str(root / "ocr_backends.py"),
        run_name="_sneeze_mineru25_backend_definition",
    )
    backend = backend_module["get_backend"]("mineru25-pro-2605")
    definition = dataclasses.asdict(backend)
    if "qualified_implementation_sha256" not in definition:
        raise RuntimeError("OCR backend definition schema is incomplete")
    definition.pop("qualified_implementation_sha256")
    definition = json.loads(
        json.dumps(
            definition,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    unsigned = {
        "schema_version": 2,
        "backend_id": "mineru25-pro-2605",
        "components": {key: components[key] for key in sorted(components)},
        "definition": definition,
    }
    return {**unsigned, "sha256": sha256_json(unsigned)}


def snapshot_identity(snapshot: Path, revision: str) -> dict[str, Any]:
    if snapshot.name != revision:
        raise RuntimeError(
            "resolved model snapshot does not match requested revision"
        )
    required_filenames = (
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "added_tokens.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    )
    missing = [
        filename
        for filename in required_filenames
        if not (snapshot / filename).is_file()
    ]
    if missing:
        raise RuntimeError(
            "pinned model snapshot is missing " + ", ".join(missing)
        )
    files = {}
    for path in sorted(
        (
            candidate
            for candidate in snapshot.rglob("*")
            if candidate.is_file()
        ),
        key=lambda candidate: candidate.relative_to(snapshot).as_posix(),
    ):
        filename = path.relative_to(snapshot).as_posix()
        files[filename] = {
            "sha256": sha256_path(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "commit": revision,
        "path": str(snapshot.resolve()),
        "files": files,
        "files_sha256": sha256_json(files),
    }


def serializable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return serializable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


class GenerationTracer:
    """Capture every native MinerU stage prompt and generation result."""

    def __init__(self, model, processor) -> None:
        self.model = model
        self.processor = processor
        self.original_generate = model.generate
        self.original_chat_template = processor.apply_chat_template
        self.traces: list[dict[str, Any]] = []
        self.prompts: list[str] = []

    def install(self) -> None:
        def traced_chat_template(*args, **kwargs):
            result = self.original_chat_template(*args, **kwargs)
            if isinstance(result, str):
                self.prompts.append(result)
            elif isinstance(result, list):
                self.prompts.extend(
                    item for item in result if isinstance(item, str)
                )
            return result

        def traced_generate(*args, **kwargs):
            started = time.monotonic()
            output = self.original_generate(*args, **kwargs)
            sequences = (
                output.sequences if hasattr(output, "sequences") else output
            )
            input_ids = kwargs.get("input_ids")
            attention_mask = kwargs.get("attention_mask")
            input_width = input_ids.shape[-1] if input_ids is not None else 0
            eos_ids = self.model.generation_config.eos_token_id
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            eos_ids = set(eos_ids or [])
            max_new_tokens = kwargs.get("max_new_tokens")
            max_length = kwargs.get("max_length")
            stage_wall_seconds = round(time.monotonic() - started, 6)
            for row_index, sequence in enumerate(sequences):
                ids = sequence.detach().cpu().tolist()
                generated = ids[input_width:]
                for token_index, token_id in enumerate(generated):
                    if token_id in eos_ids:
                        generated = generated[: token_index + 1]
                        break
                input_tokens = (
                    int(attention_mask[row_index].sum().item())
                    if attention_mask is not None
                    else input_width
                )
                reached_limit = (
                    max_new_tokens is not None
                    and len(generated) >= max_new_tokens
                ) or (
                    max_length is not None
                    and input_width + len(generated) >= max_length
                )
                reached_eos = bool(generated and generated[-1] in eos_ids)
                reached_max_tokens = reached_limit and not reached_eos
                self.traces.append(
                    {
                        "input_tokens": input_tokens,
                        "output_tokens": len(generated),
                        "last_token_id": (
                            generated[-1] if generated else None
                        ),
                        "finish_reason": (
                            "length" if reached_max_tokens else "stop"
                        ),
                        "stop_reason": (
                            "eos"
                            if reached_eos
                            else (
                                "max_tokens"
                                if reached_max_tokens
                                else "generation_stopped"
                            )
                        ),
                        "eos": reached_eos,
                        "max_tokens": reached_max_tokens,
                        "token_ids": generated,
                        "decoded_raw": self.processor.decode(
                            generated,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        ),
                        "generate_kwargs": {
                            key: serializable(value)
                            for key, value in kwargs.items()
                            if key
                            not in {
                                "input_ids",
                                "attention_mask",
                                "pixel_values",
                                "image_grid_thw",
                            }
                        },
                        "input_shapes": {
                            key: list(value.shape)
                            for key, value in kwargs.items()
                            if key
                            in {
                                "input_ids",
                                "attention_mask",
                                "pixel_values",
                                "image_grid_thw",
                            }
                            and hasattr(value, "shape")
                        },
                        "wall_seconds": stage_wall_seconds,
                    }
                )
            return output

        self.processor.apply_chat_template = traced_chat_template
        self.model.generate = traced_generate

    def reset(self) -> None:
        self.traces.clear()
        self.prompts.clear()


def validate_tuning_settings(settings: Any) -> dict[str, Any]:
    if not isinstance(settings, dict):
        raise ValueError("request settings must be an object")
    unexpected = set(settings) - TUNING_REQUEST_SETTING_KEYS
    if unexpected:
        raise ValueError(
            "request tuning settings contain unsupported controls: "
            + ", ".join(sorted(unexpected))
        )
    if settings.get("backend") != "official-two-step-transformers":
        raise ValueError(
            "request tuning settings require the official two-step backend"
        )
    if settings.get("image_analysis") is not False:
        raise ValueError(
            "request tuning settings require image_analysis=false"
        )
    batch_size = settings.get("batch_size")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or not 1 <= batch_size <= MAX_TUNING_BATCH_SIZE
    ):
        raise ValueError(
            "request tuning batch_size must be an integer from 1 through "
            f"{MAX_TUNING_BATCH_SIZE}"
        )
    tier = settings.get("memory_tier")
    budget = settings.get("memory_budget_bytes")
    if tier not in {"25", "50", "max"}:
        raise ValueError("request tuning memory_tier must be 25, 50, or max")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError(
            "request tuning memory_budget_bytes must be a positive integer"
        )
    return dict(settings)


def load_profile_settings(
    path: str,
    *,
    expected_sha256: str,
    memory_tier: str,
    allow_provisional: bool,
    model_id: str,
    revision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the MinerU stage settings from one sealed portable profile.

    The adapter is intentionally standalone, so this performs the small
    security-critical subset of profile validation again inside the isolated
    provider environment. Full schema validation remains the caller's job.
    """

    source = Path(path)
    if (
        not source.is_absolute()
        or source.is_symlink()
        or not source.is_file()
    ):
        raise ValueError(
            "execution profile must be an absolute, regular, non-symlink file"
        )
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise ValueError("execution profile SHA-256 is invalid")
    if source.stat().st_size > _PROFILE_MAX_BYTES:
        raise ValueError("execution profile exceeds the size limit")
    try:
        profile = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("could not read execution profile") from exc
    except ValueError as exc:
        raise ValueError(
            "execution profile contains a duplicate JSON object key"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise ValueError("could not read execution profile") from exc
    if not isinstance(profile, dict):
        raise ValueError("execution profile must be a JSON object")
    recorded_sha256 = profile.get("profile_sha256")
    unsigned = dict(profile)
    unsigned.pop("profile_sha256", None)
    computed_sha256 = sha256_json(unsigned)
    if (
        recorded_sha256 != computed_sha256
        or expected_sha256 != computed_sha256
    ):
        raise ValueError("execution profile content hash does not match")
    if (
        profile.get("schema_version") != 1
        or profile.get("kind") != "ocr_execution_profile"
        or profile.get("pipeline_id") != "local-mineru25"
    ):
        raise ValueError("execution profile is not a local-mineru25 profile")
    qualification = profile.get("qualification")
    state = (
        qualification.get("state")
        if isinstance(qualification, dict)
        else None
    )
    if state not in {"provisional", "qualified"}:
        raise ValueError("execution profile qualification state is invalid")
    if state != "qualified" and not allow_provisional:
        raise ValueError(
            "execution profile is provisional; explicit trial approval "
            "is required"
        )
    stages = profile.get("stages")
    markdown_stage = (
        stages.get("markdown") if isinstance(stages, dict) else None
    )
    model = (
        markdown_stage.get("model")
        if isinstance(markdown_stage, dict)
        else None
    )
    if (
        not isinstance(markdown_stage, dict)
        or markdown_stage.get("backend_id") != "mineru25-pro-2605"
        or not isinstance(model, dict)
        or model.get("requested") != model_id
        or model.get("revision") != revision
    ):
        raise ValueError("execution profile MinerU model binding differs")
    tiers = profile.get("memory_tiers")
    tier = tiers.get(memory_tier) if isinstance(tiers, dict) else None
    if not isinstance(tier, dict) or tier.get("supported") is not True:
        raise ValueError("execution profile memory tier is unsupported")
    tier_stages = tier.get("stages")
    stage = (
        tier_stages.get("markdown") if isinstance(tier_stages, dict) else None
    )
    settings = stage.get("settings") if isinstance(stage, dict) else None
    normalized = validate_tuning_settings(settings)
    if normalized.get("memory_tier") != memory_tier:
        raise ValueError("execution profile stage memory tier differs")
    if normalized.get("memory_budget_bytes") != tier.get("budget_bytes"):
        raise ValueError("execution profile stage memory budget differs")
    identity = {
        "profile_id": profile.get("profile_id"),
        "profile_sha256": computed_sha256,
        "qualification_state": state,
        "memory_tier": memory_tier,
    }
    if (
        not isinstance(identity["profile_id"], str)
        or not identity["profile_id"]
    ):
        raise ValueError("execution profile ID is invalid")
    return normalized, identity


def validate_request(
    request: dict[str, Any],
    *,
    tuning: bool = False,
    profile_settings: dict[str, Any] | None = None,
) -> None:
    attempt_id = request.get("attempt_id")
    if not isinstance(attempt_id, str) or not _SHA256_RE.fullmatch(
        attempt_id
    ):
        raise ValueError("request attempt_id must be a lowercase SHA-256")
    if request.get("schema_version") != 1:
        raise ValueError("unsupported request schema_version")
    if request.get("lane") != "best":
        raise ValueError("MinerU production adapter only supports lane=best")
    settings = request.get("settings")
    if tuning and profile_settings is not None:
        raise ValueError("tuning and execution-profile settings cannot mix")
    if tuning:
        validate_tuning_settings(settings)
    elif profile_settings is not None:
        validate_tuning_settings(settings)
        if settings != profile_settings:
            raise ValueError(
                "request settings do not match the sealed execution profile"
            )
    elif settings != QUALIFIED_REQUEST_SETTINGS:
        raise ValueError(
            "request settings must exactly match the qualified MinerU "
            "two-stage configuration"
        )
    image = request.get("input", {}).get("image")
    if not isinstance(image, dict):
        raise ValueError("request input.image must be an object")
    image_path = image.get("path")
    if not isinstance(image_path, str) or not Path(image_path).is_absolute():
        raise ValueError("request image path must be absolute")
    digest = image.get("sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError("request image SHA-256 is invalid")
    dimensions = image.get("dimensions")
    if not isinstance(dimensions, dict) or not all(
        isinstance(dimensions.get(key), int) and dimensions[key] > 0
        for key in ("w", "h")
    ):
        raise ValueError("request image dimensions are invalid")


def pipeline_configuration(client, model) -> dict[str, Any]:
    helper = client.helper
    native_client = client.client
    generation_config = getattr(model, "generation_config", None)
    to_dict = getattr(generation_config, "to_dict", None)
    return {
        "pipeline": "MinerUClient.two_step_extract+json2md",
        "backend": client.backend,
        "batching_mode": client.batching_mode,
        "prompts": serializable(client.prompts),
        "sampling_params_by_block_type": serializable(client.sampling_params),
        "layout_image_size": list(helper.layout_image_size),
        "min_image_edge": helper.min_image_edge,
        "max_image_edge_ratio": helper.max_image_edge_ratio,
        "simple_post_process": helper.simple_post_process,
        "handle_equation_block": helper.handle_equation_block,
        "abandon_list": helper.abandon_list,
        "abandon_paratext": helper.abandon_paratext,
        "image_analysis": helper.image_analysis,
        "enable_table_formula_eq_wrap": (helper.enable_table_formula_eq_wrap),
        "enable_cross_page_table_merge": (
            helper.enable_cross_page_table_merge
        ),
        "incremental_priority": client.incremental_priority,
        "batch_size": native_client.batch_size,
        "allow_truncated_content": native_client.allow_truncated_content,
        "system_prompt": native_client.system_prompt,
        "text_before_image": native_client.text_before_image,
        "model_max_length": native_client.model_max_length,
        "generation_config": (
            serializable(to_dict()) if callable(to_dict) else None
        ),
    }


def _linux_memory() -> dict[str, int | None]:
    values: dict[str, int | None] = {
        "total_bytes": None,
        "available_bytes": None,
    }
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            parsed = {}
            for line in handle:
                name, separator, remainder = line.partition(":")
                if not separator:
                    continue
                fields = remainder.split()
                if not fields or not fields[0].isdigit():
                    continue
                multiplier = 1024 if fields[1:2] == ["kB"] else 1
                parsed[name] = int(fields[0]) * multiplier
        values["total_bytes"] = parsed.get("MemTotal")
        values["available_bytes"] = parsed.get("MemAvailable")
    except OSError:
        pass
    return values


def _cgroup_v2_directory(
    membership_path: Path,
    cgroup_root: Path,
) -> Path | None:
    try:
        lines = membership_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    for line in lines:
        hierarchy, separator, remainder = line.partition(":")
        controllers, second_separator, relative = remainder.partition(":")
        if (
            separator
            and second_separator
            and hierarchy == "0"
            and controllers == ""
        ):
            parts = Path(relative.lstrip("/")).parts
            if any(part in {".", ".."} for part in parts):
                return None
            return cgroup_root.joinpath(*parts)
    return None


def _nonnegative_decimal_file(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if len(value) > 64 or not value.isdecimal():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _cgroup_v2_memory(
    *,
    membership_path: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict[str, Any]:
    """Sample cgroup v2 memory counters without exposing its local path."""

    result: dict[str, Any] = {
        "available": False,
        "memory_max_bytes": None,
        "memory_max_is_unlimited": None,
        "memory_available_bytes": None,
        "memory_current_bytes": None,
        "memory_peak_bytes": None,
    }
    directory = _cgroup_v2_directory(membership_path, cgroup_root)
    if directory is None:
        return result
    directories = []
    candidate = directory
    while True:
        directories.append(candidate)
        if candidate == cgroup_root:
            break
        candidate = candidate.parent
        if candidate != cgroup_root and cgroup_root not in candidate.parents:
            return result
    records: list[dict[str, int | str | None]] = []
    for candidate in directories:
        try:
            raw_maximum = (
                (candidate / "memory.max")
                .read_text(encoding="utf-8")
                .strip()
            )
        except (OSError, UnicodeError):
            maximum: int | str | None = None
        else:
            maximum = (
                raw_maximum
                if raw_maximum == "max"
                else (
                    int(raw_maximum)
                    if len(raw_maximum) <= 64
                    and raw_maximum.isdecimal()
                    else None
                )
            )
        records.append(
            {
                "maximum": maximum,
                "current": _nonnegative_decimal_file(
                    candidate / "memory.current"
                ),
                "peak": _nonnegative_decimal_file(
                    candidate / "memory.peak"
                ),
            }
        )
    maximums = [
        record["maximum"]
        for record in records
        if isinstance(record["maximum"], int)
        or record["maximum"] == "max"
    ]
    finite_maximums = [
        value for value in maximums if isinstance(value, int)
    ]
    if maximums:
        result["memory_max_bytes"] = (
            min(finite_maximums) if finite_maximums else None
        )
        result["memory_max_is_unlimited"] = not finite_maximums
        result["available"] = True
    finite_records = [
        record
        for record in records
        if isinstance(record["maximum"], int)
    ]
    headrooms = (
        [
            max(0, record["maximum"] - record["current"])
            for record in finite_records
        ]
        if finite_records
        and all(
            isinstance(record["current"], int)
            for record in finite_records
        )
        else []
    )
    if headrooms:
        result["memory_available_bytes"] = min(headrooms)
        result["available"] = True
    if records:
        nearest = records[0]
        for source, key in (
            ("current", "memory_current_bytes"),
            ("peak", "memory_peak_bytes"),
        ):
            value = nearest[source]
            if isinstance(value, int):
                result[key] = value
                result["available"] = True
    return result


def _peak_process_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB. The adapter is Linux/CUDA-only.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _cuda_integrated_property(properties) -> bool | None:
    value = getattr(
        properties,
        "is_integrated",
        getattr(properties, "integrated", None),
    )
    return bool(value) if value is not None else None


class CudaMemoryTracker:
    """Collect explicit allocator and device-memory high-water evidence."""

    def __init__(self, torch, device) -> None:
        self.torch = torch
        self.device = device
        self.properties = torch.cuda.get_device_properties(device)
        self.integrated = _cuda_integrated_property(self.properties)
        self.samples: list[dict[str, Any]] = []
        torch.cuda.reset_peak_memory_stats(device)
        self.sample("before_model_load")

    def sample(self, label: str) -> None:
        free_bytes = None
        total_bytes = None
        try:
            free_bytes, total_bytes = self.torch.cuda.mem_get_info(
                self.device
            )
        except Exception:
            pass
        self.samples.append(
            {
                "label": label,
                "cuda_free_bytes": free_bytes,
                "cuda_total_bytes": total_bytes,
                "torch_allocated_bytes": int(
                    self.torch.cuda.memory_allocated(self.device)
                ),
                "torch_reserved_bytes": int(
                    self.torch.cuda.memory_reserved(self.device)
                ),
                "process_peak_rss_bytes": _peak_process_rss_bytes(),
                "linux_memory": _linux_memory(),
                "cgroup_v2": _cgroup_v2_memory(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        self.sample("metadata_capture")
        cuda_free_values = [
            sample["cuda_free_bytes"]
            for sample in self.samples
            if isinstance(sample.get("cuda_free_bytes"), int)
        ]
        baseline_free = cuda_free_values[0] if cuda_free_values else None
        minimum_free = min(cuda_free_values) if cuda_free_values else None
        observed_delta = (
            baseline_free - minimum_free
            if baseline_free is not None and minimum_free is not None
            else None
        )
        linux_available_values = [
            sample["linux_memory"]["available_bytes"]
            for sample in self.samples
            if isinstance(sample.get("linux_memory"), dict)
            and isinstance(sample["linux_memory"].get("available_bytes"), int)
            and sample["linux_memory"]["available_bytes"] >= 0
        ]
        baseline_linux_available = (
            linux_available_values[0] if linux_available_values else None
        )
        minimum_linux_available = (
            min(linux_available_values) if linux_available_values else None
        )
        observed_linux_delta = (
            max(0, baseline_linux_available - minimum_linux_available)
            if baseline_linux_available is not None
            and minimum_linux_available is not None
            else None
        )
        return {
            "memory_kind": {
                True: "unified_system",
                False: "dedicated_vram",
                None: "unknown",
            }[self.integrated],
            "device_total_bytes": int(self.properties.total_memory),
            "baseline_cuda_free_bytes": baseline_free,
            "minimum_cuda_free_bytes": minimum_free,
            "observed_cuda_used_delta_bytes": observed_delta,
            "torch_peak_allocated_bytes": int(
                self.torch.cuda.max_memory_allocated(self.device)
            ),
            "torch_peak_reserved_bytes": int(
                self.torch.cuda.max_memory_reserved(self.device)
            ),
            "process_peak_rss_bytes": max(
                sample["process_peak_rss_bytes"] for sample in self.samples
            ),
            "linux_memory": dict(self.samples[-1]["linux_memory"]),
            "linux_baseline_mem_available_bytes": (baseline_linux_available),
            "linux_minimum_mem_available_bytes": (minimum_linux_available),
            "observed_linux_mem_available_delta_bytes": (
                observed_linux_delta
            ),
            "cgroup_v2_memory": _cgroup_v2_memory_summary(self.samples),
            "samples": list(self.samples),
            "authoritative_capacity_source": (
                "linux_mem_available"
                if self.integrated is True
                else (
                    "cuda_device_total"
                    if self.integrated is False
                    else "unknown"
                )
            ),
        }


def _cgroup_v2_memory_summary(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    records = [
        sample.get("cgroup_v2", {})
        for sample in samples
        if isinstance(sample.get("cgroup_v2"), dict)
    ]
    baseline_record = records[0] if records else {}
    limits = [
        record["memory_max_bytes"]
        for record in records
        if isinstance(record.get("memory_max_bytes"), int)
        and record["memory_max_bytes"] >= 0
    ]
    unlimited_observations = [
        record["memory_max_is_unlimited"]
        for record in records
        if isinstance(record.get("memory_max_is_unlimited"), bool)
    ]
    capacity = min(limits) if limits else None
    capacity_is_unlimited = (
        False
        if limits
        else (
            True
            if unlimited_observations and all(unlimited_observations)
            else None
        )
    )
    current_values = [
        record["memory_current_bytes"]
        for record in records
        if isinstance(record.get("memory_current_bytes"), int)
        and record["memory_current_bytes"] >= 0
    ]
    baseline_current = baseline_record.get("memory_current_bytes")
    if not isinstance(baseline_current, int) or baseline_current < 0:
        baseline_current = current_values[0] if current_values else None
    maximum_current = max(current_values) if current_values else None
    current_delta = (
        max(0, maximum_current - baseline_current)
        if baseline_current is not None and maximum_current is not None
        else None
    )
    peak_values = [
        record["memory_peak_bytes"]
        for record in records
        if isinstance(record.get("memory_peak_bytes"), int)
        and record["memory_peak_bytes"] >= 0
    ]
    baseline_peak = baseline_record.get("memory_peak_bytes")
    if not isinstance(baseline_peak, int) or baseline_peak < 0:
        baseline_peak = peak_values[0] if peak_values else None
    maximum_peak = max(peak_values) if peak_values else None
    peak_delta = (
        max(0, maximum_peak - baseline_peak)
        if baseline_peak is not None and maximum_peak is not None
        else None
    )
    headroom_values = [
        record["memory_available_bytes"]
        for record in records
        if isinstance(record.get("memory_available_bytes"), int)
        and record["memory_available_bytes"] >= 0
    ]
    baseline_headroom = baseline_record.get("memory_available_bytes")
    if not isinstance(baseline_headroom, int) or baseline_headroom < 0:
        baseline_headroom = headroom_values[0] if headroom_values else None
    minimum_headroom = min(headroom_values) if headroom_values else None
    headroom_delta = (
        max(0, baseline_headroom - minimum_headroom)
        if baseline_headroom is not None and minimum_headroom is not None
        else None
    )
    return {
        "available": any(
            record.get("available") is True for record in records
        ),
        "capacity_bytes": capacity,
        "capacity_is_unlimited": capacity_is_unlimited,
        "baseline_current_bytes": baseline_current,
        "maximum_current_bytes": maximum_current,
        "observed_current_delta_bytes": current_delta,
        "baseline_peak_bytes": baseline_peak,
        "maximum_peak_bytes": maximum_peak,
        "observed_peak_delta_bytes": peak_delta,
        "baseline_headroom_bytes": baseline_headroom,
        "minimum_headroom_bytes": minimum_headroom,
        "observed_headroom_delta_bytes": headroom_delta,
        "memory_peak_semantics": (
            "cgroup_lifetime_high_water_mark_not_reset_by_adapter"
        ),
    }


def apply_tuning_memory_budget(
    torch,
    device,
    settings: dict[str, Any],
) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    total_bytes = int(properties.total_memory)
    budget_bytes = int(settings["memory_budget_bytes"])
    if budget_bytes > total_bytes:
        raise RuntimeError(
            "tuning memory budget exceeds the CUDA-addressable total"
        )
    fraction = budget_bytes / total_bytes
    if not 0 < fraction <= 1:
        raise RuntimeError("tuning memory fraction is outside (0, 1]")
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    return {
        "memory_tier": settings["memory_tier"],
        "budget_bytes": budget_bytes,
        "allocator_fraction": fraction,
        "allocator_cap_applied": True,
    }


def runtime_metadata(
    model,
    processor,
    *,
    memory_tracker: CudaMemoryTracker | None = None,
    memory_budget: dict[str, Any] | None = None,
    execution_profile: dict[str, Any] | None = None,
    startup_installed_distributions: dict[str, Any],
) -> dict[str, Any]:
    implementation = implementation_record()
    if implementation["components"].get(
        "providers/mineru25.py"
    ) != sha256_path(Path(__file__).resolve()):
        raise RuntimeError(
            "executed MinerU adapter differs from installed implementation"
        )
    import torch
    import transformers

    device = torch.device(model.device)
    properties = torch.cuda.get_device_properties(device)
    attention_backend = getattr(model.config, "_attn_implementation", None)
    if not isinstance(attention_backend, str) or not attention_backend:
        attention_backend = "provider_managed"
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "engine": "transformers",
        "engine_version": transformers.__version__,
        "torch": str(torch.__version__),
        "packages": package_versions(
            [
                "torch",
                "torchvision",
                "transformers",
                "mineru-vl-utils",
                "pillow",
                "accelerate",
                "huggingface-hub",
            ]
        ),
        "installed_distributions": {
            "count": startup_installed_distributions["count"],
            "sha256": startup_installed_distributions["sha256"],
        },
        "installed_distributions_count": startup_installed_distributions[
            "count"
        ],
        "installed_distributions_sha256": startup_installed_distributions[
            "sha256"
        ],
        "attention_backend": attention_backend,
        "torch_cuda": str(torch.version.cuda),
        "model_class": model.__class__.__qualname__,
        "processor_class": processor.__class__.__qualname__,
        "dtype": str(model.dtype).removeprefix("torch."),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device": properties.name,
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_total_memory_bytes": properties.total_memory,
        "cuda_is_integrated": _cuda_integrated_property(properties),
        "gpu_memory": (
            memory_tracker.as_dict() if memory_tracker is not None else None
        ),
        "memory_budget": memory_budget,
        "execution_profile": execution_profile,
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "provider_adapter": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_path(Path(__file__).resolve()),
        },
        "implementation_sha256": implementation["sha256"],
        "implementation": implementation,
    }


def response_metadata(
    request: dict[str, Any],
    *,
    tracer: GenerationTracer,
    wall_seconds: float,
    model_id: str,
    revision: str,
    snapshot_record: dict[str, Any],
    runtime_record: dict[str, Any],
    pipeline_record: dict[str, Any],
) -> dict[str, Any]:
    traces = tracer.traces
    if not traces:
        raise RuntimeError("native MinerU pipeline recorded no generations")
    any_max = any(trace["max_tokens"] for trace in traces)
    all_eos = all(trace["eos"] for trace in traces)
    return {
        "effective_request": {
            "prompt": "\n\n--- MINERU STAGE ---\n\n".join(tracer.prompts),
            "input_artifacts": [
                {
                    **request["input"]["image"],
                    "transform": "original",
                }
            ],
            "processor_traces": [trace["input_shapes"] for trace in traces],
        },
        "model": {
            "requested": model_id,
            "revision": revision,
            "resolved_snapshot": snapshot_record["path"],
            "resolved_revision": snapshot_record["commit"],
            "snapshot_files": snapshot_record["files"],
            "snapshot_files_sha256": snapshot_record["files_sha256"],
        },
        "runtime": runtime_record,
        "generation": {
            "params": {
                "request_settings": request.get("settings", {}),
                "pipeline": pipeline_record,
                "stage_generate_kwargs": [
                    trace["generate_kwargs"] for trace in traces
                ],
            },
            "input_tokens": sum(trace["input_tokens"] for trace in traces),
            "output_tokens": sum(trace["output_tokens"] for trace in traces),
            "finish_reason": "length" if any_max else "stop",
            "stop_reason": (
                "max_tokens"
                if any_max
                else "eos"
                if all_eos
                else "pipeline_completed"
            ),
            "eos": all_eos,
            "max_tokens": any_max,
            "stage_count": len(traces),
        },
        "usage": {
            "wall_seconds": round(wall_seconds, 6),
            "cost_usd": 0.0,
        },
    }


def native_markdown(image, client) -> tuple[str, dict[str, Any]]:
    from mineru_vl_utils.post_process import json2md

    content = client.two_step_extract(image)
    markdown = json2md(content)
    if not markdown.endswith("\n"):
        markdown += "\n"
    return markdown, {"content_list": list(content)}


def resolve_snapshot(model_id: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
        )
    )


def validate_cuda_runtime(torch, device_name: str):
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("MinerU provider requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to the MinerU provider")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected CUDA device does not support bfloat16")
    return device


def error_response(
    request: dict[str, Any],
    *,
    artifacts_dir: Path,
    exc: Exception,
    phase: str,
    traceback_text: str,
    tracer: GenerationTracer | None,
) -> dict[str, Any]:
    attempt_id = request.get("attempt_id", "invalid-attempt")
    safe_name = (
        f"{attempt_id}-native.json"
        if _SHA256_RE.fullmatch(str(attempt_id))
        else f"invalid-{time.monotonic_ns()}-native.json"
    )
    artifact_path = artifacts_dir / safe_name
    write_json(
        artifact_path,
        {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "phase": phase,
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc) or repr(exc),
            },
            "traceback": traceback_text,
            "prompts": tracer.prompts if tracer is not None else [],
            "generation_traces": (
                tracer.traces if tracer is not None else []
            ),
        },
    )
    return {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "status": "error",
        "error": {
            "type": exc.__class__.__name__,
            "message": str(exc) or repr(exc),
            "phase": phase,
        },
        "output": {"format": "markdown", "text": ""},
        "artifacts": [
            {
                "relative_path": safe_name,
                "sha256": sha256_path(artifact_path),
            }
        ],
    }


def write_response(stream, response: dict[str, Any]) -> None:
    stream.write(json.dumps(response, ensure_ascii=False))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())


def main() -> None:
    args = parse_args()
    requests = read_jsonl(args.requests_jsonl)
    startup_installed_distributions = (
        startup_installed_distributions_identity()
    )
    responses_path = Path(args.responses_jsonl)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    responses_path.parent.mkdir(parents=True, exist_ok=True)

    with open(responses_path, "w", encoding="utf-8", newline="") as stream:
        tracer = None
        memory_tracker = None
        execution_settings = None
        execution_profile = None
        memory_budget = None
        try:
            profile_values = (
                args.execution_profile,
                args.profile_sha256,
                args.memory_tier,
            )
            if args.tuning and any(profile_values):
                raise ValueError(
                    "--tuning cannot be combined with an execution profile"
                )
            if any(profile_values) and not all(profile_values):
                raise ValueError(
                    "--execution-profile, --profile-sha256, and "
                    "--memory-tier must be supplied together"
                )
            if args.allow_provisional_profile and not args.execution_profile:
                raise ValueError(
                    "--allow-provisional-profile requires --execution-profile"
                )
            if not args.execution_profile:
                for request in requests:
                    validate_request(request, tuning=args.tuning)
            if args.tuning:
                if not requests:
                    raise ValueError("tuning request batch cannot be empty")
                execution_settings = validate_tuning_settings(
                    requests[0]["settings"]
                )
                if any(
                    request.get("settings") != execution_settings
                    for request in requests[1:]
                ):
                    raise ValueError(
                        "all tuning requests in one process must use the "
                        "same settings"
                    )
            elif args.execution_profile:
                if not requests:
                    raise ValueError(
                        "execution-profile request batch cannot be empty"
                    )
                execution_settings, execution_profile = load_profile_settings(
                    args.execution_profile,
                    expected_sha256=args.profile_sha256,
                    memory_tier=args.memory_tier,
                    allow_provisional=args.allow_provisional_profile,
                    model_id=args.model_id,
                    revision=args.revision,
                )
                for request in requests:
                    validate_request(
                        request,
                        profile_settings=execution_settings,
                    )
            import torch
            from mineru_vl_utils import MinerUClient
            from PIL import Image
            from transformers import (
                AutoProcessor,
                Qwen2VLForConditionalGeneration,
            )

            device = validate_cuda_runtime(torch, args.device)
            memory_tracker = CudaMemoryTracker(torch, device)
            if execution_settings is not None:
                memory_budget = apply_tuning_memory_budget(
                    torch,
                    device,
                    execution_settings,
                )
            snapshot = resolve_snapshot(args.model_id, args.revision)
            snapshot_record = snapshot_identity(snapshot, args.revision)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                snapshot,
                dtype="bfloat16",
                device_map=args.device,
                attn_implementation=args.attention_backend,
                local_files_only=True,
            )
            processor = AutoProcessor.from_pretrained(
                snapshot,
                use_fast=True,
                local_files_only=True,
            )
            memory_tracker.sample("after_model_load")
            tracer = GenerationTracer(model, processor)
            tracer.install()
            client = MinerUClient(
                backend="transformers",
                model=model,
                processor=processor,
                image_analysis=(
                    execution_settings or QUALIFIED_REQUEST_SETTINGS
                )["image_analysis"],
                batch_size=(execution_settings or QUALIFIED_REQUEST_SETTINGS)[
                    "batch_size"
                ],
                use_tqdm=False,
            )
            runtime_record = runtime_metadata(
                model,
                processor,
                memory_tracker=None,
                memory_budget=memory_budget,
                execution_profile=execution_profile,
                startup_installed_distributions=(
                    startup_installed_distributions
                ),
            )
            pipeline_record = pipeline_configuration(client, model)
        except Exception as exc:
            traceback_text = traceback.format_exc()
            for request in requests:
                write_response(
                    stream,
                    error_response(
                        request,
                        artifacts_dir=artifacts_dir,
                        exc=exc,
                        phase="initialization",
                        traceback_text=traceback_text,
                        tracer=tracer,
                    ),
                )
            return

        for request in requests:
            tracer.reset()
            started = time.monotonic()
            attempt_id = request.get("attempt_id", "invalid-attempt")
            artifact_name = f"{attempt_id}-native.json"
            artifact_path = artifacts_dir / artifact_name
            try:
                validate_request(
                    request,
                    tuning=args.tuning,
                    profile_settings=(
                        execution_settings if execution_profile else None
                    ),
                )
                image_record = request["input"]["image"]
                image_path = Path(image_record["path"])
                if sha256_path(image_path) != image_record["sha256"]:
                    raise RuntimeError(
                        "input image SHA-256 mismatch in provider"
                    )
                with Image.open(image_path) as opened:
                    if {
                        "w": opened.size[0],
                        "h": opened.size[1],
                    } != image_record["dimensions"]:
                        raise RuntimeError(
                            "input image dimensions mismatch in provider"
                        )
                    with opened.convert("RGB") as image:
                        markdown, native = native_markdown(image, client)
                memory_tracker.sample(f"after_page:{attempt_id}")
                raw_artifact = {
                    "schema_version": 1,
                    "attempt_id": attempt_id,
                    "native": native,
                    "prompts": tracer.prompts,
                    "generation_traces": tracer.traces,
                }
                write_json(artifact_path, raw_artifact)
                metadata = response_metadata(
                    request,
                    tracer=tracer,
                    wall_seconds=time.monotonic() - started,
                    model_id=args.model_id,
                    revision=args.revision,
                    snapshot_record=snapshot_record,
                    runtime_record={
                        **runtime_record,
                        "gpu_memory": memory_tracker.as_dict(),
                    },
                    pipeline_record=pipeline_record,
                )
                response = {
                    "schema_version": 1,
                    "attempt_id": attempt_id,
                    "status": "completed",
                    "output": {"format": "markdown", "text": markdown},
                    **metadata,
                    "artifacts": [
                        {
                            "relative_path": artifact_name,
                            "sha256": sha256_path(artifact_path),
                        }
                    ],
                }
            except Exception as exc:
                response = error_response(
                    request,
                    artifacts_dir=artifacts_dir,
                    exc=exc,
                    phase="page_inference",
                    traceback_text=traceback.format_exc(),
                    tracer=tracer,
                )
            write_response(stream, response)


if __name__ == "__main__":
    main()
