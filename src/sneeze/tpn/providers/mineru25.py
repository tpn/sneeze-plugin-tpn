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
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-jsonl", required=True)
    parser.add_argument("--responses-jsonl", required=True)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention-backend", default="sdpa")
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    values = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"request at line {line_number} must be an object"
                )
            values.append(value)
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
    }


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
        return {
            str(key): serializable(item) for key, item in value.items()
        }
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
                input_tokens = (
                    int(attention_mask[row_index].sum().item())
                    if attention_mask is not None
                    else input_width
                )
                reached_limit = (
                    max_new_tokens is not None
                    and len(generated) >= max_new_tokens
                ) or (
                    max_length is not None and len(ids) >= max_length
                )
                reached_eos = bool(
                    generated and generated[-1] in eos_ids
                )
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


def validate_request(request: dict[str, Any]) -> None:
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
    if settings != QUALIFIED_REQUEST_SETTINGS:
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
        "sampling_params_by_block_type": serializable(
            client.sampling_params
        ),
        "layout_image_size": list(helper.layout_image_size),
        "min_image_edge": helper.min_image_edge,
        "max_image_edge_ratio": helper.max_image_edge_ratio,
        "simple_post_process": helper.simple_post_process,
        "handle_equation_block": helper.handle_equation_block,
        "abandon_list": helper.abandon_list,
        "abandon_paratext": helper.abandon_paratext,
        "image_analysis": helper.image_analysis,
        "enable_table_formula_eq_wrap": (
            helper.enable_table_formula_eq_wrap
        ),
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


def runtime_metadata(model, processor) -> dict[str, Any]:
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
        "installed_distributions": installed_distributions_identity(),
        "attention_backend": attention_backend,
        "torch_cuda": str(torch.version.cuda),
        "model_class": model.__class__.__qualname__,
        "processor_class": processor.__class__.__qualname__,
        "dtype": str(model.dtype),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device": properties.name,
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_total_memory_bytes": properties.total_memory,
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
            "prompt": "\n\n--- MINERU STAGE ---\n\n".join(
                tracer.prompts
            ),
            "input_artifacts": [
                {
                    **request["input"]["image"],
                    "transform": "original",
                }
            ],
            "processor_traces": [
                trace["input_shapes"] for trace in traces
            ],
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
            "input_tokens": sum(
                trace["input_tokens"] for trace in traces
            ),
            "output_tokens": sum(
                trace["output_tokens"] for trace in traces
            ),
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
    responses_path = Path(args.responses_jsonl)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    responses_path.parent.mkdir(parents=True, exist_ok=True)

    with open(responses_path, "w", encoding="utf-8", newline="") as stream:
        tracer = None
        try:
            import torch
            from mineru_vl_utils import MinerUClient
            from PIL import Image
            from transformers import (
                AutoProcessor,
                Qwen2VLForConditionalGeneration,
            )

            validate_cuda_runtime(torch, args.device)
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
            tracer = GenerationTracer(model, processor)
            tracer.install()
            client = MinerUClient(
                backend="transformers",
                model=model,
                processor=processor,
                image_analysis=QUALIFIED_REQUEST_SETTINGS["image_analysis"],
                batch_size=QUALIFIED_REQUEST_SETTINGS["batch_size"],
                use_tqdm=False,
            )
            runtime_record = runtime_metadata(model, processor)
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
                validate_request(request)
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
                    runtime_record=runtime_record,
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
