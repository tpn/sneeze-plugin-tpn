"""Pinned, lightweight OCR backend definitions for the TPN plugin.

This module must remain safe to import from CLI discovery and help.  In
particular, it deliberately has no imports from Torch, Transformers, vLLM, or
provider runtime packages.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any

SCHEMA_VERSION = 1

DEEPSEEK_BACKEND_ID = "deepseek-vllm"
DEEPSEEK_MODEL_ID = "deepseek-ai/DeepSeek-OCR"
DEEPSEEK_MODEL_REVISION = "9f30c71f441d010e5429c532364a86705536c53a"
DEEPSEEK_SNAPSHOT_IDENTITY_SHA256 = (
    "d3441a0d86320dbaaf7d3fce072e0f7be2f698975de42974fe987a320f3ef9d2"
)
DEEPSEEK_SNAPSHOT_REQUIRED_FILES = (
    "config.json",
    "configuration_deepseek_v2.py",
    "conversation.py",
    "deepencoder.py",
    "model-00001-of-000001.safetensors",
    "model.safetensors.index.json",
    "modeling_deepseekocr.py",
    "modeling_deepseekv2.py",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
DEEPSEEK_SNAPSHOT_IGNORED_FILES = (
    ".gitattributes",
    ".ipynb_checkpoints/README-checkpoint.md",
    "LICENSE",
    "README.md",
    "assets/fig1.png",
    "assets/show1.jpg",
    "assets/show2.jpg",
    "assets/show3.jpg",
    "assets/show4.jpg",
)

MINERU_BACKEND_ID = "mineru25-pro-2605"
MINERU_MODEL_ID = "opendatalab/MinerU2.5-Pro-2605-1.2B"
MINERU_MODEL_REVISION = "bff20d4ae2bf202df9f45284b4d43681555a97ed"
MINERU_PROVIDER_ID = "mineru2.5-pro-2605-transformers"
MINERU_ADAPTER_SHA256 = (
    "0e64a995532b598315a82b711cf739af46ae64131dd9654e4730af3ba6ef2696"
)
MINERU_SNAPSHOT_IDENTITY_SHA256 = (
    "d817d96ca8d0fda357b67054840c86664e5c26bbbfa9eedc05387978d3a157ef"
)
MINERU_WEIGHTS_SHA256 = (
    "abf8681ca63b8dec7b67de257af47b821f179442f72998d0696ae2ed9232a5f0"
)

# Updated only after the declared implementation components have passed the
# focused unit suite and GPU qualification.  The placeholders are replaced at
# the end of a source-changing qualification cycle.
DEEPSEEK_IMPLEMENTATION_SHA256 = (
    "57a50cdd0812a0ad1f406d11200850ed69e05a4df1d22f5b2f6164904cd34aac"
)
MINERU_IMPLEMENTATION_SHA256 = (
    "688020ba33a8f381ea63fc8f9a9c7bac2bc81dd2c093b8836c6b2464850e887e"
)

_REVISION_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_BACKEND_ID_RE = re.compile(r"^[a-z0-9]+(?:[a-z0-9.-]*[a-z0-9])?$")
_EXECUTION_MODES = frozenset({"in_process", "provider_process"})
_QUALIFICATION_STATES = frozenset(
    {"qualified", "guarded", "benchmark_only", "disqualified"}
)


class BackendRegistryError(ValueError):
    """Raised when a backend definition or lookup is invalid."""


@dataclass(frozen=True)
class CapabilityQualification:
    """Qualification status for one pipeline capability."""

    capability: str
    status: str
    evidence: str

    def __post_init__(self) -> None:
        if not self.capability:
            raise BackendRegistryError("backend capability cannot be empty")
        if self.status not in _QUALIFICATION_STATES:
            raise BackendRegistryError(
                f"unsupported qualification state: {self.status}"
            )
        if not self.evidence:
            raise BackendRegistryError(
                "backend capability evidence cannot be empty"
            )


@dataclass(frozen=True)
class OcrBackendDefinition:
    """Immutable setup and execution metadata for one OCR backend."""

    backend_id: str
    aliases: tuple[str, ...]
    display_name: str
    execution_mode: str
    model_id: str
    model_revision: str
    default_env_name: str
    default_python: str
    capabilities: tuple[CapabilityQualification, ...]
    environment_kind: str
    runtime_engine: str
    runtime_engine_package: str
    qualified_torch_version: str
    qualified_torch_cuda: str
    qualified_dtype: str
    provider_id: str | None = None
    adapter_relative_path: str | None = None
    default_lane: str | None = None
    provider_prompt: str | None = None
    provider_settings: tuple[tuple[str, Any], ...] = ()
    runtime_packages: tuple[tuple[str, str], ...] = ()
    implementation_relative_paths: tuple[str, ...] = ()
    qualified_implementation_sha256: str | None = None
    qualified_adapter_sha256: str | None = None
    snapshot_identity_sha256: str | None = None
    snapshot_required_files: tuple[str, ...] = ()
    snapshot_ignored_files: tuple[str, ...] = ()
    weights_sha256: str | None = None
    default_timeout_seconds: int = 3600
    default_device: str = "cuda:0"
    default_attention_backend: str = "sdpa"

    def __post_init__(self) -> None:
        identifiers = (self.backend_id, *self.aliases)
        if any(not _BACKEND_ID_RE.fullmatch(item) for item in identifiers):
            raise BackendRegistryError(
                f"invalid backend identifier in {identifiers!r}"
            )
        if self.execution_mode not in _EXECUTION_MODES:
            raise BackendRegistryError(
                f"unsupported execution mode: {self.execution_mode}"
            )
        if not self.display_name or not self.model_id:
            raise BackendRegistryError(
                "backend display name and model ID cannot be empty"
            )
        if not _REVISION_RE.fullmatch(self.model_revision):
            raise BackendRegistryError(
                "backend model revision must be an immutable SHA"
            )
        for label, paths in (
            ("required", self.snapshot_required_files),
            ("ignored", self.snapshot_ignored_files),
        ):
            if (
                not isinstance(paths, tuple)
                or len(set(paths)) != len(paths)
                or tuple(sorted(paths)) != paths
            ):
                raise BackendRegistryError(
                    f"snapshot {label} files must be a sorted unique tuple"
                )
            for relative in paths:
                path = PurePosixPath(relative)
                if (
                    not relative
                    or path.is_absolute()
                    or str(path) != relative
                    or any(part in {".", ".."} for part in path.parts)
                ):
                    raise BackendRegistryError(
                        f"snapshot {label} files must be canonical relative "
                        "paths"
                    )
        if set(self.snapshot_required_files) & set(
            self.snapshot_ignored_files
        ):
            raise BackendRegistryError(
                "snapshot required and ignored files must be disjoint"
            )
        if not self.default_env_name or not self.default_python:
            raise BackendRegistryError(
                "backend environment defaults cannot be empty"
            )
        runtime_packages = dict(self.runtime_packages)
        if (
            not self.runtime_engine
            or not self.runtime_engine_package
            or not self.qualified_torch_version
            or not self.qualified_torch_cuda
            or not self.qualified_dtype
            or self.runtime_engine_package not in runtime_packages
        ):
            raise BackendRegistryError(
                "backend qualified runtime identity is incomplete"
            )
        if not self.capabilities:
            raise BackendRegistryError(
                "backend must declare at least one capability"
            )
        capability_names = [item.capability for item in self.capabilities]
        if len(capability_names) != len(set(capability_names)):
            raise BackendRegistryError(
                f"duplicate capabilities for backend {self.backend_id}"
            )
        if not self.implementation_relative_paths:
            raise BackendRegistryError(
                "backend must declare implementation components"
            )
        for value in self.implementation_relative_paths:
            path = PurePosixPath(value)
            if (
                path.is_absolute()
                or not path.parts
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise BackendRegistryError(
                    "backend implementation component must be a safe "
                    "relative path"
                )
        if self.execution_mode == "provider_process":
            required = (
                self.provider_id,
                self.adapter_relative_path,
                self.default_lane,
                self.provider_prompt,
                self.qualified_adapter_sha256,
            )
            if any(not item for item in required):
                raise BackendRegistryError(
                    "provider-process backends require provider metadata"
                )
        elif any(
            item is not None
            for item in (
                self.provider_id,
                self.adapter_relative_path,
                self.default_lane,
                self.provider_prompt,
            )
        ):
            raise BackendRegistryError(
                "in-process backends cannot declare provider metadata"
            )
        for label, digest in (
            (
                "qualified implementation",
                self.qualified_implementation_sha256,
            ),
            ("qualified adapter", self.qualified_adapter_sha256),
            ("snapshot identity", self.snapshot_identity_sha256),
            ("weights", self.weights_sha256),
        ):
            if digest is not None and not re.fullmatch(
                r"[0-9a-f]{64}",
                digest,
            ):
                raise BackendRegistryError(
                    f"backend {label} SHA-256 is invalid"
                )

    def capability_status(self, capability: str) -> str | None:
        for item in self.capabilities:
            if item.capability == capability:
                return item.status
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "backend_id": self.backend_id,
            "aliases": list(self.aliases),
            "display_name": self.display_name,
            "execution_mode": self.execution_mode,
            "model": {
                "requested": self.model_id,
                "revision": self.model_revision,
            },
            "environment": {
                "kind": self.environment_kind,
                "default_name": self.default_env_name,
                "python": self.default_python,
                "runtime_packages": dict(self.runtime_packages),
                "engine": self.runtime_engine,
                "engine_package": self.runtime_engine_package,
                "torch": self.qualified_torch_version,
                "torch_cuda": self.qualified_torch_cuda,
                "dtype": self.qualified_dtype,
            },
            "capabilities": [asdict(item) for item in self.capabilities],
            "provider": (
                {
                    "provider_id": self.provider_id,
                    "adapter_relative_path": self.adapter_relative_path,
                    "default_lane": self.default_lane,
                    "prompt": self.provider_prompt,
                    "settings": dict(self.provider_settings),
                    "default_timeout_seconds": (
                        self.default_timeout_seconds
                    ),
                    "default_device": self.default_device,
                    "default_attention_backend": (
                        self.default_attention_backend
                    ),
                }
                if self.execution_mode == "provider_process"
                else None
            ),
            "qualification_artifacts": {
                "implementation_sha256": (
                    self.qualified_implementation_sha256
                ),
                "implementation_components": list(
                    self.implementation_relative_paths
                ),
                "adapter_sha256": self.qualified_adapter_sha256,
                "snapshot_identity_sha256": (
                    self.snapshot_identity_sha256
                ),
                "snapshot_required_files": list(
                    self.snapshot_required_files
                ),
                "snapshot_ignored_files": list(
                    self.snapshot_ignored_files
                ),
                "weights_sha256": self.weights_sha256,
            },
        }


_BACKENDS = (
    OcrBackendDefinition(
        backend_id=DEEPSEEK_BACKEND_ID,
        aliases=("deepseek",),
        display_name="DeepSeek-OCR with vLLM",
        execution_mode="in_process",
        model_id=DEEPSEEK_MODEL_ID,
        model_revision=DEEPSEEK_MODEL_REVISION,
        default_env_name="sneeze-ocr-vllm",
        default_python="3.12.13",
        environment_kind="deepseek_vllm",
        runtime_engine="vllm",
        runtime_engine_package="vllm",
        qualified_torch_version="2.11.0+cu130",
        qualified_torch_cuda="13.0",
        qualified_dtype="bfloat16",
        runtime_packages=(
            ("huggingface-hub", "1.17.0"),
            ("pdf2image", "1.17.0"),
            ("pillow", "12.2.0"),
            ("torch", "2.11.0+cu130"),
            ("torchaudio", "2.11.0+cu130"),
            ("torchvision", "0.26.0+cu130"),
            ("tqdm", "4.67.3"),
            ("transformers", "5.9.0"),
            ("vllm", "0.21.1rc1.dev438+g4ff865c38"),
        ),
        implementation_relative_paths=(
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
            "ocr_vllm.py",
            "providers/deepseek_vllm.py",
        ),
        qualified_implementation_sha256=(
            DEEPSEEK_IMPLEMENTATION_SHA256
        ),
        snapshot_identity_sha256=DEEPSEEK_SNAPSHOT_IDENTITY_SHA256,
        snapshot_required_files=DEEPSEEK_SNAPSHOT_REQUIRED_FILES,
        snapshot_ignored_files=DEEPSEEK_SNAPSHOT_IGNORED_FILES,
        capabilities=(
            CapabilityQualification(
                capability="grounded_ocr",
                status="qualified",
                evidence=(
                    "Materially complete grounded OCR on all three cuPhoton "
                    "severe-regression pages."
                ),
            ),
            CapabilityQualification(
                capability="markdown",
                status="guarded",
                evidence=(
                    "Retained as a fail-closed fallback after severe "
                    "ungrounded Markdown regressions."
                ),
            ),
            CapabilityQualification(
                capability="html",
                status="guarded",
                evidence=(
                    "Legacy single-output mode; not in the sealed suite."
                ),
            ),
        ),
    ),
    OcrBackendDefinition(
        backend_id=MINERU_BACKEND_ID,
        aliases=("mineru25",),
        display_name="MinerU2.5-Pro-2605",
        execution_mode="provider_process",
        model_id=MINERU_MODEL_ID,
        model_revision=MINERU_MODEL_REVISION,
        default_env_name="sneeze-ocr-mineru25",
        default_python="3.12.13",
        environment_kind="mineru25_transformers",
        runtime_engine="transformers",
        runtime_engine_package="transformers",
        qualified_torch_version="2.11.0+cu130",
        qualified_torch_cuda="13.0",
        qualified_dtype="bfloat16",
        capabilities=(
            CapabilityQualification(
                capability="markdown",
                status="qualified",
                evidence=(
                    "Passed 15/15 severe repeats, 46/46 development pages, "
                    "and 14/14 sealed holdout pages."
                ),
            ),
        ),
        provider_id=MINERU_PROVIDER_ID,
        adapter_relative_path="providers/mineru25.py",
        default_lane="best",
        provider_prompt=(
            "Use the official MinerU two-stage layout and recognition "
            "pipeline and convert its structured result to Markdown."
        ),
        provider_settings=(
            ("backend", "official-two-step-transformers"),
            ("batch_size", 1),
            ("image_analysis", False),
        ),
        runtime_packages=(
            ("accelerate", "1.14.0"),
            ("huggingface-hub", "0.36.2"),
            ("mineru-vl-utils", "1.0.5"),
            ("pillow", "12.2.0"),
            ("torch", "2.11.0+cu130"),
            ("torchvision", "0.26.0+cu130"),
            ("transformers", "4.57.6"),
        ),
        implementation_relative_paths=(
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
        ),
        qualified_implementation_sha256=MINERU_IMPLEMENTATION_SHA256,
        qualified_adapter_sha256=MINERU_ADAPTER_SHA256,
        snapshot_identity_sha256=MINERU_SNAPSHOT_IDENTITY_SHA256,
        weights_sha256=MINERU_WEIGHTS_SHA256,
    ),
)

_BY_NAME: dict[str, OcrBackendDefinition] = {}
_BY_PROVIDER_ID: dict[str, OcrBackendDefinition] = {}
for _backend in _BACKENDS:
    for _name in (_backend.backend_id, *_backend.aliases):
        if _name in _BY_NAME:
            raise BackendRegistryError(f"duplicate backend name: {_name}")
        _BY_NAME[_name] = _backend
    if _backend.provider_id:
        if _backend.provider_id in _BY_PROVIDER_ID:
            raise BackendRegistryError(
                f"duplicate provider ID: {_backend.provider_id}"
            )
        _BY_PROVIDER_ID[_backend.provider_id] = _backend


def list_backends() -> tuple[OcrBackendDefinition, ...]:
    """Return all built-in backends in stable ID order."""

    return tuple(sorted(_BACKENDS, key=lambda item: item.backend_id))


def get_backend(name: str) -> OcrBackendDefinition:
    """Resolve one canonical backend ID or supported alias."""

    try:
        return _BY_NAME[name.strip().lower()]
    except (AttributeError, KeyError) as exc:
        choices = ", ".join(item.backend_id for item in list_backends())
        raise BackendRegistryError(
            f"unknown OCR backend {name!r}; expected one of: {choices}"
        ) from exc


def get_backend_for_provider(
    provider_id: str,
) -> OcrBackendDefinition:
    """Resolve the built-in backend that owns a provider specification."""

    try:
        return _BY_PROVIDER_ID[provider_id]
    except (TypeError, KeyError) as exc:
        raise BackendRegistryError(
            f"unknown built-in OCR provider ID: {provider_id!r}"
        ) from exc
