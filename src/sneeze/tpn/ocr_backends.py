"""Pinned, lightweight OCR backend definitions for the TPN plugin.

This module must remain safe to import from CLI discovery and help.  In
particular, it deliberately has no imports from Torch, Transformers, vLLM, or
provider runtime packages.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = 1

DEEPSEEK_BACKEND_ID = "deepseek-vllm"
DEEPSEEK_MODEL_ID = "deepseek-ai/DeepSeek-OCR"
DEEPSEEK_MODEL_REVISION = "9f30c71f441d010e5429c532364a86705536c53a"

MINERU_BACKEND_ID = "mineru25-pro-2605"
MINERU_MODEL_ID = "opendatalab/MinerU2.5-Pro-2605-1.2B"
MINERU_MODEL_REVISION = "bff20d4ae2bf202df9f45284b4d43681555a97ed"
MINERU_PROVIDER_ID = "mineru2.5-pro-2605-transformers"
MINERU_ADAPTER_SHA256 = (
    "2ed21528fe34a01d5e897e1e50fe8162a71b14947dc1b6bf62f8120314db20d2"
)
MINERU_SNAPSHOT_IDENTITY_SHA256 = (
    "d817d96ca8d0fda357b67054840c86664e5c26bbbfa9eedc05387978d3a157ef"
)
MINERU_WEIGHTS_SHA256 = (
    "abf8681ca63b8dec7b67de257af47b821f179442f72998d0696ae2ed9232a5f0"
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
    provider_id: str | None = None
    adapter_relative_path: str | None = None
    default_lane: str | None = None
    provider_prompt: str | None = None
    provider_settings: tuple[tuple[str, Any], ...] = ()
    runtime_packages: tuple[tuple[str, str], ...] = ()
    qualified_adapter_sha256: str | None = None
    snapshot_identity_sha256: str | None = None
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
        if not self.default_env_name or not self.default_python:
            raise BackendRegistryError(
                "backend environment defaults cannot be empty"
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
                "adapter_sha256": self.qualified_adapter_sha256,
                "snapshot_identity_sha256": (
                    self.snapshot_identity_sha256
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
        runtime_packages=(
            ("huggingface-hub", "1.17.0"),
            ("pdf2image", "1.17.0"),
            ("pillow", "12.2.0"),
            ("torch", "2.11.0"),
            ("torchaudio", "2.11.0+cu130"),
            ("torchvision", "0.26.0"),
            ("tqdm", "4.67.3"),
            ("transformers", "5.9.0"),
            ("vllm", "0.21.1rc1.dev438+g4ff865c38"),
        ),
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
