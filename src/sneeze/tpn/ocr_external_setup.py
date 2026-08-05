"""Offline provider-spec construction for hosted OCR experiments."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from . import ocr_external
from .providers import external_api

__all__ = [
    "ExternalSetupError",
    "build_external_provider_spec",
    "external_executor_bundle_identity",
    "external_provider_spec_sha256",
    "validate_external_provider_spec",
    "write_external_provider_spec",
]


class ExternalSetupError(ValueError):
    """Raised when a hosted provider spec cannot be built safely."""


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + ("\n" if pretty else "")).encode(
        "utf-8"
    )


def external_executor_bundle_identity(
    python_executable: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return the adapter/helper bundle identity required by new plans."""

    return external_api.executor_bundle_identity(python_executable)


def validate_external_provider_spec(
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one exact hosted provider spec."""

    try:
        return external_api.validate_external_provider_spec_record(spec)
    except (external_api.ExternalAdapterError, TypeError) as exc:
        raise ExternalSetupError(str(exc)) from exc


def external_provider_spec_sha256(spec: dict[str, Any]) -> str:
    """Hash the normalized provider spec as the benchmark harness does."""

    return hashlib.sha256(
        _json_bytes(validate_external_provider_spec(spec))
    ).hexdigest()


def build_external_provider_spec(
    *,
    profile_id: str,
    python_executable: str | os.PathLike[str],
    lane_id: str,
    prompt: str,
    prompt_mode: str,
    include_raw_text: bool = False,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Build a secret-free spec for one dated profile and one exact lane."""

    if lane_id == "parity" and include_raw_text:
        raise ExternalSetupError(
            "the parity lane cannot include raw text"
        )
    profile = ocr_external.get_external_provider_profile(profile_id)
    profile_sha256 = profile.pop("profile_sha256")
    executable = Path(python_executable).expanduser()
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
    ):
        raise ExternalSetupError(
            "python_executable must be an absolute executable file"
        )
    if not isinstance(include_raw_text, bool):
        raise ExternalSetupError(
            "include_raw_text must be an explicit boolean"
        )
    adapter_path = Path(external_api.__file__).resolve()
    bundle = external_executor_bundle_identity(executable)
    spec = {
        "schema_version": 1,
        "provider_id": profile_id,
        "argv": [
            str(executable.resolve()),
            "-I",
            "-S",
            "-B",
            str(adapter_path),
            "--profile-id",
            profile_id,
            "--lane-id",
            lane_id,
        ],
        "egress": "external",
        "model": {
            "requested": profile["model"]["requested"],
            "revision": profile["model"]["revision_semantics"],
        },
        "lanes": {
            lane_id: {
                "prompt": prompt,
                "settings": {
                    "profile_id": profile_id,
                    "prompt_mode": prompt_mode,
                },
                "include_raw_text": include_raw_text,
            }
        },
        "timeout_seconds": timeout_seconds,
        "secret_env_names": [profile["credential_env_name"]],
        "env": {},
        "external_policy": {
            "profile_id": profile_id,
            "profile_sha256": profile_sha256,
            "executor_bundle_sha256": bundle["bundle_sha256"],
            "python_executable": bundle["python_executable"],
            "credential_env_name": profile["credential_env_name"],
            "endpoint": profile["endpoint"],
            "lane_id": lane_id,
        },
    }
    return validate_external_provider_spec(spec)


def _write_new_atomic(path: Path, payload: bytes) -> None:
    if not path.is_absolute():
        raise ExternalSetupError("provider spec output must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".candidate",
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ExternalSetupError(
                f"refusing to replace existing provider spec: {path}"
            ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_external_provider_spec(
    *,
    output_path: str | os.PathLike[str],
    profile_id: str,
    python_executable: str | os.PathLike[str],
    lane_id: str,
    prompt: str,
    prompt_mode: str,
    include_raw_text: bool = False,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Atomically write a new secret-free spec and return its bindings."""

    spec = build_external_provider_spec(
        profile_id=profile_id,
        python_executable=python_executable,
        lane_id=lane_id,
        prompt=prompt,
        prompt_mode=prompt_mode,
        include_raw_text=include_raw_text,
        timeout_seconds=timeout_seconds,
    )
    destination = Path(output_path).expanduser()
    payload = _json_bytes(spec, pretty=True)
    _write_new_atomic(destination, payload)
    bundle = external_executor_bundle_identity(spec["argv"][0])
    return {
        "schema_version": 1,
        "kind": "external_provider_spec",
        "output_path": str(destination),
        "provider_spec_sha256": external_provider_spec_sha256(spec),
        "provider_spec_file_sha256": hashlib.sha256(payload).hexdigest(),
        "executor_bundle_sha256": bundle["bundle_sha256"],
        "provider_id": spec["provider_id"],
        "model": spec["model"],
        "lane_id": lane_id,
        "credential_env_name": spec["secret_env_names"][0],
    }
