"""Portable hashes for the code that qualifies and executes OCR stages.

Execution profiles bind exact model bytes and package versions, but those
identities are insufficient when the plugin itself contains prompt, batching,
provider, or validation logic.  This module hashes the source components
declared by each built-in backend without recording host paths.

The backend registry stores the reviewed aggregate digest.  Registration,
tuning, and production execution all recompute the same record, so changing an
in-scope source file invalidates both the installation and any older profile.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path, PurePosixPath
from typing import Any

from .ocr_path_security import TrustedPathError, validate_trusted_file


class OcrImplementationError(ValueError):
    """Raised when a backend implementation cannot be attested."""


_DEFINITION_EXCLUDED_FIELDS = frozenset({"qualified_implementation_sha256"})


def _file_sha256(path: Path) -> str:
    try:
        path = validate_trusted_file(
            path,
            label="OCR implementation component",
        )
    except TrustedPathError as exc:
        raise OcrImplementationError(str(exc)) from exc
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise OcrImplementationError(
            f"could not hash OCR implementation component: {path.name}"
        ) from exc
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _backend_definition_record(backend: Any) -> dict[str, Any]:
    """Return execution-shaping metadata without its self-referential seal."""

    value = asdict(backend)
    for name in _DEFINITION_EXCLUDED_FIELDS:
        if name not in value:
            raise OcrImplementationError(
                "OCR backend definition schema is incomplete"
            )
        value.pop(name)
    # Use the same JSON-native representation before and after a registry
    # round trip. dataclasses.asdict preserves tuples, while JSON does not.
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def backend_implementation_record(backend_id: str) -> dict[str, Any]:
    """Return the exact portable implementation identity for one backend."""

    from .ocr_backends import get_backend

    backend = get_backend(backend_id)
    root = Path(__file__).resolve().parent
    components: dict[str, str] = {}
    for raw_relative in backend.implementation_relative_paths:
        relative = PurePosixPath(raw_relative)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise OcrImplementationError(
                f"invalid implementation component for {backend.backend_id}"
            )
        candidate = root.joinpath(*relative.parts)
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise OcrImplementationError(
                f"missing implementation component for {backend.backend_id}"
            ) from exc
        if not resolved.is_file() or not resolved.is_relative_to(root):
            raise OcrImplementationError(
                f"unsafe implementation component for {backend.backend_id}"
            )
        components[relative.as_posix()] = _file_sha256(resolved)
    if not components:
        raise OcrImplementationError(
            f"backend {backend.backend_id} declares no implementation"
        )
    unsigned = {
        "schema_version": 2,
        "backend_id": backend.backend_id,
        "components": {key: components[key] for key in sorted(components)},
        "definition": _backend_definition_record(backend),
    }
    return {**unsigned, "sha256": _json_sha256(unsigned)}


def validate_backend_implementation_record(
    value: Any,
    *,
    backend_id: str,
    require_current: bool = False,
) -> dict[str, Any]:
    """Validate one portable implementation record and optional live match."""

    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "backend_id",
        "components",
        "definition",
        "sha256",
    }:
        raise OcrImplementationError(
            "OCR implementation record has unexpected or missing fields"
        )
    if value["schema_version"] != 2 or value["backend_id"] != backend_id:
        raise OcrImplementationError("OCR implementation identity differs")
    components = value["components"]
    if not isinstance(components, dict) or not components:
        raise OcrImplementationError(
            "OCR implementation components must be a nonempty object"
        )
    for relative, digest in components.items():
        path = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            path is None
            or path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef" for character in digest
            )
        ):
            raise OcrImplementationError(
                "OCR implementation component record is invalid"
            )
    digest = value["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise OcrImplementationError("OCR implementation SHA-256 is invalid")
    unsigned = {key: value[key] for key in value if key != "sha256"}
    if _json_sha256(unsigned) != digest:
        raise OcrImplementationError("OCR implementation hash mismatch")

    from .ocr_backends import get_backend

    backend = get_backend(backend_id)
    definition = value["definition"]
    expected_definition_fields = {
        item.name for item in fields(type(backend))
    } - _DEFINITION_EXCLUDED_FIELDS
    if (
        not isinstance(definition, dict)
        or set(definition) != expected_definition_fields
        or definition.get("backend_id") != backend.backend_id
    ):
        raise OcrImplementationError(
            "OCR backend definition identity is invalid"
        )
    expected = backend.qualified_implementation_sha256
    if digest != expected:
        raise OcrImplementationError(
            f"OCR implementation is not qualified for {backend.backend_id}"
        )
    if require_current and value != backend_implementation_record(backend_id):
        raise OcrImplementationError(
            f"stale OCR implementation for {backend.backend_id}"
        )
    return {
        "schema_version": 2,
        "backend_id": backend.backend_id,
        "components": {key: components[key] for key in sorted(components)},
        "definition": definition,
        "sha256": digest,
    }


__all__ = [
    "OcrImplementationError",
    "backend_implementation_record",
    "validate_backend_implementation_record",
]
