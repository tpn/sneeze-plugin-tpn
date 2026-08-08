"""Machine-local OCR backend installation records.

Portable backend and execution profiles describe configurations that can be
shared.  This registry deliberately does not: it binds one backend revision to
the executable, Hugging Face cache, model snapshot, adapter, and CUDA routing
installed on one machine.  Loading a registry validates both its content hash
and, by default, every recorded artifact.

The module imports only the Python standard library during CLI discovery.
Backend definitions and setup helpers are resolved lazily when an operation is
performed.
"""

from __future__ import annotations

import copy
import errno
import json
import os
import re
import secrets
import stat
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .ocr_path_security import (
    TrustedPathError,
    validate_trusted_directory,
    validate_trusted_file,
    validate_trusted_tree,
)

SCHEMA_VERSION = 1
REGISTRY_KIND = "sneeze.tpn.ocr_backend_installations"
REGISTRY_FILENAME = "ocr-installations.json"
_MAX_REGISTRY_BYTES = 8 * 1024 * 1024
_MAX_PTH_BYTES = 1024 * 1024
_DISTUTILS_PTH_NAME = "distutils-precedence.pth"
_DISTUTILS_PTH_SHA256 = (
    "2638ce9e2500e572a5e0de7faed6661eb569d1b696fcba07b0dd223da5f5d224"
)
_REGISTRY_DIRECTORY_MODE = 0o700
_REGISTRY_FILE_MODE = 0o600
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_CUDA_VALUE_RE = re.compile(r"^[A-Za-z0-9._:/,-]+$")
_SINGLE_GPU_SELECTOR_RE = re.compile(
    r"^(?:[0-9]+|GPU-[A-Za-z0-9][A-Za-z0-9-]*)$",
    re.IGNORECASE,
)
_DISTRIBUTION_PROBE_CODE = r"""
import hashlib
import json
from importlib.metadata import distributions

installed = sorted(
    f"{item.metadata.get('Name', item.name)}=={item.version}"
    for item in distributions()
)
digest = hashlib.sha256(
    json.dumps(
        installed,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
print(json.dumps(
    {"count": len(installed), "sha256": digest},
    sort_keys=True,
    separators=(",", ":"),
))
""".strip()


class InstallationRegistryError(ValueError):
    """Raised when a machine-local installation record is invalid."""


def default_installation_registry_path(
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the XDG location for the per-machine installation registry."""

    environment = os.environ if environ is None else environ
    configured = environment.get("XDG_CONFIG_HOME")
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            raise InstallationRegistryError(
                "XDG_CONFIG_HOME must be an absolute path"
            )
    else:
        home_value = environment.get("HOME")
        root = (
            Path(home_value).expanduser() if home_value else Path.home()
        ) / ".config"
        if not root.is_absolute():
            raise InstallationRegistryError(
                "HOME must resolve to an absolute path"
            )
    return root / "sneeze" / "tpn" / REGISTRY_FILENAME


def empty_installation_registry() -> dict[str, Any]:
    """Return a sealed empty registry document."""

    registry: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": REGISTRY_KIND,
        "installations": {},
    }
    registry["content_sha256"] = _sha256_json(registry)
    return registry


def build_backend_installation(
    backend_id: str,
    *,
    python_executable: str | os.PathLike[str],
    hf_cache: str | os.PathLike[str],
    cuda_visible_devices: str = "0",
    device: str = "cuda:0",
    attention_backend: str | None = None,
) -> dict[str, Any]:
    """Capture and seal one currently installed backend.

    The pinned snapshot must already exist beneath ``hf_cache``.  Provider
    adapters with a qualified digest must also match the backend definition;
    this prevents an unqualified local edit from being enrolled accidentally.
    """

    backend = _resolve_backend(backend_id)
    python_path = _require_executable(python_executable)
    _validate_python_environment_tree(python_path)
    cache_path = _require_directory(hf_cache, "Hugging Face cache")
    visible = _single_gpu_visible_device(
        cuda_visible_devices,
        "CUDA visible devices",
    )
    device_value = _single_gpu_logical_device(device, "CUDA device")
    attention_value = (
        None
        if attention_backend is None
        else _cuda_value(attention_backend, "attention backend")
    )

    snapshot_path = _expected_snapshot_path(backend, cache_path).resolve()
    _require_within(snapshot_path, cache_path, "model snapshot")
    snapshot = _snapshot_record(snapshot_path, cache_path)
    _check_qualified_snapshot(backend, snapshot)

    adapter = _adapter_record(backend)
    implementation = _implementation_record(backend)
    distributions = _probe_installed_distributions(python_path)
    cache_identity = _cache_identity_sha256(
        backend_id=backend.backend_id,
        revision=backend.model_revision,
        cache_path=cache_path,
        snapshot=snapshot,
    )
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "backend_id": backend.backend_id,
        "model": {
            "id": backend.model_id,
            "revision": backend.model_revision,
        },
        "python_executable": {
            "path": str(python_path),
            "sha256": _file_sha256(python_path),
            "installed_distributions_count": distributions["count"],
            "installed_distributions_sha256": distributions["sha256"],
        },
        "hf_cache": {
            "path": str(cache_path),
            "identity_sha256": cache_identity,
        },
        "snapshot": snapshot,
        "adapter": adapter,
        "implementation": implementation,
        "cuda": {
            "visible_devices": visible,
            "device": device_value,
            "attention_backend": attention_value,
        },
    }
    _assert_secret_free(record)
    record["record_sha256"] = _sha256_json(record)
    return validate_backend_installation(record, validate_current=True)


def validate_backend_installation(
    record: Mapping[str, Any],
    *,
    validate_current: bool = True,
) -> dict[str, Any]:
    """Validate one strict installation record and return a detached copy."""

    value = _plain_mapping(record, "installation")
    _exact_keys(
        value,
        {
            "schema_version",
            "backend_id",
            "model",
            "python_executable",
            "hf_cache",
            "snapshot",
            "adapter",
            "implementation",
            "cuda",
            "record_sha256",
        },
        "installation",
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != SCHEMA_VERSION
    ):
        raise InstallationRegistryError(
            "unsupported installation schema version"
        )
    backend_id = _string(value["backend_id"], "installation.backend_id")
    _digest(value["record_sha256"], "installation.record_sha256")

    model = _plain_mapping(value["model"], "installation.model")
    _exact_keys(model, {"id", "revision"}, "installation.model")
    _string(model["id"], "installation.model.id")
    _string(model["revision"], "installation.model.revision")

    executable = _plain_mapping(
        value["python_executable"],
        "installation.python_executable",
    )
    _exact_keys(
        executable,
        {
            "path",
            "sha256",
            "installed_distributions_count",
            "installed_distributions_sha256",
        },
        "installation.python_executable",
    )
    _absolute_path_value(
        executable["path"],
        "installation.python_executable.path",
    )
    _digest(
        executable["sha256"],
        "installation.python_executable.sha256",
    )
    distribution_count = executable["installed_distributions_count"]
    if type(distribution_count) is not int or distribution_count <= 0:
        raise InstallationRegistryError(
            "installation.python_executable.installed_distributions_count "
            "must be a positive integer"
        )
    _digest(
        executable["installed_distributions_sha256"],
        "installation.python_executable.installed_distributions_sha256",
    )

    cache = _plain_mapping(value["hf_cache"], "installation.hf_cache")
    _exact_keys(
        cache,
        {"path", "identity_sha256"},
        "installation.hf_cache",
    )
    _absolute_path_value(cache["path"], "installation.hf_cache.path")
    _digest(
        cache["identity_sha256"],
        "installation.hf_cache.identity_sha256",
    )

    snapshot = _validate_snapshot_schema(value["snapshot"])
    adapter = _validate_adapter_schema(value["adapter"])
    implementation = _validate_implementation_schema(
        value["implementation"],
        backend_id=backend_id,
    )
    cuda = _plain_mapping(value["cuda"], "installation.cuda")
    _exact_keys(
        cuda,
        {"visible_devices", "device", "attention_backend"},
        "installation.cuda",
    )
    _cuda_value(cuda["visible_devices"], "CUDA visible devices")
    _cuda_value(cuda["device"], "CUDA device")
    if cuda["attention_backend"] is not None:
        _cuda_value(cuda["attention_backend"], "attention backend")

    unsigned = dict(value)
    observed_record_hash = unsigned.pop("record_sha256")
    if observed_record_hash != _sha256_json(unsigned):
        raise InstallationRegistryError(
            f"installation record hash mismatch for {backend_id}"
        )
    validate_single_gpu_cuda_binding(cuda)
    _assert_secret_free(value)

    backend = _resolve_backend(backend_id)
    if backend.backend_id != backend_id:
        raise InstallationRegistryError(
            "stored installation must use the canonical backend ID"
        )
    if model["id"] != backend.model_id:
        raise InstallationRegistryError(
            f"stale installation model ID for {backend_id}"
        )
    if model["revision"] != backend.model_revision:
        raise InstallationRegistryError(
            f"stale installation model revision for {backend_id}"
        )
    _check_qualified_snapshot(backend, snapshot)
    _check_qualified_adapter(backend, adapter)
    _check_qualified_implementation(backend, implementation)

    if validate_current:
        _validate_current_artifacts(
            backend=backend,
            executable=executable,
            cache=cache,
            snapshot=snapshot,
            adapter=adapter,
            implementation=implementation,
        )
    return copy.deepcopy(value)


def validate_installation_registry(
    registry: Mapping[str, Any],
    *,
    validate_current: bool = True,
) -> dict[str, Any]:
    """Validate registry schema, hashes, backend identities, and artifacts."""

    value = _plain_mapping(registry, "registry")
    _exact_keys(
        value,
        {"schema_version", "kind", "installations", "content_sha256"},
        "registry",
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != SCHEMA_VERSION
    ):
        raise InstallationRegistryError(
            "unsupported installation registry schema version"
        )
    if value["kind"] != REGISTRY_KIND:
        raise InstallationRegistryError("invalid installation registry kind")
    _digest(value["content_sha256"], "registry.content_sha256")
    installations = _plain_mapping(
        value["installations"],
        "registry.installations",
    )
    if any(not isinstance(key, str) for key in installations):
        raise InstallationRegistryError(
            "registry installation keys must be strings"
        )
    unsigned = dict(value)
    observed_content_hash = unsigned.pop("content_sha256")
    if observed_content_hash != _sha256_json(unsigned):
        raise InstallationRegistryError(
            "installation registry content hash mismatch"
        )
    _assert_secret_free(value)

    validated: dict[str, dict[str, Any]] = {}
    for key in sorted(installations):
        backend_id = _string(key, "registry installation key")
        record = validate_backend_installation(
            installations[key],
            validate_current=validate_current,
        )
        if record["backend_id"] != backend_id:
            raise InstallationRegistryError(
                "installation registry key does not match backend ID"
            )
        validated[backend_id] = record
    result = dict(value)
    result["installations"] = validated
    return copy.deepcopy(result)


def load_installation_registry(
    path: str | os.PathLike[str] | None = None,
    *,
    validate_current: bool = True,
    missing_ok: bool = True,
) -> dict[str, Any]:
    """Load a registry, or return an empty one when absence is allowed."""

    registry_path = _registry_path(path)
    try:
        parent_descriptor = _open_secure_registry_parent(
            registry_path,
            create=False,
        )
        if parent_descriptor is None:
            if missing_ok:
                return empty_installation_registry()
            raise InstallationRegistryError(
                f"installation registry does not exist: {registry_path}"
            )
        try:
            loaded = _read_registry_at(
                parent_descriptor,
                registry_path.name,
                missing_ok=missing_ok,
            )
        finally:
            os.close(parent_descriptor)
        if loaded is None:
            return empty_installation_registry()
        value, _identity = loaded
    except InstallationRegistryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InstallationRegistryError(
            f"could not load installation registry: {type(exc).__name__}"
        ) from exc
    return validate_installation_registry(
        value,
        validate_current=validate_current,
    )


def upsert_backend_installation(
    record: Mapping[str, Any],
    *,
    path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Atomically create or update one entry while preserving all others."""

    registry_path = _registry_path(path)
    candidate = validate_backend_installation(
        record,
        validate_current=True,
    )
    parent_descriptor = _open_secure_registry_parent(
        registry_path,
        create=True,
    )
    if parent_descriptor is None:  # pragma: no cover - create=True
        raise InstallationRegistryError(
            "could not create installation registry ancestry"
        )
    try:
        with _exclusive_registry_lock(
            registry_path,
            parent_descriptor=parent_descriptor,
        ):
            loaded = _read_registry_at(
                parent_descriptor,
                registry_path.name,
                missing_ok=True,
            )
            if loaded is None:
                registry = empty_installation_registry()
                expected_identity = None
            else:
                value, expected_identity = loaded
                registry = validate_installation_registry(
                    value,
                    validate_current=True,
                )
            installations = dict(registry["installations"])
            installations[candidate["backend_id"]] = candidate
            unsigned: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "kind": REGISTRY_KIND,
                "installations": {
                    key: installations[key] for key in sorted(installations)
                },
            }
            unsigned["content_sha256"] = _sha256_json(unsigned)
            sealed = validate_installation_registry(
                unsigned,
                validate_current=True,
            )
            _atomic_write_registry(
                registry_path,
                sealed,
                parent_descriptor=parent_descriptor,
                expected_identity=expected_identity,
            )
    finally:
        os.close(parent_descriptor)
    return sealed


def register_backend_installation(
    backend_id: str,
    *,
    python_executable: str | os.PathLike[str],
    hf_cache: str | os.PathLike[str],
    cuda_visible_devices: str = "0",
    device: str = "cuda:0",
    attention_backend: str | None = None,
    path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Capture one installed backend and atomically add it to the registry."""

    record = build_backend_installation(
        backend_id,
        python_executable=python_executable,
        hf_cache=hf_cache,
        cuda_visible_devices=cuda_visible_devices,
        device=device,
        attention_backend=attention_backend,
    )
    return upsert_backend_installation(record, path=path)


def get_backend_installation(
    backend_id: str,
    *,
    path: str | os.PathLike[str] | None = None,
    validate_current: bool = True,
) -> dict[str, Any]:
    """Resolve a backend alias and return its machine-local installation."""

    backend = _resolve_backend(backend_id)
    registry = load_installation_registry(
        path,
        validate_current=validate_current,
        missing_ok=False,
    )
    try:
        return copy.deepcopy(registry["installations"][backend.backend_id])
    except KeyError as exc:
        raise InstallationRegistryError(
            f"no installation registered for {backend.backend_id}"
        ) from exc


def list_backend_installations(
    *,
    path: str | os.PathLike[str] | None = None,
    validate_current: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Return all machine-local installations in canonical backend order."""

    registry = load_installation_registry(
        path,
        validate_current=validate_current,
        missing_ok=True,
    )
    return tuple(
        copy.deepcopy(registry["installations"][key])
        for key in sorted(registry["installations"])
    )


def _resolve_backend(name: str) -> Any:
    try:
        from .ocr_backends import get_backend

        return get_backend(name)
    except (TypeError, ValueError) as exc:
        raise InstallationRegistryError(str(exc)) from exc


def _sha256_json(value: Any) -> str:
    from .ocr_backend_setup import sha256_json

    return sha256_json(value)


def _file_sha256(path: Path) -> str:
    from .ocr_backend_setup import file_sha256

    return file_sha256(path)


def _expected_snapshot_path(backend: Any, cache_path: Path) -> Path:
    from .ocr_backend_setup import expected_snapshot_path

    return expected_snapshot_path(backend, cache_path)


def _assert_secret_free(value: Any) -> None:
    try:
        from .ocr_backend_setup import make_receipt

        make_receipt("ocr_backend_installation_safety", {"value": value})
    except ValueError as exc:
        raise InstallationRegistryError(str(exc)) from exc


def _registry_path(path: str | os.PathLike[str] | None) -> Path:
    candidate = (
        default_installation_registry_path()
        if path is None
        else Path(path).expanduser()
    )
    if not candidate.is_absolute():
        raise InstallationRegistryError(
            "installation registry path must be absolute"
        )
    normalized = Path(os.path.abspath(candidate))
    if not normalized.name:
        raise InstallationRegistryError(
            "installation registry path must name a file"
        )
    return normalized


def _effective_uid() -> int:
    getter = getattr(os, "geteuid", None) or getattr(os, "getuid", None)
    if getter is None:  # pragma: no cover - supported OCR hosts are POSIX
        raise InstallationRegistryError(
            "secure installation registries require POSIX ownership controls"
        )
    return int(getter())


def _directory_open_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise InstallationRegistryError(
            "secure installation registries require POSIX file controls"
        )
    return os.O_RDONLY | directory | nofollow | getattr(os, "O_CLOEXEC", 0)


def _registry_file_open_flags(*, writable: bool) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    nonblock = getattr(os, "O_NONBLOCK", None)
    if nofollow is None or nonblock is None:
        raise InstallationRegistryError(
            "secure installation registries require POSIX file controls"
        )
    access = os.O_RDWR if writable else os.O_RDONLY
    return access | nofollow | nonblock | getattr(os, "O_CLOEXEC", 0)


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _validate_registry_directory_stat(
    metadata: os.stat_result,
    *,
    label: str,
) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise InstallationRegistryError(
            f"installation registry ancestry must contain only directories: "
            f"{label}"
        )
    if metadata.st_uid not in {0, _effective_uid()}:
        raise InstallationRegistryError(
            "installation registry ancestry must be owned by root or the "
            f"current user: {label}"
        )
    writable_by_others = stat.S_IMODE(metadata.st_mode) & 0o022
    root_owned_sticky = metadata.st_uid == 0 and bool(
        metadata.st_mode & stat.S_ISVTX
    )
    if writable_by_others and not root_owned_sticky:
        raise InstallationRegistryError(
            "installation registry ancestry must not be writable by group or "
            f"other users: {label}"
        )


def _open_registry_child_directory(
    parent_descriptor: int,
    component: str,
    *,
    create: bool,
    label: str,
) -> int | None:
    created = False
    try:
        path_stat = os.stat(
            component,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        if not create:
            return None
        try:
            os.mkdir(
                component,
                _REGISTRY_DIRECTORY_MODE,
                dir_fd=parent_descriptor,
            )
            created = True
            os.chmod(
                component,
                _REGISTRY_DIRECTORY_MODE,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            pass
        except OSError as exc:
            raise InstallationRegistryError(
                "could not create installation registry ancestry: "
                f"{type(exc).__name__}"
            ) from exc
        try:
            path_stat = os.stat(
                component,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise InstallationRegistryError(
                "could not inspect installation registry ancestry: "
                f"{type(exc).__name__}"
            ) from exc
    except OSError as exc:
        raise InstallationRegistryError(
            "could not inspect installation registry ancestry: "
            f"{type(exc).__name__}"
        ) from exc

    if stat.S_ISLNK(path_stat.st_mode):
        raise InstallationRegistryError(
            f"installation registry ancestry cannot contain symlinks: {label}"
        )
    _validate_registry_directory_stat(path_stat, label=label)
    try:
        descriptor = os.open(
            component,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        detail = (
            "changed during validation" if created else type(exc).__name__
        )
        raise InstallationRegistryError(
            "could not securely open installation registry ancestry: "
            f"{detail}"
        ) from exc
    try:
        descriptor_stat = os.fstat(descriptor)
        if not _same_file_identity(path_stat, descriptor_stat):
            raise InstallationRegistryError(
                "installation registry ancestry changed while opening"
            )
        _validate_registry_directory_stat(descriptor_stat, label=label)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _open_secure_registry_parent(
    path: Path,
    *,
    create: bool,
) -> int | None:
    """Open a trusted registry parent without following path symlinks."""

    flags = _directory_open_flags()
    try:
        descriptor = os.open(path.anchor, flags)
    except OSError as exc:
        raise InstallationRegistryError(
            "could not open installation registry filesystem root: "
            f"{type(exc).__name__}"
        ) from exc
    try:
        _validate_registry_directory_stat(
            os.fstat(descriptor),
            label=path.anchor,
        )
        current = Path(path.anchor)
        for component in path.parent.parts[1:]:
            current /= component
            child = _open_registry_child_directory(
                descriptor,
                component,
                create=create,
                label=str(current),
            )
            if child is None:
                os.close(descriptor)
                return None
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _validate_registry_leaf_stat(metadata: os.stat_result) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise InstallationRegistryError(
            "installation registry must be a regular, non-symlink file"
        )
    if metadata.st_uid != _effective_uid():
        raise InstallationRegistryError(
            "installation registry must be owned by the current user"
        )
    if stat.S_IMODE(metadata.st_mode) != _REGISTRY_FILE_MODE:
        raise InstallationRegistryError(
            "installation registry must have private permissions (0600)"
        )


def _read_limited_registry_file(descriptor: int) -> bytes:
    chunks = []
    remaining = _MAX_REGISTRY_BYTES + 1
    while remaining:
        chunk = os.read(descriptor, min(remaining, 1024 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    if len(payload) > _MAX_REGISTRY_BYTES:
        raise InstallationRegistryError(
            "installation registry exceeds the size limit"
        )
    return payload


def _read_registry_at(
    parent_descriptor: int,
    name: str,
    *,
    missing_ok: bool,
) -> tuple[dict[str, Any], tuple[int, int]] | None:
    try:
        descriptor = os.open(
            name,
            _registry_file_open_flags(writable=False),
            dir_fd=parent_descriptor,
        )
    except FileNotFoundError:
        if missing_ok:
            return None
        raise InstallationRegistryError(
            f"installation registry does not exist: {name}"
        ) from None
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENXIO, errno.ENODEV}:
            raise InstallationRegistryError(
                "installation registry must be a regular, non-symlink file"
            ) from exc
        raise InstallationRegistryError(
            f"could not open installation registry: {type(exc).__name__}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        _validate_registry_leaf_stat(before)
        if before.st_size > _MAX_REGISTRY_BYTES:
            raise InstallationRegistryError(
                "installation registry exceeds the size limit"
            )
        path_stat = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not _same_file_identity(before, path_stat):
            raise InstallationRegistryError(
                "installation registry identity changed while opening"
            )
        payload = _read_limited_registry_file(descriptor)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, field) != getattr(after, field)
            for field in stable_fields
        ):
            raise InstallationRegistryError(
                "installation registry changed while reading"
            )
        final_path_stat = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not _same_file_identity(after, final_path_stat):
            raise InstallationRegistryError(
                "installation registry identity changed while reading"
            )
    except InstallationRegistryError:
        raise
    except OSError as exc:
        raise InstallationRegistryError(
            f"could not read installation registry: {type(exc).__name__}"
        ) from exc
    finally:
        os.close(descriptor)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except InstallationRegistryError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise InstallationRegistryError(
            f"could not load installation registry: {type(exc).__name__}"
        ) from exc
    return value, (after.st_dev, after.st_ino)


def _require_executable(path: str | os.PathLike[str]) -> Path:
    try:
        return validate_trusted_file(
            _absolute_input_path(path, "Python executable"),
            label="Python executable",
            executable=True,
        )
    except TrustedPathError as exc:
        raise InstallationRegistryError(str(exc)) from exc


def _validate_python_environment_tree(python_path: Path) -> Path:
    """Reject package trees writable by a different unprivileged user."""

    environment_root = (
        python_path.parent.parent
        if python_path.parent.name == "bin"
        else python_path.parent
    )
    try:
        validated = validate_trusted_tree(
            environment_root,
            label="Python environment",
        )
        _validate_pth_import_roots(validated)
        return validated
    except TrustedPathError as exc:
        raise InstallationRegistryError(str(exc)) from exc


def _validate_pth_import_roots(environment_root: Path) -> None:
    """Validate path entries that site initialization imports from outside."""

    try:
        site_directories = {
            path.resolve(strict=True)
            for pattern in (
                "lib/python*/site-packages",
                "lib/python*/dist-packages",
                "lib64/python*/site-packages",
                "lib64/python*/dist-packages",
                "Lib/site-packages",
            )
            for path in environment_root.glob(pattern)
            if path.is_dir()
        }
        pth_files = sorted(
            path
            for site_directory in site_directories
            for path in site_directory.glob("*.pth")
            if path.is_file()
        )
    except (OSError, RuntimeError) as exc:
        raise TrustedPathError(
            "could not inspect Python environment .pth files"
        ) from exc
    for pth_path in pth_files:
        try:
            if pth_path.stat().st_size > _MAX_PTH_BYTES:
                raise TrustedPathError(
                    f"Python .pth file is larger than {_MAX_PTH_BYTES} bytes"
                )
            lines = pth_path.read_text(encoding="utf-8-sig").splitlines()
        except UnicodeError as exc:
            raise TrustedPathError(
                "Python .pth file must be UTF-8 text"
            ) from exc
        except OSError as exc:
            raise TrustedPathError(
                "could not inspect Python environment .pth file"
            ) from exc
        for raw_line in lines:
            line = raw_line.rstrip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("import ", "import\t")):
                if (
                    pth_path.name != _DISTUTILS_PTH_NAME
                    or _file_sha256(pth_path) != _DISTUTILS_PTH_SHA256
                ):
                    raise TrustedPathError(
                        "Python .pth executable line is not allowlisted"
                    )
                distutils_hack = (
                    pth_path.parent / "_distutils_hack" / "__init__.py"
                )
                try:
                    distutils_metadata = distutils_hack.lstat()
                except OSError as exc:
                    raise TrustedPathError(
                        "allowlisted distutils .pth shim is incomplete"
                    ) from exc
                if not stat.S_ISREG(distutils_metadata.st_mode):
                    raise TrustedPathError(
                        "allowlisted distutils .pth shim is incomplete"
                    )
                validate_trusted_file(
                    distutils_hack,
                    label="distutils .pth shim module",
                    resolve_symlinks=False,
                )
                continue
            if "\x00" in line:
                raise TrustedPathError(
                    "Python .pth import path must not contain NUL"
                )
            entry = Path(line)
            spelling = (
                entry if entry.is_absolute() else pth_path.parent / entry
            )
            candidate = Path(os.path.abspath(os.fspath(spelling)))
            internal = candidate.is_relative_to(environment_root)
            try:
                metadata = candidate.lstat()
            except FileNotFoundError:
                if internal:
                    continue
                raise TrustedPathError(
                    "external Python .pth import path does not exist"
                ) from None
            except OSError as exc:
                raise TrustedPathError(
                    "could not inspect Python .pth import path"
                ) from exc
            if internal:
                continue
            if stat.S_ISDIR(metadata.st_mode):
                validate_trusted_tree(
                    candidate,
                    label="Python .pth import root",
                    resolve_root_symlinks=False,
                )
            elif stat.S_ISREG(metadata.st_mode):
                validate_trusted_file(
                    candidate,
                    label="Python .pth import file",
                    resolve_symlinks=False,
                )
            else:
                raise TrustedPathError(
                    "Python .pth import path must be a file or directory"
                )


def _probe_installed_distributions(python_path: Path) -> dict[str, Any]:
    """Return the complete installed-distribution identity for one Python."""

    from .ocr_execution import (
        OcrExecutionError,
        _run_bounded_runtime_probe,
    )

    environment = {
        name: value
        for name, value in os.environ.items()
        if name in {"LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ"}
        and isinstance(value, str)
        and "\x00" not in value
    }
    environment["PYTHONNOUSERSITE"] = "1"
    try:
        completed = _run_bounded_runtime_probe(
            [
                str(python_path),
                "-I",
                "-B",
                "-c",
                _DISTRIBUTION_PROBE_CODE,
            ],
            environment=environment,
            timeout_seconds=60,
            runner=subprocess.run,
        )
    except OcrExecutionError as exc:
        if "exceeded 64 KiB" in str(exc):
            raise InstallationRegistryError(
                "installed-distribution probe output exceeded the limit"
            ) from exc
        cause = (
            type(exc.__cause__).__name__
            if exc.__cause__
            else type(exc).__name__
        )
        raise InstallationRegistryError(
            f"could not inspect installed Python distributions: {cause}"
        ) from exc
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        value = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError as exc:
        raise InstallationRegistryError(
            "installed-distribution probe returned invalid JSON"
        ) from exc
    if completed.returncode != 0 or not isinstance(value, Mapping):
        raise InstallationRegistryError("installed-distribution probe failed")
    if set(value) != {"count", "sha256"}:
        raise InstallationRegistryError(
            "installed-distribution probe returned unexpected fields"
        )
    count = value["count"]
    digest = value["sha256"]
    if type(count) is not int or count <= 0:
        raise InstallationRegistryError(
            "installed-distribution probe returned an invalid count"
        )
    _digest(digest, "installed-distribution probe SHA-256")
    return {"count": count, "sha256": digest}


def _require_directory(
    path: str | os.PathLike[str],
    label: str,
) -> Path:
    try:
        return validate_trusted_directory(
            _absolute_input_path(path, label),
            label=label,
        )
    except TrustedPathError as exc:
        raise InstallationRegistryError(str(exc)) from exc


def _absolute_input_path(
    path: str | os.PathLike[str],
    label: str,
) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise InstallationRegistryError(f"{label} must be an absolute path")
    return candidate


def _absolute_path_value(value: Any, label: str) -> Path:
    path = _string(value, label)
    candidate = Path(path)
    has_navigation_component = any(
        component in {".", ".."} for component in path.split(os.sep)
    )
    if (
        has_navigation_component
        or not candidate.is_absolute()
        or str(candidate) != path
    ):
        raise InstallationRegistryError(
            f"{label} must be a canonical absolute path"
        )
    return candidate


def _require_within(path: Path, root: Path, label: str) -> None:
    if not path.is_relative_to(root):
        raise InstallationRegistryError(
            f"{label} must remain inside the Hugging Face cache"
        )


def _snapshot_record(snapshot_path: Path, cache_path: Path) -> dict[str, Any]:
    if snapshot_path.is_symlink() or not snapshot_path.is_dir():
        raise InstallationRegistryError(
            "pinned model snapshot is missing or is a directory symlink"
        )
    try:
        validate_trusted_tree(
            snapshot_path,
            label="model snapshot",
            symlink_root=cache_path,
        )
    except TrustedPathError as exc:
        raise InstallationRegistryError(str(exc)) from exc
    files: dict[str, dict[str, Any]] = {}
    total_size = 0
    try:
        candidates = sorted(
            snapshot_path.rglob("*"),
            key=lambda item: item.relative_to(snapshot_path).as_posix(),
        )
        for candidate in candidates:
            if candidate.is_dir():
                if candidate.is_symlink():
                    raise InstallationRegistryError(
                        "model snapshot contains a directory symlink"
                    )
                continue
            resolved = candidate.resolve(strict=True)
            _require_within(resolved, cache_path, "snapshot file")
            mode = resolved.stat().st_mode
            if not stat.S_ISREG(mode):
                raise InstallationRegistryError(
                    "model snapshot contains a non-regular file"
                )
            before = resolved.stat()
            digest = _file_sha256(resolved)
            after = resolved.stat()
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise InstallationRegistryError(
                    "model snapshot changed while it was being hashed"
                )
            relative = candidate.relative_to(snapshot_path).as_posix()
            files[relative] = {
                "sha256": digest,
                "size_bytes": after.st_size,
            }
            total_size += after.st_size
    except InstallationRegistryError:
        raise
    except OSError as exc:
        raise InstallationRegistryError(
            f"could not hash model snapshot: {type(exc).__name__}"
        ) from exc
    if not files:
        raise InstallationRegistryError("model snapshot contains no files")
    model_weights = files.get("model.safetensors")
    return {
        "path": str(snapshot_path),
        "file_count": len(files),
        "size_bytes": total_size,
        "files_sha256": _sha256_json(files),
        "weights_sha256": (
            model_weights["sha256"] if model_weights else None
        ),
    }


def _adapter_path(backend: Any) -> Path | None:
    relative = backend.adapter_relative_path
    if relative is None:
        return None
    candidate = (Path(__file__).resolve().parent / relative).resolve()
    if not candidate.is_file():
        raise InstallationRegistryError(
            f"backend adapter is missing for {backend.backend_id}"
        )
    return candidate


def _adapter_record(backend: Any) -> dict[str, str] | None:
    path = _adapter_path(backend)
    if path is None:
        return None
    try:
        path = validate_trusted_file(path, label="OCR provider adapter")
    except TrustedPathError as exc:
        raise InstallationRegistryError(str(exc)) from exc
    record = {"path": str(path), "sha256": _file_sha256(path)}
    _check_qualified_adapter(backend, record)
    return record


def _implementation_record(backend: Any) -> dict[str, Any]:
    try:
        from .ocr_implementation import backend_implementation_record

        record = backend_implementation_record(backend.backend_id)
    except (OSError, ValueError) as exc:
        raise InstallationRegistryError(str(exc)) from exc
    _check_qualified_implementation(backend, record)
    return record


def _check_qualified_snapshot(
    backend: Any,
    snapshot: Mapping[str, Any],
) -> None:
    expected_files = backend.snapshot_identity_sha256
    if expected_files and snapshot["files_sha256"] != expected_files:
        raise InstallationRegistryError(
            f"model snapshot is not qualified for {backend.backend_id}"
        )
    expected_weights = backend.weights_sha256
    if expected_weights and snapshot["weights_sha256"] != expected_weights:
        raise InstallationRegistryError(
            f"model weights are not qualified for {backend.backend_id}"
        )


def _check_qualified_adapter(
    backend: Any,
    adapter: Mapping[str, Any] | None,
) -> None:
    expected = backend.qualified_adapter_sha256
    if backend.adapter_relative_path is None:
        if adapter is not None:
            raise InstallationRegistryError(
                f"backend {backend.backend_id} must not record an adapter"
            )
        return
    if adapter is None:
        raise InstallationRegistryError(
            f"backend {backend.backend_id} requires an adapter"
        )
    if expected and adapter["sha256"] != expected:
        raise InstallationRegistryError(
            f"adapter is not qualified for {backend.backend_id}"
        )


def _check_qualified_implementation(
    backend: Any,
    implementation: Mapping[str, Any],
) -> None:
    expected = backend.qualified_implementation_sha256
    if not expected or implementation.get("sha256") != expected:
        raise InstallationRegistryError(
            f"implementation is not qualified for {backend.backend_id}"
        )


def _cache_identity_sha256(
    *,
    backend_id: str,
    revision: str,
    cache_path: Path,
    snapshot: Mapping[str, Any],
) -> str:
    return _sha256_json(
        {
            "backend_id": backend_id,
            "revision": revision,
            "cache_path": str(cache_path),
            "snapshot_path": snapshot["path"],
            "snapshot_files_sha256": snapshot["files_sha256"],
        }
    )


def _validate_current_artifacts(
    *,
    backend: Any,
    executable: Mapping[str, Any],
    cache: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    adapter: Mapping[str, Any] | None,
    implementation: Mapping[str, Any],
) -> None:
    current_executable = _require_executable(executable["path"])
    if str(current_executable) != executable["path"]:
        raise InstallationRegistryError(
            f"stale Python executable path for {backend.backend_id}"
        )
    if _file_sha256(current_executable) != executable["sha256"]:
        raise InstallationRegistryError(
            f"stale Python executable hash for {backend.backend_id}"
        )
    _validate_python_environment_tree(current_executable)
    current_distributions = _probe_installed_distributions(current_executable)
    expected_distributions = {
        "count": executable["installed_distributions_count"],
        "sha256": executable["installed_distributions_sha256"],
    }
    if current_distributions != expected_distributions:
        raise InstallationRegistryError(
            f"stale Python distributions for {backend.backend_id}"
        )

    current_cache = _require_directory(
        cache["path"],
        "Hugging Face cache",
    )
    if str(current_cache) != cache["path"]:
        raise InstallationRegistryError(
            f"stale Hugging Face cache path for {backend.backend_id}"
        )
    expected_snapshot = _expected_snapshot_path(
        backend,
        current_cache,
    ).resolve()
    if str(expected_snapshot) != snapshot["path"]:
        raise InstallationRegistryError(
            f"stale model snapshot path for {backend.backend_id}"
        )
    _require_within(expected_snapshot, current_cache, "model snapshot")
    current_snapshot = _snapshot_record(expected_snapshot, current_cache)
    if current_snapshot != snapshot:
        raise InstallationRegistryError(
            f"stale or tampered model snapshot for {backend.backend_id}"
        )
    expected_cache_hash = _cache_identity_sha256(
        backend_id=backend.backend_id,
        revision=backend.model_revision,
        cache_path=current_cache,
        snapshot=current_snapshot,
    )
    if expected_cache_hash != cache["identity_sha256"]:
        raise InstallationRegistryError(
            f"stale Hugging Face cache identity for {backend.backend_id}"
        )

    current_adapter = _adapter_record(backend)
    if current_adapter != adapter:
        raise InstallationRegistryError(
            f"stale or tampered adapter for {backend.backend_id}"
        )
    current_implementation = _implementation_record(backend)
    # Registry JSON represents definition tuples as arrays. The canonical
    # digest binds the complete schema-v2 record independent of that benign
    # in-memory tuple/list distinction.
    if current_implementation["sha256"] != implementation["sha256"]:
        raise InstallationRegistryError(
            f"stale or tampered implementation for {backend.backend_id}"
        )


def _validate_snapshot_schema(value: Any) -> dict[str, Any]:
    snapshot = _plain_mapping(value, "installation.snapshot")
    _exact_keys(
        snapshot,
        {
            "path",
            "file_count",
            "size_bytes",
            "files_sha256",
            "weights_sha256",
        },
        "installation.snapshot",
    )
    _absolute_path_value(snapshot["path"], "installation.snapshot.path")
    for name in ("file_count", "size_bytes"):
        item = snapshot[name]
        if type(item) is not int or item < (1 if name == "file_count" else 0):
            raise InstallationRegistryError(
                f"installation.snapshot.{name} is invalid"
            )
    _digest(
        snapshot["files_sha256"],
        "installation.snapshot.files_sha256",
    )
    if snapshot["weights_sha256"] is not None:
        _digest(
            snapshot["weights_sha256"],
            "installation.snapshot.weights_sha256",
        )
    return snapshot


def _validate_adapter_schema(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    adapter = _plain_mapping(value, "installation.adapter")
    _exact_keys(
        adapter,
        {"path", "sha256"},
        "installation.adapter",
    )
    _absolute_path_value(adapter["path"], "installation.adapter.path")
    _digest(adapter["sha256"], "installation.adapter.sha256")
    return adapter


def _validate_implementation_schema(
    value: Any,
    *,
    backend_id: str,
) -> dict[str, Any]:
    try:
        from .ocr_implementation import (
            validate_backend_implementation_record,
        )

        return validate_backend_implementation_record(
            value,
            backend_id=backend_id,
            require_current=False,
        )
    except (OSError, ValueError) as exc:
        raise InstallationRegistryError(str(exc)) from exc


def _plain_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InstallationRegistryError(f"{label} must be a JSON object")
    return dict(value)


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    keys = set(value)
    if keys != expected or any(not isinstance(key, str) for key in value):
        raise InstallationRegistryError(
            f"{label} has unexpected or missing fields"
        )


def _string(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(character) < 32 for character in value)
    ):
        raise InstallationRegistryError(f"{label} must be a nonempty string")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise InstallationRegistryError(f"{label} is not a SHA-256 digest")
    return value


def _cuda_value(value: Any, label: str) -> str:
    text = _string(value, label)
    if len(text) > 256 or not _CUDA_VALUE_RE.fullmatch(text):
        raise InstallationRegistryError(f"{label} contains unsafe syntax")
    return text


def validate_single_gpu_cuda_binding(
    cuda: Mapping[str, Any],
    *,
    label: str = "installation.cuda",
) -> tuple[str, str]:
    """Validate CUDA routing for the single-GPU local OCR pipelines."""

    value = _plain_mapping(cuda, label)
    visible = _single_gpu_visible_device(
        value.get("visible_devices"),
        f"{label}.visible_devices",
    )
    device = _single_gpu_logical_device(
        value.get("device"),
        f"{label}.device",
    )
    return visible, device


def _single_gpu_visible_device(value: Any, label: str) -> str:
    text = _cuda_value(value, label)
    if "," in text:
        raise InstallationRegistryError(
            f"{label} must select exactly one GPU"
        )
    if not _SINGLE_GPU_SELECTOR_RE.fullmatch(text):
        raise InstallationRegistryError(
            f"{label} must be one decimal ordinal or GPU UUID selector"
        )
    return text


def _single_gpu_logical_device(value: Any, label: str) -> str:
    text = _cuda_value(value, label)
    if text != "cuda:0":
        raise InstallationRegistryError(
            f"{label} must be cuda:0 for single-GPU OCR pipelines"
        )
    return text


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InstallationRegistryError(
                f"duplicate JSON field in installation registry: {key}"
            )
        result[key] = value
    return result


@contextmanager
def _exclusive_registry_lock(
    path: Path,
    *,
    parent_descriptor: int | None = None,
) -> Iterator[None]:
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - POSIX OCR hosts have it
        raise InstallationRegistryError(
            "secure installation registry locking requires POSIX flock"
        ) from exc

    registry_path = _registry_path(path)
    owns_parent_descriptor = parent_descriptor is None
    if parent_descriptor is None:
        parent_descriptor = _open_secure_registry_parent(
            registry_path,
            create=True,
        )
        if parent_descriptor is None:  # pragma: no cover - create=True
            raise InstallationRegistryError(
                "could not create installation registry ancestry"
            )
    lock_name = f".{registry_path.name}.lock"
    try:
        descriptor = _open_registry_lock(parent_descriptor, lock_name)
    except Exception:
        if owns_parent_descriptor:
            os.close(parent_descriptor)
        raise
    locked = False
    try:
        try:
            os.fchmod(descriptor, _REGISTRY_FILE_MODE)
            lock_mode = stat.S_IMODE(os.fstat(descriptor).st_mode)
            if lock_mode != _REGISTRY_FILE_MODE:
                raise InstallationRegistryError(
                    "installation registry lock must have private "
                    "permissions (0600)"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            locked = True
        except InstallationRegistryError:
            raise
        except OSError as exc:
            raise InstallationRegistryError(
                "could not acquire installation registry lock: "
                f"{type(exc).__name__}"
            ) from exc
        yield
    finally:
        try:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
            if owns_parent_descriptor:
                os.close(parent_descriptor)


def _open_registry_lock(parent_descriptor: int, lock_name: str) -> int:
    flags = _registry_file_open_flags(writable=True) | os.O_CREAT
    try:
        descriptor = os.open(
            lock_name,
            flags,
            _REGISTRY_FILE_MODE,
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        raise InstallationRegistryError(
            f"could not open installation registry lock: {type(exc).__name__}"
        ) from exc
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.stat(
            lock_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise InstallationRegistryError(
                "installation registry lock must be a regular file"
            )
        if descriptor_stat.st_uid != _effective_uid():
            raise InstallationRegistryError(
                "installation registry lock must be owned by the current user"
            )
        if not _same_file_identity(descriptor_stat, path_stat):
            raise InstallationRegistryError(
                "installation registry lock identity changed while opening"
            )
    except InstallationRegistryError:
        os.close(descriptor)
        raise
    except OSError as exc:
        os.close(descriptor)
        raise InstallationRegistryError(
            "could not inspect installation registry lock: "
            f"{type(exc).__name__}"
        ) from exc
    return descriptor


def _create_registry_candidate(
    parent_descriptor: int,
    registry_name: str,
) -> tuple[int, str]:
    flags = _registry_file_open_flags(writable=True) | os.O_CREAT | os.O_EXCL
    for _ in range(128):
        candidate_name = f".{registry_name}.{secrets.token_hex(12)}.candidate"
        try:
            descriptor = os.open(
                candidate_name,
                flags,
                _REGISTRY_FILE_MODE,
                dir_fd=parent_descriptor,
            )
        except FileExistsError:
            continue
        try:
            os.fchmod(descriptor, _REGISTRY_FILE_MODE)
            candidate_stat = os.fstat(descriptor)
            _validate_registry_leaf_stat(candidate_stat)
            path_stat = os.stat(
                candidate_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if not _same_file_identity(candidate_stat, path_stat):
                raise InstallationRegistryError(
                    "installation registry candidate identity changed while "
                    "opening"
                )
        except Exception:
            os.close(descriptor)
            try:
                os.unlink(candidate_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
            raise
        return descriptor, candidate_name
    raise InstallationRegistryError(
        "could not allocate a unique installation registry candidate"
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:  # pragma: no cover - regular-file writes progress
            raise OSError("installation registry write made no progress")
        view = view[written:]


def _registry_leaf_stat_at(
    parent_descriptor: int,
    name: str,
) -> os.stat_result | None:
    try:
        metadata = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    _validate_registry_leaf_stat(metadata)
    return metadata


def _atomic_write_registry(
    path: Path,
    registry: Mapping[str, Any],
    *,
    parent_descriptor: int | None = None,
    expected_identity: tuple[int, int] | None,
) -> None:
    registry_path = _registry_path(path)
    payload = (
        json.dumps(
            dict(registry),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > _MAX_REGISTRY_BYTES:
        raise InstallationRegistryError(
            "installation registry exceeds the size limit"
        )
    owns_parent_descriptor = parent_descriptor is None
    if parent_descriptor is None:
        parent_descriptor = _open_secure_registry_parent(
            registry_path,
            create=True,
        )
        if parent_descriptor is None:  # pragma: no cover - create=True
            raise InstallationRegistryError(
                "could not create installation registry ancestry"
            )
    temporary_name: str | None = None
    candidate_descriptor: int | None = None
    try:
        candidate_descriptor, temporary_name = _create_registry_candidate(
            parent_descriptor,
            registry_path.name,
        )
        _write_all(candidate_descriptor, payload)
        os.fsync(candidate_descriptor)
        os.lseek(candidate_descriptor, 0, os.SEEK_SET)
        observed_payload = _read_limited_registry_file(candidate_descriptor)
        if observed_payload != payload:
            raise InstallationRegistryError(
                "installation registry candidate changed while writing"
            )
        loaded = json.loads(
            observed_payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
        validate_installation_registry(loaded, validate_current=False)
        candidate_stat = os.fstat(candidate_descriptor)
        existing = _registry_leaf_stat_at(
            parent_descriptor,
            registry_path.name,
        )
        if expected_identity is None:
            if existing is not None:
                raise InstallationRegistryError(
                    "installation registry appeared while writing"
                )
            try:
                os.link(
                    temporary_name,
                    registry_path.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise InstallationRegistryError(
                    "installation registry appeared while writing"
                ) from exc
            os.unlink(temporary_name, dir_fd=parent_descriptor)
            temporary_name = None
        else:
            if (
                existing is None
                or (
                    existing.st_dev,
                    existing.st_ino,
                )
                != expected_identity
            ):
                raise InstallationRegistryError(
                    "installation registry changed while writing"
                )
            os.replace(
                temporary_name,
                registry_path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            temporary_name = None
        installed = _registry_leaf_stat_at(
            parent_descriptor,
            registry_path.name,
        )
        if installed is None or not _same_file_identity(
            candidate_stat,
            installed,
        ):
            raise InstallationRegistryError(
                "installation registry identity changed while installing"
            )
        os.fsync(parent_descriptor)
    except InstallationRegistryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise InstallationRegistryError(
            f"could not write installation registry: {type(exc).__name__}"
        ) from exc
    finally:
        if candidate_descriptor is not None:
            os.close(candidate_descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        if owns_parent_descriptor:
            os.close(parent_descriptor)
