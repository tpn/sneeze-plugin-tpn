from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import stat
import struct
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .ocr_process_lifecycle import isolated_subprocess_signal_guard

SCHEMA_VERSION = 1
_PROTOCOL_ARGUMENTS = (
    "--requests-jsonl",
    "--responses-jsonl",
    "--artifacts-dir",
)
_SECRET_ENV_RE = re.compile(
    r"(?:AUTH|COOKIE|CREDENTIAL|KEY|PASS(?:WORD)?|SECRET|TOKEN)",
    re.IGNORECASE,
)
_SENSITIVE_LOCAL_ENV_RE = re.compile(
    r"^(?:DATABASE_URL|SENTRY_DSN|HOME|USERPROFILE|"
    r"AWS_(?:CONFIG_FILE|DEFAULT_PROFILE|PROFILE|"
    r"SHARED_CREDENTIALS_FILE)|AZURE_CONFIG_DIR|CLOUDSDK_CONFIG|"
    r"GCP_PROFILE|GOOGLE_APPLICATION_CREDENTIALS|GOOGLE_PROFILE)$",
    re.IGNORECASE,
)
_LOCAL_DSN_RE = re.compile(
    r"^(?:amqps?|cassandra|clickhouse|cockroachdb|mariadb|mongodb"
    r"(?:\+srv)?|mssql|mysql|oracle|postgres(?:ql)?|rediss?|snowflake)://",
    re.IGNORECASE,
)
_LOCAL_CREDENTIAL_URL_RE = re.compile(
    r"^[a-z][a-z0-9+.-]*://[^/@\s]+@",
    re.IGNORECASE,
)
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REVISION_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCAL_AMBIENT_ENV_ALLOWLIST = frozenset(
    {
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
    }
)
_FORCED_LOCAL_ENV = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONNOUSERSITE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
_LOCAL_SPEC_KEYS = frozenset(
    {
        "schema_version",
        "provider_id",
        "argv",
        "cwd",
        "egress",
        "model",
        "lanes",
        "timeout_seconds",
        "secret_env_names",
        "env",
    }
)


class ProviderError(ValueError):
    """Raised when a local OCR provider request is unsafe or malformed."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ProviderError("duplicate JSON field")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is not allowed: {value}")
    return parsed


def _reject_credential_like_value(value: Any, label: str) -> None:
    from .ocr_backend_setup import BackendSetupError, make_receipt

    try:
        make_receipt(
            "local_ocr_provider_value_validation",
            {"value": value},
        )
    except BackendSetupError as exc:
        raise ProviderError(
            f"{label} contains a credential-like name or value"
        ) from exc


def _reject_credential_like_environment(
    environment: Mapping[str, str],
) -> None:
    """Reject values that would make persisted provider evidence unsafe."""

    from .ocr_backend_setup import BackendSetupError, make_receipt

    try:
        # Reuse the shared receipt detector before any provider spec or
        # environment can be copied into the public evidence bundle.
        make_receipt(
            "local_ocr_provider_environment_validation",
            {"environment": dict(environment)},
        )
    except BackendSetupError as exc:
        raise ProviderError(
            "provider env contains a credential-like name or value"
        ) from exc
    if any(
        _SENSITIVE_LOCAL_ENV_RE.fullmatch(name)
        or _LOCAL_DSN_RE.match(value.strip())
        or _LOCAL_CREDENTIAL_URL_RE.match(value.strip())
        for name, value in environment.items()
    ):
        raise ProviderError(
            "provider env contains a credential-like name or value"
        )


def _validate_environment_overrides(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ProviderError("provider env must be an object")
    for key, item in value.items():
        if not isinstance(key, str) or not _ENV_NAME_RE.fullmatch(key):
            raise ProviderError(f"invalid provider environment name: {key!r}")
        if not isinstance(item, str) or "\x00" in item:
            raise ProviderError(
                f"provider environment value for {key} must be a string"
            )
        if _SECRET_ENV_RE.search(key):
            raise ProviderError(
                "provider env cannot contain secret-looking names"
            )
    _reject_credential_like_environment(value)
    for key, item in _FORCED_LOCAL_ENV.items():
        if key in value and value[key] != item:
            raise ProviderError(
                f"local-only provider env requires {key}={item}"
            )
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for value in values:
            handle.write(_json_bytes(value).decode("utf-8"))
            handle.write("\n")


def _decode_json_object(payload: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except ValueError as exc:
        raise ProviderError(
            f"Could not read provider spec {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ProviderError("Provider spec must contain one JSON object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProviderError(
            f"Could not read provider spec {path}: {exc}"
        ) from exc
    return _decode_json_object(payload, path)


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProviderError(f"{label} must be a nonempty string")
    if "\x00" in value:
        raise ProviderError(f"{label} cannot contain a NUL byte")
    return value


def _require_finite_duration(
    value: Any,
    label: str,
    *,
    allow_zero: bool = False,
) -> float:
    qualifier = "nonnegative" if allow_zero else "positive"
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProviderError(f"{label} must be a finite {qualifier} number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as exc:
        raise ProviderError(
            f"{label} must be a finite {qualifier} number"
        ) from exc
    if not math.isfinite(normalized) or (
        normalized < 0 if allow_zero else normalized <= 0
    ):
        raise ProviderError(f"{label} must be a finite {qualifier} number")
    return normalized


def _validate_local_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ProviderError("Provider spec has an unsupported schema_version")
    if not set(spec) <= _LOCAL_SPEC_KEYS:
        raise ProviderError("provider spec has unexpected fields")
    provider_id = _require_string(spec.get("provider_id"), "provider_id")
    _reject_credential_like_value(provider_id, "provider_id")

    argv = spec.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(item, str) and item for item in argv)
    ):
        raise ProviderError("provider argv must be a nonempty string list")
    if any("\x00" in argument for argument in argv):
        raise ProviderError("provider argv cannot contain a NUL byte")
    for index, argument in enumerate(argv):
        _reject_credential_like_value(argument, f"provider argv[{index}]")
    executable = Path(argv[0]).expanduser()
    if not executable.is_absolute():
        raise ProviderError("provider argv[0] must be an absolute path")
    try:
        executable_stat = executable.stat()
    except OSError as exc:
        raise ProviderError(
            f"provider executable does not exist: {executable}"
        ) from exc
    if not stat.S_ISREG(executable_stat.st_mode) or not os.access(
        executable, os.X_OK
    ):
        raise ProviderError(
            f"provider executable is not an executable file: {executable}"
        )
    for argument in argv:
        if any(
            argument == protocol_arg
            or argument.startswith(f"{protocol_arg}=")
            for protocol_arg in _PROTOCOL_ARGUMENTS
        ):
            raise ProviderError(
                "provider argv cannot supply protocol path arguments"
            )

    cwd_value = spec.get("cwd")
    if cwd_value is not None:
        cwd = Path(_require_string(cwd_value, "cwd")).expanduser()
        if not cwd.is_absolute() or not cwd.is_dir():
            raise ProviderError(
                "provider cwd must be an existing absolute directory"
            )

    if spec.get("egress", "none") != "none":
        raise ProviderError(
            "production OCR providers must be local-only (egress=none)"
        )

    model = spec.get("model")
    if not isinstance(model, dict):
        raise ProviderError("provider model must be an object")
    if set(model) != {"requested", "revision"}:
        raise ProviderError(
            "provider model must contain exactly requested and revision"
        )
    _require_string(model.get("requested"), "model.requested")
    revision = _require_string(model.get("revision"), "model.revision")
    _reject_credential_like_value(model, "provider model")
    if not _REVISION_RE.fullmatch(revision):
        raise ProviderError(
            "model.revision must be an immutable 40- or 64-character "
            "lowercase hexadecimal revision"
        )

    lanes = spec.get("lanes")
    if not isinstance(lanes, dict) or not lanes:
        raise ProviderError("provider lanes must be a nonempty object")
    for lane, lane_spec in lanes.items():
        _require_string(lane, "lane name")
        if not isinstance(lane_spec, dict):
            raise ProviderError(f"provider lane {lane} must be an object")
        if not set(lane_spec) <= {
            "prompt",
            "settings",
            "include_raw_text",
        }:
            raise ProviderError(f"provider lane {lane} has unexpected fields")
        prompt = _require_string(
            lane_spec.get("prompt"), f"lanes.{lane}.prompt"
        )
        settings = lane_spec.get("settings", {})
        if not isinstance(settings, dict):
            raise ProviderError(f"lanes.{lane}.settings must be an object")
        _reject_credential_like_value(prompt, f"lanes.{lane}.prompt")
        _reject_credential_like_value(settings, f"lanes.{lane}.settings")
        include_raw = lane_spec.get("include_raw_text", False)
        if not isinstance(include_raw, bool):
            raise ProviderError(
                f"lanes.{lane}.include_raw_text must be boolean"
            )

    _require_finite_duration(
        spec.get("timeout_seconds", 1800),
        "timeout_seconds",
    )

    secret_names = spec.get("secret_env_names", [])
    if not isinstance(secret_names, list) or not all(
        isinstance(item, str) and item for item in secret_names
    ):
        raise ProviderError("secret_env_names must be a string list")
    if secret_names:
        raise ProviderError(
            "local-only production providers cannot inherit secret "
            "environment variables"
        )

    _validate_environment_overrides(spec.get("env", {}))


def _normalize_local_spec_paths(spec: dict[str, Any]) -> None:
    try:
        spec["argv"][0] = str(
            Path(spec["argv"][0]).expanduser().resolve(strict=True)
        )
        if spec.get("cwd") is not None:
            spec["cwd"] = str(
                Path(spec["cwd"]).expanduser().resolve(strict=True)
            )
    except OSError as exc:
        raise ProviderError(
            f"Could not resolve provider executable or cwd: {exc}"
        ) from exc


def load_local_provider_spec(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Load and validate a local-only production provider specification."""

    spec_path = Path(path).expanduser().resolve(strict=True)
    spec = _read_json(spec_path)
    _validate_local_spec(spec)
    _normalize_local_spec_paths(spec)
    return spec


def _validate_provider_run_ancestor(path: Path) -> None:
    """Reject ancestry that another unprivileged user could replace."""

    try:
        path_stat = path.lstat()
    except OSError as exc:
        raise ProviderError(
            f"could not inspect provider run directory ancestry: {path}"
        ) from exc
    if stat.S_ISLNK(path_stat.st_mode):
        raise ProviderError(
            "provider run directory ancestry cannot contain symlinks"
        )
    if not stat.S_ISDIR(path_stat.st_mode):
        raise ProviderError(
            "provider run directory ancestry must contain only directories"
        )
    if path_stat.st_uid not in {0, os.geteuid()}:
        raise ProviderError(
            "provider run directory ancestry must be owned by root or "
            "the current user"
        )
    writable_by_others = stat.S_IMODE(path_stat.st_mode) & 0o022
    if writable_by_others:
        sticky = bool(path_stat.st_mode & stat.S_ISVTX)
        if not sticky:
            raise ProviderError(
                "provider run directory ancestry is writable by other users"
            )


def _prepare_provider_run_parent(parent: Path) -> None:
    """Create missing ancestors privately and validate the complete chain."""

    current = Path(parent.anchor)
    components = parent.parts[1:]
    for component in components:
        _validate_provider_run_ancestor(current)
        current /= component
        try:
            current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                # A racing creator is validated below; symlinks and unsafe
                # directories are not accepted.
                pass
            except OSError as exc:
                raise ProviderError(
                    "could not create provider run directory ancestry"
                ) from exc
            else:
                try:
                    current.chmod(0o700, follow_symlinks=False)
                except OSError as exc:
                    raise ProviderError(
                        "could not secure provider run directory ancestry"
                    ) from exc
        except OSError as exc:
            raise ProviderError(
                "could not inspect provider run directory ancestry"
            ) from exc
    _validate_provider_run_ancestor(current)


def validate_provider_run_dir(
    run_dir: str | os.PathLike[str],
) -> Path:
    """Create or validate a new, empty, absolute evidence directory."""

    candidate = Path(run_dir).expanduser()
    if not candidate.is_absolute():
        raise ProviderError("provider run directory must be absolute")
    candidate = Path(os.path.abspath(candidate))
    _prepare_provider_run_parent(candidate.parent)

    open_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        parent_fd = os.open(candidate.parent, open_flags)
    except OSError as exc:
        raise ProviderError(
            "could not safely open provider run directory parent"
        ) from exc
    try:
        # Revalidate the pathname after opening it and bind all leaf creation
        # and inspection to the verified parent descriptor.
        _validate_provider_run_ancestor(candidate.parent)
        parent_path_stat = candidate.parent.lstat()
        parent_fd_stat = os.fstat(parent_fd)
        if (
            parent_path_stat.st_dev != parent_fd_stat.st_dev
            or parent_path_stat.st_ino != parent_fd_stat.st_ino
        ):
            raise ProviderError(
                "provider run directory ancestry changed during setup"
            )

        created = False
        try:
            initial_stat = os.stat(
                candidate.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            try:
                os.mkdir(candidate.name, mode=0o700, dir_fd=parent_fd)
                created = True
                os.chmod(
                    candidate.name,
                    0o700,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                # Validate the object that won the creation race below.
                pass
            except (OSError, ValueError) as exc:
                raise ProviderError(
                    "could not create provider run directory"
                ) from exc
            try:
                initial_stat = os.stat(
                    candidate.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ProviderError(
                    "could not inspect provider run directory"
                ) from exc
        except OSError as exc:
            raise ProviderError(
                "could not inspect provider run directory"
            ) from exc

        if stat.S_ISLNK(initial_stat.st_mode):
            raise ProviderError("provider run directory cannot be a symlink")
        if not stat.S_ISDIR(initial_stat.st_mode):
            raise ProviderError("provider run path is not a directory")

        try:
            directory_fd = os.open(
                candidate.name,
                open_flags,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise ProviderError(
                "could not safely open provider run directory"
            ) from exc
        try:
            directory_stat = os.fstat(directory_fd)
            if (
                directory_stat.st_dev != initial_stat.st_dev
                or directory_stat.st_ino != initial_stat.st_ino
            ):
                raise ProviderError(
                    "provider run directory changed during setup"
                )
            if directory_stat.st_uid != os.geteuid():
                raise ProviderError(
                    "provider run directory must be owned by the current user"
                )
            if created:
                try:
                    os.fchmod(directory_fd, 0o700)
                except OSError as exc:
                    raise ProviderError(
                        "could not secure provider run directory"
                    ) from exc
                directory_stat = os.fstat(directory_fd)
            if stat.S_IMODE(directory_stat.st_mode) != 0o700:
                raise ProviderError(
                    "provider run directory must have private usable "
                    "permissions (0700)"
                )
            if os.listdir(directory_fd):
                raise ProviderError("provider run directory must be empty")
            final_parent_stat = candidate.parent.lstat()
            final_stat = candidate.lstat()
            if (
                final_parent_stat.st_dev != parent_fd_stat.st_dev
                or final_parent_stat.st_ino != parent_fd_stat.st_ino
                or stat.S_ISLNK(final_stat.st_mode)
                or final_stat.st_dev != directory_stat.st_dev
                or final_stat.st_ino != directory_stat.st_ino
            ):
                raise ProviderError(
                    "provider run directory changed during setup"
                )
        except OSError as exc:
            raise ProviderError(
                "could not validate provider run directory"
            ) from exc
        finally:
            os.close(directory_fd)
    finally:
        os.close(parent_fd)
    return candidate


def build_local_provider_environment(
    spec: Mapping[str, Any],
    *,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a minimal environment from safe ambient and spec values."""

    source = os.environ if base_environment is None else base_environment
    overrides = _validate_environment_overrides(spec.get("env", {}))
    environment = {
        key: value
        for key, value in source.items()
        if key in _LOCAL_AMBIENT_ENV_ALLOWLIST
        and "\x00" not in key
        and "\x00" not in value
    }
    environment["PATH"] = "/usr/bin:/bin"
    environment.update(overrides)
    environment.update(_FORCED_LOCAL_ENV)
    environment["NO_PROXY"] = "127.0.0.1,localhost"
    environment["no_proxy"] = "127.0.0.1,localhost"
    _reject_credential_like_environment(environment)
    return environment


def _recorded_provider_environment(
    environment: Mapping[str, str],
    spec: Mapping[str, Any],
) -> dict[str, str]:
    del spec
    _reject_credential_like_environment(environment)
    return dict(sorted(environment.items()))


def _png_dimensions(path: Path) -> dict[str, int] | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError:
        return None
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return {"w": width, "h": height}


def _normalize_pages(
    pages: Sequence[Mapping[str, Any]],
    *,
    spec: Mapping[str, Any],
    spec_hash: str,
    lane: str,
) -> list[dict[str, Any]]:
    if isinstance(pages, (str, bytes)) or not isinstance(pages, Sequence):
        raise ProviderError("pages must be a nonempty sequence")
    if not pages:
        raise ProviderError("pages must be a nonempty sequence")
    lane_specs = spec["lanes"]
    if lane not in lane_specs:
        raise ProviderError(f"provider does not define lane: {lane}")
    lane_spec = lane_specs[lane]
    if lane_spec.get("include_raw_text", False):
        raise ProviderError(
            "production provider lanes cannot request extracted raw text"
        )

    requests = []
    page_ids = set()
    for index, page in enumerate(pages):
        label = f"pages[{index}]"
        if not isinstance(page, Mapping):
            raise ProviderError(f"{label} must be an object")
        page_id = _require_string(page.get("page_id"), f"{label}.page_id")
        if any(ord(character) < 32 for character in page_id):
            raise ProviderError(
                f"{label}.page_id contains control characters"
            )
        if page_id in page_ids:
            raise ProviderError(f"duplicate page_id: {page_id}")
        page_ids.add(page_id)

        image_path = Path(
            _require_string(page.get("image_path"), f"{label}.image_path")
        ).expanduser()
        if not image_path.is_absolute() or not image_path.is_file():
            raise ProviderError(
                f"{label}.image_path must be an existing absolute file"
            )
        image_path = image_path.resolve(strict=True)
        image_sha256 = _require_string(
            page.get("image_sha256"), f"{label}.image_sha256"
        )
        if not _SHA256_RE.fullmatch(image_sha256):
            raise ProviderError(f"{label}.image_sha256 is invalid")
        actual_hash = _file_sha256(image_path)
        if actual_hash != image_sha256:
            raise ProviderError(
                f"{label} image SHA-256 mismatch: expected "
                f"{image_sha256}, got {actual_hash}"
            )

        dimensions = page.get("image_dims")
        if (
            not isinstance(dimensions, Mapping)
            or set(dimensions) != {"w", "h"}
            or not all(
                isinstance(dimensions[key], int)
                and not isinstance(dimensions[key], bool)
                and dimensions[key] > 0
                for key in ("w", "h")
            )
        ):
            raise ProviderError(
                f"{label}.image_dims must contain positive integer w and h"
            )
        dimensions = {"w": dimensions["w"], "h": dimensions["h"]}
        actual_dimensions = _png_dimensions(image_path)
        if actual_dimensions != dimensions:
            raise ProviderError(
                f"{label} image dimensions mismatch: expected "
                f"{dimensions}, got {actual_dimensions}"
            )

        instruction = _require_string(
            page.get("instruction"), f"{label}.instruction"
        )
        if instruction != lane_spec["prompt"]:
            raise ProviderError(
                f"{label}.instruction does not match lanes.{lane}.prompt"
            )
        settings = page.get("settings")
        if not isinstance(settings, Mapping):
            raise ProviderError(f"{label}.settings must be an object")
        settings = dict(settings)
        if settings != lane_spec.get("settings", {}):
            raise ProviderError(
                f"{label}.settings do not match lanes.{lane}.settings"
            )

        identity = {
            "provider_spec_sha256": spec_hash,
            "lane": lane,
            "page_id": page_id,
            "image_sha256": image_sha256,
            "instruction_sha256": _sha256_text(instruction),
            "settings": settings,
        }
        requests.append(
            {
                "schema_version": SCHEMA_VERSION,
                "attempt_id": _sha256_json(identity),
                "case_id": page_id,
                "page_id": page_id,
                "lane": lane,
                "input": {
                    "image": {
                        "path": str(image_path),
                        "sha256": image_sha256,
                        "dimensions": dimensions,
                    },
                    "raw_text": None,
                },
                "instruction": instruction,
                "instruction_sha256": _sha256_text(instruction),
                "settings": settings,
            }
        )
    return requests


def _nested_value(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _has_nested(value: Mapping[str, Any], dotted_path: str) -> bool:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def _safe_artifact_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _is_nonnegative_integer(value: Any) -> bool:
    return (
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
    )


def _is_nonnegative_number(value: Any) -> bool:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value < 0
    ):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_provider_response(
    response: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    spec: Mapping[str, Any],
    artifacts_dir: str | os.PathLike[str],
) -> list[str]:
    """Return missing or invalid provenance fields for a completion."""

    gaps = []
    if response.get("schema_version") != SCHEMA_VERSION:
        gaps.append("schema_version")
    if response.get("attempt_id") != request.get("attempt_id"):
        gaps.append("attempt_id")
    if response.get("status") != "completed":
        gaps.append("status")
        return gaps

    output = response.get("output")
    if not isinstance(output, Mapping):
        gaps.append("output")
    else:
        if output.get("format") != "markdown":
            gaps.append("output.format")
        if not isinstance(output.get("text"), str):
            gaps.append("output.text")

    required = (
        "effective_request.prompt",
        "effective_request.input_artifacts",
        "model.requested",
        "model.revision",
        "model.resolved_snapshot",
        "runtime.python",
        "runtime.engine",
        "runtime.engine_version",
        "runtime.packages",
        "runtime.attention_backend",
        "generation.params",
        "generation.input_tokens",
        "generation.output_tokens",
        "generation.finish_reason",
        "generation.stop_reason",
        "generation.eos",
        "generation.max_tokens",
        "usage.wall_seconds",
        "usage.cost_usd",
        "artifacts",
    )
    gaps.extend(path for path in required if not _has_nested(response, path))

    prompt = _nested_value(response, "effective_request.prompt")
    if not isinstance(prompt, str) or not prompt:
        gaps.append("effective_request.prompt")
    effective_inputs = _nested_value(
        response, "effective_request.input_artifacts"
    )
    expected_image = request["input"]["image"]
    original = None
    if isinstance(effective_inputs, list):
        original = next(
            (
                item
                for item in effective_inputs
                if isinstance(item, Mapping)
                and item.get("transform") == "original"
                and item.get("path") == expected_image["path"]
                and item.get("sha256") == expected_image["sha256"]
                and item.get("dimensions") == expected_image["dimensions"]
            ),
            None,
        )
    if original is None:
        gaps.append("effective_request.original_image")

    model = spec["model"]
    if _nested_value(response, "model.requested") != model["requested"]:
        gaps.append("model.requested_mismatch")
    if _nested_value(response, "model.revision") != model["revision"]:
        gaps.append("model.revision_mismatch")
    resolved_snapshot = _nested_value(response, "model.resolved_snapshot")
    if not isinstance(resolved_snapshot, str) or not resolved_snapshot:
        gaps.append("model.resolved_snapshot")
    elif (
        resolved_snapshot != model["revision"]
        and Path(resolved_snapshot).name != model["revision"]
    ):
        gaps.append("model.resolved_snapshot_mismatch")

    for path in (
        "model.requested",
        "model.revision",
        "runtime.python",
        "runtime.engine",
        "runtime.engine_version",
        "generation.finish_reason",
    ):
        value = _nested_value(response, path)
        if not isinstance(value, str) or not value:
            gaps.append(path)

    packages = _nested_value(response, "runtime.packages")
    if (
        not isinstance(packages, Mapping)
        or not packages
        or not all(
            isinstance(key, str) and key and isinstance(value, str) and value
            for key, value in packages.items()
        )
    ):
        gaps.append("runtime.packages")
    attention_backend = _nested_value(response, "runtime.attention_backend")
    if not isinstance(attention_backend, str) or not attention_backend:
        gaps.append("runtime.attention_backend")
    if not isinstance(_nested_value(response, "generation.params"), Mapping):
        gaps.append("generation.params")
    input_tokens = _nested_value(response, "generation.input_tokens")
    if not _is_nonnegative_integer(input_tokens):
        gaps.append("generation.input_tokens")
    if not _is_nonnegative_integer(
        _nested_value(response, "generation.output_tokens")
    ):
        gaps.append("generation.output_tokens")
    finish_reason = _nested_value(response, "generation.finish_reason")
    stop_reason = _nested_value(response, "generation.stop_reason")
    if not (
        (isinstance(stop_reason, str) and bool(stop_reason))
        or (
            isinstance(stop_reason, int) and not isinstance(stop_reason, bool)
        )
    ):
        gaps.append("generation.stop_reason")
    eos = _nested_value(response, "generation.eos")
    max_tokens = _nested_value(response, "generation.max_tokens")
    if not isinstance(eos, bool):
        gaps.append("generation.eos")
    if not isinstance(max_tokens, bool):
        gaps.append("generation.max_tokens")
    if isinstance(eos, bool) and isinstance(max_tokens, bool):
        if eos and max_tokens:
            gaps.append("generation.termination_conflict")
        elif not eos and not max_tokens and stop_reason is None:
            gaps.append("generation.termination_unknown")
    if finish_reason == "length" and max_tokens is not True:
        gaps.append("generation.length_without_max_tokens")

    if not _is_nonnegative_number(
        _nested_value(response, "usage.wall_seconds")
    ):
        gaps.append("usage.wall_seconds")
    cost = _nested_value(response, "usage.cost_usd")
    if cost is not None and not _is_nonnegative_number(cost):
        gaps.append("usage.cost_usd")

    root = Path(artifacts_dir)
    artifacts = response.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        gaps.append("artifacts")
        artifacts = []
    seen_artifacts = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            gaps.append(f"artifacts.{index}")
            continue
        relative = artifact.get("relative_path")
        if isinstance(relative, str):
            if relative in seen_artifacts:
                gaps.append(f"artifacts.{index}.duplicate_path")
            seen_artifacts.add(relative)
        artifact_path = _safe_artifact_path(root, relative)
        expected_hash = artifact.get("sha256")
        if artifact_path is None:
            gaps.append(f"artifacts.{index}.relative_path")
            continue
        if (
            not isinstance(expected_hash, str)
            or not _SHA256_RE.fullmatch(expected_hash)
            or _file_sha256(artifact_path) != expected_hash
        ):
            gaps.append(f"artifacts.{index}.sha256")
    return list(dict.fromkeys(gaps))


def _read_response_jsonl(
    path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    values = []
    errors = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(
                        line,
                        object_pairs_hook=_unique_json_object,
                        parse_constant=_reject_json_constant,
                        parse_float=_parse_finite_json_float,
                    )
                except ValueError as exc:
                    errors.append(
                        "malformed response JSON at line "
                        f"{line_number}: {exc}"
                    )
                    continue
                if not isinstance(value, dict):
                    errors.append(
                        f"response at line {line_number} is not an object"
                    )
                    continue
                values.append(value)
    except OSError as exc:
        errors.append(f"could not read provider responses: {exc}")
    return values, errors


def _signal_process_group(pgid: int, signal_number: int) -> bool:
    try:
        os.killpg(pgid, signal_number)
    except ProcessLookupError:
        return False
    except OSError:
        return False
    return True


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _terminate_process_group(
    process: subprocess.Popen[bytes],
    grace_seconds: float,
) -> list[str]:
    signals = []
    deadline = time.monotonic() + grace_seconds
    if _signal_process_group(process.pid, signal.SIGTERM):
        signals.append("SIGTERM")
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    while _process_group_exists(process.pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    if _process_group_exists(process.pid):
        if _signal_process_group(process.pid, signal.SIGKILL):
            signals.append("SIGKILL")
    if process.poll() is None:
        process.wait()
    return signals


def _argv_file_hashes(argv: Sequence[str]) -> list[dict[str, str]]:
    hashes = []
    seen = set()
    for value in argv:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute() or not candidate.is_file():
            continue
        resolved = candidate.resolve(strict=True)
        if resolved in seen:
            continue
        seen.add(resolved)
        hashes.append(
            {"path": str(resolved), "sha256": _file_sha256(resolved)}
        )
    return hashes


def _run_provider_process(
    *,
    argv: list[str],
    cwd: str | None,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
    terminate_grace_seconds: float,
) -> dict[str, Any]:
    timeout_seconds = _require_finite_duration(
        timeout_seconds,
        "timeout_seconds",
    )
    terminate_grace_seconds = _require_finite_duration(
        terminate_grace_seconds,
        "terminate_grace_seconds",
        allow_zero=True,
    )
    with isolated_subprocess_signal_guard() as lifecycle:
        return _run_provider_process_guarded(
            argv=argv,
            cwd=cwd,
            environment=environment,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=timeout_seconds,
            terminate_grace_seconds=terminate_grace_seconds,
            lifecycle=lifecycle,
        )


def _run_provider_process_guarded(
    *,
    argv: list[str],
    cwd: str | None,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
    terminate_grace_seconds: float,
    lifecycle: Any,
) -> dict[str, Any]:
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    timed_out = False
    error = None
    termination_signals: list[str] = []
    returncode = 127
    with (
        stdout_path.open("wb") as stdout_handle,
        stderr_path.open("wb") as stderr_handle,
    ):
        try:
            with lifecycle.spawn_window():
                process = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=dict(environment),
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    start_new_session=True,
                )
            try:
                returncode = lifecycle.wait(
                    process,
                    timeout_seconds=timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                error = (
                    f"provider timed out after {timeout_seconds:g} seconds"
                )
        except OSError as exc:
            error = f"could not execute provider: {exc}"
            returncode = 127
        finally:
            # Remain inside the passive signal guard until the entire private
            # group is gone, on normal completion and on every failure path.
            with lifecycle.cleanup_window():
                if process is not None:
                    termination_signals.extend(
                        _terminate_process_group(
                            process,
                            terminate_grace_seconds,
                        )
                    )
                    returncode = process.returncode
    return {
        "argv": argv,
        "cwd": cwd,
        "returncode": returncode,
        "timed_out": timed_out,
        "error": error,
        "termination_signals": list(dict.fromkeys(termination_signals)),
        "wall_seconds": round(time.monotonic() - started, 6),
        "stdout_sha256": _file_sha256(stdout_path),
        "stderr_sha256": _file_sha256(stderr_path),
    }


def _provider_error_text(response: Mapping[str, Any]) -> str:
    error = response.get("error")
    if isinstance(error, Mapping):
        error_type = error.get("type")
        message = error.get("message")
        if error_type and message:
            return f"provider error {error_type}: {message}"
        if message:
            return f"provider error: {message}"
    if isinstance(error, str) and error:
        return f"provider error: {error}"
    return "provider returned error status"


def _normalized_metadata(
    *,
    request: Mapping[str, Any],
    response: Mapping[str, Any] | None,
    spec: Mapping[str, Any],
    spec_hash: str,
    environment_hash: str,
    lane: str,
    request_hash: str,
    response_hash: str | None,
    run_dir: Path,
    errors: Sequence[str],
) -> dict[str, Any]:
    status = "completed" if response is not None and not errors else "error"
    metadata: dict[str, Any] = {
        "status": status,
        "backend": "local_provider",
        "provider": {
            "provider_id": spec["provider_id"],
            "lane": lane,
            "provider_spec_sha256": spec_hash,
            "provider_environment_sha256": environment_hash,
        },
        "image": request["input"]["image"],
        "prompt_mode": f"provider:{lane}",
        "output_format": "md",
        "evidence": {
            "run_dir": str(run_dir),
            "request_sha256": request_hash,
            "response_sha256": response_hash,
        },
    }
    if response is None:
        metadata["error"] = list(errors)
        return metadata
    generation = response.get("generation", {})
    effective_request = response.get("effective_request", {})
    metadata.update(
        {
            "prompt": effective_request.get("prompt"),
            "instruction": request["instruction"],
            "model": response.get("model"),
            "runtime": response.get("runtime"),
            "sampling_params": generation.get("params"),
            "prompt_token_count": generation.get("input_tokens"),
            "generated_token_count": generation.get("output_tokens"),
            "finish_reason": generation.get("finish_reason"),
            "stop_reason": generation.get("stop_reason"),
            "termination": {
                "eos": generation.get("eos"),
                "length": generation.get("finish_reason") == "length",
                "max_tokens": generation.get("max_tokens"),
            },
            "usage": response.get("usage"),
            "artifacts": response.get("artifacts"),
        }
    )
    if errors:
        metadata["error"] = list(errors)
    return metadata


def run_local_provider_batch(
    *,
    spec_path: str | os.PathLike[str],
    lane: str,
    run_dir: str | os.PathLike[str],
    pages: Sequence[Mapping[str, Any]],
    out: Callable[[str], None] | None = None,
    timeout_seconds: float | None = None,
    terminate_grace_seconds: float = 10.0,
) -> dict[str, Any]:
    """Run one local provider process for a complete page request batch."""

    spec_source = Path(spec_path).expanduser().resolve(strict=True)
    try:
        spec_source_bytes = spec_source.read_bytes()
    except OSError as exc:
        raise ProviderError(
            f"Could not read provider spec {spec_source}: {exc}"
        ) from exc
    spec = _decode_json_object(spec_source_bytes, spec_source)
    _validate_local_spec(spec)
    _normalize_local_spec_paths(spec)
    spec_source_hash = hashlib.sha256(spec_source_bytes).hexdigest()
    lane = _require_string(lane, "lane")
    spec_hash = _sha256_json(spec)
    requests = _normalize_pages(
        pages,
        spec=spec,
        spec_hash=spec_hash,
        lane=lane,
    )
    timeout = spec.get("timeout_seconds", 1800)
    if timeout_seconds is not None:
        timeout = timeout_seconds
    timeout = _require_finite_duration(timeout, "timeout_seconds")
    terminate_grace_seconds = _require_finite_duration(
        terminate_grace_seconds,
        "terminate_grace_seconds",
        allow_zero=True,
    )
    destination = validate_provider_run_dir(run_dir)

    requests_path = destination / "requests.jsonl"
    responses_path = destination / "responses.jsonl"
    artifacts_dir = destination / "artifacts"
    stdout_path = destination / "stdout.log"
    stderr_path = destination / "stderr.log"
    artifacts_dir.mkdir()
    (destination / "provider-spec.json").write_bytes(spec_source_bytes)
    _write_json(destination / "provider-spec.normalized.json", spec)
    _write_jsonl(requests_path, requests)
    argv = [
        *spec["argv"],
        "--requests-jsonl",
        str(requests_path),
        "--responses-jsonl",
        str(responses_path),
        "--artifacts-dir",
        str(artifacts_dir),
    ]
    environment = build_local_provider_environment(spec)
    recorded_environment = _recorded_provider_environment(environment, spec)
    environment_hash = _sha256_json(recorded_environment)
    started_at = datetime.now(UTC).isoformat()
    run_record = {
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "provider_id": spec["provider_id"],
        "lane": lane,
        "page_count": len(requests),
        "provider_spec_path": str(spec_source),
        "provider_spec_source_sha256": spec_source_hash,
        "provider_spec_sha256": spec_hash,
        "provider_environment": recorded_environment,
        "provider_environment_sha256": environment_hash,
        "argv_files": _argv_file_hashes(spec["argv"]),
        "requests_jsonl_sha256": _file_sha256(requests_path),
        "timeout_seconds": timeout,
    }
    _write_json(destination / "run.json", run_record)
    if out:
        out(
            f"Running local OCR provider {spec['provider_id']} "
            f"lane={lane} for {len(requests)} page(s)."
        )
    process_record = _run_provider_process(
        argv=argv,
        cwd=spec.get("cwd"),
        environment=environment,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_seconds=timeout,
        terminate_grace_seconds=terminate_grace_seconds,
    )

    protocol_errors = []
    if process_record["error"]:
        protocol_errors.append(process_record["error"])
    if process_record["returncode"] != 0 and not process_record["error"]:
        protocol_errors.append(
            f"provider exited with status {process_record['returncode']}"
        )
    if not responses_path.exists():
        responses_path.touch()
        protocol_errors.append("provider did not create responses JSONL")
    responses, parse_errors = _read_response_jsonl(responses_path)
    protocol_errors.extend(parse_errors)

    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids = set()
    missing_id_lines = 0
    for response in responses:
        attempt_id = response.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            missing_id_lines += 1
            continue
        if attempt_id in by_id:
            duplicate_ids.add(attempt_id)
        else:
            by_id[attempt_id] = response
    if missing_id_lines:
        protocol_errors.append(
            f"{missing_id_lines} response object(s) had no valid attempt_id"
        )
    if duplicate_ids:
        protocol_errors.append(
            "duplicate response IDs: " + ", ".join(sorted(duplicate_ids))
        )
    expected_ids = {request["attempt_id"] for request in requests}
    unexpected_ids = set(by_id) - expected_ids
    if unexpected_ids:
        protocol_errors.append(
            "unexpected response IDs: " + ", ".join(sorted(unexpected_ids))
        )

    results = []
    for request in requests:
        attempt_id = request["attempt_id"]
        response = by_id.get(attempt_id)
        errors = []
        metadata_gaps = []
        if attempt_id in duplicate_ids:
            errors.append("provider returned a duplicate response ID")
        if response is None:
            errors.append("provider response is missing")
        elif response.get("status") == "error":
            errors.append(_provider_error_text(response))
        else:
            metadata_gaps = validate_provider_response(
                response,
                request=request,
                spec=spec,
                artifacts_dir=artifacts_dir,
            )
            if metadata_gaps:
                errors.append("provider response provenance is incomplete")
        output_text = None
        if response is not None:
            output = response.get("output")
            if isinstance(output, Mapping) and isinstance(
                output.get("text"), str
            ):
                output_text = output["text"]
        request_hash = _sha256_json(request)
        response_hash = (
            _sha256_json(response) if response is not None else None
        )
        valid = not errors
        normalized = _normalized_metadata(
            request=request,
            response=response,
            spec=spec,
            spec_hash=spec_hash,
            environment_hash=environment_hash,
            lane=lane,
            request_hash=request_hash,
            response_hash=response_hash,
            run_dir=destination,
            errors=errors,
        )
        results.append(
            {
                "page_id": request["page_id"],
                "attempt_id": attempt_id,
                "valid": valid,
                "status": response.get("status") if response else "missing",
                "errors": errors,
                "metadata_gaps": metadata_gaps,
                "output_text": output_text,
                "output_sha256": (
                    _sha256_text(output_text)
                    if output_text is not None
                    else None
                ),
                "request_sha256": request_hash,
                "response_sha256": response_hash,
                "request": request,
                "response": response,
                "normalized_metadata": normalized,
            }
        )

    protocol_errors = list(dict.fromkeys(protocol_errors))
    batch = {
        "schema_version": SCHEMA_VERSION,
        "ok": not protocol_errors
        and all(result["valid"] for result in results),
        "provider_id": spec["provider_id"],
        "lane": lane,
        "provider_spec_sha256": spec_hash,
        "provider_environment": recorded_environment,
        "hashes": {
            "provider_spec_source_sha256": spec_source_hash,
            "provider_spec_sha256": spec_hash,
            "provider_environment_sha256": environment_hash,
            "requests_jsonl_sha256": _file_sha256(requests_path),
            "responses_jsonl_sha256": _file_sha256(responses_path),
        },
        "paths": {
            "run_dir": str(destination),
            "provider_spec_source": str(destination / "provider-spec.json"),
            "provider_spec_normalized": str(
                destination / "provider-spec.normalized.json"
            ),
            "requests_jsonl": str(requests_path),
            "responses_jsonl": str(responses_path),
            "artifacts_dir": str(artifacts_dir),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        },
        "process": process_record,
        "protocol_errors": protocol_errors,
        "unexpected_response_ids": sorted(unexpected_ids),
        "results": results,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
    }
    _write_json(destination / "batch.json", batch)
    return batch
