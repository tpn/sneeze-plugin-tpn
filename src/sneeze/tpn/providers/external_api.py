"""Fail-closed JSONL adapter for approved hosted OCR experiments.

The adapter has one small stdlib HTTPS transport, but no call is made until an
external plan, separate approval, provider spec, executor bundle, every JSONL
request, and a durable pessimistic budget reservation have been validated.
Tests inject an in-memory transport; importing this module never reads the
environment or opens a socket.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import http.client
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import ssl
import struct
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 256 * 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9._:/+-]{1,256}$")
_RETRYABLE_HTTP_STATUS = frozenset({408, 409, 425, 429})
_ISOLATED_STARTUP_FLAGS = ("-I", "-S", "-B")
_SPEC_KEYS = frozenset(
    {
        "schema_version",
        "provider_id",
        "argv",
        "egress",
        "model",
        "lanes",
        "timeout_seconds",
        "secret_env_names",
        "env",
        "external_policy",
    }
)


class ExternalAdapterError(ValueError):
    """Raised before or after a definitively delivered hosted request."""


class DeliveryUnknownError(RuntimeError):
    """Raised when transport cannot prove whether a request was delivered."""


class ConfirmedDeliveryError(RuntimeError):
    """Raised when a confirmed response cannot be safely represented."""


class _LazyExternalPlanningModule:
    """Load the sibling helper without importing package initializers."""

    _module: Any | None = None

    @staticmethod
    def path() -> Path:
        return Path(__file__).resolve().parents[1] / "ocr_external.py"

    def load(self) -> Any:
        if self._module is None:
            path = self.path()
            module_spec = importlib.util.spec_from_file_location(
                "_sneeze_tpn_external_planning_executor",
                path,
            )
            if module_spec is None or module_spec.loader is None:
                raise ExternalAdapterError(
                    "could not load the external planning helper"
                )
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            self._module = module
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self.load(), name)


ocr_external = _LazyExternalPlanningModule()


@dataclass(frozen=True)
class TransportResult:
    status_code: int
    response_headers: dict[str, str]
    body: bytes
    wall_seconds: float


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, UnicodeError, ValueError) as exc:
        raise ExternalAdapterError("value is not canonical JSON") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ExternalAdapterError(f"{label} exceeds the size limit")
        value = json.loads(path.read_text(encoding="utf-8"))
    except ExternalAdapterError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise ExternalAdapterError(f"could not read {label}") from exc
    if not isinstance(value, dict):
        raise ExternalAdapterError(f"{label} must be one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ExternalAdapterError(
                "requests JSONL exceeds the size limit"
            )
        lines = path.read_text(encoding="utf-8").splitlines()
    except ExternalAdapterError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ExternalAdapterError("could not read requests JSONL") from exc
    values = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError as exc:
            raise ExternalAdapterError(
                f"invalid request JSON at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ExternalAdapterError(
                f"request at line {line_number} must be an object"
            )
        values.append(value)
    if not values:
        raise ExternalAdapterError("requests JSONL is empty")
    return values


def _exact_keys(value: Any, keys: frozenset[str], label: str) -> dict:
    if not isinstance(value, dict) or frozenset(value) != keys:
        raise ExternalAdapterError(f"{label} has unexpected fields")
    return value


def _safe_string(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not _SAFE_VALUE_RE.fullmatch(value)
    ):
        raise ExternalAdapterError(f"{label} is invalid")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ExternalAdapterError(f"{label} must be a SHA-256 digest")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ExternalAdapterError(f"{label} must be a positive integer")
    return value


def executor_bundle_identity(
    python_executable: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Hash this adapter and every non-stdlib executor helper it imports."""

    files = []
    for module_name, path in (
        ("sneeze.tpn.ocr_external", ocr_external.path().resolve()),
        ("sneeze.tpn.providers.external_api", Path(__file__).resolve()),
    ):
        files.append(
            {
                "module": module_name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_path(path),
            }
        )
    executable = Path(python_executable or sys.executable).expanduser()
    if not executable.is_absolute() or not executable.is_file():
        raise ExternalAdapterError("executor Python executable is invalid")
    executable = executable.resolve(strict=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "startup_flags": list(_ISOLATED_STARTUP_FLAGS),
        "python_executable": {
            "path": str(executable),
            "size_bytes": executable.stat().st_size,
            "sha256": _sha256_path(executable),
        },
        "files": files,
    }
    return {**payload, "bundle_sha256": _sha256_json(payload)}


def _require_isolated_startup() -> None:
    checks = {
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
        "no_user_site": sys.flags.no_user_site,
        "ignore_environment": sys.flags.ignore_environment,
        "safe_path": sys.flags.safe_path,
        "dont_write_bytecode": sys.flags.dont_write_bytecode,
    }
    if not all(checks.values()):
        missing = ", ".join(
            name for name, enabled in checks.items() if not enabled
        )
        raise ExternalAdapterError(
            "external adapter requires isolated no-site startup; "
            f"missing={missing}"
        )


def _require_self_digest(
    value: dict[str, Any], field: str, label: str
) -> str:
    digest = _sha256(value.get(field), f"{label}.{field}")
    without_digest = {
        key: item for key, item in value.items() if key != field
    }
    if _sha256_json(without_digest) != digest:
        raise ExternalAdapterError(f"{label} digest does not match contents")
    return digest


def _isolated_executor_preflight(
    *,
    provider_spec_path: str | os.PathLike[str],
    plan_path: str | os.PathLike[str],
    approval_path: str | os.PathLike[str],
) -> None:
    """Attest all non-stdlib code before importing it with a key present."""

    _require_isolated_startup()
    spec = _read_json(Path(provider_spec_path), "external provider spec")
    plan = _read_json(Path(plan_path), "external plan")
    approval = _read_json(Path(approval_path), "external approval")
    plan_digest = _require_self_digest(
        plan, "plan_sha256", "external plan"
    )
    _require_self_digest(
        approval, "approval_sha256", "external approval"
    )
    if approval.get("plan_sha256") != plan_digest:
        raise ExternalAdapterError(
            "external approval is bound to a different plan"
        )
    executor = plan.get("executor")
    if not isinstance(executor, dict):
        raise ExternalAdapterError("external plan executor is invalid")
    expected_bundle = _sha256(
        executor.get("adapter_bundle_sha256"),
        "external plan executor bundle",
    )
    expected_spec = _sha256(
        executor.get("provider_spec_sha256"),
        "external plan provider spec",
    )
    if _sha256_json(spec) != expected_spec:
        raise ExternalAdapterError(
            "external provider spec differs from the approved plan"
        )
    bundle = executor_bundle_identity()
    if bundle["bundle_sha256"] != expected_bundle:
        raise ExternalAdapterError(
            "external executor differs from the approved plan"
        )
    policy = spec.get("external_policy")
    if not isinstance(policy, dict):
        raise ExternalAdapterError("external provider policy is invalid")
    if (
        policy.get("executor_bundle_sha256") != expected_bundle
        or policy.get("python_executable")
        != bundle["python_executable"]
    ):
        raise ExternalAdapterError(
            "external provider policy executor identity is stale"
        )


def validate_external_provider_spec_record(
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Validate the exact hosted provider spec used by this adapter."""

    spec = json.loads(_json_bytes(spec))
    _exact_keys(spec, _SPEC_KEYS, "external provider spec")
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ExternalAdapterError("unsupported provider spec schema")
    profile_id = _safe_string(spec.get("provider_id"), "provider_id")
    profile = ocr_external.get_external_provider_profile(profile_id)
    profile_sha256 = profile.pop("profile_sha256")
    if spec.get("egress") != "external":
        raise ExternalAdapterError(
            "external provider egress must be external"
        )
    model = _exact_keys(
        spec.get("model"),
        frozenset({"requested", "revision"}),
        "model",
    )
    if model != {
        "requested": profile["model"]["requested"],
        "revision": profile["model"]["revision_semantics"],
    }:
        raise ExternalAdapterError(
            "provider spec model does not match profile"
        )
    lanes = spec.get("lanes")
    if not isinstance(lanes, dict) or len(lanes) != 1:
        raise ExternalAdapterError(
            "external spec must define exactly one lane"
        )
    lane_id, lane = next(iter(lanes.items()))
    _safe_string(lane_id, "lane_id")
    lane = _exact_keys(
        lane,
        frozenset({"prompt", "settings", "include_raw_text"}),
        "lane",
    )
    if not isinstance(lane.get("prompt"), str) or not lane["prompt"].strip():
        raise ExternalAdapterError("lane prompt must be nonempty")
    if not isinstance(lane.get("include_raw_text"), bool):
        raise ExternalAdapterError(
            "include_raw_text must be explicit boolean"
        )
    if lane_id == "parity" and lane["include_raw_text"]:
        raise ExternalAdapterError(
            "the parity lane cannot include raw text"
        )
    settings = _exact_keys(
        lane.get("settings"),
        frozenset({"profile_id", "prompt_mode"}),
        "lane settings",
    )
    if settings.get("profile_id") != profile_id:
        raise ExternalAdapterError("lane profile does not match provider")
    _safe_string(settings.get("prompt_mode"), "prompt_mode")
    policy = _exact_keys(
        spec.get("external_policy"),
        frozenset(
            {
                "profile_id",
                "profile_sha256",
                "executor_bundle_sha256",
                "python_executable",
                "credential_env_name",
                "endpoint",
                "lane_id",
            }
        ),
        "external_policy",
    )
    if spec.get("secret_env_names") != [profile["credential_env_name"]]:
        raise ExternalAdapterError(
            "spec must expose exactly one credential name"
        )
    if spec.get("env") != {}:
        raise ExternalAdapterError("external provider env must be empty")
    timeout = spec.get("timeout_seconds")
    if (
        not isinstance(timeout, int)
        or isinstance(timeout, bool)
        or not 1 <= timeout <= 600
    ):
        raise ExternalAdapterError(
            "timeout_seconds must be from 1 through 600"
        )
    argv = spec.get("argv")
    adapter = Path(__file__).resolve()
    if (
        not isinstance(argv, list)
        or len(argv) != 9
        or argv[1:] != [
            *_ISOLATED_STARTUP_FLAGS,
            str(adapter),
            "--profile-id",
            profile_id,
            "--lane-id",
            lane_id,
        ]
    ):
        raise ExternalAdapterError(
            "provider argv is not the exact adapter argv"
        )
    executable = Path(argv[0]).expanduser()
    if not executable.is_absolute() or not executable.is_file():
        raise ExternalAdapterError("provider Python executable is invalid")
    executable = executable.resolve(strict=True)
    spec["argv"][0] = str(executable)
    bundle = executor_bundle_identity(executable)
    if policy != {
        "profile_id": profile_id,
        "profile_sha256": profile_sha256,
        "executor_bundle_sha256": bundle["bundle_sha256"],
        "python_executable": bundle["python_executable"],
        "credential_env_name": profile["credential_env_name"],
        "endpoint": profile["endpoint"],
        "lane_id": lane_id,
    }:
        raise ExternalAdapterError("external policy is stale or mismatched")
    return spec


def external_provider_spec_sha256(spec: dict[str, Any]) -> str:
    return _sha256_json(validate_external_provider_spec_record(spec))


def _absolute_file(path: str | os.PathLike[str], label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or not candidate.is_file():
        raise ExternalAdapterError(f"{label} must be an absolute file")
    return candidate.resolve(strict=True)


def _absolute_output(path: str | os.PathLike[str], label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise ExternalAdapterError(f"{label} must be absolute")
    if candidate.exists():
        raise ExternalAdapterError(f"{label} already exists")
    if not candidate.parent.is_dir():
        raise ExternalAdapterError(f"{label} parent does not exist")
    return candidate


def _png_dimensions(payload: bytes) -> dict[str, int] | None:
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", payload[16:24])
    return {"w": width, "h": height}


def _request_attempt_id(
    request: dict[str, Any], plan: dict[str, Any]
) -> str:
    identity = {
        "provider_spec_sha256": plan["executor"][
            "provider_spec_sha256"
        ],
        "manifest_sha256": plan["manifest_sha256"],
        "lane": request["lane"],
        "repeat": request["repeat"],
        "case_id": request["case_id"],
        "experiment_sha256": request["experiment_sha256"],
    }
    return _sha256_json(identity)


def _validate_request(
    request: dict[str, Any],
    *,
    plan: dict[str, Any],
    spec: dict[str, Any],
    lane_id: str,
) -> dict[str, Any]:
    _exact_keys(
        request,
        frozenset(
            {
                "schema_version",
                "attempt_id",
                "experiment_sha256",
                "case_id",
                "lane",
                "repeat",
                "input",
                "instruction",
                "instruction_sha256",
                "settings",
            }
        ),
        "provider request",
    )
    if request.get("schema_version") != SCHEMA_VERSION:
        raise ExternalAdapterError("request schema_version is invalid")
    _sha256(request.get("attempt_id"), "attempt_id")
    _sha256(request.get("experiment_sha256"), "experiment_sha256")
    case_id = _safe_string(request.get("case_id"), "case_id")
    if request.get("lane") != lane_id:
        raise ExternalAdapterError("request lane does not match adapter lane")
    repeat = _positive_int(request.get("repeat"), "repeat")
    if repeat > plan["execution"]["repeats"]:
        raise ExternalAdapterError("request repeat exceeds plan")
    cases = {case["case_id"]: case for case in plan["cases"]}
    if case_id not in cases:
        raise ExternalAdapterError("request case is outside the plan")
    case = cases[case_id]
    plan_lane = next(
        lane for lane in plan["lanes"] if lane["lane_id"] == lane_id
    )
    spec_lane = spec["lanes"][lane_id]
    if request.get("instruction") != plan_lane["prompt"]:
        raise ExternalAdapterError("request prompt does not match plan")
    if request["instruction"] != spec_lane["prompt"]:
        raise ExternalAdapterError("request prompt does not match spec")
    if request.get("instruction_sha256") != hashlib.sha256(
        request["instruction"].encode("utf-8")
    ).hexdigest():
        raise ExternalAdapterError("request prompt digest is invalid")
    if request.get("settings") != spec_lane["settings"]:
        raise ExternalAdapterError("request settings do not match spec")
    inputs = _exact_keys(
        request.get("input"),
        frozenset({"image", "raw_text"}),
        "request input",
    )
    image = _exact_keys(
        inputs.get("image"),
        frozenset({"path", "sha256", "dimensions"}),
        "request image",
    )
    image_path = _absolute_file(image.get("path"), "request image path")
    if image_path.stat().st_size > MAX_IMAGE_BYTES:
        raise ExternalAdapterError("request image exceeds the size limit")
    image_bytes = image_path.read_bytes()
    if hashlib.sha256(image_bytes).hexdigest() != case["image_sha256"]:
        raise ExternalAdapterError("request image digest does not match plan")
    if image.get("sha256") != case["image_sha256"]:
        raise ExternalAdapterError("request image digest metadata is invalid")
    if image.get("dimensions") != case["dimensions"]:
        raise ExternalAdapterError(
            "request image dimensions do not match plan"
        )
    if _png_dimensions(image_bytes) != case["dimensions"]:
        raise ExternalAdapterError("decoded PNG dimensions do not match plan")
    raw_record = inputs.get("raw_text")
    raw_text = None
    if plan_lane["include_raw_text"]:
        raw_record = _exact_keys(
            raw_record,
            frozenset({"path", "sha256", "text"}),
            "request raw text",
        )
        raw_path = _absolute_file(raw_record.get("path"), "raw text path")
        if raw_path.stat().st_size > MAX_JSON_BYTES:
            raise ExternalAdapterError("raw text exceeds the size limit")
        raw_bytes = raw_path.read_bytes()
        if hashlib.sha256(raw_bytes).hexdigest() != case["raw_text_sha256"]:
            raise ExternalAdapterError("raw text digest does not match plan")
        try:
            raw_text = raw_bytes.decode("utf-8")
        except UnicodeError as exc:
            raise ExternalAdapterError("raw text is not UTF-8") from exc
        if raw_record.get("text") != raw_text:
            raise ExternalAdapterError("inline raw text does not match file")
        if raw_record.get("sha256") != case["raw_text_sha256"]:
            raise ExternalAdapterError("raw text digest metadata is invalid")
    elif raw_record is not None:
        raise ExternalAdapterError("request disclosed unapproved raw text")
    if request["attempt_id"] != _request_attempt_id(request, plan):
        raise ExternalAdapterError("request attempt_id is not plan-bound")
    return {
        "request": request,
        "case": case,
        "lane": plan_lane,
        "image_bytes": image_bytes,
        "raw_text": raw_text,
    }


def _materialize_body(
    item: dict[str, Any], plan: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = item["request"]
    rendered = ocr_external.render_sanitized_external_request(
        plan,
        case_id=request["case_id"],
        lane_id=request["lane"],
        repeat=request["repeat"],
    )
    sanitized_body = rendered["body"]
    image_marker = (
        f"<IMAGE_BASE64_OMITTED sha256={item['case']['image_sha256']}>"
    )
    image_base64 = base64.b64encode(item["image_bytes"]).decode("ascii")
    raw_marker = None
    if item["raw_text"] is not None:
        raw_marker = (
            "<RAW_TEXT_UTF8_OMITTED sha256="
            f"{item['case']['raw_text_sha256']}>"
        )

    def replace(value: Any) -> Any:
        if isinstance(value, str):
            value = value.replace(image_marker, image_base64)
            if raw_marker is not None:
                value = value.replace(raw_marker, item["raw_text"])
            return value
        if isinstance(value, list):
            return [replace(entry) for entry in value]
        if isinstance(value, dict):
            return {key: replace(entry) for key, entry in value.items()}
        return value

    materialized = replace(sanitized_body)
    serialized = _json_bytes(materialized)
    if (
        b"IMAGE_BASE64_OMITTED" in serialized
        or b"RAW_TEXT_UTF8_OMITTED" in serialized
    ):
        raise ExternalAdapterError(
            "request placeholder materialization failed"
        )
    if plan["provider_profile"]["provider"] == "openai":
        transmitted_prompt = materialized["input"][0]["content"][0]["text"]
    else:
        transmitted_prompt = materialized["messages"][0]["content"][0][
            "text"
        ]
    prompt_bytes = transmitted_prompt.encode("utf-8")
    provenance = {
        "sanitized_body": sanitized_body,
        "sanitized_body_sha256": _sha256_json(sanitized_body),
        "transmitted_body_sha256": _sha256_json(materialized),
        "transmitted_prompt": {
            "sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "characters": len(transmitted_prompt),
            "utf8_bytes": len(prompt_bytes),
            "includes_raw_text": item["raw_text"] is not None,
            "raw_text_sha256": item["case"].get("raw_text_sha256"),
        },
        "content_shape": {
            "provider": plan["provider_profile"]["provider"],
            "image_media_type": "image/png",
            "image_encoding": "base64",
            "image_sha256": item["case"]["image_sha256"],
            "image_dimensions": item["case"]["dimensions"],
        },
    }
    return materialized, provenance


def _whitelisted_headers(headers: list[tuple[str, str]]) -> dict[str, str]:
    allowed = {
        "content-type",
        "request-id",
        "x-request-id",
        "openai-request-id",
        "anthropic-request-id",
    }
    result = {}
    for name, value in headers:
        lowered = name.casefold()
        if lowered in allowed and _SAFE_VALUE_RE.fullmatch(value):
            result[lowered] = value
    return dict(sorted(result.items()))


def send_https_json(
    *,
    profile: dict[str, Any],
    credential: str,
    body: dict[str, Any],
    timeout_seconds: int,
) -> TransportResult:
    """Send direct default-context HTTPS without a proxy or redirect."""

    if any(
        os.environ.get(name) for name in ("SSL_CERT_DIR", "SSL_CERT_FILE")
    ):
        raise ExternalAdapterError(
            "custom TLS trust environment is forbidden"
        )
    endpoint = profile["endpoint"]
    if endpoint["scheme"] != "https" or endpoint["method"] != "POST":
        raise ExternalAdapterError(
            "only fixed HTTPS POST endpoints are allowed"
        )
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "sneeze-plugin-tpn-external-ocr/1",
    }
    if profile["provider"] == "openai":
        headers["Authorization"] = f"Bearer {credential}"
    else:
        headers["x-api-key"] = credential
        headers.update(profile["request_metadata"])
    connection = http.client.HTTPSConnection(
        endpoint["host"],
        port=443,
        timeout=timeout_seconds,
        context=ssl.create_default_context(),
    )
    started = time.monotonic()
    try:
        connection.request(
            "POST", endpoint["path"], body=_json_bytes(body), headers=headers
        )
        response = connection.getresponse()
        payload = response.read(MAX_RESPONSE_BYTES + 1)
        if len(payload) > MAX_RESPONSE_BYTES:
            raise ConfirmedDeliveryError(
                "provider response exceeds size limit"
            )
        return TransportResult(
            status_code=response.status,
            response_headers=_whitelisted_headers(response.getheaders()),
            body=payload,
            wall_seconds=round(time.monotonic() - started, 6),
        )
    except ExternalAdapterError:
        raise
    except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
        raise DeliveryUnknownError(
            "hosted request delivery state is unknown"
        ) from exc
    finally:
        connection.close()


def _parse_json_response(result: TransportResult) -> dict[str, Any]:
    try:
        value = json.loads(result.body)
    except (UnicodeError, ValueError) as exc:
        raise ExternalAdapterError("provider returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ExternalAdapterError("provider response must be an object")
    return value


def _validate_transport_result(value: Any) -> TransportResult:
    if not isinstance(value, TransportResult):
        raise ConfirmedDeliveryError("transport returned an invalid result")
    if (
        not isinstance(value.status_code, int)
        or isinstance(value.status_code, bool)
        or not 100 <= value.status_code <= 599
    ):
        raise ConfirmedDeliveryError("transport HTTP status is invalid")
    if (
        not isinstance(value.body, bytes)
        or len(value.body) > MAX_RESPONSE_BYTES
    ):
        raise ConfirmedDeliveryError("transport response body is invalid")
    if (
        not isinstance(value.wall_seconds, (int, float))
        or isinstance(value.wall_seconds, bool)
        or value.wall_seconds < 0
    ):
        raise ConfirmedDeliveryError("transport wall time is invalid")
    if not isinstance(value.response_headers, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.response_headers.items()
    ):
        raise ConfirmedDeliveryError(
            "transport response headers are invalid"
        )
    return TransportResult(
        status_code=value.status_code,
        response_headers=_whitelisted_headers(
            list(value.response_headers.items())
        ),
        body=value.body,
        wall_seconds=float(value.wall_seconds),
    )


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ExternalAdapterError(f"{label} must be a nonnegative integer")
    return value


def _safe_optional(value: Any) -> str | None:
    if isinstance(value, str) and _SAFE_VALUE_RE.fullmatch(value):
        return value
    return None


def _openai_reasoning_record(item: dict[str, Any]) -> dict[str, Any]:
    required = frozenset({"id", "type", "summary"})
    optional = frozenset({"content", "encrypted_content", "status"})
    keys = frozenset(item)
    if not required <= keys or keys - required - optional:
        raise ExternalAdapterError(
            "OpenAI reasoning item has unexpected fields"
        )
    reasoning_id = _safe_string(item.get("id"), "OpenAI reasoning id")
    status = item.get("status")
    if status not in {None, "in_progress", "completed", "incomplete"}:
        raise ExternalAdapterError("OpenAI reasoning status is invalid")

    summary = item.get("summary")
    if not isinstance(summary, list):
        raise ExternalAdapterError("OpenAI reasoning summary is invalid")
    summary_hashes = []
    for block in summary:
        if (
            not isinstance(block, dict)
            or frozenset(block) != {"type", "text"}
            or block.get("type") != "summary_text"
            or not isinstance(block.get("text"), str)
        ):
            raise ExternalAdapterError(
                "OpenAI reasoning summary block is invalid"
            )
        summary_hashes.append(
            hashlib.sha256(block["text"].encode()).hexdigest()
        )

    content = item.get("content")
    if content is not None and not isinstance(content, list):
        raise ExternalAdapterError("OpenAI reasoning content is invalid")
    content_hashes = []
    for block in content or []:
        if (
            not isinstance(block, dict)
            or frozenset(block) != {"type", "text"}
            or block.get("type") != "reasoning_text"
            or not isinstance(block.get("text"), str)
        ):
            raise ExternalAdapterError(
                "OpenAI reasoning content block is invalid"
            )
        content_hashes.append(
            hashlib.sha256(block["text"].encode()).hexdigest()
        )

    encrypted = item.get("encrypted_content")
    if encrypted is not None and not isinstance(encrypted, str):
        raise ExternalAdapterError(
            "OpenAI encrypted reasoning content is invalid"
        )
    return {
        "id": reasoning_id,
        "status": status,
        "summary_count": len(summary_hashes),
        "summary_sha256": summary_hashes,
        "content_count": len(content_hashes),
        "content_sha256": content_hashes,
        "encrypted_content_present": encrypted is not None,
        "encrypted_content_sha256": (
            hashlib.sha256(encrypted.encode()).hexdigest()
            if encrypted is not None
            else None
        ),
    }


def _openai_response(value: dict[str, Any]) -> dict[str, Any]:
    response_id = _safe_string(value.get("id"), "OpenAI response id")
    model = _safe_string(value.get("model"), "OpenAI response model")
    status = _safe_string(value.get("status"), "OpenAI response status")
    if status not in {"completed", "failed", "incomplete", "cancelled"}:
        raise ExternalAdapterError("OpenAI response is not terminal")
    usage = value.get("usage")
    if not isinstance(usage, dict):
        raise ExternalAdapterError("OpenAI response has no usage")
    input_tokens = _nonnegative_int(
        usage.get("input_tokens"), "OpenAI input_tokens"
    )
    output_tokens = _nonnegative_int(
        usage.get("output_tokens"), "OpenAI output_tokens"
    )
    cached_tokens = 0
    cache_write_tokens = 0
    details = usage.get("input_tokens_details")
    if details is not None:
        if not isinstance(details, dict):
            raise ExternalAdapterError(
                "OpenAI input_tokens_details must be an object"
            )
        if "cached_tokens" in details:
            cached_tokens = _nonnegative_int(
                details["cached_tokens"], "OpenAI cached_tokens"
            )
        if "cache_write_tokens" in details:
            cache_write_tokens = _nonnegative_int(
                details["cache_write_tokens"],
                "OpenAI cache_write_tokens",
            )
    if cached_tokens + cache_write_tokens > input_tokens:
        raise ExternalAdapterError(
            "OpenAI cached_tokens plus cache_write_tokens exceeds "
            "input_tokens"
        )
    texts = []
    refusal_hashes = []
    reasoning_records = []
    output = value.get("output")
    if not isinstance(output, list):
        raise ExternalAdapterError("OpenAI response output is invalid")
    for item in output:
        if not isinstance(item, dict):
            raise ExternalAdapterError("OpenAI output item is invalid")
        if item.get("type") == "reasoning":
            reasoning_records.append(_openai_reasoning_record(item))
            continue
        if item.get("type") != "message":
            raise ExternalAdapterError("OpenAI returned a non-message output")
        message_status = item.get("status")
        if message_status not in {"completed", "incomplete"}:
            raise ExternalAdapterError("OpenAI message status is unknown")
        if status == "completed" and message_status != "completed":
            raise ExternalAdapterError("OpenAI message is incomplete")
        content = item.get("content")
        if not isinstance(content, list):
            raise ExternalAdapterError("OpenAI message content is invalid")
        for block in content:
            if not isinstance(block, dict):
                raise ExternalAdapterError("OpenAI content block is invalid")
            if block.get("type") == "output_text" and isinstance(
                block.get("text"), str
            ):
                texts.append(block["text"])
            elif block.get("type") == "refusal" and isinstance(
                block.get("refusal"), str
            ):
                refusal_hashes.append(
                    hashlib.sha256(block["refusal"].encode()).hexdigest()
                )
            else:
                raise ExternalAdapterError(
                    "OpenAI content type is not allowed"
                )
    incomplete = value.get("incomplete_details")
    incomplete_reason = (
        _safe_optional(incomplete.get("reason"))
        if isinstance(incomplete, dict)
        else None
    )
    if status == "completed" and incomplete_reason is not None:
        raise ExternalAdapterError(
            "completed OpenAI response has incomplete_details"
        )
    if status == "incomplete" and incomplete_reason not in {
        "max_output_tokens",
        "content_filter",
    }:
        raise ExternalAdapterError(
            "OpenAI incomplete reason is undocumented or missing"
        )
    max_tokens = incomplete_reason == "max_output_tokens"
    refused = bool(refusal_hashes)
    finish_reason = "refusal" if refused else "length" if max_tokens \
        else "stop"
    stop_reason = "refusal" if refused else incomplete_reason or status
    return {
        "response_id": response_id,
        "provider_model": model,
        "provider_status": status,
        "text": "".join(texts),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_tokens,
        "cache_write_input_tokens": cache_write_tokens,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "eos": False,
        "max_tokens": max_tokens,
        "refusal": {
            "detected": refused,
            "source": "provider_content_block",
            "count": len(refusal_hashes),
            "sha256": refusal_hashes,
        },
        "whitelist": {
            "id": response_id,
            "model": model,
            "status": status,
            "incomplete_reason": incomplete_reason,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_input_tokens": cached_tokens,
                "cache_write_input_tokens": cache_write_tokens,
            },
            "reasoning": {
                "count": len(reasoning_records),
                "items": reasoning_records,
            },
            "refusal": {
                "detected": refused,
                "count": len(refusal_hashes),
                "sha256": refusal_hashes,
            },
        },
    }


def _anthropic_response(value: dict[str, Any]) -> dict[str, Any]:
    response_id = _safe_string(value.get("id"), "Anthropic message id")
    model = _safe_string(value.get("model"), "Anthropic response model")
    if value.get("type") != "message" or value.get("role") != "assistant":
        raise ExternalAdapterError("Anthropic response identity is invalid")
    stop_reason = _safe_string(
        value.get("stop_reason"), "Anthropic stop_reason"
    )
    terminal_reasons = {
        "end_turn",
        "stop_sequence",
        "max_tokens",
        "model_context_window_exceeded",
        "refusal",
        "tool_use",
        "pause_turn",
    }
    if stop_reason not in terminal_reasons:
        raise ExternalAdapterError(
            "Anthropic stop_reason is undocumented or nonterminal"
        )
    usage = value.get("usage")
    if not isinstance(usage, dict):
        raise ExternalAdapterError("Anthropic response has no usage")
    input_tokens = _nonnegative_int(
        usage.get("input_tokens"), "Anthropic input_tokens"
    )
    output_tokens = _nonnegative_int(
        usage.get("output_tokens"), "Anthropic output_tokens"
    )
    for cache_key in (
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        if cache_key in usage and _nonnegative_int(
            usage[cache_key], f"Anthropic {cache_key}"
        ) != 0:
            raise ExternalAdapterError(
                "unpriced Anthropic cache usage returned"
            )
    texts = []
    ignored_thinking_hashes = []
    content = value.get("content")
    if not isinstance(content, list):
        raise ExternalAdapterError("Anthropic content is invalid")
    for block in content:
        if not isinstance(block, dict):
            raise ExternalAdapterError("Anthropic content block is invalid")
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif block_type in {"thinking", "redacted_thinking"}:
            material = block.get("thinking", block.get("data", ""))
            if not isinstance(material, str):
                raise ExternalAdapterError(
                    "Anthropic thinking block is invalid"
                )
            ignored_thinking_hashes.append(
                hashlib.sha256(material.encode()).hexdigest()
            )
        else:
            raise ExternalAdapterError(
                "Anthropic content type is not allowed"
            )
    refused = stop_reason == "refusal"
    max_tokens = stop_reason in {
        "max_tokens",
        "model_context_window_exceeded",
    }
    finish_reason = "refusal" if refused else "length" if max_tokens \
        else "stop"
    provider_status = (
        "completed"
        if stop_reason in {"end_turn", "stop_sequence"}
        else "refused"
        if refused
        else "incomplete"
    )
    return {
        "response_id": response_id,
        "provider_model": model,
        "provider_status": provider_status,
        "text": "".join(texts),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "eos": False,
        "max_tokens": max_tokens,
        "refusal": {
            "detected": refused,
            "source": "provider_stop_reason",
            "thinking_block_count": len(ignored_thinking_hashes),
            "thinking_sha256": ignored_thinking_hashes,
        },
        "whitelist": {
            "id": response_id,
            "model": model,
            "stop_reason": stop_reason,
            "stop_sequence_sha256": (
                hashlib.sha256(value["stop_sequence"].encode()).hexdigest()
                if isinstance(value.get("stop_sequence"), str)
                else None
            ),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            "refusal": {
                "detected": refused,
                "source": "provider_stop_reason",
            },
            "thinking_block_count": len(ignored_thinking_hashes),
            "thinking_sha256": ignored_thinking_hashes,
        },
    }


def _parse_provider_response(
    value: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    parsed = (
        _openai_response(value)
        if profile["provider"] == "openai"
        else _anthropic_response(value)
    )
    requested = profile["model"]["requested"]
    returned = parsed["provider_model"]
    if returned != requested:
        raise ExternalAdapterError("provider returned an unapproved model")
    return parsed


def _computed_cost(
    parsed: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    pricing = profile["pricing"]
    input_tokens = parsed["input_tokens"]
    output_tokens = parsed["output_tokens"]
    if profile["provider"] == "openai":
        tier = (
            "short_context"
            if input_tokens <= pricing["short_context_at_or_below_tokens"]
            else "long_context"
        )
        rates = pricing[tier]
        cached = parsed["cached_input_tokens"]
        cache_write = parsed["cache_write_input_tokens"]
        uncached = input_tokens - cached - cache_write
        input_cost = (
            Decimal(uncached) * Decimal(rates["input"])
            + Decimal(cached) * Decimal(rates["cached_input"])
            + Decimal(cache_write) * Decimal(rates["cache_write"])
        ) / Decimal(1_000_000)
        output_rate = Decimal(rates["output"])
        rate_record = {
            "tier": tier,
            "input": rates["input"],
            "cached_input": rates["cached_input"],
            "cache_write": rates["cache_write"],
            "output": rates["output"],
        }
    else:
        input_cost = (
            Decimal(input_tokens) * Decimal(pricing["input"])
            / Decimal(1_000_000)
        )
        output_rate = Decimal(pricing["output"])
        rate_record = {
            "tier": "fixed",
            "input": pricing["input"],
            "output": pricing["output"],
        }
    output_cost = (
        Decimal(output_tokens) * output_rate / Decimal(1_000_000)
    )
    total = input_cost + output_cost
    return {
        "kind": "computed_from_dated_profile",
        "verified_at": pricing["verified_at"],
        "source": pricing["source"],
        "rates_per_million_tokens": rate_record,
        "input_tokens": input_tokens,
        "cached_input_tokens": parsed["cached_input_tokens"],
        "cache_write_input_tokens": parsed["cache_write_input_tokens"],
        "output_tokens": output_tokens,
        "computed_usd": format(total, ".6f"),
    }


def _load_ledger(handle) -> dict[str, Any] | None:
    handle.seek(0)
    payload = handle.read()
    if not payload:
        return None
    try:
        value = json.loads(payload)
    except ValueError as exc:
        raise ExternalAdapterError("budget ledger is corrupt") from exc
    if not isinstance(value, dict):
        raise ExternalAdapterError("budget ledger is invalid")
    return value


def _write_ledger(handle, ledger: dict[str, Any]) -> None:
    handle.seek(0)
    handle.truncate()
    handle.write(_json_bytes(ledger).decode("utf-8"))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def _ledger_identity(plan: dict, approval: dict) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "approval_sha256": approval["approval_sha256"],
        "budget_usd": plan["execution"]["budget_usd"],
        "pessimistic_max_usd": plan["execution"]["cost_estimate"][
            "pessimistic_max_usd"
        ],
        "reserved_usd": "0.000000",
        "delivery_unknown": False,
        "reservations": {},
    }


def _validate_ledger_state(
    ledger: dict[str, Any], plan: dict, approval: dict
) -> None:
    _exact_keys(
        ledger,
        frozenset(
            {
                "schema_version",
                "plan_sha256",
                "approval_sha256",
                "budget_usd",
                "pessimistic_max_usd",
                "reserved_usd",
                "delivery_unknown",
                "reservations",
            }
        ),
        "budget ledger",
    )
    expected = _ledger_identity(plan, approval)
    for key in (
        "schema_version",
        "plan_sha256",
        "approval_sha256",
        "budget_usd",
        "pessimistic_max_usd",
    ):
        if ledger.get(key) != expected[key]:
            raise ExternalAdapterError("budget ledger identity mismatch")
    if not isinstance(ledger.get("delivery_unknown"), bool):
        raise ExternalAdapterError("budget ledger state is invalid")
    reservations = ledger.get("reservations")
    if not isinstance(reservations, dict):
        raise ExternalAdapterError("budget ledger reservations are invalid")
    per_attempt = plan["execution"]["cost_estimate"][
        "pessimistic_per_attempt_usd"
    ]
    reserved_total = Decimal("0")
    saw_unknown = False
    for reservation_id, record in reservations.items():
        _sha256(reservation_id, "ledger reservation_id")
        record = _exact_keys(
            record,
            frozenset({"slot", "state", "reserved_usd"}),
            "ledger reservation",
        )
        slot = _exact_keys(
            record.get("slot"),
            frozenset(
                {
                    "plan_sha256",
                    "case_id",
                    "lane_id",
                    "repeat",
                    "api_attempt",
                }
            ),
            "ledger reservation slot",
        )
        if slot.get("plan_sha256") != plan["plan_sha256"]:
            raise ExternalAdapterError("ledger slot plan mismatch")
        if slot.get("case_id") not in {
            case["case_id"] for case in plan["cases"]
        }:
            raise ExternalAdapterError("ledger slot case is outside plan")
        if slot.get("lane_id") not in {
            lane["lane_id"] for lane in plan["lanes"]
        }:
            raise ExternalAdapterError("ledger slot lane is outside plan")
        if not isinstance(slot.get("repeat"), int) or not (
            1 <= slot["repeat"] <= plan["execution"]["repeats"]
        ):
            raise ExternalAdapterError("ledger slot repeat is outside plan")
        if not isinstance(slot.get("api_attempt"), int) or not (
            1
            <= slot["api_attempt"]
            <= plan["execution"]["max_api_attempts_per_case"]
        ):
            raise ExternalAdapterError(
                "ledger slot API attempt is outside plan"
            )
        if reservation_id != _sha256_json(slot):
            raise ExternalAdapterError("ledger reservation digest mismatch")
        if record.get("reserved_usd") != per_attempt:
            raise ExternalAdapterError("ledger reservation cost mismatch")
        if record.get("state") not in {
            "reserved",
            "delivered",
            "delivery_unknown",
        }:
            raise ExternalAdapterError("ledger reservation state is invalid")
        saw_unknown = saw_unknown or record["state"] == "delivery_unknown"
        reserved_total += Decimal(per_attempt)
    if ledger.get("reserved_usd") != format(reserved_total, ".6f"):
        raise ExternalAdapterError("ledger reserved total is invalid")
    if ledger["delivery_unknown"] is not saw_unknown:
        raise ExternalAdapterError("ledger delivery_unknown state is invalid")
    limit = min(
        Decimal(ledger["budget_usd"]),
        Decimal(ledger["pessimistic_max_usd"]),
    )
    if reserved_total > limit:
        raise ExternalAdapterError("ledger exceeds approved budget")


def _initialize_or_validate_ledger(
    path: Path, plan: dict, approval: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        ledger = _load_ledger(handle)
        expected = _ledger_identity(plan, approval)
        if ledger is None:
            _write_ledger(handle, expected)
            return
        _validate_ledger_state(ledger, plan, approval)
        if ledger.get("delivery_unknown") is True:
            raise ExternalAdapterError(
                "budget ledger is blocked by delivery_unknown"
            )


def _reserve_attempt(
    path: Path,
    *,
    plan: dict,
    approval: dict,
    case_id: str,
    lane_id: str,
    repeat: int,
    api_attempt: int,
) -> tuple[str, str]:
    if case_id not in {case["case_id"] for case in plan["cases"]}:
        raise ExternalAdapterError("reservation case is outside plan")
    if lane_id not in {lane["lane_id"] for lane in plan["lanes"]}:
        raise ExternalAdapterError("reservation lane is outside plan")
    if not 1 <= repeat <= plan["execution"]["repeats"]:
        raise ExternalAdapterError("reservation repeat is outside plan")
    if not 1 <= api_attempt <= plan["execution"][
        "max_api_attempts_per_case"
    ]:
        raise ExternalAdapterError("reservation API attempt is outside plan")
    slot = {
        "plan_sha256": plan["plan_sha256"],
        "case_id": case_id,
        "lane_id": lane_id,
        "repeat": repeat,
        "api_attempt": api_attempt,
    }
    reservation_id = _sha256_json(slot)
    per_attempt = plan["execution"]["cost_estimate"][
        "pessimistic_per_attempt_usd"
    ]
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        ledger = _load_ledger(handle)
        if ledger is None:
            raise ExternalAdapterError("budget ledger disappeared")
        _validate_ledger_state(ledger, plan, approval)
        if ledger.get("delivery_unknown") is True:
            raise ExternalAdapterError(
                "budget ledger is blocked by delivery_unknown"
            )
        reservations = ledger["reservations"]
        if reservation_id in reservations:
            raise ExternalAdapterError("API attempt was already reserved")
        next_total = Decimal(ledger["reserved_usd"]) + Decimal(per_attempt)
        limit = min(
            Decimal(ledger["budget_usd"]),
            Decimal(ledger["pessimistic_max_usd"]),
        )
        if next_total > limit:
            raise ExternalAdapterError("pessimistic API budget is exhausted")
        reservations[reservation_id] = {
            "slot": slot,
            "state": "reserved",
            "reserved_usd": per_attempt,
        }
        ledger["reserved_usd"] = format(next_total, ".6f")
        _write_ledger(handle, ledger)
        return reservation_id, per_attempt


def _mark_reservation(
    path: Path,
    reservation_id: str,
    state: str,
    *,
    plan: dict,
    approval: dict,
) -> None:
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        ledger = _load_ledger(handle)
        if ledger is None or reservation_id not in ledger["reservations"]:
            raise ExternalAdapterError("budget reservation is missing")
        _validate_ledger_state(ledger, plan, approval)
        ledger["reservations"][reservation_id]["state"] = state
        if state == "delivery_unknown":
            ledger["delivery_unknown"] = True
        _write_ledger(handle, ledger)


def _write_json_new(path: Path, value: Any) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _artifact(
    *,
    path: Path,
    request: dict,
    plan: dict,
    approval: dict,
    spec_hash: str,
    bundle_hash: str,
    attempts: list[dict[str, Any]],
) -> dict[str, str]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": request["attempt_id"],
        "plan_sha256": plan["plan_sha256"],
        "approval_sha256": approval["approval_sha256"],
        "provider_spec_sha256": spec_hash,
        "executor_bundle_sha256": bundle_hash,
        "raw_response_persisted": False,
        "attempts": attempts,
    }
    _write_json_new(path, value)
    return {"relative_path": path.name, "sha256": _sha256_path(path)}


def _runtime_metadata(bundle: dict[str, Any]) -> dict[str, Any]:
    try:
        package_version = importlib.metadata.version("sneeze-plugin-tpn")
    except importlib.metadata.PackageNotFoundError:
        package_version = "editable-unknown"
    return {
        "python": platform.python_version(),
        "engine": "stdlib-http.client",
        "engine_version": platform.python_version(),
        "packages": {"sneeze-plugin-tpn": package_version},
        "attention_backend": "remote_provider_managed",
        "tls": {
            "context": "ssl.create_default_context",
            "openssl": ssl.OPENSSL_VERSION,
            "trust_configuration": "system_default_no_environment_override",
            "proxies": "disabled_direct_connection",
            "redirects": "disabled",
        },
        "executor_bundle": bundle,
    }


def _error_response(
    request: dict, error_type: str, artifact: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": request["attempt_id"],
        "status": "error",
        "error": {"type": error_type, "message": error_type},
        "output": {"format": "markdown", "text": ""},
        "artifacts": [artifact],
    }


def _completed_response(
    *,
    item: dict,
    parsed: dict,
    profile: dict,
    spec: dict,
    plan: dict,
    runtime: dict,
    request_provenance: dict[str, Any],
    cost: dict,
    reserved_usd: str,
    attempts: int,
    wall_seconds: float,
    artifact: dict[str, str],
) -> dict[str, Any]:
    request = item["request"]
    refused = parsed["refusal"]["detected"]
    truncated = parsed["max_tokens"]
    provider_incomplete = parsed["provider_status"] != "completed"
    failed = refused or truncated or provider_incomplete
    if refused:
        error_type = "ProviderRefusal"
    elif truncated:
        error_type = "ProviderMaxTokens"
    elif provider_incomplete:
        error_type = "ProviderIncomplete"
    else:
        error_type = None
    return {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": request["attempt_id"],
        "status": "error" if failed else "completed",
        "error": (
            {"type": error_type, "message": error_type}
            if error_type is not None
            else None
        ),
        "output": {"format": "markdown", "text": parsed["text"]},
        "effective_request": {
            "prompt": request["instruction"],
            "transmitted_prompt": request_provenance[
                "transmitted_prompt"
            ],
            "content_shape": request_provenance["content_shape"],
            "sanitized_body": request_provenance["sanitized_body"],
            "sanitized_body_sha256": request_provenance[
                "sanitized_body_sha256"
            ],
            "transmitted_body_sha256": request_provenance[
                "transmitted_body_sha256"
            ],
            "input_artifacts": [
                {**request["input"]["image"], "transform": "original"}
            ],
        },
        "model": {
            "requested": profile["model"]["requested"],
            "revision": spec["model"]["revision"],
            "resolved_snapshot": parsed["provider_model"],
            "provider_returned_model": parsed["provider_model"],
        },
        "provider": {
            "provider_id": spec["provider_id"],
            "response_id": parsed["response_id"],
            "status": parsed["provider_status"],
        },
        "runtime": runtime,
        "generation": {
            "params": {
                "profile_id": spec["provider_id"],
                "prompt_sha256": request["instruction_sha256"],
                "image_sha256": request["input"]["image"]["sha256"],
                "max_input_tokens": plan["execution"][
                    "max_input_tokens_per_case"
                ],
                "max_output_tokens": plan["execution"][
                    "max_output_tokens"
                ],
                "input_cap_semantics": plan["execution"][
                    "cost_estimate"
                ]["input_cap_semantics"],
                "request": {
                    "endpoint": profile["endpoint"],
                    "request_metadata": profile.get(
                        "request_metadata", {}
                    ),
                    "body": request_provenance["sanitized_body"],
                    "transmitted_body_sha256": request_provenance[
                        "transmitted_body_sha256"
                    ],
                },
            },
            "input_tokens": parsed["input_tokens"],
            "output_tokens": parsed["output_tokens"],
            "finish_reason": parsed["finish_reason"],
            "stop_reason": parsed["stop_reason"],
            "eos": parsed["eos"],
            "max_tokens": parsed["max_tokens"],
            "refusal": parsed["refusal"],
        },
        "usage": {
            "wall_seconds": round(wall_seconds, 6),
            "cost_usd": float(Decimal(cost["computed_usd"])),
            "cost": cost,
            "pessimistic_reserved_usd": reserved_usd,
            "api_attempts": attempts,
        },
        "artifacts": [artifact],
    }


def _credential_from_environment(name: str) -> str | None:
    return os.environ.get(name)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _fresh_plan_approval_gate(
    *,
    plan_path: Path,
    approval_path: Path,
    expected_plan_sha256: str,
    expected_approval_sha256: str,
    now: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = ocr_external.validate_external_upload_plan(
        _read_json(plan_path, "external plan")
    )
    if plan["plan_sha256"] != expected_plan_sha256:
        raise ExternalAdapterError("external plan changed during execution")
    approval = ocr_external.validate_external_upload_approval(
        plan,
        _read_json(approval_path, "external approval"),
        now=now,
    )
    if approval["approval_sha256"] != expected_approval_sha256:
        raise ExternalAdapterError(
            "external approval changed during execution"
        )
    return plan, approval


def _redact_credential(value: Any, credential: str) -> tuple[Any, int]:
    if isinstance(value, str):
        count = value.count(credential)
        return value.replace(credential, "[REDACTED_CREDENTIAL]"), count
    if isinstance(value, list):
        redacted = []
        count = 0
        for entry in value:
            item, item_count = _redact_credential(entry, credential)
            redacted.append(item)
            count += item_count
        return redacted, count
    if isinstance(value, dict):
        redacted = {}
        count = 0
        for key, entry in value.items():
            item, item_count = _redact_credential(entry, credential)
            redacted[key] = item
            count += item_count
        return redacted, count
    return value, 0


def run_external_adapter(
    *,
    requests_path: str | os.PathLike[str],
    responses_path: str | os.PathLike[str],
    artifacts_dir: str | os.PathLike[str],
    provider_spec_path: str | os.PathLike[str],
    plan_path: str | os.PathLike[str],
    approval_path: str | os.PathLike[str],
    budget_ledger_path: str | os.PathLike[str],
    profile_id: str,
    lane_id: str,
    transport: Callable[..., TransportResult] = send_https_json,
    credential_reader: Callable[[str], str | None] = (
        _credential_from_environment
    ),
    now: str | None = None,
    now_reader: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    """Validate all bindings, then execute an approved request slice."""

    request_file = _absolute_file(requests_path, "requests JSONL")
    spec_file = _absolute_file(provider_spec_path, "provider spec")
    plan_file = _absolute_file(plan_path, "external plan")
    approval_file = _absolute_file(approval_path, "external approval")
    response_file = _absolute_output(responses_path, "responses JSONL")
    artifact_root = Path(artifacts_dir).expanduser()
    if not artifact_root.is_absolute() or not artifact_root.is_dir():
        raise ExternalAdapterError(
            "artifacts_dir must be an absolute directory"
        )
    if any(artifact_root.iterdir()):
        raise ExternalAdapterError("artifacts_dir must be empty")
    ledger_path = Path(budget_ledger_path).expanduser()
    if not ledger_path.is_absolute():
        raise ExternalAdapterError("budget ledger path must be absolute")
    if any(
        os.environ.get(name) for name in ("SSL_CERT_DIR", "SSL_CERT_FILE")
    ):
        raise ExternalAdapterError(
            "custom TLS trust environment is forbidden"
        )

    spec = validate_external_provider_spec_record(
        _read_json(spec_file, "provider spec")
    )
    spec_hash = external_provider_spec_sha256(spec)
    plan = ocr_external.validate_external_upload_plan(
        _read_json(plan_file, "external plan")
    )
    approval_now = now if now is not None else now_reader()
    approval = ocr_external.validate_external_upload_approval(
        plan,
        _read_json(approval_file, "external approval"),
        now=approval_now,
    )
    profile = plan["provider_profile"]
    bundle = executor_bundle_identity()
    if (
        spec["provider_id"] != profile_id
        or profile["profile_id"] != profile_id
    ):
        raise ExternalAdapterError("profile identity mismatch")
    if spec["external_policy"]["lane_id"] != lane_id:
        raise ExternalAdapterError("lane identity mismatch")
    plan_lanes = {lane["lane_id"]: lane for lane in plan["lanes"]}
    if set(plan_lanes) != {lane_id}:
        raise ExternalAdapterError("adapter plan must contain its exact lane")
    plan_lane = plan_lanes[lane_id]
    spec_lane = spec["lanes"][lane_id]
    if (
        plan_lane["prompt"] != spec_lane["prompt"]
        or plan_lane["prompt_mode"]
        != spec_lane["settings"]["prompt_mode"]
        or plan_lane["include_raw_text"]
        is not spec_lane["include_raw_text"]
    ):
        raise ExternalAdapterError("plan lane does not match provider spec")
    if plan["executor"]["provider_spec_sha256"] != spec_hash:
        raise ExternalAdapterError("plan provider spec digest mismatch")
    if plan["executor"]["adapter_bundle_sha256"] != bundle["bundle_sha256"]:
        raise ExternalAdapterError("plan executor bundle digest mismatch")
    requests = _read_jsonl(request_file)
    items = [
        _validate_request(
            request,
            plan=plan,
            spec=spec,
            lane_id=lane_id,
        )
        for request in requests
    ]
    repeats = {item["request"]["repeat"] for item in items}
    if len(repeats) != 1:
        raise ExternalAdapterError("one adapter run must contain one repeat")
    if {item["request"]["case_id"] for item in items} != {
        case["case_id"] for case in plan["cases"]
    }:
        raise ExternalAdapterError(
            "request slice must contain every plan case"
        )
    if len({item["request"]["case_id"] for item in items}) != len(items):
        raise ExternalAdapterError("request slice contains duplicate cases")
    _initialize_or_validate_ledger(ledger_path, plan, approval)

    credential_name = profile["credential_env_name"]
    credential = credential_reader(credential_name)
    if (
        not isinstance(credential, str)
        or not credential
        or "\x00" in credential
        or "\n" in credential
        or "\r" in credential
    ):
        raise ExternalAdapterError(
            "required provider credential is unavailable"
        )

    runtime = _runtime_metadata(bundle)
    response_rows = []
    delivery_unknown = False
    with response_file.open("x", encoding="utf-8", newline="") as stream:
        for item in items:
            request = item["request"]
            attempt_records = []
            parsed = None
            total_wall = 0.0
            total_reserved = Decimal("0")
            terminal_error = None
            request_provenance = None
            for api_attempt in range(
                1, plan["execution"]["max_api_attempts_per_case"] + 1
            ):
                fresh_now = now if now is not None else now_reader()
                fresh_plan, fresh_approval = _fresh_plan_approval_gate(
                    plan_path=plan_file,
                    approval_path=approval_file,
                    expected_plan_sha256=plan["plan_sha256"],
                    expected_approval_sha256=approval["approval_sha256"],
                    now=fresh_now,
                )
                reservation_id, reserved = _reserve_attempt(
                    ledger_path,
                    plan=fresh_plan,
                    approval=fresh_approval,
                    case_id=request["case_id"],
                    lane_id=request["lane"],
                    repeat=request["repeat"],
                    api_attempt=api_attempt,
                )
                total_reserved += Decimal(reserved)
                body, request_provenance = _materialize_body(
                    item, fresh_plan
                )
                upload_now = now if now is not None else now_reader()
                _fresh_plan_approval_gate(
                    plan_path=plan_file,
                    approval_path=approval_file,
                    expected_plan_sha256=fresh_plan["plan_sha256"],
                    expected_approval_sha256=fresh_approval[
                        "approval_sha256"
                    ],
                    now=upload_now,
                )
                try:
                    result = transport(
                        profile=profile,
                        credential=credential,
                        body=body,
                        timeout_seconds=spec["timeout_seconds"],
                    )
                    result = _validate_transport_result(result)
                except DeliveryUnknownError:
                    _mark_reservation(
                        ledger_path,
                        reservation_id,
                        "delivery_unknown",
                        plan=fresh_plan,
                        approval=fresh_approval,
                    )
                    attempt_records.append(
                        {
                            "api_attempt": api_attempt,
                            "reservation_id": reservation_id,
                            "reserved_usd": reserved,
                            "delivery": "unknown",
                        }
                    )
                    terminal_error = "DeliveryUnknown"
                    delivery_unknown = True
                    break
                except ConfirmedDeliveryError:
                    _mark_reservation(
                        ledger_path,
                        reservation_id,
                        "delivered",
                        plan=fresh_plan,
                        approval=fresh_approval,
                    )
                    attempt_records.append(
                        {
                            "api_attempt": api_attempt,
                            "reservation_id": reservation_id,
                            "reserved_usd": reserved,
                            "delivery": "confirmed",
                            "provider_response": "rejected",
                        }
                    )
                    terminal_error = "ProviderTransportResultRejected"
                    break
                _mark_reservation(
                    ledger_path,
                    reservation_id,
                    "delivered",
                    plan=fresh_plan,
                    approval=fresh_approval,
                )
                total_wall += result.wall_seconds
                raw_hash = hashlib.sha256(result.body).hexdigest()
                attempt_record = {
                    "api_attempt": api_attempt,
                    "reservation_id": reservation_id,
                    "reserved_usd": reserved,
                    "delivery": "confirmed",
                    "http_status": result.status_code,
                    "headers": result.response_headers,
                    "raw_body_sha256": raw_hash,
                }
                try:
                    response_value = _parse_json_response(result)
                except ExternalAdapterError:
                    attempt_record["response_parse"] = "rejected"
                    attempt_records.append(attempt_record)
                    retryable = (
                        result.status_code in _RETRYABLE_HTTP_STATUS
                        or 500 <= result.status_code < 600
                    )
                    if not retryable:
                        terminal_error = "ProviderResponseInvalid"
                        break
                    continue
                if 200 <= result.status_code < 300:
                    try:
                        parsed = _parse_provider_response(
                            response_value, profile
                        )
                    except ExternalAdapterError:
                        attempt_record["provider_response"] = "rejected"
                        attempt_records.append(attempt_record)
                        terminal_error = "ProviderResponseRejected"
                        break
                    if (
                        parsed["input_tokens"]
                        > plan["execution"]["max_input_tokens_per_case"]
                        or parsed["output_tokens"]
                        > plan["execution"]["max_output_tokens"]
                    ):
                        attempt_record["provider_response"] = (
                            "observed_usage_exceeded_reservation"
                        )
                        attempt_records.append(attempt_record)
                        parsed = None
                        terminal_error = "ProviderUsageExceededReservation"
                        break
                    parsed, credential_redactions = _redact_credential(
                        parsed, credential
                    )
                    parsed["credential_redactions"] = credential_redactions
                    attempt_record["provider_response"] = parsed["whitelist"]
                    attempt_records.append(attempt_record)
                    break
                error = response_value.get("error")
                attempt_record["provider_error"] = {
                    "type": _safe_optional(
                        error.get("type") if isinstance(error, dict) else None
                    ),
                    "code": _safe_optional(
                        error.get("code") if isinstance(error, dict) else None
                    ),
                }
                attempt_records.append(attempt_record)
                retryable = result.status_code in _RETRYABLE_HTTP_STATUS or (
                    500 <= result.status_code < 600
                )
                if not retryable:
                    terminal_error = f"ProviderHTTP{result.status_code}"
                    break
            attempt_records, _ = _redact_credential(
                attempt_records, credential
            )
            artifact_path = artifact_root / (
                f"{request['attempt_id']}-external.json"
            )
            artifact = _artifact(
                path=artifact_path,
                request=request,
                plan=plan,
                approval=approval,
                spec_hash=spec_hash,
                bundle_hash=bundle["bundle_sha256"],
                attempts=attempt_records,
            )
            if parsed is None:
                terminal_error = terminal_error or "ProviderAttemptsExhausted"
                response = _error_response(request, terminal_error, artifact)
            else:
                cost = _computed_cost(parsed, profile)
                response = _completed_response(
                    item=item,
                    parsed=parsed,
                    profile=profile,
                    spec=spec,
                    plan=plan,
                    runtime=runtime,
                    request_provenance=request_provenance,
                    cost=cost,
                    reserved_usd=format(total_reserved, ".6f"),
                    attempts=len(attempt_records),
                    wall_seconds=total_wall,
                    artifact=artifact,
                )
            stream.write(_json_bytes(response).decode("utf-8") + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            response_rows.append(response)
            if delivery_unknown:
                break
    return {
        "response_count": len(response_rows),
        "delivery_unknown": delivery_unknown,
        "all_completed": len(response_rows) == len(items)
        and all(row["status"] == "completed" for row in response_rows),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--external-provider-spec", required=True)
    parser.add_argument("--external-plan", required=True)
    parser.add_argument("--external-approval", required=True)
    parser.add_argument("--external-budget-ledger", required=True)
    parser.add_argument("--requests-jsonl", required=True)
    parser.add_argument("--responses-jsonl", required=True)
    parser.add_argument("--artifacts-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _isolated_executor_preflight(
            provider_spec_path=args.external_provider_spec,
            plan_path=args.external_plan,
            approval_path=args.external_approval,
        )
        result = run_external_adapter(
            requests_path=args.requests_jsonl,
            responses_path=args.responses_jsonl,
            artifacts_dir=args.artifacts_dir,
            provider_spec_path=args.external_provider_spec,
            plan_path=args.external_plan,
            approval_path=args.external_approval,
            budget_ledger_path=args.external_budget_ledger,
            profile_id=args.profile_id,
            lane_id=args.lane_id,
        )
    except ValueError as exc:
        print(f"external adapter refused: {exc}", file=sys.stderr)
        return 2
    return 2 if result["delivery_unknown"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
