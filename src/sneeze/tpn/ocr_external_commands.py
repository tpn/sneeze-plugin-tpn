"""Credential-free command helpers for hosted OCR experiment approval."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .ocr_benchmark import (
    BenchmarkError,
    _png_dimensions,
    _validate_destination,
    file_sha256,
    load_manifest,
)
from .ocr_external import (
    ExternalApprovalError,
    ExternalPlanError,
    build_external_upload_approval_record,
    build_external_upload_plan,
    get_external_provider_profile,
    render_sanitized_external_request,
    validate_external_upload_plan,
)
from .ocr_external_setup import (
    ExternalSetupError,
    build_external_provider_spec,
    external_executor_bundle_identity,
    external_provider_spec_sha256,
)

DEFAULT_EXTERNAL_PROMPT = "Convert the document to markdown."


class ExternalCommandError(ValueError):
    """Raised when an offline hosted-provider command is unsafe."""


def _json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    options: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    suffix = "\n" if pretty else ""
    return (json.dumps(value, **options) + suffix).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    if not path.is_absolute():
        raise ExternalCommandError("external output paths must be absolute")
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
            raise ExternalCommandError(
                f"refusing to replace existing file: {path}"
            ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _verified_cases(
    manifest: list[dict[str, Any]],
    input_root: Path,
    *,
    include_raw_text: bool,
) -> list[dict[str, Any]]:
    cases = []
    for case in manifest:
        stem = case["stem"]
        image_path = input_root / f"{stem}.png"
        image_hash = file_sha256(str(image_path))
        if image_hash != case["input"]["image_sha256"]:
            raise ExternalCommandError(
                f"{case['case_id']} image SHA-256 does not match manifest"
            )
        dimensions = _png_dimensions(image_path)
        if dimensions != case["input"]["dimensions"]:
            raise ExternalCommandError(
                f"{case['case_id']} image dimensions do not match manifest"
            )
        record = {
            "case_id": case["case_id"],
            "image_sha256": image_hash,
            "dimensions": dimensions,
        }
        if include_raw_text:
            raw_path = input_root / f"{stem}.txt"
            raw_hash = file_sha256(str(raw_path))
            if raw_hash != case["input"]["raw_text_sha256"]:
                raise ExternalCommandError(
                    f"{case['case_id']} raw-text SHA-256 does not match "
                    "manifest"
                )
            record["raw_text_sha256"] = raw_hash
        cases.append(record)
    return cases


def create_external_plan_bundle(
    *,
    profile_id: str,
    manifest_path: str | os.PathLike[str],
    input_root: str | os.PathLike[str],
    dest_dir: str | os.PathLike[str],
    python_executable: str | os.PathLike[str],
    lane_id: str = "parity",
    prompt: str = DEFAULT_EXTERNAL_PROMPT,
    prompt_mode: str = "parity",
    include_raw_text: bool = False,
    repeats: int = 1,
    max_api_attempts_per_case: int = 1,
    max_input_tokens_per_case: int = 262_144,
    max_output_tokens: int = 8_192,
    budget_usd: str,
    valid_hours: int = 24,
    timeout_seconds: int = 120,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Write a byte-free, credential-free plan/spec/preview bundle."""

    try:
        source = Path(manifest_path).expanduser().resolve(strict=True)
        source_bytes = source.read_bytes()
        root = Path(input_root).expanduser().resolve(strict=True)
        destination = Path(dest_dir).expanduser()
        if not destination.is_absolute():
            raise ExternalCommandError("plan destination must be absolute")
        _validate_destination(root, destination)
        if destination.exists() and any(destination.iterdir()):
            raise ExternalCommandError(
                "plan destination must be new or empty"
            )
        manifest = load_manifest(source, _source_bytes=source_bytes)
        cases = _verified_cases(
            manifest,
            root,
            include_raw_text=include_raw_text,
        )
        spec = build_external_provider_spec(
            profile_id=profile_id,
            python_executable=python_executable,
            lane_id=lane_id,
            prompt=prompt,
            prompt_mode=prompt_mode,
            include_raw_text=include_raw_text,
            timeout_seconds=timeout_seconds,
        )
        bundle = external_executor_bundle_identity(spec["argv"][0])
        created = (now or datetime.now(UTC)).astimezone(UTC).replace(
            microsecond=0
        )
        if not isinstance(valid_hours, int) or valid_hours <= 0:
            raise ExternalCommandError("valid_hours must be positive")
        plan = build_external_upload_plan(
            profile_id=profile_id,
            manifest_sha256=_sha256_bytes(source_bytes),
            provider_spec_sha256=external_provider_spec_sha256(spec),
            adapter_bundle_sha256=bundle["bundle_sha256"],
            cases=cases,
            lanes=[
                {
                    "lane_id": lane_id,
                    "prompt": prompt,
                    "prompt_mode": prompt_mode,
                    "include_raw_text": include_raw_text,
                }
            ],
            budget_usd=budget_usd,
            created_at=_timestamp(created),
            expires_at=_timestamp(created + timedelta(hours=valid_hours)),
            repeats=repeats,
            max_api_attempts_per_case=max_api_attempts_per_case,
            max_input_tokens_per_case=max_input_tokens_per_case,
            max_output_tokens=max_output_tokens,
        )
        previews = [
            render_sanitized_external_request(
                plan,
                case_id=case["case_id"],
                lane_id=lane_id,
                repeat=repeat,
            )
            for repeat in range(1, repeats + 1)
            for case in cases
        ]
    except (
        BenchmarkError,
        ExternalApprovalError,
        ExternalPlanError,
        ExternalSetupError,
        OSError,
        ValueError,
    ) as exc:
        if isinstance(exc, ExternalCommandError):
            raise
        raise ExternalCommandError(str(exc)) from exc

    spec_payload = _json_bytes(spec)
    plan_payload = _json_bytes(plan)
    previews_payload = b"".join(
        _json_bytes(preview, pretty=False) + b"\n" for preview in previews
    )
    destination.mkdir(parents=True, exist_ok=True)
    spec_path = destination / "provider-spec.json"
    plan_path = destination / "upload-plan.json"
    previews_path = destination / "request-previews.jsonl"
    _write_new(spec_path, spec_payload)
    _write_new(plan_path, plan_payload)
    _write_new(previews_path, previews_payload)
    profile = get_external_provider_profile(profile_id)
    return {
        "schema_version": 1,
        "kind": "external_plan_bundle",
        "profile_id": profile_id,
        "model": profile["model"]["requested"],
        "case_count": len(cases),
        "lane_id": lane_id,
        "repeats": repeats,
        "raw_text_included": include_raw_text,
        "provider_spec_path": str(spec_path),
        "provider_spec_file_sha256": _sha256_bytes(spec_payload),
        "provider_spec_sha256": plan["executor"][
            "provider_spec_sha256"
        ],
        "upload_plan_path": str(plan_path),
        "upload_plan_file_sha256": _sha256_bytes(plan_payload),
        "plan_sha256": plan["plan_sha256"],
        "request_previews_path": str(previews_path),
        "request_previews_sha256": _sha256_bytes(previews_payload),
        "pessimistic_max_usd": plan["execution"]["cost_estimate"][
            "pessimistic_max_usd"
        ],
        "budget_usd": plan["execution"]["budget_usd"],
        "expires_at": plan["expires_at"],
        "retention_policy_id": profile["retention"]["policy_id"],
        "required_retention_acknowledgement": profile["retention"][
            "required_acknowledgement"
        ],
        "credential_env_name": profile["credential_env_name"],
        "source_bytes_written": False,
        "credentials_read": False,
    }


def create_external_approval(
    *,
    plan_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    approved_by: str,
    accept_retention: bool,
    valid_hours: int = 24,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Write a separate hash-bound approval; never read a credential."""

    if accept_retention is not True:
        raise ExternalCommandError(
            "retention policy was not accepted; pass --accept-retention"
        )
    current = (now or datetime.now(UTC)).astimezone(UTC).replace(
        microsecond=0
    )
    if not isinstance(valid_hours, int) or valid_hours <= 0:
        raise ExternalCommandError("valid_hours must be positive")
    try:
        source = Path(plan_path).expanduser().resolve(strict=True)
        plan = validate_external_upload_plan(
            json.loads(source.read_text(encoding="utf-8"))
        )
        plan_expiry = datetime.strptime(
            plan["expires_at"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=UTC)
        approval_expiry = min(
            current + timedelta(hours=valid_hours),
            plan_expiry,
        )
        profile = plan["provider_profile"]
        approval = build_external_upload_approval_record(
            plan,
            approved_by=approved_by,
            approved_at=_timestamp(current),
            expires_at=_timestamp(approval_expiry),
            retention_acknowledgement=profile["retention"][
                "required_acknowledgement"
            ],
        )
    except (
        ExternalApprovalError,
        ExternalPlanError,
        OSError,
        ValueError,
    ) as exc:
        raise ExternalCommandError(str(exc)) from exc
    destination = Path(output_path).expanduser()
    payload = _json_bytes(approval)
    _write_new(destination, payload)
    return {
        "schema_version": 1,
        "kind": "external_upload_approval",
        "output_path": str(destination),
        "approval_file_sha256": _sha256_bytes(payload),
        "approval_sha256": approval["approval_sha256"],
        "plan_sha256": approval["plan_sha256"],
        "provider_profile_id": approval["provider_profile_id"],
        "model": approval["model"],
        "approved_by": approval["approved_by"],
        "expires_at": approval["expires_at"],
        "retention_policy_id": approval["retention"]["policy_id"],
        "credentials_read": False,
    }
