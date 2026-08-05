"""Offline planning for explicitly approved external OCR experiments.

This module deliberately has no transport implementation.  It builds and
validates content-addressed upload plans and approval records, then renders a
sanitized representation of the request that an eventual, separately
approved executor would send.  Image and raw-text bytes are represented only
by digest-bound placeholders.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import Any

__all__ = [
    "EXTERNAL_SCHEMA_VERSION",
    "ExternalApprovalError",
    "ExternalPlanError",
    "build_external_execution_binding",
    "build_external_upload_approval_record",
    "build_external_upload_plan",
    "get_external_provider_profile",
    "list_external_provider_profiles",
    "render_sanitized_external_request",
    "validate_external_upload_approval",
    "validate_external_upload_plan",
]

EXTERNAL_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MONEY_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]{1,6})?$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_APPROVER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9@._+-]{0,127}$")
_SECRET_VALUE_RES = (
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bBearer\s+\S+", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),
    re.compile(
        r"\b(?:api[_-]?key|authorization|password|secret|token)\s*[:=]"
        r"\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(r"https?://[^/@\s:]+:[^/@\s]+@", re.IGNORECASE),
)


class ExternalPlanError(ValueError):
    """Raised when an external upload plan is malformed or stale."""


class ExternalApprovalError(ValueError):
    """Raised when an external upload approval is malformed or stale."""


_OPENAI_RETENTION_ACKNOWLEDGEMENT = (
    "I accept the documented OpenAI API default abuse-monitoring retention "
    "policy for this upload plan; store=false is not Zero Data Retention."
)
_ANTHROPIC_RETENTION_ACKNOWLEDGEMENT = (
    "I accept mandatory 30-day retention for Claude Fable 5 API inputs and "
    "outputs; Zero Data Retention is unavailable."
)

_PROVIDER_PROFILES: dict[str, dict[str, Any]] = {
    "openai-responses-gpt-5.6-sol": {
        "schema_version": EXTERNAL_SCHEMA_VERSION,
        "kind": "external_provider_policy_profile",
        "profile_id": "openai-responses-gpt-5.6-sol",
        "profile_date": "2026-08-04",
        "provider": "openai",
        "endpoint": {
            "scheme": "https",
            "host": "api.openai.com",
            "path": "/v1/responses",
            "method": "POST",
            "api": "responses",
        },
        "credential_env_name": "OPENAI_API_KEY",
        "model": {
            "requested": "gpt-5.6-sol",
            "revision_semantics": "provider_managed_model_id",
            "fallback": "disabled",
        },
        "limits": {
            "context_tokens": 1_050_000,
            "max_input_tokens": 922_000,
            "max_output_tokens": 128_000,
            "default_experiment_output_tokens": 8_192,
        },
        "request_policy": {
            "image_transport": "inline_base64",
            "image_detail": "original",
            "background": False,
            "store": False,
            "service_tier": "default",
            "truncation": "disabled",
            "reasoning_effort": "none",
            "prompt_cache_options": {"mode": "explicit"},
            "prompt_cache_breakpoint_count": 0,
            "tools": "disabled",
        },
        "retention": {
            "policy_id": "openai-api-default-2026-08-04",
            "verified_at": "2026-08-04",
            "default_abuse_monitoring_max_days": 30,
            "zero_data_retention_assumed": False,
            "store_false_is_zero_data_retention": False,
            "image_safety_review_exception_documented": True,
            "required_acknowledgement": (
                _OPENAI_RETENTION_ACKNOWLEDGEMENT
            ),
        },
        "pricing": {
            "currency": "USD",
            "unit": "million_tokens",
            "verified_at": "2026-08-04",
            "short_context_at_or_below_tokens": 272_000,
            "short_context": {
                "input": "5.00",
                "cached_input": "0.50",
                "cache_write": "6.25",
                "output": "30.00",
            },
            "long_context": {
                "input": "10.00",
                "cached_input": "1.00",
                "cache_write": "12.50",
                "output": "45.00",
            },
            "source": "https://developers.openai.com/api/docs/pricing",
        },
        "sources": [
            (
                "https://developers.openai.com/api/docs/guides/"
                "model-guidance?model=gpt-5.6"
            ),
            (
                "https://developers.openai.com/api/docs/guides/"
                "images-vision#choose-an-image-detail-level"
            ),
            (
                "https://developers.openai.com/api/docs/guides/"
                "prompt-caching"
            ),
            (
                "https://developers.openai.com/api/docs/guides/"
                "your-data#data-retention-controls-for-abuse-monitoring"
            ),
            "https://developers.openai.com/api/docs/pricing",
        ],
    },
    "anthropic-messages-claude-fable-5": {
        "schema_version": EXTERNAL_SCHEMA_VERSION,
        "kind": "external_provider_policy_profile",
        "profile_id": "anthropic-messages-claude-fable-5",
        "profile_date": "2026-08-04",
        "provider": "anthropic",
        "endpoint": {
            "scheme": "https",
            "host": "api.anthropic.com",
            "path": "/v1/messages",
            "method": "POST",
            "api": "messages",
        },
        "request_metadata": {"anthropic-version": "2023-06-01"},
        "credential_env_name": "ANTHROPIC_API_KEY",
        "model": {
            "requested": "claude-fable-5",
            "revision_semantics": (
                "provider_documented_fixed_snapshot_model_id"
            ),
            "fallback": "disabled",
        },
        "limits": {
            "context_tokens": 1_000_000,
            "max_output_tokens": 128_000,
            "default_experiment_output_tokens": 8_192,
        },
        "request_policy": {
            "image_transport": "inline_base64",
            "image_order": "image_then_text",
            "provider_image_processing": {
                "max_edge_pixels": 2_576,
                "max_visual_tokens": 4_784,
                "oversize_behavior": "provider_downscale",
            },
            "thinking_type": "adaptive",
            "thinking_display": "omitted",
            "effort": "low",
            "tools": "disabled",
        },
        "retention": {
            "policy_id": "anthropic-fable-5-mandatory-30d-2026-08-04",
            "verified_at": "2026-08-04",
            "mandatory_days": 30,
            "zero_data_retention_available": False,
            "required_acknowledgement": (
                _ANTHROPIC_RETENTION_ACKNOWLEDGEMENT
            ),
        },
        "pricing": {
            "currency": "USD",
            "unit": "million_tokens",
            "verified_at": "2026-08-04",
            "input": "10.00",
            "output": "50.00",
            "source": (
                "https://platform.claude.com/docs/en/about-claude/"
                "models/overview"
            ),
        },
        "sources": [
            (
                "https://platform.claude.com/docs/en/about-claude/models/"
                "introducing-claude-fable-5-and-claude-mythos-5"
            ),
            (
                "https://platform.claude.com/docs/en/about-claude/"
                "models/overview"
            ),
            (
                "https://platform.claude.com/docs/en/manage-claude/"
                "api-and-data-retention"
            ),
            (
                "https://platform.claude.com/docs/en/build-with-claude/"
                "effort"
            ),
            (
                "https://platform.claude.com/docs/en/build-with-claude/"
                "vision"
            ),
            "https://platform.claude.com/docs/en/api/messages",
        ],
    },
}

_PLAN_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "provider_profile",
        "provider_profile_sha256",
        "executor",
        "manifest_sha256",
        "cases",
        "lanes",
        "execution",
        "transmission",
        "created_at",
        "expires_at",
        "plan_sha256",
    }
)
_APPROVAL_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "plan_sha256",
        "provider_profile_id",
        "model",
        "approved_by",
        "approved_at",
        "expires_at",
        "scope",
        "retention",
        "approval_sha256",
    }
)


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
        raise ExternalPlanError(
            f"value is not canonical JSON data: {exc}"
        ) from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _without_hash(value: dict[str, Any], key: str) -> dict[str, Any]:
    unhashed = copy.deepcopy(value)
    unhashed.pop(key, None)
    return unhashed


def _require_exact_keys(
    value: Any,
    keys: frozenset[str],
    label: str,
    *,
    error_type: type[ValueError] = ExternalPlanError,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise error_type(f"{label} must be an object")
    actual = frozenset(value)
    if actual != keys:
        missing = sorted(keys - actual)
        raise error_type(
            f"{label} has unexpected fields; missing={missing}, "
            f"extra_count={len(actual - keys)}"
        )
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExternalPlanError(f"{label} must be a nonempty string")
    if "\x00" in value:
        raise ExternalPlanError(f"{label} cannot contain a NUL byte")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ExternalPlanError(f"{label} must be valid UTF-8 text") from exc
    return value


def _reject_secret_like_value(
    value: str,
    label: str,
    *,
    error_type: type[ValueError] = ExternalPlanError,
) -> None:
    if any(pattern.search(value) for pattern in _SECRET_VALUE_RES):
        raise error_type(
            f"{label} appears to contain a credential or secret value"
        )


def _require_safe_id(value: Any, label: str) -> str:
    identifier = _require_text(value, label)
    _reject_secret_like_value(identifier, label)
    if not _SAFE_ID_RE.fullmatch(identifier):
        raise ExternalPlanError(f"{label} must be a safe identifier")
    return identifier


def _require_approver(
    value: Any,
    *,
    error_type: type[ValueError] = ExternalPlanError,
) -> str:
    try:
        approver = _require_text(value, "approved_by")
    except ExternalPlanError as exc:
        raise error_type(str(exc)) from exc
    _reject_secret_like_value(
        approver, "approved_by", error_type=error_type
    )
    if not _APPROVER_RE.fullmatch(approver):
        raise error_type(
            "approved_by must be a safe principal identifier"
        )
    return approver


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ExternalPlanError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ExternalPlanError(f"{label} must be a positive integer")
    return value


def _require_money(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _MONEY_RE.fullmatch(value):
        raise ExternalPlanError(
            f"{label} must be a canonical nonnegative USD decimal string"
        )
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise ExternalPlanError(f"{label} is not a valid USD amount") \
            from exc
    if amount <= 0:
        raise ExternalPlanError(f"{label} must be greater than zero")
    return value


def _pessimistic_cost_estimate(
    *,
    profile: dict[str, Any],
    case_count: int,
    lane_count: int,
    repeats: int,
    max_api_attempts_per_case: int,
    max_input_tokens_per_case: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    pricing = profile["pricing"]
    # Providers do not expose an enforceable input-token limit for these
    # image requests.  The user-selected input limit is therefore a scope
    # check on returned usage, not a spending control.  Reserve against the
    # model's entire documented context ceiling so a delivered request cannot
    # cost more than its ledger reservation merely because image tokenization
    # exceeded the estimate.
    reserved_input_tokens = profile["limits"]["context_tokens"]
    if profile["provider"] == "openai":
        threshold = pricing["short_context_at_or_below_tokens"]
        tier = (
            "short_context"
            if reserved_input_tokens <= threshold
            else "long_context"
        )
        tier_rates = pricing[tier]
        input_rate_basis = max(
            ("input", "cached_input", "cache_write"),
            key=lambda key: Decimal(tier_rates[key]),
        )
        input_rate = Decimal(tier_rates[input_rate_basis])
        output_rate = Decimal(tier_rates["output"])
    else:
        tier = "fixed"
        input_rate_basis = "input"
        input_rate = Decimal(pricing["input"])
        output_rate = Decimal(pricing["output"])
    attempt_count = (
        case_count
        * lane_count
        * repeats
        * max_api_attempts_per_case
    )
    million = Decimal(1_000_000)
    raw_per_attempt = (
        Decimal(reserved_input_tokens) * input_rate / million
        + Decimal(max_output_tokens) * output_rate / million
    )
    rounded_per_attempt = raw_per_attempt.quantize(
        Decimal("0.000001"), rounding=ROUND_CEILING
    )
    rounded_cost = Decimal(attempt_count) * rounded_per_attempt
    return {
        "method": "provider_context_ceiling_reservation_v2",
        "currency": "USD",
        "pricing_verified_at": pricing["verified_at"],
        "rate_tier": tier,
        "input_rate_basis": input_rate_basis,
        "input_rate_per_million_tokens": format(input_rate, "f"),
        "output_rate_per_million_tokens": format(output_rate, "f"),
        "case_count": case_count,
        "lane_count": lane_count,
        "repeats": repeats,
        "max_api_attempts_per_case": max_api_attempts_per_case,
        "attempt_count": attempt_count,
        "max_input_tokens_per_case": max_input_tokens_per_case,
        "billable_input_reservation_tokens": reserved_input_tokens,
        "input_cap_semantics": (
            "post_response_scope_check_only; spending reservation uses "
            "the full documented provider context ceiling"
        ),
        "max_output_tokens_per_attempt": max_output_tokens,
        "pessimistic_per_attempt_usd": format(
            rounded_per_attempt, ".6f"
        ),
        "pessimistic_max_usd": format(rounded_cost, ".6f"),
        "rounding": "ceiling_to_0.000001_usd",
    }


def _parse_timestamp(
    value: Any,
    label: str,
    *,
    error_type: type[ValueError] = ExternalPlanError,
) -> datetime:
    if not isinstance(value, str) or not _TIMESTAMP_RE.fullmatch(value):
        raise error_type(f"{label} must be an RFC 3339 UTC timestamp")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=UTC
        )
    except ValueError as exc:
        raise error_type(f"{label} is not a valid UTC timestamp") from exc


def get_external_provider_profile(profile_id: str) -> dict[str, Any]:
    """Return an isolated copy of a dated hosted-provider policy profile."""

    if not isinstance(profile_id, str):
        raise ExternalPlanError(
            "external provider profile_id must be a string"
        )
    try:
        profile = copy.deepcopy(_PROVIDER_PROFILES[profile_id])
    except KeyError as exc:
        choices = ", ".join(sorted(_PROVIDER_PROFILES))
        raise ExternalPlanError(
            f"unknown external provider profile; choose one of: {choices}"
        ) from exc
    profile["profile_sha256"] = _sha256_json(profile)
    return profile


def list_external_provider_profiles() -> list[str]:
    """Return the stable profile identifiers supported by this module."""

    return sorted(_PROVIDER_PROFILES)


def _normalize_cases(cases: Any) -> list[dict[str, Any]]:
    if not isinstance(cases, list) or not cases:
        raise ExternalPlanError("cases must be a nonempty list")
    normalized = []
    seen = set()
    allowed = frozenset(
        {"case_id", "image_sha256", "dimensions", "raw_text_sha256"}
    )
    required = frozenset({"case_id", "image_sha256", "dimensions"})
    for index, source in enumerate(cases, 1):
        label = f"cases[{index}]"
        if not isinstance(source, dict):
            raise ExternalPlanError(f"{label} must be an object")
        actual = frozenset(source)
        if not required <= actual or not actual <= allowed:
            raise ExternalPlanError(f"{label} has unexpected fields")
        case_id = _require_safe_id(source.get("case_id"), f"{label}.case_id")
        if case_id in seen:
            raise ExternalPlanError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        dimensions = _require_exact_keys(
            source.get("dimensions"),
            frozenset({"w", "h"}),
            f"{label}.dimensions",
        )
        case = {
            "case_id": case_id,
            "image_sha256": _require_sha256(
                source.get("image_sha256"),
                f"{label}.image_sha256",
            ),
            "dimensions": {
                "w": _require_positive_int(
                    dimensions.get("w"), f"{label}.dimensions.w"
                ),
                "h": _require_positive_int(
                    dimensions.get("h"), f"{label}.dimensions.h"
                ),
            },
            "media_type": "image/png",
        }
        if "raw_text_sha256" in source:
            case["raw_text_sha256"] = _require_sha256(
                source.get("raw_text_sha256"),
                f"{label}.raw_text_sha256",
            )
        normalized.append(case)
    return normalized


def _validate_normalized_cases(cases: Any) -> list[dict[str, Any]]:
    if not isinstance(cases, list) or not cases:
        raise ExternalPlanError("cases must be a nonempty list")
    sources = []
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            raise ExternalPlanError(f"cases[{index}] must be an object")
        expected = {
            "case_id",
            "image_sha256",
            "dimensions",
            "media_type",
        }
        if "raw_text_sha256" in case:
            expected.add("raw_text_sha256")
        _require_exact_keys(case, frozenset(expected), f"cases[{index}]")
        if case.get("media_type") != "image/png":
            raise ExternalPlanError(
                f"cases[{index}].media_type must be image/png"
            )
        source = {key: copy.deepcopy(value) for key, value in case.items()}
        source.pop("media_type")
        sources.append(source)
    normalized = _normalize_cases(sources)
    if normalized != cases:
        raise ExternalPlanError("cases are not in canonical form")
    return normalized


def _normalize_lanes(lanes: Any) -> list[dict[str, Any]]:
    if not isinstance(lanes, list) or not lanes:
        raise ExternalPlanError("lanes must be a nonempty list")
    normalized = []
    seen = set()
    keys = frozenset(
        {"lane_id", "prompt", "prompt_mode", "include_raw_text"}
    )
    for index, source in enumerate(lanes, 1):
        label = f"lanes[{index}]"
        lane = _require_exact_keys(source, keys, label)
        lane_id = _require_safe_id(lane.get("lane_id"), f"{label}.lane_id")
        if lane_id in seen:
            raise ExternalPlanError(f"duplicate lane_id: {lane_id}")
        seen.add(lane_id)
        prompt = _require_text(lane.get("prompt"), f"{label}.prompt")
        _reject_secret_like_value(prompt, f"{label}.prompt")
        prompt_mode = _require_safe_id(
            lane.get("prompt_mode"), f"{label}.prompt_mode"
        )
        include_raw = lane.get("include_raw_text")
        if not isinstance(include_raw, bool):
            raise ExternalPlanError(
                f"{label}.include_raw_text must be boolean"
            )
        normalized.append(
            {
                "lane_id": lane_id,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "prompt_mode": prompt_mode,
                "include_raw_text": include_raw,
            }
        )
    return normalized


def _validate_normalized_lanes(lanes: Any) -> list[dict[str, Any]]:
    if not isinstance(lanes, list) or not lanes:
        raise ExternalPlanError("lanes must be a nonempty list")
    sources = []
    keys = frozenset(
        {
            "lane_id",
            "prompt",
            "prompt_sha256",
            "prompt_mode",
            "include_raw_text",
        }
    )
    for index, lane in enumerate(lanes, 1):
        _require_exact_keys(lane, keys, f"lanes[{index}]")
        _require_sha256(
            lane.get("prompt_sha256"), f"lanes[{index}].prompt_sha256"
        )
        source = copy.deepcopy(lane)
        source.pop("prompt_sha256")
        sources.append(source)
    normalized = _normalize_lanes(sources)
    if normalized != lanes:
        raise ExternalPlanError("lane prompts or digests are not canonical")
    return normalized


def build_external_upload_plan(
    *,
    profile_id: str,
    manifest_sha256: str,
    provider_spec_sha256: str,
    adapter_bundle_sha256: str,
    cases: list[dict[str, Any]],
    lanes: list[dict[str, Any]],
    budget_usd: str,
    created_at: str,
    expires_at: str,
    repeats: int = 1,
    max_api_attempts_per_case: int = 1,
    max_input_tokens_per_case: int,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic, byte-free external upload plan.

    The result is not permission to transmit anything.  It must be paired with
    a separately supplied approval record accepted by
    :func:`validate_external_upload_approval`.
    """

    profile_with_hash = get_external_provider_profile(profile_id)
    profile_hash = profile_with_hash.pop("profile_sha256")
    normalized_cases = _normalize_cases(cases)
    normalized_lanes = _normalize_lanes(lanes)
    created = _parse_timestamp(created_at, "created_at")
    expires = _parse_timestamp(expires_at, "expires_at")
    if expires <= created:
        raise ExternalPlanError("expires_at must be after created_at")
    repeats = _require_positive_int(repeats, "repeats")
    attempts = _require_positive_int(
        max_api_attempts_per_case, "max_api_attempts_per_case"
    )
    if attempts > 3:
        raise ExternalPlanError(
            "max_api_attempts_per_case cannot exceed 3"
        )
    if max_output_tokens is None:
        max_output_tokens = profile_with_hash["limits"][
            "default_experiment_output_tokens"
        ]
    max_output_tokens = _require_positive_int(
        max_output_tokens, "max_output_tokens"
    )
    if max_output_tokens > profile_with_hash["limits"][
        "max_output_tokens"
    ]:
        raise ExternalPlanError(
            "max_output_tokens exceeds the provider profile limit"
        )
    max_input_tokens_per_case = _require_positive_int(
        max_input_tokens_per_case, "max_input_tokens_per_case"
    )
    max_input_tokens = profile_with_hash["limits"].get(
        "max_input_tokens"
    )
    if (
        max_input_tokens is not None
        and max_input_tokens_per_case > max_input_tokens
    ):
        raise ExternalPlanError(
            "max_input_tokens_per_case exceeds the provider profile limit"
        )
    if (
        max_input_tokens_per_case + max_output_tokens
        > profile_with_hash["limits"]["context_tokens"]
    ):
        raise ExternalPlanError(
            "input and output token caps exceed the model context limit"
        )
    budget = _require_money(budget_usd, "budget_usd")
    if any(lane["include_raw_text"] for lane in normalized_lanes):
        if any("raw_text_sha256" not in case for case in normalized_cases):
            raise ExternalPlanError(
                "every case needs raw_text_sha256 for a raw-text lane"
            )
    raw_lane_ids = [
        lane["lane_id"]
        for lane in normalized_lanes
        if lane["include_raw_text"]
    ]
    cost_estimate = _pessimistic_cost_estimate(
        profile=profile_with_hash,
        case_count=len(normalized_cases),
        lane_count=len(normalized_lanes),
        repeats=repeats,
        max_api_attempts_per_case=attempts,
        max_input_tokens_per_case=max_input_tokens_per_case,
        max_output_tokens=max_output_tokens,
    )
    if Decimal(budget) < Decimal(cost_estimate["pessimistic_max_usd"]):
        raise ExternalPlanError(
            "budget_usd is below the pessimistic cost estimate"
        )
    plan = {
        "schema_version": EXTERNAL_SCHEMA_VERSION,
        "kind": "external_upload_plan",
        "provider_profile": profile_with_hash,
        "provider_profile_sha256": profile_hash,
        "executor": {
            "protocol": "sneeze-external-ocr-v1",
            "provider_spec_sha256": _require_sha256(
                provider_spec_sha256, "provider_spec_sha256"
            ),
            "adapter_bundle_sha256": _require_sha256(
                adapter_bundle_sha256, "adapter_bundle_sha256"
            ),
        },
        "manifest_sha256": _require_sha256(
            manifest_sha256, "manifest_sha256"
        ),
        "cases": normalized_cases,
        "lanes": normalized_lanes,
        "execution": {
            "repeats": repeats,
            "max_api_attempts_per_case": attempts,
            "max_input_tokens_per_case": max_input_tokens_per_case,
            "max_output_tokens": max_output_tokens,
            "budget_usd": budget,
            "cost_estimate": cost_estimate,
            "fallback": "disabled",
        },
        "transmission": {
            "requires_external_upload_approval": True,
            "images": {
                "media_type": "image/png",
                "encoding": "base64",
                "transport": "inline",
            },
            "raw_text": {
                "encoding": "utf-8",
                "transport": "inline",
                "included_lane_ids": raw_lane_ids,
            },
            "plan_contains_source_bytes": False,
        },
        "created_at": created_at,
        "expires_at": expires_at,
    }
    plan["plan_sha256"] = _sha256_json(plan)
    return validate_external_upload_plan(plan)


def validate_external_upload_plan(
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Validate and return an isolated copy of an external upload plan."""

    plan = copy.deepcopy(plan)
    _require_exact_keys(plan, _PLAN_KEYS, "external upload plan")
    if plan.get("schema_version") != EXTERNAL_SCHEMA_VERSION:
        raise ExternalPlanError("unsupported external plan schema_version")
    if plan.get("kind") != "external_upload_plan":
        raise ExternalPlanError("external plan kind is invalid")
    profile = plan.get("provider_profile")
    if not isinstance(profile, dict):
        raise ExternalPlanError("provider_profile must be an object")
    profile_id = _require_text(profile.get("profile_id"), "profile_id")
    expected_profile = get_external_provider_profile(profile_id)
    expected_profile_hash = expected_profile.pop("profile_sha256")
    if profile != expected_profile:
        raise ExternalPlanError(
            "provider_profile does not match the current dated policy"
        )
    if plan.get("provider_profile_sha256") != expected_profile_hash:
        raise ExternalPlanError("provider_profile_sha256 does not match")
    executor = _require_exact_keys(
        plan.get("executor"),
        frozenset(
            {
                "protocol",
                "provider_spec_sha256",
                "adapter_bundle_sha256",
            }
        ),
        "executor",
    )
    if executor.get("protocol") != "sneeze-external-ocr-v1":
        raise ExternalPlanError("executor protocol is unsupported")
    _require_sha256(
        executor.get("provider_spec_sha256"),
        "executor.provider_spec_sha256",
    )
    _require_sha256(
        executor.get("adapter_bundle_sha256"),
        "executor.adapter_bundle_sha256",
    )
    _require_sha256(plan.get("manifest_sha256"), "manifest_sha256")
    cases = _validate_normalized_cases(plan.get("cases"))
    lanes = _validate_normalized_lanes(plan.get("lanes"))

    execution = _require_exact_keys(
        plan.get("execution"),
        frozenset(
            {
                "repeats",
                "max_api_attempts_per_case",
                "max_input_tokens_per_case",
                "max_output_tokens",
                "budget_usd",
                "cost_estimate",
                "fallback",
            }
        ),
        "execution",
    )
    _require_positive_int(execution.get("repeats"), "execution.repeats")
    attempts = _require_positive_int(
        execution.get("max_api_attempts_per_case"),
        "execution.max_api_attempts_per_case",
    )
    if attempts > 3:
        raise ExternalPlanError(
            "execution.max_api_attempts_per_case cannot exceed 3"
        )
    output_limit = _require_positive_int(
        execution.get("max_output_tokens"),
        "execution.max_output_tokens",
    )
    if output_limit > expected_profile["limits"]["max_output_tokens"]:
        raise ExternalPlanError(
            "execution.max_output_tokens exceeds provider profile limit"
        )
    input_limit = _require_positive_int(
        execution.get("max_input_tokens_per_case"),
        "execution.max_input_tokens_per_case",
    )
    max_input_tokens = expected_profile["limits"].get("max_input_tokens")
    if max_input_tokens is not None and input_limit > max_input_tokens:
        raise ExternalPlanError(
            "execution.max_input_tokens_per_case exceeds provider profile "
            "limit"
        )
    if input_limit + output_limit > expected_profile["limits"][
        "context_tokens"
    ]:
        raise ExternalPlanError(
            "execution token caps exceed the model context limit"
        )
    budget = _require_money(
        execution.get("budget_usd"), "execution.budget_usd"
    )
    expected_cost = _pessimistic_cost_estimate(
        profile=expected_profile,
        case_count=len(cases),
        lane_count=len(lanes),
        repeats=execution["repeats"],
        max_api_attempts_per_case=attempts,
        max_input_tokens_per_case=input_limit,
        max_output_tokens=output_limit,
    )
    if execution.get("cost_estimate") != expected_cost:
        raise ExternalPlanError(
            "execution.cost_estimate does not match the dated profile rates"
        )
    if Decimal(budget) < Decimal(expected_cost["pessimistic_max_usd"]):
        raise ExternalPlanError(
            "execution budget is below the pessimistic cost estimate"
        )
    if execution.get("fallback") != "disabled":
        raise ExternalPlanError("external model fallback must be disabled")

    raw_lane_ids = [
        lane["lane_id"] for lane in lanes if lane["include_raw_text"]
    ]
    if raw_lane_ids and any("raw_text_sha256" not in case for case in cases):
        raise ExternalPlanError(
            "every case needs raw_text_sha256 for a raw-text lane"
        )
    expected_transmission = {
        "requires_external_upload_approval": True,
        "images": {
            "media_type": "image/png",
            "encoding": "base64",
            "transport": "inline",
        },
        "raw_text": {
            "encoding": "utf-8",
            "transport": "inline",
            "included_lane_ids": raw_lane_ids,
        },
        "plan_contains_source_bytes": False,
    }
    if plan.get("transmission") != expected_transmission:
        raise ExternalPlanError("transmission policy does not match lanes")
    created = _parse_timestamp(plan.get("created_at"), "created_at")
    expires = _parse_timestamp(plan.get("expires_at"), "expires_at")
    if expires <= created:
        raise ExternalPlanError("expires_at must be after created_at")
    _require_sha256(plan.get("plan_sha256"), "plan_sha256")
    expected_plan_hash = _sha256_json(_without_hash(plan, "plan_sha256"))
    if plan["plan_sha256"] != expected_plan_hash:
        raise ExternalPlanError("plan_sha256 does not match plan contents")
    return plan


def build_external_upload_approval_record(
    plan: dict[str, Any],
    *,
    approved_by: str,
    approved_at: str,
    expires_at: str,
    retention_acknowledgement: str,
) -> dict[str, Any]:
    """Build a digest-bound approval record without asserting authority.

    The returned digest detects mutation but is not a cryptographic identity
    signature.  The caller remains responsible for establishing that
    ``approved_by`` is authorized to approve the upload.
    """

    plan = validate_external_upload_plan(plan)
    profile = plan["provider_profile"]
    required_acknowledgement = profile["retention"][
        "required_acknowledgement"
    ]
    if retention_acknowledgement != required_acknowledgement:
        raise ExternalApprovalError(
            "retention_acknowledgement must exactly match the provider policy"
        )
    approval = {
        "schema_version": EXTERNAL_SCHEMA_VERSION,
        "kind": "external_upload_approval",
        "plan_sha256": plan["plan_sha256"],
        "provider_profile_id": profile["profile_id"],
        "model": profile["model"]["requested"],
        "approved_by": _require_approver(
            approved_by, error_type=ExternalApprovalError
        ),
        "approved_at": approved_at,
        "expires_at": expires_at,
        "scope": {
            "case_ids": [case["case_id"] for case in plan["cases"]],
            "lane_ids": [lane["lane_id"] for lane in plan["lanes"]],
            "repeats": plan["execution"]["repeats"],
            "max_api_attempts_per_case": plan["execution"][
                "max_api_attempts_per_case"
            ],
            "max_output_tokens": plan["execution"]["max_output_tokens"],
            "max_input_tokens_per_case": plan["execution"][
                "max_input_tokens_per_case"
            ],
            "budget_usd": plan["execution"]["budget_usd"],
            "cost_estimate": copy.deepcopy(
                plan["execution"]["cost_estimate"]
            ),
            "fallback": "disabled",
            "executor": copy.deepcopy(plan["executor"]),
        },
        "retention": {
            "policy_id": profile["retention"]["policy_id"],
            "acknowledgement": retention_acknowledgement,
        },
    }
    approval["approval_sha256"] = _sha256_json(approval)
    return validate_external_upload_approval(
        plan, approval, now=approved_at
    )


def validate_external_upload_approval(
    plan: dict[str, Any],
    approval: dict[str, Any],
    *,
    now: str | None = None,
) -> dict[str, Any]:
    """Validate a separate approval against the exact upload plan."""

    plan = validate_external_upload_plan(plan)
    approval = copy.deepcopy(approval)
    _require_exact_keys(
        approval,
        _APPROVAL_KEYS,
        "external upload approval",
        error_type=ExternalApprovalError,
    )
    if approval.get("schema_version") != EXTERNAL_SCHEMA_VERSION:
        raise ExternalApprovalError(
            "unsupported external approval schema_version"
        )
    if approval.get("kind") != "external_upload_approval":
        raise ExternalApprovalError("external approval kind is invalid")
    if approval.get("plan_sha256") != plan["plan_sha256"]:
        raise ExternalApprovalError("approval is for a different plan")
    profile = plan["provider_profile"]
    if approval.get("provider_profile_id") != profile["profile_id"]:
        raise ExternalApprovalError(
            "approval provider profile does not match the plan"
        )
    if approval.get("model") != profile["model"]["requested"]:
        raise ExternalApprovalError("approval model does not match the plan")
    _require_approver(
        approval.get("approved_by"), error_type=ExternalApprovalError
    )
    approved = _parse_timestamp(
        approval.get("approved_at"),
        "approved_at",
        error_type=ExternalApprovalError,
    )
    approval_expires = _parse_timestamp(
        approval.get("expires_at"),
        "approval expires_at",
        error_type=ExternalApprovalError,
    )
    plan_created = _parse_timestamp(plan["created_at"], "created_at")
    plan_expires = _parse_timestamp(plan["expires_at"], "expires_at")
    if approved < plan_created:
        raise ExternalApprovalError(
            "approval cannot predate the upload plan"
        )
    if approval_expires <= approved:
        raise ExternalApprovalError(
            "approval expires_at must be after approved_at"
        )
    if approval_expires > plan_expires:
        raise ExternalApprovalError(
            "approval cannot outlive the upload plan"
        )
    if now is None:
        current = datetime.now(UTC).replace(microsecond=0)
    else:
        current = _parse_timestamp(
            now, "now", error_type=ExternalApprovalError
        )
    if current < approved:
        raise ExternalApprovalError("approval is not yet valid")
    if current >= approval_expires:
        raise ExternalApprovalError("approval has expired")

    expected_scope = {
        "case_ids": [case["case_id"] for case in plan["cases"]],
        "lane_ids": [lane["lane_id"] for lane in plan["lanes"]],
        "repeats": plan["execution"]["repeats"],
        "max_api_attempts_per_case": plan["execution"][
            "max_api_attempts_per_case"
        ],
        "max_output_tokens": plan["execution"]["max_output_tokens"],
        "max_input_tokens_per_case": plan["execution"][
            "max_input_tokens_per_case"
        ],
        "budget_usd": plan["execution"]["budget_usd"],
        "cost_estimate": copy.deepcopy(
            plan["execution"]["cost_estimate"]
        ),
        "fallback": "disabled",
        "executor": copy.deepcopy(plan["executor"]),
    }
    if approval.get("scope") != expected_scope:
        raise ExternalApprovalError(
            "approval scope does not exactly match the plan"
        )
    expected_retention = {
        "policy_id": profile["retention"]["policy_id"],
        "acknowledgement": profile["retention"][
            "required_acknowledgement"
        ],
    }
    if approval.get("retention") != expected_retention:
        raise ExternalApprovalError(
            "approval does not accept the exact retention policy"
        )
    approval_hash = approval.get("approval_sha256")
    if not isinstance(approval_hash, str) or not _SHA256_RE.fullmatch(
        approval_hash
    ):
        raise ExternalApprovalError(
            "approval_sha256 must be a lowercase SHA-256 digest"
        )
    expected_hash = _sha256_json(
        _without_hash(approval, "approval_sha256")
    )
    if approval_hash != expected_hash:
        raise ExternalApprovalError(
            "approval_sha256 does not match approval contents"
        )
    return approval


def _find_case(plan: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in plan["cases"]:
        if case["case_id"] == case_id:
            return case
    raise ExternalPlanError(f"case_id is not in the plan: {case_id}")


def _find_lane(plan: dict[str, Any], lane_id: str) -> dict[str, Any]:
    for lane in plan["lanes"]:
        if lane["lane_id"] == lane_id:
            return lane
    raise ExternalPlanError(f"lane_id is not in the plan: {lane_id}")


def _render_instruction(
    lane: dict[str, Any], case: dict[str, Any]
) -> str:
    if not lane["include_raw_text"]:
        return lane["prompt"]
    raw_digest = case["raw_text_sha256"]
    return (
        f"{lane['prompt']}\n\n"
        f"<RAW_TEXT_UTF8_OMITTED sha256={raw_digest}>"
    )


def _render_openai_body(
    profile: dict[str, Any],
    case: dict[str, Any],
    instruction: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    image_placeholder = (
        "data:image/png;base64,"
        f"<IMAGE_BASE64_OMITTED sha256={case['image_sha256']}>"
    )
    return {
        "model": profile["model"]["requested"],
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {
                        "type": "input_image",
                        "image_url": image_placeholder,
                        "detail": "original",
                    },
                ],
            }
        ],
        "max_output_tokens": max_output_tokens,
        "background": False,
        "reasoning": {"effort": "none"},
        "prompt_cache_options": {"mode": "explicit"},
        "service_tier": "default",
        "store": False,
        "truncation": "disabled",
    }


def _render_anthropic_body(
    profile: dict[str, Any],
    case: dict[str, Any],
    instruction: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    image_placeholder = (
        f"<IMAGE_BASE64_OMITTED sha256={case['image_sha256']}>"
    )
    return {
        "model": profile["model"]["requested"],
        "max_tokens": max_output_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_placeholder,
                        },
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ],
        "thinking": {"type": "adaptive", "display": "omitted"},
        "output_config": {"effort": "low"},
    }


def render_sanitized_external_request(
    plan: dict[str, Any],
    *,
    case_id: str,
    lane_id: str,
    repeat: int,
) -> dict[str, Any]:
    """Render a deterministic, non-executable request with byte placeholders.

    The result never contains request headers, credential values, proxy
    settings, image bytes, or raw-text bytes.  It is suitable for review and
    hashing, not for direct submission to a provider.
    """

    plan = validate_external_upload_plan(plan)
    case_id = _require_safe_id(case_id, "case_id")
    lane_id = _require_safe_id(lane_id, "lane_id")
    repeat = _require_positive_int(repeat, "repeat")
    if repeat > plan["execution"]["repeats"]:
        raise ExternalPlanError("repeat exceeds the approved plan repeats")
    case = _find_case(plan, case_id)
    lane = _find_lane(plan, lane_id)
    profile = plan["provider_profile"]
    instruction = _render_instruction(lane, case)
    output_limit = plan["execution"]["max_output_tokens"]
    if profile["provider"] == "openai":
        body = _render_openai_body(
            profile, case, instruction, output_limit
        )
        request_metadata: dict[str, str] = {}
    elif profile["provider"] == "anthropic":
        body = _render_anthropic_body(
            profile, case, instruction, output_limit
        )
        request_metadata = copy.deepcopy(profile["request_metadata"])
    else:  # pragma: no cover - guarded by the exact profile comparison
        raise ExternalPlanError("unsupported external provider")
    attempt_material = {
        "plan_sha256": plan["plan_sha256"],
        "case_id": case_id,
        "lane_id": lane_id,
        "repeat": repeat,
    }
    attempt_id = "external-" + _sha256_json(attempt_material)
    return {
        "schema_version": EXTERNAL_SCHEMA_VERSION,
        "kind": "sanitized_external_request",
        "plan_sha256": plan["plan_sha256"],
        "attempt_id": attempt_id,
        "case_id": case_id,
        "lane_id": lane_id,
        "repeat": repeat,
        "endpoint": copy.deepcopy(profile["endpoint"]),
        "credential_env_name": profile["credential_env_name"],
        "executor": copy.deepcopy(plan["executor"]),
        "request_metadata": request_metadata,
        "body": body,
        "policy_binding": {
            "provider_profile_sha256": plan[
                "provider_profile_sha256"
            ],
            "model_revision_semantics": profile["model"][
                "revision_semantics"
            ],
            "retention": copy.deepcopy(profile["retention"]),
            "fallback": "disabled",
        },
        "redactions": {
            "image": {
                "sha256": case["image_sha256"],
                "dimensions": copy.deepcopy(case["dimensions"]),
                "bytes_included": False,
            },
            "raw_text": {
                "sha256": case.get("raw_text_sha256"),
                "bytes_included": False,
                "transmitted_for_lane": lane["include_raw_text"],
            },
        },
    }


def build_external_execution_binding(
    plan: dict[str, Any],
    approval: dict[str, Any],
    *,
    now: str,
) -> dict[str, Any]:
    """Bind an approval to a plan without reading or returning credentials.

    An eventual executor can use this record to select the one allowed
    credential *name*.  This helper never reads process environment variables
    and never accepts or returns a credential value.
    """

    plan = validate_external_upload_plan(plan)
    approval = validate_external_upload_approval(plan, approval, now=now)
    profile = plan["provider_profile"]
    return {
        "schema_version": EXTERNAL_SCHEMA_VERSION,
        "kind": "external_execution_binding",
        "plan_sha256": plan["plan_sha256"],
        "approval_sha256": approval["approval_sha256"],
        "provider_profile_sha256": plan["provider_profile_sha256"],
        "provider_profile_id": profile["profile_id"],
        "provider": profile["provider"],
        "model": profile["model"]["requested"],
        "endpoint": copy.deepcopy(profile["endpoint"]),
        "credential_env_name": profile["credential_env_name"],
        "executor": copy.deepcopy(plan["executor"]),
        "scope": copy.deepcopy(approval["scope"]),
        "retention_policy_id": profile["retention"]["policy_id"],
        "valid_until": approval["expires_at"],
    }
