from __future__ import annotations

import copy
import json

import pytest

CASE_ID = "synthetic-page-01"
IMAGE_SHA256 = "a" * 64
RAW_TEXT_SHA256 = "b" * 64
MANIFEST_SHA256 = "c" * 64
PROVIDER_SPEC_SHA256 = "d" * 64
ADAPTER_BUNDLE_SHA256 = "e" * 64
CREATED_AT = "2026-08-03T12:00:00Z"
APPROVED_AT = "2026-08-03T12:05:00Z"
EXPIRES_AT = "2026-08-03T14:00:00Z"
APPROVAL_EXPIRES_AT = "2026-08-03T13:00:00Z"


def _cases(*, with_raw_text: bool = True) -> list[dict]:
    case = {
        "case_id": CASE_ID,
        "image_sha256": IMAGE_SHA256,
        "dimensions": {"w": 64, "h": 96},
    }
    if with_raw_text:
        case["raw_text_sha256"] = RAW_TEXT_SHA256
    return [case]


def _lanes(*, include_raw_text: bool = False) -> list[dict]:
    return [
        {
            "lane_id": "markdown",
            "prompt": "Transcribe this synthetic page to Markdown.",
            "prompt_mode": "image_only",
            "include_raw_text": include_raw_text,
        }
    ]


def _plan(
    profile_id: str = "openai-responses-gpt-5.6-sol",
    *,
    include_raw_text: bool = False,
    repeats: int = 2,
    max_input_tokens_per_case: int = 32_768,
    max_output_tokens: int = 4096,
) -> dict:
    from sneeze.tpn.ocr_external import build_external_upload_plan

    return build_external_upload_plan(
        profile_id=profile_id,
        manifest_sha256=MANIFEST_SHA256,
        provider_spec_sha256=PROVIDER_SPEC_SHA256,
        adapter_bundle_sha256=ADAPTER_BUNDLE_SHA256,
        cases=_cases(),
        lanes=_lanes(include_raw_text=include_raw_text),
        repeats=repeats,
        max_api_attempts_per_case=1,
        max_input_tokens_per_case=max_input_tokens_per_case,
        max_output_tokens=max_output_tokens,
        budget_usd="100.00",
        created_at=CREATED_AT,
        expires_at=EXPIRES_AT,
    )


def _approval(plan: dict) -> dict:
    from sneeze.tpn.ocr_external import (
        build_external_upload_approval_record,
    )

    acknowledgement = plan["provider_profile"]["retention"][
        "required_acknowledgement"
    ]
    return build_external_upload_approval_record(
        plan,
        approved_by="synthetic-reviewer@example.test",
        approved_at=APPROVED_AT,
        expires_at=APPROVAL_EXPIRES_AT,
        retention_acknowledgement=acknowledgement,
    )


def test_provider_profiles_pin_hosts_models_retention_and_dated_prices():
    from sneeze.tpn.ocr_external import (
        get_external_provider_profile,
        list_external_provider_profiles,
    )

    assert list_external_provider_profiles() == [
        "anthropic-messages-claude-fable-5",
        "openai-responses-gpt-5.6-sol",
    ]

    openai = get_external_provider_profile(
        "openai-responses-gpt-5.6-sol"
    )
    assert openai["endpoint"] == {
        "scheme": "https",
        "host": "api.openai.com",
        "path": "/v1/responses",
        "method": "POST",
        "api": "responses",
    }
    assert openai["credential_env_name"] == "OPENAI_API_KEY"
    assert openai["model"] == {
        "requested": "gpt-5.6-sol",
        "revision_semantics": "provider_managed_model_id",
        "fallback": "disabled",
    }
    assert openai["retention"]["default_abuse_monitoring_max_days"] == 30
    assert openai["retention"]["store_false_is_zero_data_retention"] \
        is False
    assert openai["pricing"]["verified_at"] == "2026-08-03"
    assert openai["pricing"]["short_context"]["output"] == "30.00"
    assert openai["pricing"]["source"].startswith(
        "https://developers.openai.com/"
    )
    assert len(openai["profile_sha256"]) == 64

    anthropic = get_external_provider_profile(
        "anthropic-messages-claude-fable-5"
    )
    assert anthropic["endpoint"] == {
        "scheme": "https",
        "host": "api.anthropic.com",
        "path": "/v1/messages",
        "method": "POST",
        "api": "messages",
    }
    assert anthropic["request_metadata"] == {
        "anthropic-version": "2023-06-01"
    }
    assert anthropic["credential_env_name"] == "ANTHROPIC_API_KEY"
    assert anthropic["model"]["requested"] == "claude-fable-5"
    assert anthropic["model"]["revision_semantics"] == (
        "provider_documented_fixed_snapshot_model_id"
    )
    assert anthropic["retention"]["mandatory_days"] == 30
    assert anthropic["retention"]["zero_data_retention_available"] \
        is False
    assert anthropic["pricing"] == {
        "currency": "USD",
        "unit": "million_tokens",
        "verified_at": "2026-08-03",
        "input": "10.00",
        "output": "50.00",
        "source": (
            "https://platform.claude.com/docs/en/about-claude/"
            "models/overview"
        ),
    }

    openai["endpoint"]["host"] = "example.invalid"
    fresh = get_external_provider_profile(
        "openai-responses-gpt-5.6-sol"
    )
    assert fresh["endpoint"]["host"] == "api.openai.com"


def test_upload_plan_is_deterministic_digest_bound_and_byte_free():
    from sneeze.tpn.ocr_external import validate_external_upload_plan

    first = _plan(include_raw_text=True)
    second = _plan(include_raw_text=True)

    assert first == second
    assert validate_external_upload_plan(first) == first
    assert len(first["provider_profile_sha256"]) == 64
    assert "profile_sha256" not in first["provider_profile"]
    assert first["executor"] == {
        "protocol": "sneeze-external-ocr-v1",
        "provider_spec_sha256": PROVIDER_SPEC_SHA256,
        "adapter_bundle_sha256": ADAPTER_BUNDLE_SHA256,
    }
    estimate = first["execution"]["cost_estimate"]
    assert estimate["method"] == "provider_context_ceiling_reservation_v2"
    assert estimate["rate_tier"] == "long_context"
    assert estimate["input_rate_basis"] == "cache_write"
    assert estimate["input_rate_per_million_tokens"] == "12.50"
    assert estimate["output_rate_per_million_tokens"] == "45.00"
    assert estimate["attempt_count"] == 2
    assert estimate["billable_input_reservation_tokens"] == 1_050_000
    assert estimate["pessimistic_per_attempt_usd"] == "13.309320"
    assert estimate["pessimistic_max_usd"] == "26.618640"
    assert first["transmission"] == {
        "requires_external_upload_approval": True,
        "images": {
            "media_type": "image/png",
            "encoding": "base64",
            "transport": "inline",
        },
        "raw_text": {
            "encoding": "utf-8",
            "transport": "inline",
            "included_lane_ids": ["markdown"],
        },
        "plan_contains_source_bytes": False,
    }
    assert first["lanes"][0]["prompt_mode"] == "image_only"
    assert len(first["lanes"][0]["prompt_sha256"]) == 64
    assert first["cases"][0]["media_type"] == "image/png"
    serialized = json.dumps(first, sort_keys=True)
    assert "OPENAI_API_KEY" in serialized
    assert "data:image/png;base64" not in serialized
    assert "RAW_TEXT_UTF8_OMITTED" not in serialized
    assert "authorization" not in serialized.lower()
    assert "bearer" not in serialized.lower()
    assert "headers" not in serialized.lower()
    assert "proxy" not in serialized.lower()


def test_upload_plan_requires_explicit_safe_data_scope():
    from sneeze.tpn.ocr_external import (
        ExternalPlanError,
        build_external_upload_plan,
    )

    base = {
        "profile_id": "openai-responses-gpt-5.6-sol",
        "manifest_sha256": MANIFEST_SHA256,
        "provider_spec_sha256": PROVIDER_SPEC_SHA256,
        "adapter_bundle_sha256": ADAPTER_BUNDLE_SHA256,
        "cases": _cases(),
        "lanes": _lanes(),
        "budget_usd": "1.00",
        "created_at": CREATED_AT,
        "expires_at": EXPIRES_AT,
        "max_input_tokens_per_case": 32_768,
    }

    missing_flag = copy.deepcopy(base)
    missing_flag["lanes"][0].pop("include_raw_text")
    with pytest.raises(ExternalPlanError, match="unexpected fields"):
        build_external_upload_plan(**missing_flag)

    extra_case_field = copy.deepcopy(base)
    extra_case_field["cases"][0]["image_path"] = "/synthetic/page.png"
    with pytest.raises(ExternalPlanError, match="unexpected fields"):
        build_external_upload_plan(**extra_case_field)

    secret_prompt = copy.deepcopy(base)
    secret_prompt["lanes"][0]["prompt"] = (
        "Authorization: Bearer synthetic-secret-token"
    )
    with pytest.raises(ExternalPlanError, match="credential or secret"):
        build_external_upload_plan(**secret_prompt)

    raw_lane_without_digest = copy.deepcopy(base)
    raw_lane_without_digest["lanes"][0]["include_raw_text"] = True
    raw_lane_without_digest["cases"] = _cases(with_raw_text=False)
    with pytest.raises(ExternalPlanError, match="raw_text_sha256"):
        build_external_upload_plan(**raw_lane_without_digest)

    too_many_attempts = copy.deepcopy(base)
    too_many_attempts["max_api_attempts_per_case"] = 4
    with pytest.raises(ExternalPlanError, match="cannot exceed 3"):
        build_external_upload_plan(**too_many_attempts)

    insufficient_budget = copy.deepcopy(base)
    insufficient_budget["budget_usd"] = "0.01"
    with pytest.raises(ExternalPlanError, match="pessimistic"):
        build_external_upload_plan(**insufficient_budget)


def test_openai_cost_estimate_switches_above_long_context_boundary():
    short_plan = _plan(max_input_tokens_per_case=272_000)
    short = short_plan["execution"]["cost_estimate"]
    assert short["rate_tier"] == "long_context"
    assert short["input_rate_per_million_tokens"] == "12.50"
    assert short["output_rate_per_million_tokens"] == "45.00"
    assert short["pessimistic_max_usd"] == "26.618640"

    long_plan = _plan(max_input_tokens_per_case=272_001)
    long = long_plan["execution"]["cost_estimate"]
    assert long["rate_tier"] == "long_context"
    assert long["input_rate_basis"] == "cache_write"
    assert long["input_rate_per_million_tokens"] == "12.50"
    assert long["output_rate_per_million_tokens"] == "45.00"
    assert long["pessimistic_per_attempt_usd"] == "13.309320"
    assert long["pessimistic_max_usd"] == "26.618640"


def test_upload_plan_validation_rejects_mutation_and_stale_profile():
    from sneeze.tpn.ocr_external import (
        ExternalPlanError,
        validate_external_upload_plan,
    )

    plan = _plan()
    mutated = copy.deepcopy(plan)
    mutated["execution"]["max_output_tokens"] = 5000
    with pytest.raises(ExternalPlanError, match="cost_estimate"):
        validate_external_upload_plan(mutated)

    mutated = copy.deepcopy(plan)
    mutated["provider_profile"]["endpoint"]["host"] = "proxy.invalid"
    with pytest.raises(ExternalPlanError, match="dated policy"):
        validate_external_upload_plan(mutated)

    mutated = copy.deepcopy(plan)
    mutated["executor"]["adapter_bundle_sha256"] = "f" * 64
    with pytest.raises(ExternalPlanError, match="plan_sha256"):
        validate_external_upload_plan(mutated)

    mutated = copy.deepcopy(plan)
    mutated["credential_value"] = "synthetic-secret"
    with pytest.raises(ExternalPlanError, match="unexpected fields"):
        validate_external_upload_plan(mutated)


def test_separate_approval_binds_scope_retention_budget_and_expiry():
    from sneeze.tpn.ocr_external import (
        ExternalApprovalError,
        build_external_execution_binding,
        validate_external_upload_approval,
    )

    plan = _plan()
    approval = _approval(plan)
    assert validate_external_upload_approval(
        plan, approval, now="2026-08-03T12:30:00Z"
    ) == approval

    binding = build_external_execution_binding(
        plan, approval, now="2026-08-03T12:30:00Z"
    )
    assert binding["plan_sha256"] == plan["plan_sha256"]
    assert binding["approval_sha256"] == approval["approval_sha256"]
    assert binding["credential_env_name"] == "OPENAI_API_KEY"
    assert binding["executor"] == plan["executor"]
    assert binding["scope"]["budget_usd"] == "100.00"
    assert binding["scope"]["cost_estimate"] == (
        plan["execution"]["cost_estimate"]
    )
    serialized = json.dumps(binding, sort_keys=True)
    assert "synthetic-secret" not in serialized
    assert "authorization" not in serialized.lower()
    assert "bearer" not in serialized.lower()
    assert "headers" not in serialized.lower()
    assert "proxy" not in serialized.lower()

    wrong_plan = copy.deepcopy(approval)
    wrong_plan["plan_sha256"] = "d" * 64
    with pytest.raises(ExternalApprovalError, match="different plan"):
        validate_external_upload_approval(
            plan, wrong_plan, now="2026-08-03T12:30:00Z"
        )

    wrong_scope = copy.deepcopy(approval)
    wrong_scope["scope"]["budget_usd"] = "101.00"
    with pytest.raises(ExternalApprovalError, match="scope"):
        validate_external_upload_approval(
            plan, wrong_scope, now="2026-08-03T12:30:00Z"
        )

    wrong_retention = copy.deepcopy(approval)
    wrong_retention["retention"]["policy_id"] = "different-policy"
    with pytest.raises(ExternalApprovalError, match="retention"):
        validate_external_upload_approval(
            plan, wrong_retention, now="2026-08-03T12:30:00Z"
        )

    with pytest.raises(ExternalApprovalError, match="expired"):
        validate_external_upload_approval(
            plan, approval, now=APPROVAL_EXPIRES_AT
        )


def test_approval_rejects_secret_identity_and_inexact_retention():
    from sneeze.tpn.ocr_external import (
        ExternalApprovalError,
        build_external_upload_approval_record,
    )

    plan = _plan("anthropic-messages-claude-fable-5")
    acknowledgement = plan["provider_profile"]["retention"][
        "required_acknowledgement"
    ]
    with pytest.raises(ExternalApprovalError, match="exactly match"):
        build_external_upload_approval_record(
            plan,
            approved_by="synthetic-reviewer",
            approved_at=APPROVED_AT,
            expires_at=APPROVAL_EXPIRES_AT,
            retention_acknowledgement="30 days is probably acceptable",
        )
    with pytest.raises(ExternalApprovalError, match="credential or secret"):
        build_external_upload_approval_record(
            plan,
            approved_by="sk-ant-syntheticcredentialvalue",
            approved_at=APPROVED_AT,
            expires_at=APPROVAL_EXPIRES_AT,
            retention_acknowledgement=acknowledgement,
        )


def test_openai_sanitized_request_is_exact_bounded_and_non_executable():
    from sneeze.tpn.ocr_external import render_sanitized_external_request

    plan = _plan(max_output_tokens=6144)
    request = render_sanitized_external_request(
        plan, case_id=CASE_ID, lane_id="markdown", repeat=1
    )
    assert request == render_sanitized_external_request(
        plan, case_id=CASE_ID, lane_id="markdown", repeat=1
    )
    assert request["endpoint"] == {
        "scheme": "https",
        "host": "api.openai.com",
        "path": "/v1/responses",
        "method": "POST",
        "api": "responses",
    }
    assert request["request_metadata"] == {}
    body = request["body"]
    assert body["model"] == "gpt-5.6-sol"
    assert body["max_output_tokens"] == 6144
    assert body["background"] is False
    assert body["reasoning"] == {"effort": "none"}
    assert body["service_tier"] == "default"
    assert body["store"] is False
    assert body["truncation"] == "disabled"
    image = body["input"][0]["content"][1]
    assert image == {
        "type": "input_image",
        "image_url": (
            "data:image/png;base64,"
            f"<IMAGE_BASE64_OMITTED sha256={IMAGE_SHA256}>"
        ),
        "detail": "original",
    }
    assert request["redactions"]["image"]["bytes_included"] is False
    serialized = json.dumps(request, sort_keys=True)
    assert "authorization" not in serialized.lower()
    assert "bearer" not in serialized.lower()
    assert "headers" not in serialized.lower()
    assert "proxy" not in serialized.lower()
    assert "synthetic-secret-token" not in serialized


def test_anthropic_sanitized_request_has_fable_policy_and_no_fallback():
    from sneeze.tpn.ocr_external import render_sanitized_external_request

    plan = _plan(
        "anthropic-messages-claude-fable-5",
        include_raw_text=True,
        max_output_tokens=7168,
    )
    estimate = plan["execution"]["cost_estimate"]
    assert estimate["rate_tier"] == "fixed"
    assert estimate["input_rate_per_million_tokens"] == "10.00"
    assert estimate["output_rate_per_million_tokens"] == "50.00"
    assert estimate["billable_input_reservation_tokens"] == 1_000_000
    assert estimate["pessimistic_per_attempt_usd"] == "10.358400"
    assert estimate["pessimistic_max_usd"] == "20.716800"
    request = render_sanitized_external_request(
        plan, case_id=CASE_ID, lane_id="markdown", repeat=2
    )
    assert request["endpoint"]["host"] == "api.anthropic.com"
    assert request["endpoint"]["path"] == "/v1/messages"
    assert request["request_metadata"] == {
        "anthropic-version": "2023-06-01"
    }
    body = request["body"]
    assert body["model"] == "claude-fable-5"
    assert body["max_tokens"] == 7168
    assert body["thinking"] == {
        "type": "adaptive",
        "display": "omitted",
    }
    assert body["output_config"] == {"effort": "low"}
    assert "fallback" not in body
    assert "tools" not in body
    image = body["messages"][0]["content"][1]
    assert image == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": f"<IMAGE_BASE64_OMITTED sha256={IMAGE_SHA256}>",
        },
    }
    prompt = body["messages"][0]["content"][0]["text"]
    assert prompt.endswith(
        f"<RAW_TEXT_UTF8_OMITTED sha256={RAW_TEXT_SHA256}>"
    )
    policy = request["policy_binding"]
    assert policy["retention"]["mandatory_days"] == 30
    assert policy["retention"]["zero_data_retention_available"] is False
    assert policy["fallback"] == "disabled"


def test_external_module_has_no_transport_sdk_or_environment_access():
    import ast
    import inspect

    import sneeze.tpn.ocr_external as external

    tree = ast.parse(inspect.getsource(external))
    imported_roots = {
        node.names[0].name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    }
    imported_roots.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported_roots <= {
        "__future__",
        "copy",
        "datetime",
        "decimal",
        "hashlib",
        "json",
        "re",
        "typing",
    }
