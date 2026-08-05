from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

import pytest

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
    "nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)
IMAGE_SHA256 = hashlib.sha256(PNG_BYTES).hexdigest()
RAW_TEXT = "Synthetic extracted text only."
RAW_TEXT_SHA256 = hashlib.sha256(RAW_TEXT.encode()).hexdigest()
MANIFEST_SHA256 = "a" * 64
EXPERIMENT_SHA256 = "b" * 64
CASE_ID = "synthetic-page-01"
LANE_ID = "markdown"
PROMPT = "Transcribe this synthetic page to Markdown."
CREATED_AT = "2026-08-03T10:00:00Z"
APPROVED_AT = "2026-08-03T10:05:00Z"
APPROVAL_EXPIRES_AT = "2026-08-03T18:00:00Z"
PLAN_EXPIRES_AT = "2026-08-03T20:00:00Z"
RUN_NOW = "2026-08-03T11:00:00Z"
SYNTHETIC_CREDENTIAL = "synthetic-credential-value"


def _json_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _fixture(
    tmp_path: Path,
    *,
    profile_id: str = "openai-responses-gpt-5.6-sol",
    include_raw_text: bool = False,
    max_api_attempts: int = 1,
) -> dict:
    from sneeze.tpn.ocr_external import (
        build_external_upload_approval_record,
        build_external_upload_plan,
    )
    from sneeze.tpn.ocr_external_setup import (
        build_external_provider_spec,
        external_executor_bundle_identity,
        external_provider_spec_sha256,
    )

    image_path = (tmp_path / "synthetic.png").resolve()
    image_path.write_bytes(PNG_BYTES)
    raw_path = (tmp_path / "synthetic.txt").resolve()
    raw_path.write_text(RAW_TEXT, encoding="utf-8")
    spec = build_external_provider_spec(
        profile_id=profile_id,
        python_executable=sys.executable,
        lane_id=LANE_ID,
        prompt=PROMPT,
        prompt_mode="image_only",
        include_raw_text=include_raw_text,
        timeout_seconds=30,
    )
    spec_sha256 = external_provider_spec_sha256(spec)
    bundle = external_executor_bundle_identity(sys.executable)
    plan = build_external_upload_plan(
        profile_id=profile_id,
        manifest_sha256=MANIFEST_SHA256,
        provider_spec_sha256=spec_sha256,
        adapter_bundle_sha256=bundle["bundle_sha256"],
        cases=[
            {
                "case_id": CASE_ID,
                "image_sha256": IMAGE_SHA256,
                "dimensions": {"w": 1, "h": 1},
                "raw_text_sha256": RAW_TEXT_SHA256,
            }
        ],
        lanes=[
            {
                "lane_id": LANE_ID,
                "prompt": PROMPT,
                "prompt_mode": "image_only",
                "include_raw_text": include_raw_text,
            }
        ],
        budget_usd="100.00",
        created_at=CREATED_AT,
        expires_at=PLAN_EXPIRES_AT,
        repeats=1,
        max_api_attempts_per_case=max_api_attempts,
        max_input_tokens_per_case=32_768,
        max_output_tokens=4096,
    )
    acknowledgement = plan["provider_profile"]["retention"][
        "required_acknowledgement"
    ]
    approval = build_external_upload_approval_record(
        plan,
        approved_by="synthetic-reviewer@example.test",
        approved_at=APPROVED_AT,
        expires_at=APPROVAL_EXPIRES_AT,
        retention_acknowledgement=acknowledgement,
    )
    spec_path = (tmp_path / "provider.json").resolve()
    plan_path = (tmp_path / "plan.json").resolve()
    approval_path = (tmp_path / "approval.json").resolve()
    _write_json(spec_path, spec)
    _write_json(plan_path, plan)
    _write_json(approval_path, approval)
    request = {
        "schema_version": 1,
        "attempt_id": "",
        "experiment_sha256": EXPERIMENT_SHA256,
        "case_id": CASE_ID,
        "lane": LANE_ID,
        "repeat": 1,
        "input": {
            "image": {
                "path": str(image_path),
                "sha256": IMAGE_SHA256,
                "dimensions": {"w": 1, "h": 1},
            },
            "raw_text": (
                {
                    "path": str(raw_path),
                    "sha256": RAW_TEXT_SHA256,
                    "text": RAW_TEXT,
                }
                if include_raw_text
                else None
            ),
        },
        "instruction": PROMPT,
        "instruction_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "settings": spec["lanes"][LANE_ID]["settings"],
    }
    request["attempt_id"] = _json_sha256(
        {
            "provider_spec_sha256": spec_sha256,
            "manifest_sha256": MANIFEST_SHA256,
            "lane": LANE_ID,
            "repeat": 1,
            "case_id": CASE_ID,
            "experiment_sha256": EXPERIMENT_SHA256,
        }
    )
    requests_path = (tmp_path / "requests.jsonl").resolve()
    _write_jsonl(requests_path, [request])
    artifacts_dir = (tmp_path / "artifacts").resolve()
    artifacts_dir.mkdir()
    return {
        "profile_id": profile_id,
        "spec": spec,
        "spec_path": spec_path,
        "plan": plan,
        "plan_path": plan_path,
        "approval": approval,
        "approval_path": approval_path,
        "request": request,
        "requests_path": requests_path,
        "responses_path": (tmp_path / "responses.jsonl").resolve(),
        "artifacts_dir": artifacts_dir,
        "ledger_path": (tmp_path / "budget-ledger.json").resolve(),
    }


def _run(
    fixture: dict,
    transport,
    *,
    credential_reader=None,
    now_reader=None,
):
    from sneeze.tpn.providers.external_api import run_external_adapter

    credential_reader = credential_reader or (
        lambda name: SYNTHETIC_CREDENTIAL
    )
    now_reader = now_reader or (lambda: RUN_NOW)
    return run_external_adapter(
        requests_path=fixture["requests_path"],
        responses_path=fixture["responses_path"],
        artifacts_dir=fixture["artifacts_dir"],
        provider_spec_path=fixture["spec_path"],
        plan_path=fixture["plan_path"],
        approval_path=fixture["approval_path"],
        budget_ledger_path=fixture["ledger_path"],
        profile_id=fixture["profile_id"],
        lane_id=LANE_ID,
        transport=transport,
        credential_reader=credential_reader,
        now_reader=now_reader,
    )


def _read_response(fixture: dict) -> dict:
    return json.loads(
        fixture["responses_path"].read_text(encoding="utf-8").strip()
    )


def test_setup_spec_binds_interpreter_adapter_profile_and_no_secret(tmp_path):
    from sneeze.tpn.ocr_external_setup import (
        ExternalSetupError,
        external_executor_bundle_identity,
        external_provider_spec_sha256,
        validate_external_provider_spec,
        write_external_provider_spec,
    )

    fixture = _fixture(tmp_path)
    spec = fixture["spec"]
    bundle = external_executor_bundle_identity(sys.executable)

    assert validate_external_provider_spec(spec) == spec
    assert spec["external_policy"]["executor_bundle_sha256"] == (
        bundle["bundle_sha256"]
    )
    assert spec["external_policy"]["python_executable"] == (
        bundle["python_executable"]
    )
    assert spec["secret_env_names"] == ["OPENAI_API_KEY"]
    assert spec["argv"][0] == str(Path(sys.executable).resolve())
    assert spec["argv"][1:4] == ["-I", "-S", "-B"]
    assert Path(spec["argv"][4]).name == "external_api.py"
    assert bundle["startup_flags"] == ["-I", "-S", "-B"]
    assert SYNTHETIC_CREDENTIAL not in json.dumps(spec)
    assert len(external_provider_spec_sha256(spec)) == 64
    assert [item["module"] for item in bundle["files"]] == [
        "sneeze.tpn.ocr_external",
        "sneeze.tpn.providers.external_api",
    ]

    output = (tmp_path / "written-provider.json").resolve()
    receipt = write_external_provider_spec(
        output_path=output,
        profile_id=fixture["profile_id"],
        python_executable=sys.executable,
        lane_id="separate-lane",
        prompt=PROMPT,
        prompt_mode="image_only",
    )
    assert output.is_file()
    assert receipt["executor_bundle_sha256"] == bundle["bundle_sha256"]
    with pytest.raises(ExternalSetupError, match="refusing to replace"):
        write_external_provider_spec(
            output_path=output,
            profile_id=fixture["profile_id"],
            python_executable=sys.executable,
            lane_id="separate-lane",
            prompt=PROMPT,
            prompt_mode="image_only",
        )
    with pytest.raises(ExternalSetupError, match="parity lane"):
        write_external_provider_spec(
            output_path=(tmp_path / "invalid-parity.json").resolve(),
            profile_id=fixture["profile_id"],
            python_executable=sys.executable,
            lane_id="parity",
            prompt=PROMPT,
            prompt_mode="parity",
            include_raw_text=True,
        )


def test_isolated_no_site_executor_attests_before_credential_read(tmp_path):
    import os
    import subprocess
    from datetime import UTC, datetime, timedelta

    from sneeze.tpn.ocr_external import (
        build_external_upload_approval_record,
        build_external_upload_plan,
    )
    from sneeze.tpn.ocr_external_setup import (
        external_executor_bundle_identity,
        external_provider_spec_sha256,
    )

    fixture = _fixture(tmp_path)
    current = datetime.now(UTC).replace(microsecond=0)

    def timestamp(value):
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")

    plan = build_external_upload_plan(
        profile_id=fixture["profile_id"],
        manifest_sha256=MANIFEST_SHA256,
        provider_spec_sha256=external_provider_spec_sha256(
            fixture["spec"]
        ),
        adapter_bundle_sha256=external_executor_bundle_identity(
            sys.executable
        )["bundle_sha256"],
        cases=[
            {
                "case_id": CASE_ID,
                "image_sha256": IMAGE_SHA256,
                "dimensions": {"w": 1, "h": 1},
            }
        ],
        lanes=[
            {
                "lane_id": LANE_ID,
                "prompt": PROMPT,
                "prompt_mode": "image_only",
                "include_raw_text": False,
            }
        ],
        budget_usd="100.00",
        created_at=timestamp(current - timedelta(minutes=1)),
        expires_at=timestamp(current + timedelta(hours=1)),
        repeats=1,
        max_api_attempts_per_case=1,
        max_input_tokens_per_case=32_768,
        max_output_tokens=4096,
    )
    approval = build_external_upload_approval_record(
        plan,
        approved_by="synthetic-reviewer@example.test",
        approved_at=timestamp(current),
        expires_at=timestamp(current + timedelta(minutes=30)),
        retention_acknowledgement=plan["provider_profile"]["retention"][
            "required_acknowledgement"
        ],
    )
    _write_json(fixture["plan_path"], plan)
    _write_json(fixture["approval_path"], approval)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
    }
    completed = subprocess.run(
        [
            *fixture["spec"]["argv"],
            "--external-provider-spec",
            str(fixture["spec_path"]),
            "--external-plan",
            str(fixture["plan_path"]),
            "--external-approval",
            str(fixture["approval_path"]),
            "--external-budget-ledger",
            str(fixture["ledger_path"]),
            "--requests-jsonl",
            str(fixture["requests_path"]),
            "--responses-jsonl",
            str(fixture["responses_path"]),
            "--artifacts-dir",
            str(fixture["artifacts_dir"]),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "required provider credential is unavailable" in completed.stderr
    assert "No module named" not in completed.stderr
    assert not fixture["responses_path"].exists()


def test_openai_fake_transport_end_to_end_is_sanitized_and_plan_bound(
    tmp_path,
):
    from sneeze.tpn.providers.external_api import TransportResult

    fixture = _fixture(tmp_path)
    calls = []
    clock_calls = []
    reasoning_summary = "Sensitive synthetic reasoning summary."
    reasoning_content = "Sensitive synthetic internal reasoning content."
    encrypted_reasoning = "encrypted-synthetic-reasoning-payload"

    def clock():
        clock_calls.append(RUN_NOW)
        return RUN_NOW

    def transport(*, profile, credential, body, timeout_seconds):
        calls.append(body)
        assert profile["endpoint"]["host"] == "api.openai.com"
        assert credential == SYNTHETIC_CREDENTIAL
        assert timeout_seconds == 30
        image = body["input"][0]["content"][1]
        assert image["image_url"].startswith("data:image/png;base64,")
        assert "OMITTED" not in image["image_url"]
        assert body["background"] is False
        assert body["prompt_cache_options"] == {"mode": "explicit"}
        response = {
            "id": "resp_synthetic_1",
            "model": "gpt-5.6-sol",
            "status": "completed",
            "output": [
                {
                    "id": "rs_synthetic_1",
                    "type": "reasoning",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": reasoning_summary,
                        }
                    ],
                    "content": [
                        {
                            "type": "reasoning_text",
                            "text": reasoning_content,
                        }
                    ],
                    "encrypted_content": encrypted_reasoning,
                    "status": "completed",
                },
                {
                    "type": "message",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "# Synthetic\n"}
                    ],
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 4,
                "input_tokens_details": {
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                },
            },
            "untrusted": SYNTHETIC_CREDENTIAL,
        }
        return TransportResult(
            status_code=200,
            response_headers={
                "x-request-id": "request_synthetic_1",
                "authorization": SYNTHETIC_CREDENTIAL,
            },
            body=json.dumps(response).encode(),
            wall_seconds=0.25,
        )

    result = _run(fixture, transport, now_reader=clock)

    assert result == {
        "response_count": 1,
        "delivery_unknown": False,
        "all_completed": True,
    }
    assert len(calls) == 1
    assert len(clock_calls) >= 3
    response = _read_response(fixture)
    assert response["status"] == "completed"
    assert response["output"]["text"] == "# Synthetic\n"
    assert response["model"]["resolved_snapshot"] == "gpt-5.6-sol"
    assert response["generation"]["input_tokens"] == 100
    assert response["generation"]["finish_reason"] == "stop"
    params = response["generation"]["params"]
    assert params["input_cap_semantics"] == (
        "post_response_scope_check_only; spending reservation uses the full "
        "documented provider context ceiling"
    )
    assert "IMAGE_BASE64_OMITTED" in json.dumps(params["request"]["body"])
    assert response["effective_request"]["transmitted_prompt"][
        "sha256"
    ] == hashlib.sha256(PROMPT.encode()).hexdigest()
    persisted = fixture["responses_path"].read_text(encoding="utf-8")
    artifact_path = next(fixture["artifacts_dir"].iterdir())
    artifact = artifact_path.read_text(encoding="utf-8")
    assert SYNTHETIC_CREDENTIAL not in persisted
    assert SYNTHETIC_CREDENTIAL not in artifact
    assert "authorization" not in artifact.lower()
    assert base64.b64encode(PNG_BYTES).decode() not in artifact
    assert reasoning_summary not in persisted
    assert reasoning_content not in persisted
    assert encrypted_reasoning not in persisted
    assert reasoning_summary not in artifact
    assert reasoning_content not in artifact
    assert encrypted_reasoning not in artifact
    artifact_record = json.loads(artifact)
    safe_reasoning = artifact_record["attempts"][0]["provider_response"][
        "reasoning"
    ]
    assert safe_reasoning == {
        "count": 1,
        "items": [
            {
                "id": "rs_synthetic_1",
                "status": "completed",
                "summary_count": 1,
                "summary_sha256": [
                    hashlib.sha256(reasoning_summary.encode()).hexdigest()
                ],
                "content_count": 1,
                "content_sha256": [
                    hashlib.sha256(reasoning_content.encode()).hexdigest()
                ],
                "encrypted_content_present": True,
                "encrypted_content_sha256": hashlib.sha256(
                    encrypted_reasoning.encode()
                ).hexdigest(),
            }
        ],
    }
    cost = response["usage"]["cost"]
    assert cost["cached_input_tokens"] == 0
    assert cost["cache_write_input_tokens"] == 0
    assert cost["rates_per_million_tokens"]["cache_write"] == "6.25"
    assert cost["computed_usd"] == "0.000620"
    ledger = json.loads(fixture["ledger_path"].read_text())
    reservation = next(iter(ledger["reservations"].values()))
    assert reservation["slot"] == {
        "plan_sha256": fixture["plan"]["plan_sha256"],
        "case_id": CASE_ID,
        "lane_id": LANE_ID,
        "repeat": 1,
        "api_attempt": 1,
    }
    assert "experiment_sha256" not in reservation["slot"]
    assert "request_attempt_id" not in reservation["slot"]


@pytest.mark.parametrize(
    ("cached_tokens", "cache_write_tokens"),
    [(11, 0), (0, 11), (6, 5)],
)
def test_openai_rejects_cache_usage_larger_than_total_input(
    cached_tokens,
    cache_write_tokens,
):
    from sneeze.tpn.providers.external_api import (
        ExternalAdapterError,
        _openai_response,
    )

    response = {
        "id": "resp_synthetic_invalid_usage",
        "model": "gpt-5.6-sol",
        "status": "completed",
        "output": [],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 1,
            "input_tokens_details": {
                "cached_tokens": cached_tokens,
                "cache_write_tokens": cache_write_tokens,
            },
        },
    }

    with pytest.raises(ExternalAdapterError, match="exceeds input_tokens"):
        _openai_response(response)


def test_openai_cache_accounting_remains_generic():
    from sneeze.tpn.ocr_external import get_external_provider_profile
    from sneeze.tpn.providers.external_api import (
        _computed_cost,
        _openai_response,
    )

    response = {
        "id": "resp_synthetic_cached_usage",
        "model": "gpt-5.6-sol",
        "status": "completed",
        "output": [],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 4,
            "input_tokens_details": {
                "cached_tokens": 20,
                "cache_write_tokens": 30,
            },
        },
    }
    profile = get_external_provider_profile(
        "openai-responses-gpt-5.6-sol"
    )
    profile.pop("profile_sha256")
    parsed = _openai_response(response)
    cost = _computed_cost(parsed, profile)

    assert parsed["cached_input_tokens"] == 20
    assert parsed["cache_write_input_tokens"] == 30
    assert cost["computed_usd"] == "0.000568"


@pytest.mark.parametrize(
    ("cached_tokens", "cache_write_tokens"), [(1, 0), (0, 1)]
)
def test_openai_cache_disabled_profile_fails_closed_on_returned_usage(
    cached_tokens,
    cache_write_tokens,
):
    from sneeze.tpn.ocr_external import get_external_provider_profile
    from sneeze.tpn.providers.external_api import (
        ExternalAdapterError,
        _parse_provider_response,
    )

    response = {
        "id": "resp_synthetic_unexpected_cache_usage",
        "model": "gpt-5.6-sol",
        "status": "completed",
        "output": [],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 4,
            "input_tokens_details": {
                "cached_tokens": cached_tokens,
                "cache_write_tokens": cache_write_tokens,
            },
        },
    }
    profile = get_external_provider_profile(
        "openai-responses-gpt-5.6-sol"
    )
    profile.pop("profile_sha256")

    with pytest.raises(ExternalAdapterError, match="caching was disabled"):
        _parse_provider_response(response, profile)


@pytest.mark.parametrize(
    "reasoning_item",
    [
        {
            "id": "rs_extra",
            "type": "reasoning",
            "summary": [],
            "unexpected": "not-documented",
        },
        {
            "id": "rs_bad_summary",
            "type": "reasoning",
            "summary": [
                {
                    "type": "summary_text",
                    "text": "sensitive summary",
                    "unexpected": True,
                }
            ],
        },
        {
            "id": "rs_bad_content",
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "output_text", "text": "not reasoning"}],
        },
        {
            "id": "rs_bad_encrypted",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": 123,
        },
    ],
)
def test_openai_rejects_undocumented_reasoning_item_shapes(reasoning_item):
    from sneeze.tpn.providers.external_api import (
        ExternalAdapterError,
        _openai_response,
    )

    response = {
        "id": "resp_synthetic_invalid_reasoning",
        "model": "gpt-5.6-sol",
        "status": "completed",
        "output": [reasoning_item],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 1,
            "input_tokens_details": {
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
        },
    }

    with pytest.raises(ExternalAdapterError, match="OpenAI .*reasoning"):
        _openai_response(response)


def test_anthropic_raw_text_provenance_is_exact_but_artifact_is_redacted(
    tmp_path,
):
    from sneeze.tpn.providers.external_api import TransportResult

    fixture = _fixture(
        tmp_path,
        profile_id="anthropic-messages-claude-fable-5",
        include_raw_text=True,
    )
    transmitted_prompt = f"{PROMPT}\n\n{RAW_TEXT}"
    thinking_signature = "synthetic-omitted-thinking-signature"

    def transport(*, profile, credential, body, timeout_seconds):
        assert profile["endpoint"]["host"] == "api.anthropic.com"
        content = body["messages"][0]["content"]
        assert content[0]["type"] == "image"
        assert content[1]["type"] == "text"
        assert content[1]["text"] == (
            transmitted_prompt
        )
        assert body["thinking"] == {
            "type": "adaptive",
            "display": "omitted",
        }
        response = {
            "id": "msg_synthetic_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-fable-5",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "",
                    "signature": thinking_signature,
                },
                {"type": "text", "text": "# Fable\n"},
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 120,
                "output_tokens": 5,
                "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None,
                "output_tokens_details": {"thinking_tokens": 3},
            },
        }
        return TransportResult(200, {}, json.dumps(response).encode(), 0.2)

    result = _run(fixture, transport)

    assert result["all_completed"] is True
    response = _read_response(fixture)
    transmitted = response["effective_request"]["transmitted_prompt"]
    assert transmitted["sha256"] == hashlib.sha256(
        transmitted_prompt.encode()
    ).hexdigest()
    assert transmitted["includes_raw_text"] is True
    assert transmitted["raw_text_sha256"] == RAW_TEXT_SHA256
    assert response["generation"]["output_tokens_details"] == {
        "thinking_tokens": 3
    }
    sanitized = json.dumps(
        response["effective_request"]["sanitized_body"], sort_keys=True
    )
    assert "RAW_TEXT_UTF8_OMITTED" in sanitized
    artifact = next(fixture["artifacts_dir"].iterdir()).read_text()
    assert RAW_TEXT not in artifact
    assert thinking_signature not in artifact
    assert thinking_signature not in fixture["responses_path"].read_text()
    artifact_record = json.loads(artifact)
    provider_response = artifact_record["attempts"][0][
        "provider_response"
    ]
    assert provider_response["usage"]["output_tokens_details"] == {
        "thinking_tokens": 3
    }
    assert provider_response["thinking_hash_sources"] == ["signature"]
    assert provider_response["thinking_sha256"] == [
        hashlib.sha256(thinking_signature.encode()).hexdigest()
    ]


def test_invalid_scope_refuses_before_credential_or_transport(tmp_path):
    from sneeze.tpn.providers.external_api import (
        ExternalAdapterError,
        TransportResult,
    )

    fixture = _fixture(tmp_path)
    request = fixture["request"]
    request["instruction"] = "Changed after approval."
    _write_jsonl(fixture["requests_path"], [request])

    def credential_reader(name):
        raise AssertionError("credential must not be read")

    def transport(**kwargs):
        raise AssertionError("transport must not run")

    with pytest.raises(ExternalAdapterError, match="prompt"):
        _run(
            fixture,
            transport,
            credential_reader=credential_reader,
        )
    assert not fixture["responses_path"].exists()
    assert not fixture["ledger_path"].exists()
    assert TransportResult is not None


def test_delivery_unknown_stops_and_durably_blocks_replay(tmp_path):
    from sneeze.tpn.providers.external_api import (
        DeliveryUnknownError,
        ExternalAdapterError,
    )

    fixture = _fixture(tmp_path, max_api_attempts=2)
    calls = []

    def transport(**kwargs):
        calls.append(1)
        raise DeliveryUnknownError("synthetic unknown delivery")

    result = _run(fixture, transport)

    assert result["delivery_unknown"] is True
    assert len(calls) == 1
    assert _read_response(fixture)["error"]["type"] == "DeliveryUnknown"
    ledger = json.loads(fixture["ledger_path"].read_text())
    assert ledger["delivery_unknown"] is True
    assert next(iter(ledger["reservations"].values()))["state"] == (
        "delivery_unknown"
    )

    replay = dict(fixture)
    replay["responses_path"] = (tmp_path / "replay-responses.jsonl").resolve()
    replay["artifacts_dir"] = (tmp_path / "replay-artifacts").resolve()
    replay["artifacts_dir"].mkdir()

    def credential_reader(name):
        raise AssertionError("blocked replay must not read credential")

    with pytest.raises(ExternalAdapterError, match="delivery_unknown"):
        _run(replay, transport, credential_reader=credential_reader)
    assert len(calls) == 1


def test_confirmed_non_json_http_error_gets_safe_row_and_artifact(tmp_path):
    from sneeze.tpn.providers.external_api import TransportResult

    fixture = _fixture(tmp_path)

    def transport(**kwargs):
        return TransportResult(
            status_code=400,
            response_headers={"content-type": "text/html"},
            body=b"<html>synthetic error</html>",
            wall_seconds=0.1,
        )

    result = _run(fixture, transport)

    assert result["all_completed"] is False
    response = _read_response(fixture)
    assert response["error"]["type"] == "ProviderResponseInvalid"
    artifact = json.loads(
        next(fixture["artifacts_dir"].iterdir()).read_text()
    )
    attempt = artifact["attempts"][0]
    assert attempt["delivery"] == "confirmed"
    assert attempt["response_parse"] == "rejected"
    assert "synthetic error" not in json.dumps(artifact)
