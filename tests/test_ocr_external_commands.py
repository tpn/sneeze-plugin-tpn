from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pytest

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
    "x8AAwMCAO+Xk6cAAAAASUVORK5CYII="
)
RAW_TEXT = "SYNTHETIC-RAW-TEXT-PAYLOAD-DO-NOT-PERSIST"
OPENAI_CREDENTIAL = "sk-synthetic-openai-credential-do-not-read"
ANTHROPIC_CREDENTIAL = "sk-ant-synthetic-credential-do-not-read"
PROFILE_ID = "openai-responses-gpt-5.6-sol"
CASE_ID = "synthetic-page-01"
STEM = "paper/page-01"
NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    input_root = (tmp_path / "input").resolve()
    page_stem = input_root / STEM
    page_stem.parent.mkdir(parents=True)
    image_path = page_stem.with_suffix(".png")
    raw_path = page_stem.with_suffix(".txt")
    image_path.write_bytes(PNG_BYTES)
    raw_path.write_text(RAW_TEXT, encoding="utf-8")
    case = {
        "schema_version": 1,
        "case_id": CASE_ID,
        "stem": STEM,
        "input": {
            "image_sha256": _sha256(PNG_BYTES),
            "dimensions": {"w": 1, "h": 1},
            "raw_text_sha256": _sha256(RAW_TEXT.encode()),
        },
        "reference": {
            "grounded_ocr_sha256": "a" * 64,
            "markdown_sha256": "b" * 64,
        },
        "assertions": [{"type": "quality_status", "value": "pass"}],
    }
    manifest_path = (tmp_path / "manifest.jsonl").resolve()
    manifest_path.write_text(
        json.dumps(case, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, input_root, image_path, raw_path


def _create_bundle(
    tmp_path: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, Path, Path, Path]:
    from sneeze.tpn.ocr_external_commands import create_external_plan_bundle

    manifest_path, input_root, image_path, raw_path = _write_fixture(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_CREDENTIAL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_CREDENTIAL)
    destination = (tmp_path / "plan-bundle").resolve()
    receipt = create_external_plan_bundle(
        profile_id=PROFILE_ID,
        manifest_path=manifest_path,
        input_root=input_root,
        dest_dir=destination,
        python_executable=Path(sys.executable).resolve(),
        lane_id="best",
        prompt="Convert the synthetic document to Markdown.",
        prompt_mode="image_and_raw_text",
        include_raw_text=True,
        repeats=2,
        max_api_attempts_per_case=1,
        max_input_tokens_per_case=4096,
        max_output_tokens=1024,
        budget_usd="100.000000",
        valid_hours=2,
        timeout_seconds=30,
        now=NOW,
    )
    return receipt, destination, image_path, raw_path


def test_plan_bundle_hashes_dimensions_and_contains_no_source_or_credentials(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.ocr_external import validate_external_upload_plan
    from sneeze.tpn.ocr_external_setup import external_provider_spec_sha256

    receipt, destination, image_path, raw_path = _create_bundle(
        tmp_path,
        monkeypatch=monkeypatch,
    )
    spec_path = Path(receipt["provider_spec_path"])
    plan_path = Path(receipt["upload_plan_path"])
    previews_path = Path(receipt["request_previews_path"])
    spec_payload = spec_path.read_bytes()
    plan_payload = plan_path.read_bytes()
    previews_payload = previews_path.read_bytes()
    spec = json.loads(spec_payload)
    plan = validate_external_upload_plan(json.loads(plan_payload))
    previews = [
        json.loads(line)
        for line in previews_payload.splitlines()
        if line.strip()
    ]

    assert sorted(path.name for path in destination.iterdir()) == [
        "provider-spec.json",
        "request-previews.jsonl",
        "upload-plan.json",
    ]
    assert receipt["provider_spec_file_sha256"] == _sha256(spec_payload)
    assert receipt["provider_spec_sha256"] == (
        external_provider_spec_sha256(spec)
    )
    assert receipt["upload_plan_file_sha256"] == _sha256(plan_payload)
    assert receipt["plan_sha256"] == plan["plan_sha256"]
    assert receipt["request_previews_sha256"] == _sha256(previews_payload)
    assert receipt["case_count"] == 1
    assert receipt["repeats"] == 2
    assert receipt["source_bytes_written"] is False
    assert receipt["credentials_read"] is False
    assert plan["cases"] == [
        {
            "case_id": CASE_ID,
            "image_sha256": _sha256(PNG_BYTES),
            "dimensions": {"w": 1, "h": 1},
            "media_type": "image/png",
            "raw_text_sha256": _sha256(RAW_TEXT.encode()),
        }
    ]
    assert plan["transmission"]["plan_contains_source_bytes"] is False
    assert len(previews) == 2
    assert all(
        preview["redactions"]["image"]
        == {
            "sha256": _sha256(PNG_BYTES),
            "dimensions": {"w": 1, "h": 1},
            "bytes_included": False,
        }
        for preview in previews
    )
    assert all(
        preview["redactions"]["raw_text"]
        == {
            "sha256": _sha256(RAW_TEXT.encode()),
            "bytes_included": False,
            "transmitted_for_lane": True,
        }
        for preview in previews
    )

    persisted = b"\n".join(
        path.read_bytes() for path in sorted(destination.iterdir())
    )
    forbidden = (
        PNG_BYTES,
        base64.b64encode(PNG_BYTES),
        RAW_TEXT.encode(),
        OPENAI_CREDENTIAL.encode(),
        ANTHROPIC_CREDENTIAL.encode(),
        str(image_path).encode(),
        str(raw_path).encode(),
    )
    assert all(value not in persisted for value in forbidden)


def test_plan_bundle_rejects_manifest_dimension_drift(tmp_path, monkeypatch):
    from sneeze.tpn.ocr_external_commands import (
        ExternalCommandError,
        create_external_plan_bundle,
    )

    manifest_path, input_root, _image_path, _raw_path = _write_fixture(
        tmp_path
    )
    case = json.loads(manifest_path.read_text(encoding="utf-8"))
    case["input"]["dimensions"] = {"w": 2, "h": 1}
    manifest_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_CREDENTIAL)

    with pytest.raises(
        ExternalCommandError,
        match="image dimensions do not match manifest",
    ):
        create_external_plan_bundle(
            profile_id=PROFILE_ID,
            manifest_path=manifest_path,
            input_root=input_root,
            dest_dir=(tmp_path / "invalid-plan").resolve(),
            python_executable=Path(sys.executable).resolve(),
            budget_usd="100.000000",
        )


def test_external_cli_commands_are_offline_and_approval_requires_retention(
    tmp_path,
    monkeypatch,
):
    import http.client

    from sneeze.command import CommandError

    from sneeze.tpn.commands import (
        OcrExternalApprove,
        OcrExternalList,
        OcrExternalPlan,
    )

    manifest_path, input_root, _image_path, _raw_path = _write_fixture(
        tmp_path
    )
    credential_reads = []
    network_calls = []
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_CREDENTIAL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_CREDENTIAL)
    environment_type = type(os.environ)
    original_getitem = environment_type.__getitem__

    def guarded_getitem(environment, key):
        normalized = os.fsdecode(key)
        if normalized in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}:
            credential_reads.append(normalized)
            raise AssertionError(f"credential was read: {normalized}")
        return original_getitem(environment, key)

    def blocked_network(*args, **kwargs):
        network_calls.append((args, kwargs))
        raise AssertionError("offline external command attempted network I/O")

    monkeypatch.setattr(environment_type, "__getitem__", guarded_getitem)
    monkeypatch.setattr(socket, "create_connection", blocked_network)
    monkeypatch.setattr(http.client, "HTTPSConnection", blocked_network)

    out = StringIO()
    list_command = OcrExternalList(None, out, None)
    list_command.format = "json"
    list_command.run()
    profiles = json.loads(out.getvalue())
    assert {profile["profile_id"] for profile in profiles} == {
        "openai-responses-gpt-5.6-sol",
        "anthropic-messages-claude-fable-5",
    }

    plan_dir = (tmp_path / "cli-plan").resolve()
    out = StringIO()
    plan_command = OcrExternalPlan(None, out, None)
    plan_command.profile = PROFILE_ID
    plan_command.manifest = str(manifest_path)
    plan_command.input_root = str(input_root)
    plan_command.dest_dir = str(plan_dir)
    plan_command.python_executable = str(Path(sys.executable).resolve())
    plan_command.lane = "best"
    plan_command.prompt = "Convert the synthetic document to Markdown."
    plan_command.prompt_mode = "image_and_raw_text"
    plan_command.include_raw_text = True
    plan_command.repeats = 1
    plan_command.max_api_attempts = 1
    plan_command.max_input_tokens = 4096
    plan_command.max_output_tokens = 1024
    plan_command.budget_usd = "100.000000"
    plan_command.valid_hours = 2
    plan_command.timeout_seconds = 30
    plan_command.run()
    plan_receipt = json.loads(out.getvalue())
    assert plan_receipt["kind"] == "external_plan_bundle"
    assert plan_receipt["credentials_read"] is False

    approval_path = (tmp_path / "approval.json").resolve()
    approval_command = OcrExternalApprove(None, StringIO(), None)
    approval_command.plan = plan_receipt["upload_plan_path"]
    approval_command.output = str(approval_path)
    approval_command.approved_by = "synthetic-reviewer@example.test"
    approval_command.accept_retention = False
    with pytest.raises(
        CommandError,
        match="retention policy was not accepted",
    ):
        approval_command.run()
    assert not approval_path.exists()

    out = StringIO()
    approval_command = OcrExternalApprove(None, out, None)
    approval_command.plan = plan_receipt["upload_plan_path"]
    approval_command.output = str(approval_path)
    approval_command.approved_by = "synthetic-reviewer@example.test"
    approval_command.accept_retention = True
    approval_command.valid_hours = 1
    approval_command.run()
    approval_receipt = json.loads(out.getvalue())
    assert approval_receipt["kind"] == "external_upload_approval"
    assert approval_receipt["plan_sha256"] == plan_receipt["plan_sha256"]
    assert approval_receipt["credentials_read"] is False
    assert credential_reads == []
    assert network_calls == []


def test_external_cli_import_and_help_never_access_network_or_credentials(
    tmp_path,
):
    credential_sentinel = (tmp_path / "credential-accessed").resolve()
    network_sentinel = (tmp_path / "network-accessed").resolve()
    code = r"""
import os
import sys
from pathlib import Path

credential_sentinel = Path(sys.argv[1])
network_sentinel = Path(sys.argv[2])
credential_names = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
environment_type = type(os.environ)
original_getitem = environment_type.__getitem__

def guarded_getitem(environment, key):
    normalized = os.fsdecode(key)
    if normalized in credential_names:
        credential_sentinel.write_text(normalized, encoding="utf-8")
        raise AssertionError("credential environment access: " + normalized)
    return original_getitem(environment, key)

environment_type.__getitem__ = guarded_getitem

def audit(event, _args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        network_sentinel.write_text(event, encoding="utf-8")
        raise AssertionError("network access: " + event)

sys.addaudithook(audit)

from sneeze import cli

instance = cli.CLI(
    program_name="sne",
    module_names=["sneeze", "sneeze.tpn"],
    introspect=True,
    auto_plugins=False,
)
names = (
    "ocr-external-list",
    "ocr-external-plan",
    "ocr-external-approve",
)
assert set(names) <= set(instance._commands_by_name)
for name in names:
    command_line = instance._commands_by_name[name]
    try:
        command_line.run(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("help did not exit for " + name)
"""
    environment = dict(os.environ)
    environment.update(
        {
            "OPENAI_API_KEY": OPENAI_CREDENTIAL,
            "ANTHROPIC_API_KEY": ANTHROPIC_CREDENTIAL,
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(credential_sentinel),
            str(network_sentinel),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    for name in (
        "ocr-external-list",
        "ocr-external-plan",
        "ocr-external-approve",
    ):
        assert f"Usage: sne {name}" in completed.stdout
    assert not credential_sentinel.exists()
    assert not network_sentinel.exists()
