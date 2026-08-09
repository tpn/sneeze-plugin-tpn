from __future__ import annotations

import base64
import hashlib
import json
import os
import shlex
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

VALID_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
    "x8AAwMCAO+Xk6cAAAAASUVORK5CYII="
)
MODEL_REVISION = "1" * 40
EXACT_OUTPUT = "Table 6\nreal page prose\n"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_sha256(value) -> str:
    return _sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _synthetic_credential_sentinel() -> str:
    return "-".join(("synthetic", "test", "credential", "sentinel"))


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False) + "\n" for value in values
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_test_git(path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=path,
        env={"PATH": os.defpath},
        check=True,
        capture_output=True,
    )


def _init_git_repository(path: Path) -> None:
    path.mkdir(parents=True)
    for args in (
        ("init", "--quiet"),
        ("config", "user.name", "OCR Benchmark Test"),
        ("config", "user.email", "ocr-benchmark@example.invalid"),
    ):
        _run_test_git(path, *args)
    tracked = path / "src" / "tracked.py"
    tracked.parent.mkdir()
    tracked.write_text("initial\n", encoding="utf-8")
    _run_test_git(path, "add", "src/tracked.py")
    _run_test_git(path, "commit", "--quiet", "-m", "initial")


def _case_record(stem: str = "paper/page-01") -> dict:
    raw_text = b"Table 6 real page prose"
    grounded_ocr = b"Table 6 grounded page prose"
    markdown = b"Reference Table 6\nreal page prose\n"
    return {
        "schema_version": 1,
        "case_id": "sample-page-01",
        "stem": stem,
        "input": {
            "image_sha256": _sha256(VALID_PNG_BYTES),
            "dimensions": {"w": 1, "h": 1},
            "raw_text_sha256": _sha256(raw_text),
        },
        "reference": {
            "grounded_ocr_sha256": _sha256(grounded_ocr),
            "markdown_sha256": _sha256(markdown),
        },
        "assertions": [
            {"type": "quality_status", "value": "pass"},
            {
                "type": "regex_occurrences",
                "pattern": r"\bTable\s+6\b",
                "ignore_case": True,
                "min": 1,
            },
            {
                "type": "metric_range",
                "path": (
                    "quality.table_completion.raw_token_multiset_recall"
                ),
                "min": 0.55,
            },
        ],
    }


def _write_case_tree(input_root: Path, case: dict) -> None:
    stem = input_root / case["stem"]
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".png").write_bytes(VALID_PNG_BYTES)
    stem.with_suffix(".txt").write_text(
        "Table 6 real page prose",
        encoding="utf-8",
    )
    stem.with_suffix(".ocr").write_text(
        "Table 6 grounded page prose",
        encoding="utf-8",
    )
    stem.with_suffix(".md").write_text(
        "Reference Table 6\nreal page prose\n",
        encoding="utf-8",
    )


def _write_manifest(tmp_path: Path) -> tuple[Path, Path, dict]:
    input_root = tmp_path / "corpus"
    input_root.mkdir()
    case = _case_record()
    _write_case_tree(input_root, case)
    manifest_path = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest_path, [case])
    return manifest_path, input_root, case


def _write_fake_provider(tmp_path: Path) -> Path:
    provider_path = tmp_path / "fake_provider.py"
    provider_path.write_text(
        textwrap.dedent(
            f"""
            import argparse
            import hashlib
            import json
            import sys
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--requests-jsonl", required=True)
            parser.add_argument("--responses-jsonl", required=True)
            parser.add_argument("--artifacts-dir", required=True)
            args = parser.parse_args()

            requests = [
                json.loads(line)
                for line in Path(args.requests_jsonl).read_text().splitlines()
                if line.strip()
            ]
            artifacts_dir = Path(args.artifacts_dir)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            responses = []
            for request in requests:
                artifact_name = request["attempt_id"] + ".json"
                artifact = artifacts_dir / artifact_name
                artifact_bytes = b'{{"wire": true}}\\n'
                artifact.write_bytes(artifact_bytes)
                responses.append(
                    {{
                        "schema_version": 1,
                        "attempt_id": request["attempt_id"],
                        "status": "completed",
                        "output": {{
                            "format": "markdown",
                            "text": {EXACT_OUTPUT!r},
                        }},
                        "effective_request": {{
                            "prompt": request["instruction"],
                            "input_artifacts": [
                                {{
                                    **request["input"]["image"],
                                    "transform": "original",
                                }}
                            ],
                        }},
                        "model": {{
                            "requested": "example/fake-ocr",
                            "revision": {MODEL_REVISION!r},
                            "resolved_snapshot": {MODEL_REVISION!r},
                        }},
                        "runtime": {{
                            "python": sys.version.split()[0],
                            "engine": "fake",
                            "engine_version": "1.0",
                            "packages": {{}},
                            "attention_backend": "not_applicable",
                        }},
                        "generation": {{
                            "params": request["settings"],
                            "input_tokens": 10,
                            "output_tokens": 8,
                            "finish_reason": "stop",
                            "stop_reason": None,
                            "eos": True,
                            "max_tokens": False,
                        }},
                        "usage": {{
                            "wall_seconds": 0.01,
                            "cost_usd": 0.0,
                        }},
                        "artifacts": [
                            {{
                                "relative_path": artifact_name,
                                "sha256": hashlib.sha256(
                                    artifact_bytes
                                ).hexdigest(),
                            }}
                        ],
                    }}
                )
            Path(args.responses_jsonl).write_text(
                "".join(json.dumps(row) + "\\n" for row in responses)
            )
            print("fake provider stdout")
            print("fake provider stderr", file=sys.stderr)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return provider_path


def _provider_record(provider_path: Path, *, egress: str = "none") -> dict:
    return {
        "schema_version": 1,
        "provider_id": "fake-ocr",
        "argv": [sys.executable, str(provider_path)],
        "cwd": str(provider_path.parent),
        "egress": egress,
        "model": {
            "requested": "example/fake-ocr",
            "revision": MODEL_REVISION,
        },
        "lanes": {
            "parity": {
                "prompt": "Transcribe this page literally into Markdown.",
                "settings": {"temperature": 0.0},
                "include_raw_text": False,
            },
            "best": {
                "prompt": "Use the supplied text as transcription evidence.",
                "settings": {"temperature": 0.0},
                "include_raw_text": True,
            },
        },
        "timeout_seconds": 30,
        "secret_env_names": [],
    }


def _write_provider_spec(
    tmp_path: Path,
    *,
    egress: str = "none",
) -> tuple[Path, Path]:
    provider_path = _write_fake_provider(tmp_path)
    spec_path = tmp_path / "provider.json"
    _write_json(spec_path, _provider_record(provider_path, egress=egress))
    return spec_path, provider_path


def _write_external_bundle(
    tmp_path: Path,
    manifest_path: Path,
    input_root: Path,
) -> tuple[Path, Path, Path]:
    from sneeze.tpn.ocr_external_commands import (
        create_external_approval,
        create_external_plan_bundle,
    )

    receipt = create_external_plan_bundle(
        profile_id="openai-responses-gpt-5.6-sol",
        manifest_path=manifest_path,
        input_root=input_root,
        dest_dir=(tmp_path / "external-plan").resolve(),
        python_executable=Path(sys.executable).resolve(),
        lane_id="parity",
        prompt="Transcribe this page literally into Markdown.",
        prompt_mode="parity",
        include_raw_text=False,
        repeats=1,
        max_api_attempts_per_case=1,
        max_input_tokens_per_case=4096,
        max_output_tokens=1024,
        budget_usd="100.000000",
        valid_hours=1,
    )
    plan_path = Path(receipt["upload_plan_path"])
    approval_path = plan_path.with_name("upload-approval.json")
    create_external_approval(
        plan_path=plan_path,
        output_path=approval_path,
        approved_by="test-user",
        accept_retention=True,
        valid_hours=1,
    )
    return Path(receipt["provider_spec_path"]), plan_path, approval_path


def _argument(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _write_fake_external_success(argv: list[str]) -> None:
    requests = _read_jsonl(Path(_argument(argv, "--requests-jsonl")))
    responses_path = Path(_argument(argv, "--responses-jsonl"))
    artifacts_dir = Path(_argument(argv, "--artifacts-dir"))
    responses = []
    for request in requests:
        artifact_name = request["attempt_id"] + ".json"
        artifact = artifacts_dir / artifact_name
        artifact_bytes = b'{"synthetic":true}\n'
        artifact.write_bytes(artifact_bytes)
        responses.append(
            {
                "schema_version": 1,
                "attempt_id": request["attempt_id"],
                "status": "completed",
                "output": {"format": "markdown", "text": EXACT_OUTPUT},
                "effective_request": {
                    "prompt": request["instruction"],
                    "input_artifacts": [
                        {
                            **request["input"]["image"],
                            "transform": "original",
                        }
                    ],
                },
                "model": {
                    "requested": "gpt-5.6-sol",
                    "revision": "provider_managed_model_id",
                    "resolved_snapshot": "gpt-5.6-sol",
                },
                "runtime": {
                    "python": sys.version.split()[0],
                    "engine": "openai-responses",
                    "engine_version": "v1",
                    "packages": {},
                    "attention_backend": "provider_managed",
                },
                "generation": {
                    "params": {"max_output_tokens": 1024},
                    "input_tokens": 10,
                    "output_tokens": 8,
                    "finish_reason": "stop",
                    "stop_reason": None,
                    "eos": True,
                    "max_tokens": False,
                },
                "usage": {"wall_seconds": 0.01, "cost_usd": 0.001},
                "artifacts": [
                    {
                        "relative_path": artifact_name,
                        "sha256": _sha256(artifact_bytes),
                    }
                ],
            }
        )
    _write_jsonl(responses_path, responses)


def test_prepare_run_validates_and_copies_exact_inputs(tmp_path):
    from sneeze.tpn.ocr_benchmark import load_manifest, prepare_run

    manifest_path, input_root, case = _write_manifest(tmp_path)
    manifest = load_manifest(str(manifest_path))
    dest_dir = tmp_path / "scratch-run"

    prepare_run(manifest, str(input_root), str(dest_dir))

    copied_stem = dest_dir / "inputs" / case["stem"]
    copied_image = copied_stem.with_suffix(".png")
    copied_text = copied_stem.with_suffix(".txt")
    assert copied_image.read_bytes() == VALID_PNG_BYTES
    assert copied_text.read_text(encoding="utf-8") == (
        "Table 6 real page prose"
    )
    assert (
        _sha256(copied_image.read_bytes()) == (case["input"]["image_sha256"])
    )
    assert (
        _sha256(copied_text.read_bytes())
        == (case["input"]["raw_text_sha256"])
    )


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        ("image_sha256", "(?i)sha-?256"),
        ("dimensions", "dimension"),
    ],
)
def test_prepare_run_rejects_fixture_drift(
    tmp_path,
    mutation,
    error_match,
):
    from sneeze.tpn.ocr_benchmark import load_manifest, prepare_run

    manifest_path, input_root, case = _write_manifest(tmp_path)
    if mutation == "image_sha256":
        case["input"]["image_sha256"] = "0" * 64
    else:
        case["input"]["dimensions"] = {"w": 2, "h": 1}
    _write_jsonl(manifest_path, [case])

    with pytest.raises(ValueError, match=error_match):
        prepare_run(
            load_manifest(str(manifest_path)),
            str(input_root),
            str(tmp_path / "scratch-run"),
        )


def test_prepare_run_rejects_unsafe_or_nonempty_destination(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.ocr_benchmark import load_manifest, prepare_run

    manifest_path, input_root, _case = _write_manifest(tmp_path)
    manifest = load_manifest(str(manifest_path))

    with pytest.raises(ValueError, match="input root"):
        prepare_run(
            manifest,
            str(input_root),
            str(input_root / "benchmark-output"),
        )

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "owned-by-user").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)(nonempty|must be empty)"):
        prepare_run(manifest, str(input_root), str(nonempty))
    assert (nonempty / "owned-by-user").read_text() == "preserve"

    protected = tmp_path / "canonical" / "papers"
    protected.mkdir(parents=True)
    config_home = tmp_path / "config"
    policy = config_home / "sneeze" / "tpn" / "ocr-protected-roots"
    policy.parent.mkdir(parents=True)
    policy.write_text(f"{protected}\n", encoding="utf-8")
    policy.chmod(0o600)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    protected_output = protected / "benchmark-output"
    with pytest.raises(ValueError, match="protected corpora"):
        prepare_run(manifest, str(input_root), str(protected_output))
    assert not protected_output.exists()


def test_load_provider_spec_keeps_argv_and_exact_revision(tmp_path):
    from sneeze.tpn.ocr_benchmark import load_provider_spec

    spec_path, provider_path = _write_provider_spec(tmp_path)

    spec = load_provider_spec(str(spec_path))

    assert spec["argv"] == [
        str(Path(sys.executable).resolve()),
        str(provider_path),
    ]
    assert spec["model"]["revision"] == MODEL_REVISION
    assert spec["egress"] == "none"

    record = _provider_record(provider_path)
    record["argv"][0] = "python"
    _write_json(spec_path, record)
    with pytest.raises(ValueError, match="absolute path"):
        load_provider_spec(str(spec_path))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("env", {"BAD=NAME": "value"}, "valid environment names"),
        ("env", {"VALID_NAME": "bad\x00value"}, "NUL-free"),
        (
            "secret_env_names",
            ["BAD=SECRET"],
            "valid environment names",
        ),
        (
            "secret_env_names",
            ["BAD\x00SECRET"],
            "valid environment names",
        ),
    ),
)
def test_load_provider_spec_rejects_invalid_environment_entries(
    tmp_path,
    field,
    value,
    message,
):
    from sneeze.tpn.ocr_benchmark import BenchmarkError, load_provider_spec

    spec_path, provider_path = _write_provider_spec(tmp_path)
    record = _provider_record(provider_path)
    record[field] = value
    _write_json(spec_path, record)

    with pytest.raises(BenchmarkError, match=message):
        load_provider_spec(spec_path)


@pytest.mark.parametrize("surface", ("argv", "env"))
def test_load_provider_spec_rejects_credential_shaped_values(
    tmp_path,
    surface,
):
    from sneeze.tpn.ocr_benchmark import BenchmarkError, load_provider_spec

    spec_path, provider_path = _write_provider_spec(tmp_path)
    record = _provider_record(provider_path)
    synthetic_value = "hf_" + "x" * 20
    if surface == "argv":
        record["argv"].append(synthetic_value)
    else:
        record["env"] = {"CONFIG_VALUE": synthetic_value}
    _write_json(spec_path, record)

    with pytest.raises(BenchmarkError, match="credential-like") as exc_info:
        load_provider_spec(spec_path)
    assert synthetic_value not in str(exc_info.value)


def test_benchmark_inputs_reject_duplicate_json_fields(tmp_path):
    from sneeze.tpn.ocr_benchmark import (
        BenchmarkError,
        load_manifest,
        load_provider_spec,
    )

    manifest_path, _input_root, _case = _write_manifest(tmp_path)
    manifest_payload = manifest_path.read_text(encoding="utf-8").replace(
        '"schema_version": 1',
        '"schema_version": 1, "schema_version": 1',
        1,
    )
    manifest_path.write_text(manifest_payload, encoding="utf-8")
    with pytest.raises(BenchmarkError, match="duplicate JSON field"):
        load_manifest(manifest_path)

    spec_path, _provider_path = _write_provider_spec(tmp_path)
    spec_payload = spec_path.read_text(encoding="utf-8").replace(
        '"schema_version": 1,',
        '"schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    spec_path.write_text(spec_payload, encoding="utf-8")
    with pytest.raises(BenchmarkError, match="duplicate JSON field"):
        load_provider_spec(spec_path)


def test_manifest_rejects_overflowed_exponent_assertion_bound(tmp_path):
    from sneeze.tpn.ocr_benchmark import BenchmarkError, load_manifest

    manifest_path, _input_root, _case = _write_manifest(tmp_path)
    payload = manifest_path.read_text(encoding="utf-8").replace(
        '"min": 1',
        '"min": 1e400',
        1,
    )
    manifest_path.write_text(payload, encoding="utf-8")

    with pytest.raises(BenchmarkError, match="non-finite JSON number"):
        load_manifest(manifest_path)


@pytest.mark.parametrize("missing_source", ("manifest", "provider spec"))
def test_run_benchmark_wraps_missing_source_errors(
    tmp_path,
    missing_source,
):
    from sneeze.tpn.ocr_benchmark import BenchmarkError, run_benchmark

    manifest_path, input_root, _case = _write_manifest(tmp_path)
    spec_path, _provider_path = _write_provider_spec(tmp_path)
    missing_path = tmp_path / "missing.json"

    kwargs = {
        "manifest_path": manifest_path,
        "input_root": input_root,
        "provider_spec_path": spec_path,
        "dest_dir": tmp_path / "run",
        "lanes": ("best",),
        "repeats": 1,
    }
    if missing_source == "manifest":
        kwargs["manifest_path"] = missing_path
    else:
        kwargs["provider_spec_path"] = missing_path

    with pytest.raises(BenchmarkError, match="Could not resolve"):
        run_benchmark(**kwargs)


def test_manifest_and_provider_reject_unsafe_output_components(tmp_path):
    from sneeze.tpn.ocr_benchmark import load_manifest, load_provider_spec

    manifest_path, _input_root, case = _write_manifest(tmp_path)
    case["case_id"] = "../../outside"
    _write_jsonl(manifest_path, [case])
    with pytest.raises(ValueError, match="safe path component"):
        load_manifest(str(manifest_path))

    spec_path, provider_path = _write_provider_spec(tmp_path)
    spec = _provider_record(provider_path)
    spec["lanes"] = {"../outside": spec["lanes"]["best"]}
    _write_json(spec_path, spec)
    with pytest.raises(ValueError, match="safe path component"):
        load_provider_spec(str(spec_path))


def test_git_provenance_rejects_unrelated_parent_repository(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_benchmark as benchmark

    outer = tmp_path / "outer-repository"
    _init_git_repository(outer)
    sensitive_name = "private-credential-do-not-record.txt"
    (outer / sensitive_name).write_text(
        "unrelated private material\n",
        encoding="utf-8",
    )
    export = outer / "source-export"
    export.mkdir()

    calls = []
    real_run = subprocess.run

    def recording_run(argv, *args, **kwargs):
        calls.append(tuple(argv))
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(benchmark.subprocess, "run", recording_run)
    context = benchmark._git_repository_context(export)
    metadata = benchmark._git_metadata(export, context=context)
    evidence_root = tmp_path / "evidence"
    diff = benchmark._snapshot_git_diff(evidence_root, context=context)

    assert len(calls) == 1
    assert calls[0][-2:] == ("rev-parse", "--show-toplevel")
    assert "status" not in calls[0]
    assert "diff" not in calls[0]
    assert metadata == {
        "available": False,
        "unavailable_reason": "git_toplevel_mismatch",
        "sha": None,
        "branch": None,
        "status": None,
    }
    assert diff["available"] is False
    assert diff["unavailable_reason"] == "git_toplevel_mismatch"
    assert diff["returncode"] is None
    diff_path = evidence_root / diff["evidence_path"]
    assert diff_path.read_bytes() == b""
    assert diff["size_bytes"] == 0
    recorded = json.dumps({"repository": metadata, "git_diff": diff})
    assert sensitive_name not in recorded


def test_git_provenance_handles_missing_git_without_leaking_error(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_benchmark as benchmark

    source_export = tmp_path / "source-export"
    source_export.mkdir()
    sensitive_detail = "private-parent-repository-name"

    def missing_git(*_args, **_kwargs):
        raise FileNotFoundError(sensitive_detail)

    monkeypatch.setattr(benchmark.subprocess, "run", missing_git)
    context = benchmark._git_repository_context(source_export)
    metadata = benchmark._git_metadata(source_export, context=context)
    evidence_root = tmp_path / "evidence"
    diff = benchmark._snapshot_git_diff(evidence_root, context=context)

    assert context["available"] is False
    assert context["unavailable_reason"] == "git_probe_launch_failed"
    assert metadata == {
        "available": False,
        "unavailable_reason": "git_probe_launch_failed",
        "sha": None,
        "branch": None,
        "status": None,
    }
    assert diff["available"] is False
    assert diff["unavailable_reason"] == "git_probe_launch_failed"
    assert diff["returncode"] is None
    recorded = json.dumps({"repository": metadata, "git_diff": diff})
    assert sensitive_detail not in recorded


def test_git_provenance_resolves_executable_before_sanitizing_path(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_benchmark as benchmark

    repository = tmp_path / "repository"
    _init_git_repository(repository)
    wrapper = tmp_path / "custom-git"
    wrapper.write_text('#!/bin/sh\nexec git "$@"\n', encoding="utf-8")
    wrapper.chmod(0o700)
    monkeypatch.setattr(
        benchmark.shutil,
        "which",
        lambda _name, path=None: str(wrapper),
    )
    calls = []
    real_run = subprocess.run

    def recording_run(argv, *args, **kwargs):
        calls.append((tuple(argv), dict(kwargs["env"])))
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(benchmark.subprocess, "run", recording_run)
    context = benchmark._git_repository_context(repository)
    metadata = benchmark._git_metadata(repository, context=context)

    assert metadata["available"] is True
    assert calls
    assert all(call[0][0] == str(wrapper.resolve()) for call in calls)
    assert all(call[1]["PATH"] == os.defpath for call in calls)
    assert str(wrapper) not in json.dumps(metadata)


def test_later_git_launch_error_is_a_detail_free_benchmark_error(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_benchmark as benchmark

    repository = tmp_path / "repository"
    repository.mkdir()
    sensitive_detail = "private-parent-repository-name"
    calls = 0

    def disappearing_git(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(
                args=["git"],
                returncode=0,
                stdout=f"{repository}\n",
                stderr="",
            )
        raise FileNotFoundError(sensitive_detail)

    monkeypatch.setattr(benchmark.subprocess, "run", disappearing_git)
    context = benchmark._git_repository_context(repository)

    with pytest.raises(
        benchmark.BenchmarkError,
        match="Git provenance probe could not be launched",
    ) as exc_info:
        benchmark._git_metadata(repository, context=context)

    assert sensitive_detail not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_git_provenance_avoids_worktree_status_and_diff(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_benchmark as benchmark

    repository = tmp_path / "exact-repository"
    _init_git_repository(repository)
    (repository / "src" / "tracked.py").write_text(
        "changed in public source\n",
        encoding="utf-8",
    )
    sensitive_name = "private-credential-do-not-record.txt"
    (repository / sensitive_name).write_text(
        "unrelated private material\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "ambient-control.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "ambient-work-tree"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-value")
    monkeypatch.setenv("PATH", "/ambient/bin")
    monkeypatch.setenv("LC_ALL", "ambient-locale")
    monkeypatch.setenv("TZ", "ambient-timezone")
    calls = []
    real_run = subprocess.run

    def recording_run(argv, *args, **kwargs):
        calls.append(tuple(argv))
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(benchmark.subprocess, "run", recording_run)

    context = benchmark._git_repository_context(repository)
    assert context["environment"] == {
        "PATH": os.defpath,
        "LC_ALL": "C",
        "TZ": "UTC",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    metadata = benchmark._git_metadata(repository, context=context)
    evidence_root = tmp_path / "evidence"
    diff = benchmark._snapshot_git_diff(evidence_root, context=context)

    assert metadata["available"] is True
    assert metadata["unavailable_reason"] is None
    assert metadata["sha"]
    assert isinstance(metadata["branch"], str)
    assert metadata["status"] is None
    assert diff["available"] is False
    assert diff["unavailable_reason"] == "working_tree_git_diff_disabled"
    assert diff["returncode"] is None
    assert (evidence_root / diff["evidence_path"]).read_bytes() == b""
    assert all("status" not in call for call in calls)
    assert all("diff" not in call for call in calls)
    assert sensitive_name not in json.dumps(
        {"repository": metadata, "git_diff": diff}
    )


def test_git_provenance_does_not_invoke_repository_clean_filter(tmp_path):
    import sneeze.tpn.ocr_benchmark as benchmark

    repository = tmp_path / "repository"
    _init_git_repository(repository)
    marker = tmp_path / "clean-filter-ran"
    filter_script = tmp_path / "clean-filter.py"
    filter_script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[1]).write_text('ran', encoding='utf-8')\n"
        "sys.stdout.buffer.write(sys.stdin.buffer.read())\n",
        encoding="utf-8",
    )
    command = " ".join(
        shlex.quote(value)
        for value in (sys.executable, str(filter_script), str(marker))
    )
    _run_test_git(repository, "config", "filter.unsafe.clean", command)
    (repository / ".gitattributes").write_text(
        "src/tracked.py filter=unsafe\n",
        encoding="utf-8",
    )
    tracked = repository / "src" / "tracked.py"
    tracked.write_text("dirty raw source\n", encoding="utf-8")

    context = benchmark._git_repository_context(repository)
    metadata = benchmark._git_metadata(repository, context=context)
    evidence_root = tmp_path / "evidence"
    diff = benchmark._snapshot_git_diff(evidence_root, context=context)
    source_tree = benchmark._snapshot_source_tree(repository, evidence_root)

    assert metadata["available"] is True
    assert metadata["status"] is None
    assert diff["available"] is False
    assert diff["unavailable_reason"] == "working_tree_git_diff_disabled"
    assert marker.exists() is False
    source_record = next(
        record
        for record in source_tree["files"]
        if record["path"] == "src/tracked.py"
    )
    assert (evidence_root / source_record["evidence_path"]).read_bytes() == (
        b"dirty raw source\n"
    )


def test_experiment_identity_binds_changed_git_head(tmp_path):
    import sneeze.tpn.ocr_benchmark as benchmark

    repository = tmp_path / "repository"
    _init_git_repository(repository)

    def identity(evidence_name: str) -> tuple[dict, str]:
        context = benchmark._git_repository_context(repository)
        metadata = benchmark._git_metadata(repository, context=context)
        evidence_root = tmp_path / evidence_name
        diff = benchmark._snapshot_git_diff(evidence_root, context=context)
        payload = benchmark._experiment_identity_payload(
            manifest_file_sha256="1" * 64,
            provider_spec_file_sha256="2" * 64,
            environment={},
            argv_files=[
                {
                    "argv_index": 0,
                    "path": "/usr/bin/python",
                    "size_bytes": 1,
                    "sha256": "3" * 64,
                }
            ],
            harness_files=[{"sha256": "4" * 64}],
            git_metadata=metadata,
            git_diff=diff,
            source_tree_sha256="5" * 64,
            external_identity=None,
        )
        return payload, _json_sha256(payload)

    first_payload, first_digest = identity("first-evidence")
    marker = repository / "release-marker.txt"
    marker.write_text("new commit outside source scope\n", encoding="utf-8")
    _run_test_git(repository, "add", "release-marker.txt")
    _run_test_git(repository, "commit", "--quiet", "-m", "new head")
    second_payload, second_digest = identity("second-evidence")

    assert first_payload["repository"]["sha"] != second_payload[
        "repository"
    ]["sha"]
    assert first_payload["git_diff_sha256"] == second_payload[
        "git_diff_sha256"
    ]
    assert first_payload["source_tree_sha256"] == second_payload[
        "source_tree_sha256"
    ]
    assert first_digest != second_digest


def test_source_tree_snapshots_and_binds_untracked_source(tmp_path):
    import sneeze.tpn.ocr_benchmark as benchmark

    repository = tmp_path / "repository"
    _init_git_repository(repository)
    untracked = repository / "src" / "untracked.py"
    untracked.write_text("first untracked source\n", encoding="utf-8")
    cache = repository / "src" / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"generated cache")

    first_root = tmp_path / "first-evidence"
    first_root.mkdir()
    first = benchmark._snapshot_source_tree(repository, first_root)
    records = {record["path"]: record for record in first["files"]}
    assert set(records) == {"src/tracked.py", "src/untracked.py"}
    untracked_record = records["src/untracked.py"]
    copied = first_root / untracked_record["evidence_path"]
    assert copied.read_bytes() == untracked.read_bytes()
    assert untracked_record["sha256"] == _sha256(untracked.read_bytes())
    assert first["aggregate_sha256"] == _json_sha256(
        {
            key: first[key]
            for key in (
                "schema_version",
                "scope_paths",
                "excluded_directory_names",
                "excluded_file_suffixes",
                "file_count",
                "files",
            )
        }
    )

    untracked.write_text("second untracked source\n", encoding="utf-8")
    second_root = tmp_path / "second-evidence"
    second_root.mkdir()
    second = benchmark._snapshot_source_tree(repository, second_root)
    assert first["aggregate_sha256"] != second["aggregate_sha256"]

    common = {
        "manifest_file_sha256": "1" * 64,
        "provider_spec_file_sha256": "2" * 64,
        "environment": {},
        "argv_files": [],
        "harness_files": [],
        "git_metadata": {
            "available": False,
            "unavailable_reason": "git_toplevel_unavailable",
            "sha": None,
        },
        "git_diff": {
            "available": False,
            "unavailable_reason": "git_toplevel_unavailable",
            "returncode": None,
            "sha256": "3" * 64,
        },
        "external_identity": None,
    }
    first_identity = benchmark._experiment_identity_payload(
        **common,
        source_tree_sha256=first["aggregate_sha256"],
    )
    second_identity = benchmark._experiment_identity_payload(
        **common,
        source_tree_sha256=second["aggregate_sha256"],
    )
    assert _json_sha256(first_identity) != _json_sha256(second_identity)


def test_source_tree_rejects_unsafe_symlink(tmp_path):
    import sneeze.tpn.ocr_benchmark as benchmark

    repository = tmp_path / "repository"
    _init_git_repository(repository)
    outside = tmp_path / "outside-private-source.py"
    outside.write_text("must not be followed\n", encoding="utf-8")
    (repository / "src" / "unsafe.py").symlink_to(outside)
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()

    with pytest.raises(benchmark.BenchmarkError, match="unsafe symlink"):
        benchmark._snapshot_source_tree(repository, evidence_root)


def test_run_benchmark_isolates_lanes_and_preserves_evidence(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_benchmark as benchmark

    manifest_path, input_root, case = _write_manifest(tmp_path)
    spec_path, _provider_path = _write_provider_spec(tmp_path)
    dest_dir = tmp_path / "benchmark-run"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["env"] = {
        "PYTHONHASHSEED": "7",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    }
    _write_json(spec_path, spec)
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FAKE_BACKEND")
    monkeypatch.setenv("TORCH_LOGS", "must-be-scrubbed")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-recorded")

    summary = benchmark.run_benchmark(
        manifest_path=str(manifest_path),
        input_root=str(input_root),
        provider_spec_path=str(spec_path),
        dest_dir=str(dest_dir),
        lanes=("parity", "best"),
        repeats=2,
        allow_external=False,
        out=None,
    )

    assert summary["all_pass"] is True
    run_record = json.loads((dest_dir / "run.json").read_text())
    assert run_record["repository"]["root"] == str(
        Path(__file__).resolve().parents[1]
    )
    assert run_record["repository"]["available"] is True
    assert run_record["repository"]["unavailable_reason"] is None
    assert run_record["repository"]["sha"]
    assert isinstance(run_record["repository"]["branch"], str)
    assert run_record["repository"]["status"] is None
    assert run_record["evidence"]["git_diff"]["available"] is False
    assert run_record["evidence"]["git_diff"]["unavailable_reason"] == (
        "working_tree_git_diff_disabled"
    )
    assert run_record["provider"]["environment"][
        "VLLM_ATTENTION_BACKEND"
    ] == "FLASH_ATTN"
    assert run_record["provider"]["environment"][
        "PYTHONHASHSEED"
    ] == "7"
    assert "TORCH_LOGS" not in run_record["provider"]["environment"]
    assert "OPENAI_API_KEY" not in run_record["provider"]["environment"]
    assert run_record["provider"]["argv_files"]
    assert run_record["harness"]["manifest_file_sha256"] == _sha256(
        manifest_path.read_bytes()
    )
    assert run_record["harness"]["provider_spec_file_sha256"] == _sha256(
        spec_path.read_bytes()
    )
    evidence_root = dest_dir / "evidence"
    assert (evidence_root / "manifest.jsonl").read_bytes() == (
        manifest_path.read_bytes()
    )
    assert (evidence_root / "provider-spec.json").read_bytes() == (
        spec_path.read_bytes()
    )
    source_tree = run_record["evidence"]["source_tree"]
    assert source_tree["file_count"] == len(source_tree["files"])
    assert source_tree["scope_paths"] == list(
        benchmark._GIT_PROVENANCE_PATHS
    )
    assert source_tree["aggregate_sha256"] == _json_sha256(
        {
            key: source_tree[key]
            for key in (
                "schema_version",
                "scope_paths",
                "excluded_directory_names",
                "excluded_file_suffixes",
                "file_count",
                "files",
            )
        }
    )
    inventory_path = evidence_root / source_tree["inventory_evidence_path"]
    assert _sha256(inventory_path.read_bytes()) == source_tree[
        "inventory_file_sha256"
    ]
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert inventory == {
        key: value
        for key, value in source_tree.items()
        if key not in {"inventory_evidence_path", "inventory_file_sha256"}
    }
    source_records = {
        record["path"]: record for record in source_tree["files"]
    }
    benchmark_record = source_records["src/sneeze/tpn/ocr_benchmark.py"]
    benchmark_copy = evidence_root / benchmark_record["evidence_path"]
    assert _sha256(benchmark_copy.read_bytes()) == benchmark_record["sha256"]
    experiment_identity = benchmark._experiment_identity_payload(
        manifest_file_sha256=run_record["harness"][
            "manifest_file_sha256"
        ],
        provider_spec_file_sha256=run_record["harness"][
            "provider_spec_file_sha256"
        ],
        environment=run_record["provider"]["environment"],
        argv_files=run_record["evidence"]["argv_files"],
        harness_files=run_record["evidence"]["harness_files"],
        git_metadata=run_record["repository"],
        git_diff=run_record["evidence"]["git_diff"],
        source_tree_sha256=source_tree["aggregate_sha256"],
        external_identity=None,
    )
    assert _json_sha256(experiment_identity) == run_record["harness"][
        "experiment_sha256"
    ]
    adapter_snapshot = next((evidence_root / "provider").iterdir())
    adapter_bytes = adapter_snapshot.read_bytes()
    _provider_path.write_text("mutated after run", encoding="utf-8")
    _write_json(spec_path, {"mutated": True})
    assert adapter_snapshot.read_bytes() == adapter_bytes
    assert json.loads((evidence_root / "provider-spec.json").read_text())[
        "provider_id"
    ] == "fake-ocr"
    results = _read_jsonl(dest_dir / "results.jsonl")
    assert len(results) == 4
    assert {(row["lane"], row["repeat"]) for row in results} == {
        ("parity", 1),
        ("parity", 2),
        ("best", 1),
        ("best", 2),
    }
    assert all(row["case_id"] == case["case_id"] for row in results)
    assert all(
        row["output_sha256"] == _sha256(EXACT_OUTPUT.encode())
        for row in results
    )
    assert all(row["metadata_complete"] is True for row in results)

    reference_hashes = set(case["reference"].values())
    for lane in ("parity", "best"):
        for repeat in (1, 2):
            repeat_dir = dest_dir / lane / f"repeat-{repeat:03d}"
            requests_path = repeat_dir / "requests.jsonl"
            responses_path = repeat_dir / "responses.jsonl"
            requests = _read_jsonl(requests_path)
            responses = _read_jsonl(responses_path)
            assert len(requests) == len(responses) == 1
            request = requests[0]
            assert request["input"]["image"]["path"].startswith(
                str(dest_dir / "inputs")
            )
            if lane == "parity":
                assert request["input"]["raw_text"] is None
            else:
                assert request["input"]["raw_text"]["text"] == (
                    "Table 6 real page prose"
                )
            request_json = json.dumps(request, sort_keys=True)
            assert "reference" not in request_json.casefold()
            assert all(
                digest not in request_json for digest in reference_hashes
            )
            assert responses[0]["output"]["text"] == EXACT_OUTPUT
            assert (repeat_dir / "stdout.log").read_text().strip() == (
                "fake provider stdout"
            )
            assert (repeat_dir / "stderr.log").read_text().strip() == (
                "fake provider stderr"
            )
            output_path = repeat_dir / "outputs" / f"{case['case_id']}.md"
            assert output_path.read_text(encoding="utf-8") == EXACT_OUTPUT


def test_run_benchmark_rejects_caller_bound_provider_spec_drift_before_exec(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_benchmark as benchmark

    manifest_path, input_root, _case = _write_manifest(tmp_path)
    spec_path, _provider_path = _write_provider_spec(tmp_path)
    expected = _sha256(spec_path.read_bytes())
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    def unexpected_popen(*_args, **_kwargs):
        raise AssertionError("provider must not execute after spec drift")

    monkeypatch.setattr(subprocess, "Popen", unexpected_popen)
    destination = tmp_path / "benchmark-run"
    with pytest.raises(
        benchmark.BenchmarkError,
        match="caller-bound SHA-256",
    ):
        benchmark.run_benchmark(
            manifest_path=manifest_path,
            input_root=input_root,
            provider_spec_path=spec_path,
            expected_provider_spec_file_sha256=expected,
            dest_dir=destination,
            lanes=("best",),
            repeats=1,
        )
    assert not destination.exists()


def test_run_benchmark_rejects_cross_user_writable_destination(tmp_path):
    from sneeze.tpn.ocr_benchmark import BenchmarkError, run_benchmark

    manifest_path, input_root, _case = _write_manifest(tmp_path)
    spec_path, _provider_path = _write_provider_spec(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o775)

    with pytest.raises(BenchmarkError, match="writable by group"):
        run_benchmark(
            manifest_path=manifest_path,
            input_root=input_root,
            provider_spec_path=spec_path,
            dest_dir=shared / "benchmark-run",
            lanes=("best",),
            repeats=1,
        )


@pytest.mark.parametrize(
    "value",
    (
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param(10**10000, id="huge-integer"),
    ),
)
def test_benchmark_usage_validator_rejects_nonfinite_numbers(value):
    from sneeze.tpn.ocr_benchmark import _is_nonnegative_number

    assert _is_nonnegative_number(value) is False


def test_experiment_identity_includes_provider_argv_file_bytes(tmp_path):
    from sneeze.tpn.ocr_benchmark import run_benchmark

    manifest_path, input_root, _case = _write_manifest(tmp_path)
    spec_path, provider_path = _write_provider_spec(tmp_path)

    first_dest = tmp_path / "first-run"
    run_benchmark(
        manifest_path=str(manifest_path),
        input_root=str(input_root),
        provider_spec_path=str(spec_path),
        dest_dir=str(first_dest),
        lanes=("best",),
        repeats=1,
    )
    first_run = json.loads((first_dest / "run.json").read_text())
    first_request = _read_jsonl(
        first_dest / "best" / "repeat-001" / "requests.jsonl"
    )[0]

    provider_path.write_text(
        provider_path.read_text(encoding="utf-8")
        + "\n# provider identity changed\n",
        encoding="utf-8",
    )
    second_dest = tmp_path / "second-run"
    run_benchmark(
        manifest_path=str(manifest_path),
        input_root=str(input_root),
        provider_spec_path=str(spec_path),
        dest_dir=str(second_dest),
        lanes=("best",),
        repeats=1,
    )
    second_run = json.loads((second_dest / "run.json").read_text())
    second_request = _read_jsonl(
        second_dest / "best" / "repeat-001" / "requests.jsonl"
    )[0]

    assert first_run["provider"]["argv_files"][-1]["sha256"] != (
        second_run["provider"]["argv_files"][-1]["sha256"]
    )
    assert first_run["harness"]["experiment_sha256"] != (
        second_run["harness"]["experiment_sha256"]
    )
    assert first_request["attempt_id"] != second_request["attempt_id"]


def test_manifest_and_spec_evidence_use_initial_source_bytes(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_benchmark as benchmark

    manifest_path, input_root, _case = _write_manifest(tmp_path)
    spec_path, _provider_path = _write_provider_spec(tmp_path)
    manifest_bytes = manifest_path.read_bytes()
    spec_bytes = spec_path.read_bytes()
    original_prepare = benchmark.prepare_run

    def mutate_sources_after_prepare(*args, **kwargs):
        prepared = original_prepare(*args, **kwargs)
        manifest_path.write_text('{"mutated": true}\n', encoding="utf-8")
        spec_path.write_text('{"mutated": true}\n', encoding="utf-8")
        return prepared

    monkeypatch.setattr(
        benchmark,
        "prepare_run",
        mutate_sources_after_prepare,
    )
    dest = tmp_path / "run"
    summary = benchmark.run_benchmark(
        manifest_path=manifest_path,
        input_root=input_root,
        provider_spec_path=spec_path,
        dest_dir=dest,
        lanes=("best",),
        repeats=1,
    )

    run = json.loads((dest / "run.json").read_text())
    assert summary["all_pass"] is True
    assert run["harness"]["manifest_file_sha256"] == _sha256(
        manifest_bytes
    )
    assert run["harness"]["provider_spec_file_sha256"] == _sha256(
        spec_bytes
    )
    assert (dest / "evidence" / "manifest.jsonl").read_bytes() == (
        manifest_bytes
    )
    assert (dest / "evidence" / "provider-spec.json").read_bytes() == (
        spec_bytes
    )


def test_provider_source_drift_stops_repeats(tmp_path):
    from sneeze.tpn.ocr_benchmark import BenchmarkError, run_benchmark

    manifest_path, input_root, _case = _write_manifest(tmp_path)
    spec_path, provider_path = _write_provider_spec(tmp_path)
    provider_path.write_text(
        provider_path.read_text(encoding="utf-8").replace(
            'print("fake provider stdout")',
            'print("fake provider stdout")\n'
            'Path(__file__).write_text('
            'Path(__file__).read_text() + "\\n# self-mutated\\n")',
        ),
        encoding="utf-8",
    )
    dest = tmp_path / "run"

    with pytest.raises(BenchmarkError, match="argv file drift"):
        run_benchmark(
            manifest_path=manifest_path,
            input_root=input_root,
            provider_spec_path=spec_path,
            dest_dir=dest,
            lanes=("best",),
            repeats=2,
        )

    assert (dest / "best" / "repeat-001" / "responses.jsonl").is_file()
    assert not (dest / "best" / "repeat-002").exists()
    drift = json.loads((dest / "provider-drift.json").read_text())
    assert drift["repeat"] == 1


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (
            '"artifacts": [',
            '"artifacts": None, "discarded_artifacts": [',
            "artifacts",
        ),
        (
            '"attempt_id": request["attempt_id"],',
            '"attempt_id": [],',
            "invalid response IDs",
        ),
        (
            '"format": "markdown",',
            '"format": None,',
            "output format",
        ),
        (
            "json.dumps(row)",
            "'{not-json'",
            "Invalid JSON",
        ),
        (
            '"wall_seconds": 0.01,',
            '"wall_seconds": float("inf"),',
            "non-finite JSON number",
        ),
        (
            '"wall_seconds": 0.01,',
            '"wall_seconds": 1e400,',
            "non-finite JSON number",
        ),
    ],
)
def test_malformed_provider_responses_become_durable_errors(
    tmp_path,
    old,
    new,
    expected,
):
    from sneeze.tpn.ocr_benchmark import run_benchmark

    manifest_path, input_root, _case = _write_manifest(tmp_path)
    spec_path, provider_path = _write_provider_spec(tmp_path)
    source = provider_path.read_text(encoding="utf-8")
    assert old in source
    provider_path.write_text(source.replace(old, new), encoding="utf-8")

    summary = run_benchmark(
        manifest_path=str(manifest_path),
        input_root=str(input_root),
        provider_spec_path=str(spec_path),
        dest_dir=str(tmp_path / "benchmark-run"),
        lanes=("best",),
        repeats=1,
        allow_external=False,
        out=None,
    )

    assert summary["all_pass"] is False
    result = _read_jsonl(tmp_path / "benchmark-run" / "results.jsonl")[0]
    evidence = json.dumps(result)
    assert expected.casefold() in evidence.casefold()


def test_provider_timeout_terminates_descendant_process(tmp_path):
    from sneeze.tpn.ocr_benchmark import run_benchmark

    manifest_path, input_root, _case = _write_manifest(tmp_path)
    provider_path = tmp_path / "timeout_provider.py"
    child_pid_path = tmp_path / "child.pid"
    provider_path.write_text(
        textwrap.dedent(
            f"""
            import subprocess
            import sys
            import time
            from pathlib import Path

            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"]
            )
            Path({str(child_pid_path)!r}).write_text(str(child.pid))
            time.sleep(60)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    spec = _provider_record(provider_path)
    spec["timeout_seconds"] = 1
    spec_path = tmp_path / "provider.json"
    _write_json(spec_path, spec)

    summary = run_benchmark(
        manifest_path=str(manifest_path),
        input_root=str(input_root),
        provider_spec_path=str(spec_path),
        dest_dir=str(tmp_path / "benchmark-run"),
        lanes=("best",),
        repeats=1,
        allow_external=False,
        out=None,
    )

    assert summary["all_pass"] is False
    result = _read_jsonl(tmp_path / "benchmark-run" / "results.jsonl")[0]
    assert result["subprocess"]["returncode"] == 124
    child_pid = int(child_pid_path.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and Path(f"/proc/{child_pid}").exists():
        time.sleep(0.05)
    assert not Path(f"/proc/{child_pid}").exists()


def test_provider_interrupt_terminates_descendant_and_reaps_provider(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn import ocr_benchmark

    provider_path = tmp_path / "interrupted_provider.py"
    process_ids_path = tmp_path / "process-ids"
    provider_path.write_text(
        textwrap.dedent(
            f"""
            import os
            import subprocess
            import sys
            import time
            from pathlib import Path

            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"]
            )
            Path({str(process_ids_path)!r}).write_text(
                f"{{os.getpid()}} {{child.pid}}"
            )
            time.sleep(60)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    original_wait = subprocess.Popen.wait
    interrupted = False

    def interrupt_first_wait(process, timeout=None):
        nonlocal interrupted
        if not interrupted:
            deadline = time.monotonic() + 2
            while (
                time.monotonic() < deadline and not process_ids_path.is_file()
            ):
                time.sleep(0.01)
            assert process_ids_path.is_file()
            interrupted = True
            raise KeyboardInterrupt
        return original_wait(process, timeout=timeout)

    monkeypatch.setattr(
        ocr_benchmark.subprocess.Popen,
        "wait",
        interrupt_first_wait,
    )
    with open(os.devnull, "wb") as output:
        with pytest.raises(KeyboardInterrupt):
            ocr_benchmark._run_provider_process(
                [sys.executable, str(provider_path)],
                cwd=str(tmp_path),
                environment=dict(os.environ),
                stdout_handle=output,
                stderr_handle=output,
                timeout_seconds=30,
            )

    provider_pid, child_pid = (
        int(value) for value in process_ids_path.read_text().split()
    )
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and Path(f"/proc/{child_pid}").exists():
        time.sleep(0.05)
    assert not Path(f"/proc/{provider_pid}").exists()
    assert not Path(f"/proc/{child_pid}").exists()


def test_evaluator_uses_verified_snapshot_after_source_mutation(tmp_path):
    from sneeze.tpn.ocr_benchmark import run_benchmark

    manifest_path, input_root, case = _write_manifest(tmp_path)
    spec_path, provider_path = _write_provider_spec(tmp_path)
    grounded_path = (input_root / case["stem"]).with_suffix(".ocr")
    original_grounded = grounded_path.read_bytes()
    source = provider_path.read_text(encoding="utf-8")
    marker = "args = parser.parse_args()\n"
    mutation = (
        marker
        + f"Path({str(grounded_path)!r}).write_text('mutated source')\n"
    )
    assert marker in source
    provider_path.write_text(
        source.replace(marker, mutation),
        encoding="utf-8",
    )
    dest_dir = tmp_path / "benchmark-run"

    summary = run_benchmark(
        manifest_path=str(manifest_path),
        input_root=str(input_root),
        provider_spec_path=str(spec_path),
        dest_dir=str(dest_dir),
        lanes=("best",),
        repeats=1,
        allow_external=False,
        out=None,
    )

    assert summary["all_pass"] is True
    assert grounded_path.read_text() == "mutated source"
    evaluator_grounded = (
        dest_dir / "evaluator" / f"{case['stem']}.ocr"
    )
    assert evaluator_grounded.read_bytes() == original_grounded
    assert _sha256(evaluator_grounded.read_bytes()) == (
        case["reference"]["grounded_ocr_sha256"]
    )


@pytest.mark.parametrize(
    ("old", "new", "gap"),
    [
        (
            f'"resolved_snapshot": {MODEL_REVISION!r}',
            '"resolved_snapshot": None',
            "model.resolved_snapshot",
        ),
        (
            '"output_tokens": 8',
            '"output_tokens": None',
            "generation.output_tokens",
        ),
        (
            '"finish_reason": "stop"',
            '"finish_reason": None',
            "generation.finish_reason",
        ),
        (
            '"finish_reason": "stop"',
            '"finish_reason": "length"',
            "generation.length_without_max_tokens",
        ),
        ('"eos": True', '"eos": None', "generation.eos"),
        (
            '"max_tokens": False',
            '"max_tokens": None',
            "generation.max_tokens",
        ),
    ],
)
def test_run_benchmark_rejects_placeholder_provenance(
    tmp_path,
    old,
    new,
    gap,
):
    from sneeze.tpn.ocr_benchmark import run_benchmark

    manifest_path, input_root, _case = _write_manifest(tmp_path)
    spec_path, provider_path = _write_provider_spec(tmp_path)
    source = provider_path.read_text(encoding="utf-8")
    assert old in source
    provider_path.write_text(source.replace(old, new), encoding="utf-8")

    summary = run_benchmark(
        manifest_path=str(manifest_path),
        input_root=str(input_root),
        provider_spec_path=str(spec_path),
        dest_dir=str(tmp_path / "benchmark-run"),
        lanes=("best",),
        repeats=1,
        allow_external=False,
        out=None,
    )

    assert summary["all_pass"] is False
    result = _read_jsonl(tmp_path / "benchmark-run" / "results.jsonl")[0]
    assert gap in result["provenance"]["missing_or_invalid"]


@pytest.mark.parametrize(
    "mutation",
    (
        "local_secret",
        "external_noncredential_secret",
        "external_implicit_raw_text",
        "credential_option",
        "credential_url",
        "credential_header",
        "proxy_override",
        "external_unallowlisted_env",
    ),
)
def test_provider_spec_rejects_unsafe_external_configuration(
    tmp_path,
    mutation,
):
    from sneeze.tpn.ocr_benchmark import load_provider_spec

    sentinel = _synthetic_credential_sentinel()
    provider_path = _write_fake_provider(tmp_path)
    egress = "none" if mutation == "local_secret" else "external"
    spec = _provider_record(provider_path, egress=egress)
    if mutation == "local_secret":
        spec["secret_env_names"] = ["TEST_API_TOKEN"]
    elif mutation == "external_noncredential_secret":
        spec["secret_env_names"] = ["BENCHMARK_VALUE"]
    elif mutation == "external_implicit_raw_text":
        del spec["lanes"]["best"]["include_raw_text"]
    elif mutation == "credential_option":
        spec["argv"].extend(["--api-key-file", sentinel])
    elif mutation == "credential_url":
        spec["argv"].append(
            f"https://example.invalid/v1?access_token={sentinel}"
        )
    elif mutation == "credential_header":
        spec["argv"].append(
            f"--header=Authorization: Bearer {sentinel}"
        )
    elif mutation == "proxy_override":
        spec["env"] = {
            "HTTPS_PROXY": f"https://user:{sentinel}@example.invalid"
        }
    elif mutation == "external_unallowlisted_env":
        spec["env"] = {"AWS_CONFIG_FILE": sentinel}
    spec_path = tmp_path / "provider.json"
    _write_json(spec_path, spec)

    with pytest.raises(ValueError) as caught:
        load_provider_spec(str(spec_path))

    assert sentinel not in str(caught.value)


def test_external_provider_receives_only_allowlisted_environment(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.ocr_benchmark import (
        _provider_environment,
        _recorded_provider_environment,
    )
    from sneeze.tpn.ocr_external_setup import build_external_provider_spec

    sentinel = _synthetic_credential_sentinel()
    spec = build_external_provider_spec(
        profile_id="openai-responses-gpt-5.6-sol",
        python_executable=Path(sys.executable).resolve(),
        lane_id="parity",
        prompt="Transcribe this page literally into Markdown.",
        prompt_mode="parity",
    )
    monkeypatch.setenv("OPENAI_API_KEY", sentinel)
    monkeypatch.setenv("HOME", "/sensitive/home")
    monkeypatch.setenv("HTTPS_PROXY", "https://sensitive-proxy.invalid")
    monkeypatch.setenv("PYTHONPATH", "/sensitive/python")
    monkeypatch.setenv("SSL_CERT_FILE", "/sensitive/custom-ca.pem")
    monkeypatch.setenv(
        "AWS_SHARED_CREDENTIALS_FILE",
        "/sensitive/cloud-config",
    )
    monkeypatch.setenv("RANDOM_BENCHMARK_VALUE", "sensitive-ambient")
    monkeypatch.setenv("PATH", "/usr/bin:/sensitive/path")
    environment = _provider_environment(spec)

    assert environment["OPENAI_API_KEY"] == sentinel
    assert environment["PATH"] == "/bin:/usr/bin"
    assert environment["PYTHONNOUSERSITE"] == "1"
    for name in (
        "HOME",
        "HTTPS_PROXY",
        "PYTHONPATH",
        "SSL_CERT_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "RANDOM_BENCHMARK_VALUE",
    ):
        assert name not in environment
    recorded = _recorded_provider_environment(environment, spec)
    assert recorded["PATH"] == "/bin:/usr/bin"
    assert "OPENAI_API_KEY" not in recorded


def test_external_provider_credential_leaks_are_redacted_and_fail(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.ocr_benchmark import run_benchmark

    sentinel = _synthetic_credential_sentinel()
    encoded_sentinel = base64.urlsafe_b64encode(
        sentinel.encode("utf-8")
    ).decode("ascii")
    manifest_path, input_root, _case = _write_manifest(tmp_path)
    spec_path, plan_path, approval_path = _write_external_bundle(
        tmp_path,
        manifest_path,
        input_root,
    )
    monkeypatch.setenv("OPENAI_API_KEY", sentinel)
    dest_dir = tmp_path / "benchmark-run"

    def leak_provider(
        argv,
        *,
        environment,
        stdout_handle,
        stderr_handle,
        **_kwargs,
    ):
        credential = environment["OPENAI_API_KEY"]
        encoded = base64.urlsafe_b64encode(
            credential.encode("utf-8")
        ).decode("ascii")
        stdout_handle.write(credential.encode("utf-8"))
        stderr_handle.write(encoded.encode("ascii"))
        requests = _read_jsonl(Path(_argument(argv, "--requests-jsonl")))
        responses_path = Path(_argument(argv, "--responses-jsonl"))
        artifacts_dir = Path(_argument(argv, "--artifacts-dir"))
        (artifacts_dir / f"{credential}.json").write_text(
            encoded,
            encoding="utf-8",
        )
        _write_jsonl(
            responses_path,
            [
                {
                    "schema_version": 1,
                    "attempt_id": request["attempt_id"],
                    "status": "completed",
                    "output": {"format": "markdown", "text": credential},
                    "credential_copy": encoded,
                }
                for request in requests
            ],
        )
        return 0, None

    monkeypatch.setattr(
        "sneeze.tpn.ocr_benchmark._run_provider_process",
        leak_provider,
    )

    summary = run_benchmark(
        manifest_path=str(manifest_path),
        input_root=str(input_root),
        provider_spec_path=str(spec_path),
        dest_dir=str(dest_dir),
        lanes=("parity",),
        repeats=1,
        allow_external=True,
        external_plan_path=plan_path,
        external_approval_path=approval_path,
        out=None,
    )

    assert summary["all_pass"] is False
    result = _read_jsonl(dest_dir / "results.jsonl")[0]
    assert result["passed"] is False
    assert "credential leak detected" in result["error"]
    assert sentinel not in result["error"]
    subprocess_record = result["subprocess"]
    assert subprocess_record["credential_leak_detected"] is True
    assert subprocess_record["credential_scan_complete"] is True
    assert all(
        subprocess_record["credential_redactions"][surface] > 0
        for surface in ("stdout", "stderr", "responses", "artifacts")
    )
    for path in dest_dir.rglob("*"):
        assert sentinel not in path.name
        if path == dest_dir / "evidence" / "harness" / "git-diff.patch":
            continue
        if path.is_file():
            payload = path.read_bytes()
            assert sentinel.encode("utf-8") not in payload
            assert encoded_sentinel.encode("ascii") not in payload


def test_external_provider_requires_bound_plan_and_separate_approval(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.ocr_benchmark import run_benchmark

    manifest_path, input_root, _case = _write_manifest(tmp_path)
    spec_path, plan_path, approval_path = _write_external_bundle(
        tmp_path,
        manifest_path,
        input_root,
    )

    with pytest.raises(ValueError, match="external"):
        run_benchmark(
            manifest_path=str(manifest_path),
            input_root=str(input_root),
            provider_spec_path=str(spec_path),
            dest_dir=str(tmp_path / "benchmark-run"),
            lanes=("parity",),
            repeats=1,
            allow_external=False,
            external_plan_path=plan_path,
            external_approval_path=approval_path,
            out=None,
        )

    with pytest.raises(ValueError, match="external-plan"):
        run_benchmark(
            manifest_path=str(manifest_path),
            input_root=str(input_root),
            provider_spec_path=str(spec_path),
            dest_dir=str(tmp_path / "benchmark-run-authorized"),
            lanes=("parity",),
            repeats=1,
            allow_external=True,
            out=None,
        )

    captured = {}

    def fake_provider(argv, *, environment, **_kwargs):
        captured["argv"] = argv
        captured["environment"] = environment
        _write_fake_external_success(argv)
        return 0, None

    monkeypatch.setattr(
        "sneeze.tpn.ocr_benchmark._run_provider_process",
        fake_provider,
    )
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        _synthetic_credential_sentinel(),
    )
    summary = run_benchmark(
        manifest_path=str(manifest_path),
        input_root=str(input_root),
        provider_spec_path=str(spec_path),
        dest_dir=str(tmp_path / "benchmark-run-authorized"),
        lanes=("parity",),
        repeats=1,
        allow_external=True,
        external_plan_path=plan_path,
        external_approval_path=approval_path,
        out=None,
    )
    assert summary["all_pass"] is True
    for argument in (
        "--external-provider-spec",
        "--external-plan",
        "--external-approval",
        "--external-budget-ledger",
    ):
        assert argument in captured["argv"]
    assert captured["environment"]["OPENAI_API_KEY"] == (
        _synthetic_credential_sentinel()
    )
    run_record = json.loads(
        (tmp_path / "benchmark-run-authorized" / "run.json").read_text()
    )
    assert run_record["provider"]["external_execution"]["plan_sha256"]
    assert "external_authorization" not in run_record["provider"]


def test_evaluate_assertions_catches_exact_regressions():
    from sneeze.tpn.ocr_benchmark import evaluate_assertions

    good_quality = {
        "status": "pass",
        "axis": {"terms": 2},
        "table_completion": {"raw_token_multiset_recall": 0.90},
    }
    assertions = [
        {"type": "quality_status", "value": "pass"},
        {
            "type": "metric_range",
            "path": "quality.axis.terms",
            "max": 63,
        },
        {
            "type": "literal_occurrences",
            "text": "To train the PSF fitting model",
            "casefold": True,
            "max": 0,
        },
        {
            "type": "regex_occurrences",
            "pattern": r"\bTable\s+8\b",
            "ignore_case": True,
            "min": 1,
        },
        {
            "type": "regex_occurrences",
            "pattern": r"\bTable\s+9\b",
            "ignore_case": True,
            "min": 1,
        },
        {
            "type": "metric_range",
            "path": ("quality.table_completion.raw_token_multiset_recall"),
            "min": 0.80,
        },
    ]

    outcomes = evaluate_assertions(
        "Table 8\nrows\nTable 9\nmore rows",
        good_quality,
        assertions,
    )
    assert all(outcome["passed"] for outcome in outcomes)

    axis_quality = {
        **good_quality,
        "axis": {"terms": 1103},
    }
    outcomes = evaluate_assertions(
        " ".join(str(value) for value in range(20, 2226, 2)),
        axis_quality,
        assertions[:2],
    )
    assert outcomes[1]["passed"] is False

    duev_text = "To train the PSF fitting model, invented text.\n\n" * 98
    outcomes = evaluate_assertions(
        duev_text,
        good_quality,
        [assertions[2]],
    )
    assert outcomes[0]["passed"] is False
    assert outcomes[0]["actual"] == 98

    supported_captions = "Figure 2 real caption.\n\n" * 98
    outcomes = evaluate_assertions(
        supported_captions,
        good_quality,
        [assertions[2]],
    )
    assert outcomes[0]["passed"] is True

    outcomes = evaluate_assertions(
        supported_captions,
        good_quality,
        [
            {
                "type": "metric_range",
                "path": (
                    "quality.repetition.max_unsupported_repeat."
                    "nonoverlap_occurrences"
                ),
                "max": 2,
            }
        ],
    )
    assert outcomes[0]["passed"] is True
    assert outcomes[0]["actual"] == 0

    truncated_quality = {
        **good_quality,
        "table_completion": {"raw_token_multiset_recall": 0.117},
    }
    outcomes = evaluate_assertions(
        "| 1 | 2 |\n| 3 | 4 |",
        truncated_quality,
        assertions[3:],
    )
    assert [outcome["passed"] for outcome in outcomes] == [
        False,
        False,
        False,
    ]


def test_run_benchmark_rejects_boolean_repeat_count(tmp_path):
    from sneeze.tpn.ocr_benchmark import BenchmarkError, run_benchmark

    with pytest.raises(BenchmarkError, match="positive integer"):
        run_benchmark(
            manifest_path=tmp_path / "missing-manifest.jsonl",
            input_root=tmp_path / "missing-inputs",
            provider_spec_path=tmp_path / "missing-provider.json",
            dest_dir=tmp_path / "missing-output",
            repeats=True,
        )


def test_assertions_can_be_scoped_to_provider_lanes(tmp_path):
    from sneeze.tpn.ocr_benchmark import evaluate_assertions, load_manifest

    quality = {
        "status": "pass",
        "table_completion": {"raw_token_multiset_recall": 0.4},
    }
    assertions = [
        {"type": "quality_status", "value": "pass"},
        {
            "type": "metric_range",
            "path": "quality.table_completion.raw_token_multiset_recall",
            "min": 0.8,
            "lanes": ["best", "parity"],
        },
    ]
    grounded = evaluate_assertions(
        "complete grounded OCR",
        quality,
        assertions,
        lane="grounded_ocr",
    )
    markdown = evaluate_assertions(
        "truncated markdown",
        quality,
        assertions,
        lane="best",
    )
    assert [item["passed"] for item in grounded] == [True]
    assert [item["passed"] for item in markdown] == [True, False]

    manifest_path, _input_root, case = _write_manifest(tmp_path)
    case["assertions"][0]["lanes"] = ["best", "best"]
    _write_jsonl(manifest_path, [case])
    with pytest.raises(ValueError, match="nonempty unique list"):
        load_manifest(manifest_path)

    case["assertions"][0]["lanes"] = [["best"]]
    _write_jsonl(manifest_path, [case])
    with pytest.raises(ValueError, match=r"lanes\[1\]"):
        load_manifest(manifest_path)


@pytest.mark.parametrize(
    "lanes",
    (pytest.param([], id="empty"), pytest.param([["best"]], id="nested")),
)
def test_benchmark_rejects_invalid_selected_lanes(tmp_path, lanes):
    from sneeze.tpn.ocr_benchmark import BenchmarkError, run_benchmark

    manifest_path, input_root, _case = _write_manifest(tmp_path)
    spec_path, _provider_path = _write_provider_spec(tmp_path)

    with pytest.raises(BenchmarkError, match="lanes"):
        run_benchmark(
            manifest_path=manifest_path,
            input_root=input_root,
            provider_spec_path=spec_path,
            dest_dir=tmp_path / "invalid-lanes-run",
            lanes=lanes,
            repeats=1,
        )


def test_benchmark_does_not_vacuously_pass_uncovered_lane(tmp_path):
    from sneeze.tpn.ocr_benchmark import run_benchmark

    manifest_path, input_root, case = _write_manifest(tmp_path)
    for assertion in case["assertions"]:
        assertion["lanes"] = ["parity"]
    _write_jsonl(manifest_path, [case])
    spec_path, _provider_path = _write_provider_spec(tmp_path)

    summary = run_benchmark(
        manifest_path=manifest_path,
        input_root=input_root,
        provider_spec_path=spec_path,
        dest_dir=tmp_path / "uncovered-lane-run",
        lanes=("best",),
        repeats=1,
    )

    assert summary["all_pass"] is False
    assert summary["passed_attempt_count"] == 0


def test_summarize_results_quantifies_nondeterminism():
    from sneeze.tpn.ocr_benchmark import summarize_results

    def result(repeat, output_hash, *, passed=True, complete=True):
        return {
            "case_id": "sample-page-01",
            "lane": "parity",
            "repeat": repeat,
            "status": "completed",
            "metadata_complete": complete,
            "output_sha256": output_hash,
            "normalized_output_sha256": output_hash,
            "assertions": [{"type": "quality_status", "passed": passed}],
        }

    summary = summarize_results(
        [
            result(1, "a" * 64),
            result(2, "a" * 64),
            result(3, "b" * 64, passed=False),
        ]
    )

    assert summary["all_pass"] is False
    assert summary["attempt_count"] == 3
    assert summary["passed_attempt_count"] == 2
    group = summary["groups"][0]
    assert group["case_id"] == "sample-page-01"
    assert group["lane"] == "parity"
    assert group["repeat_count"] == 3
    assert group["pass_count"] == 2
    assert group["all_pass"] is False
    assert group["unique_output_hash_count"] == 2
    assert group["output_hash_counts"] == {
        "a" * 64: 2,
        "b" * 64: 1,
    }
    assert group["modal_output_fraction"] == pytest.approx(2 / 3)
    assert group["termination"]["eos_count"] == 0
    assert group["termination"]["max_tokens_count"] == 0
    assert group["ranges"]["output_tokens"] == {
        "min": None,
        "max": None,
    }
    assert group["ranges"]["unsupported_repeat_occurrences"] == {
        "min": 0,
        "max": 0,
    }

    incomplete = summarize_results([result(1, "a" * 64, complete=False)])
    assert incomplete["all_pass"] is False
