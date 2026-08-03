from __future__ import annotations

import base64
import hashlib
import json
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


def test_prepare_run_rejects_unsafe_or_nonempty_destination(tmp_path):
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


def test_run_benchmark_isolates_lanes_and_preserves_evidence(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.ocr_benchmark import run_benchmark

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

    summary = run_benchmark(
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
    assert run_record["repository"]["sha"]
    assert isinstance(run_record["repository"]["branch"], str)
    assert run_record["repository"]["status"]
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


def test_external_provider_requires_explicit_authorization(tmp_path):
    from sneeze.tpn.ocr_benchmark import run_benchmark

    manifest_path, input_root, _case = _write_manifest(tmp_path)
    spec_path, provider_path = _write_provider_spec(
        tmp_path,
        egress="external",
    )
    marker_path = tmp_path / "provider-started"
    original = provider_path.read_text(encoding="utf-8")
    provider_path.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker_path)!r}).write_text('started')\n" + original,
        encoding="utf-8",
    )

    with pytest.raises(
        (PermissionError, RuntimeError, ValueError),
        match="external",
    ):
        run_benchmark(
            manifest_path=str(manifest_path),
            input_root=str(input_root),
            provider_spec_path=str(spec_path),
            dest_dir=str(tmp_path / "benchmark-run"),
            lanes=("parity",),
            repeats=1,
            allow_external=False,
            out=None,
        )

    assert not marker_path.exists()

    with pytest.raises(ValueError, match="external_authorization"):
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
    assert not marker_path.exists()

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["external_authorization"] = {
        "approved_by": "test-user",
        "approved_at": "2026-08-03T00:00:00Z",
        "scope": "synthetic one-pixel fixture",
        "retention": "test-only",
        "budget_usd": 0.0,
    }
    _write_json(spec_path, spec)
    summary = run_benchmark(
        manifest_path=str(manifest_path),
        input_root=str(input_root),
        provider_spec_path=str(spec_path),
        dest_dir=str(tmp_path / "benchmark-run-authorized"),
        lanes=("parity",),
        repeats=1,
        allow_external=True,
        out=None,
    )
    assert summary["all_pass"] is True
    assert marker_path.read_text() == "started"


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
