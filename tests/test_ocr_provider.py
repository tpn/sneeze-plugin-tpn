import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
    "nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)
MODEL_ID = "example/local-ocr"
MODEL_REVISION = "a" * 40
PROMPT = "Use the native staged parser and return Markdown."
SETTINGS = {"batch_size": 1, "temperature": 0.0}


FAKE_PROVIDER = r"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MODEL_ID = "example/local-ocr"
MODEL_REVISION = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


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
artifacts = Path(args.artifacts_dir)
artifacts.mkdir(parents=True, exist_ok=True)
mode = os.environ.get("FAKE_PROVIDER_MODE", "valid")
print(f"request_count={len(requests)}", flush=True)

if mode == "timeout":
    marker = artifacts / "grandchild-ticks.txt"
    ready = artifacts / "grandchild-ready.txt"
    child_code = (
        "import pathlib,signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(" + repr(str(ready)) + ").write_text('ready'); "
        "p=pathlib.Path(" + repr(str(marker)) + "); "
        "[(p.open('a').write('x'), time.sleep(0.02)) for _ in range(3000)]"
    )
    child = subprocess.Popen([sys.executable, "-c", child_code])
    deadline = time.monotonic() + 0.8
    while not ready.is_file() and time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError("timeout grandchild exited before ready")
        time.sleep(0.01)
    if not ready.is_file():
        raise RuntimeError("timeout grandchild did not become ready")
    time.sleep(60)

responses = []
for index, request in enumerate(requests):
    if mode == "missing" and index == len(requests) - 1:
        continue
    attempt_id = request["attempt_id"]
    artifact_name = f"{attempt_id}-raw.json"
    artifact_path = artifacts / artifact_name
    artifact_bytes = (
        json.dumps({"attempt_id": attempt_id}, sort_keys=True) + "\n"
    ).encode()
    artifact_path.write_bytes(artifact_bytes)
    artifact = {
        "relative_path": artifact_name,
        "sha256": sha256_bytes(artifact_bytes),
    }
    if mode == "outside_artifact" and index == 0:
        outside = artifacts.parent / "outside.txt"
        outside.write_text("outside")
        artifact = {
            "relative_path": "../outside.txt",
            "sha256": sha256_bytes(b"outside"),
        }
    if mode == "bad_artifact_hash" and index == 0:
        artifact["sha256"] = "0" * 64
    response = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "status": "completed",
        "output": {
            "format": "markdown",
            "text": f"# {request['page_id']}\n\nComplete page text.\n",
        },
        "effective_request": {
            "prompt": "native-wrapper\n" + request["instruction"],
            "input_artifacts": [
                {
                    **request["input"]["image"],
                    "transform": "original",
                }
            ],
        },
        "model": {
            "requested": MODEL_ID,
            "revision": MODEL_REVISION,
            "resolved_snapshot": "/cache/snapshots/" + MODEL_REVISION,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "engine": "fake-transformers",
            "engine_version": "1.2.3",
            "packages": {"fake-engine": "1.2.3"},
            "attention_backend": "not_applicable",
        },
        "generation": {
            "params": request["settings"],
            "input_tokens": 12,
            "output_tokens": 8,
            "finish_reason": "stop",
            "stop_reason": "eos_token",
            "eos": True,
            "max_tokens": False,
        },
        "usage": {"wall_seconds": 0.01, "cost_usd": 0.0},
        "artifacts": [artifact],
    }
    if mode == "model_mismatch" and index == 0:
        response["model"]["revision"] = "b" * 40
    if mode == "null_backend" and index == 0:
        response["runtime"]["attention_backend"] = None
    if mode == "null_input_tokens" and index == 0:
        response["generation"]["input_tokens"] = None
    if mode == "provider_error" and index == 0:
        response = {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "status": "error",
            "error": {"type": "RuntimeError", "message": "failed page"},
        }
    responses.append(response)

responses_path = Path(args.responses_jsonl)
with responses_path.open("w") as handle:
    if mode == "malformed":
        handle.write("{not-json}\n")
        handle.flush()
    for response in responses:
        handle.write(json.dumps(response, sort_keys=True) + "\n")
        handle.flush()
    if mode == "duplicate" and responses:
        handle.write(json.dumps(responses[0], sort_keys=True) + "\n")
        handle.flush()

if mode == "nonzero":
    raise SystemExit(3)
"""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_provider_spec(tmp_path: Path, *, mode: str = "valid") -> Path:
    adapter = tmp_path / "fake_provider.py"
    adapter.write_text(FAKE_PROVIDER, encoding="utf-8")
    spec = {
        "schema_version": 1,
        "provider_id": "fake-local-provider",
        "argv": [sys.executable, str(adapter)],
        "cwd": str(tmp_path),
        "egress": "none",
        "model": {
            "requested": MODEL_ID,
            "revision": MODEL_REVISION,
        },
        "lanes": {
            "best": {
                "prompt": PROMPT,
                "include_raw_text": False,
                "settings": SETTINGS,
            }
        },
        "timeout_seconds": 10,
        "secret_env_names": [],
        "env": {"FAKE_PROVIDER_MODE": mode},
    }
    spec_path = tmp_path / "provider.json"
    _write_json(spec_path, spec)
    return spec_path


def _write_page(tmp_path: Path, page_id: str) -> dict:
    image_path = tmp_path / f"{page_id}.png"
    image_path.write_bytes(PNG_BYTES)
    return {
        "page_id": page_id,
        "image_path": str(image_path),
        "image_sha256": _sha256(PNG_BYTES),
        "image_dims": {"w": 1, "h": 1},
        "instruction": PROMPT,
        "settings": SETTINGS,
    }


def test_load_local_provider_spec_is_strictly_local(tmp_path):
    from sneeze.tpn.ocr_provider import (
        ProviderError,
        load_local_provider_spec,
    )

    spec_path = _write_provider_spec(tmp_path)
    spec = load_local_provider_spec(spec_path)

    assert spec["provider_id"] == "fake-local-provider"
    assert spec["model"]["revision"] == MODEL_REVISION

    for field, value, message in (
        ("egress", "external", "local-only"),
        ("secret_env_names", ["OPENAI_API_KEY"], "cannot inherit secret"),
    ):
        modified = dict(spec)
        modified[field] = value
        bad_path = tmp_path / f"bad-{field}.json"
        _write_json(bad_path, modified)
        with pytest.raises(ProviderError, match=message):
            load_local_provider_spec(bad_path)

    modified = json.loads(json.dumps(spec))
    modified["model"]["revision"] = "main"
    bad_path = tmp_path / "mutable-revision.json"
    _write_json(bad_path, modified)
    with pytest.raises(ProviderError, match="immutable"):
        load_local_provider_spec(bad_path)


@pytest.mark.parametrize(
    "timeout",
    (
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ),
)
def test_local_provider_spec_rejects_nonfinite_timeout(tmp_path, timeout):
    from sneeze.tpn.ocr_provider import (
        ProviderError,
        load_local_provider_spec,
    )

    spec_path = _write_provider_spec(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["timeout_seconds"] = timeout
    _write_json(spec_path, spec)

    with pytest.raises(ProviderError, match="finite"):
        load_local_provider_spec(spec_path)


@pytest.mark.parametrize(
    ("argument", "value"),
    tuple(
        pytest.param(argument, value, id=f"{argument}-{label}")
        for argument in ("timeout_seconds", "terminate_grace_seconds")
        for label, value in (
            ("nan", float("nan")),
            ("positive-infinity", float("inf")),
            ("negative-infinity", float("-inf")),
        )
    ),
)
def test_local_provider_request_rejects_nonfinite_durations(
    tmp_path,
    argument,
    value,
):
    from sneeze.tpn.ocr_provider import (
        ProviderError,
        run_local_provider_batch,
    )

    run_dir = tmp_path / "run"
    kwargs = {argument: value}
    with pytest.raises(ProviderError, match=rf"{argument}.*finite"):
        run_local_provider_batch(
            spec_path=_write_provider_spec(tmp_path),
            lane="best",
            run_dir=run_dir,
            pages=[_write_page(tmp_path, "page-01")],
            **kwargs,
        )

    assert not run_dir.exists()


@pytest.mark.parametrize(
    ("argument", "value"),
    tuple(
        pytest.param(argument, value, id=f"{argument}-{label}")
        for argument in ("timeout_seconds", "terminate_grace_seconds")
        for label, value in (
            ("nan", float("nan")),
            ("positive-infinity", float("inf")),
            ("negative-infinity", float("-inf")),
        )
    ),
)
def test_provider_process_boundary_rejects_nonfinite_durations(
    tmp_path,
    argument,
    value,
):
    from sneeze.tpn.ocr_provider import ProviderError, _run_provider_process

    kwargs = {
        "timeout_seconds": 1.0,
        "terminate_grace_seconds": 0.0,
    }
    kwargs[argument] = value
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"

    with pytest.raises(ProviderError, match=rf"{argument}.*finite"):
        _run_provider_process(
            argv=[sys.executable, "-c", "raise SystemExit(0)"],
            cwd=str(tmp_path),
            environment={},
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            **kwargs,
        )

    assert not stdout_path.exists()
    assert not stderr_path.exists()


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        pytest.param(
            'float("nan")',
            "non-finite JSON number",
            id="nan",
        ),
        pytest.param(
            'float("inf")',
            "non-finite JSON number",
            id="infinity",
        ),
        pytest.param(
            "1e400",
            "non-finite JSON number",
            id="overflowed-exponent",
        ),
    ),
)
def test_local_provider_rejects_nonfinite_usage_json(
    tmp_path,
    expression,
    expected,
):
    from sneeze.tpn.ocr_provider import run_local_provider_batch

    spec_path = _write_provider_spec(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    adapter = Path(spec["argv"][1])
    source = adapter.read_text(encoding="utf-8")
    adapter.write_text(
        source.replace(
            '"wall_seconds": 0.01,',
            f'"wall_seconds": {expression},',
        ),
        encoding="utf-8",
    )

    batch = run_local_provider_batch(
        spec_path=spec_path,
        lane="best",
        run_dir=tmp_path / "run",
        pages=[_write_page(tmp_path, "page-01")],
    )

    assert batch["ok"] is False
    assert expected in " ".join(batch["protocol_errors"])


@pytest.mark.parametrize(
    "value",
    (
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param(10**10000, id="huge-integer"),
    ),
)
def test_provider_usage_validator_rejects_nonfinite_numbers(value):
    from sneeze.tpn.ocr_provider import _is_nonnegative_number

    assert _is_nonnegative_number(value) is False


def test_load_local_provider_spec_normalizes_tilde_paths(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.ocr_provider import (
        load_local_provider_spec,
        run_local_provider_batch,
    )

    fake_home = tmp_path / "home"
    provider_cwd = fake_home / "provider"
    provider_cwd.mkdir(parents=True)
    executable = fake_home / "python"
    executable.symlink_to(Path(sys.executable).resolve())
    spec_path = _write_provider_spec(tmp_path)
    spec = json.loads(spec_path.read_text())
    spec["argv"][0] = "~/python"
    spec["cwd"] = "~/provider"
    _write_json(spec_path, spec)
    monkeypatch.setenv("HOME", str(fake_home))

    loaded = load_local_provider_spec(spec_path)

    assert loaded["argv"][0] == str(Path(sys.executable).resolve())
    assert loaded["cwd"] == str(provider_cwd.resolve())

    page = _write_page(tmp_path, "page-01")
    batch = run_local_provider_batch(
        spec_path=spec_path,
        lane="best",
        run_dir=tmp_path / "run",
        pages=[page],
    )
    assert batch["ok"] is True
    normalized = json.loads(
        (tmp_path / "run" / "provider-spec.normalized.json").read_text()
    )
    assert normalized["argv"][0] == str(Path(sys.executable).resolve())
    assert normalized["cwd"] == str(provider_cwd.resolve())


def test_load_local_provider_spec_rejects_nul_argv(tmp_path):
    from sneeze.tpn.ocr_provider import (
        ProviderError,
        load_local_provider_spec,
    )

    spec_path = _write_provider_spec(tmp_path)
    spec = json.loads(spec_path.read_text())
    spec["argv"].append("embedded\x00nul")
    _write_json(spec_path, spec)

    with pytest.raises(ProviderError, match="NUL byte"):
        load_local_provider_spec(spec_path)


def test_local_provider_spec_rejects_duplicate_json_fields(tmp_path):
    from sneeze.tpn.ocr_provider import (
        ProviderError,
        load_local_provider_spec,
    )

    spec_path = _write_provider_spec(tmp_path)
    payload = spec_path.read_text(encoding="utf-8").replace(
        '"provider_id": "fake-local-provider",',
        '"provider_id": "shadow",\n  "provider_id": "fake-local-provider",',
        1,
    )
    spec_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ProviderError, match="duplicate JSON field"):
        load_local_provider_spec(spec_path)


@pytest.mark.parametrize(
    "surface",
    ("argv", "model", "prompt", "settings", "unexpected"),
)
def test_local_provider_spec_rejects_credential_shaped_persisted_values(
    tmp_path,
    surface,
):
    from sneeze.tpn.ocr_provider import (
        ProviderError,
        run_local_provider_batch,
    )

    spec_path = _write_provider_spec(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    synthetic_value = "hf_" + "x" * 20
    if surface == "argv":
        spec["argv"].append(synthetic_value)
    elif surface == "model":
        spec["model"]["requested"] = synthetic_value
    elif surface == "prompt":
        spec["lanes"]["best"]["prompt"] = synthetic_value
    elif surface == "settings":
        spec["lanes"]["best"]["settings"]["safe_value"] = synthetic_value
    else:
        spec["unexpected_value"] = synthetic_value
    _write_json(spec_path, spec)
    run_dir = tmp_path / "public-evidence"

    with pytest.raises(
        ProviderError,
        match="credential-like|unexpected fields",
    ) as exc_info:
        run_local_provider_batch(
            spec_path=spec_path,
            lane="best",
            run_dir=run_dir,
            pages=[_write_page(tmp_path, "page-01")],
        )

    assert synthetic_value not in str(exc_info.value)
    assert not run_dir.exists()


def test_local_provider_response_rejects_duplicate_json_fields(tmp_path):
    from sneeze.tpn.ocr_provider import _read_response_jsonl

    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        '{"status":"completed","status":"error"}\n',
        encoding="utf-8",
    )

    values, errors = _read_response_jsonl(responses)

    assert values == []
    assert len(errors) == 1
    assert "duplicate JSON field" in errors[0]


def test_local_provider_environment_scrubs_secrets_and_inference_controls(
    tmp_path,
):
    from sneeze.tpn.ocr_provider import (
        build_local_provider_environment,
        load_local_provider_spec,
    )

    spec = load_local_provider_spec(_write_provider_spec(tmp_path))
    environment = build_local_provider_environment(
        spec,
        base_environment={
            "PATH": "/usr/bin",
            "HOME": "/home/example",
            "OPENAI_API_KEY": "secret",
            "SESSION_TOKEN": "secret",
            "AWS_PROFILE": "production",
            "AWS_CONFIG_FILE": "/home/example/.aws/config",
            "AZURE_CONFIG_DIR": "/home/example/.azure",
            "CLOUDSDK_CONFIG": "/home/example/.config/gcloud",
            "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/google.json",
            "DATABASE_URL": "postgresql://ambient.invalid/database",
            "SENTRY_DSN": "https://ambient.invalid/1",
            "PYTHONPATH": "/parent/source",
            "VIRTUAL_ENV": "/parent/env",
            "CUDA_VISIBLE_DEVICES": "7",
            "PYTHONHASHSEED": "random",
            "VLLM_ATTENTION_BACKEND": "ambient-backend",
            "SAFE_VALUE": "not-allowlisted",
            "LANG": "C.UTF-8",
            "TZ": "UTC",
        },
    )

    assert environment["PATH"] == "/usr/bin:/bin"
    assert environment["LANG"] == "C.UTF-8"
    assert environment["TZ"] == "UTC"
    assert environment["FAKE_PROVIDER_MODE"] == "valid"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    for name in (
        "HOME",
        "OPENAI_API_KEY",
        "SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_CONFIG_FILE",
        "AZURE_CONFIG_DIR",
        "CLOUDSDK_CONFIG",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "DATABASE_URL",
        "SENTRY_DSN",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CUDA_VISIBLE_DEVICES",
        "PYTHONHASHSEED",
        "VLLM_ATTENTION_BACKEND",
        "SAFE_VALUE",
    ):
        assert name not in environment


@pytest.mark.parametrize(
    ("name", "value"),
    (
        (
            "SAFE_OVERRIDE",
            "https://example:not-a-real-secret@example.invalid/model",
        ),
        ("SAFE_OVERRIDE", "Bearer not-a-real-token-value"),
        ("SAFE_OVERRIDE", "token=not-a-real-token-value"),
        ("SAFE_OVERRIDE", "sk-ant-not-a-real-token-value"),
        ("SAFE_OVERRIDE", "hf_" + "x" * 20),
        ("SAFE_OVERRIDE", "github_pat_" + "x" * 20),
        (
            "SAFE_OVERRIDE",
            "postgresql://example:not-a-real-secret@db.invalid/database",
        ),
        ("DATABASE_URL", "sqlite:////tmp/example.db"),
        (
            "SENTRY_DSN",
            "https://public-placeholder@o0.ingest.sentry.invalid/1",
        ),
    ),
)
def test_local_provider_rejects_credential_like_environment_evidence(
    tmp_path,
    name,
    value,
):
    from sneeze.tpn.ocr_provider import (
        ProviderError,
        run_local_provider_batch,
    )

    spec_path = _write_provider_spec(tmp_path)
    spec = json.loads(spec_path.read_text())
    spec["env"][name] = value
    _write_json(spec_path, spec)
    run_dir = tmp_path / "public-evidence"

    with pytest.raises(ProviderError, match="credential-like") as exc_info:
        run_local_provider_batch(
            spec_path=spec_path,
            lane="best",
            run_dir=run_dir,
            pages=[_write_page(tmp_path, "page-01")],
        )

    assert value not in str(exc_info.value)
    assert not run_dir.exists()


@pytest.mark.parametrize(
    "name",
    (
        "PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONEXECUTABLE",
        "PYTHONINSPECT",
        "LD_PRELOAD",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_DEBUG_OUTPUT",
        "DYLD_INSERT_LIBRARIES",
        "GCONV_PATH",
        "GLIBC_TUNABLES",
    ),
)
def test_local_provider_rejects_execution_environment_overrides(
    tmp_path,
    name,
):
    from sneeze.tpn.ocr_provider import (
        ProviderError,
        load_local_provider_spec,
    )

    spec_path = _write_provider_spec(tmp_path)
    spec = json.loads(spec_path.read_text())
    spec["env"][name] = "/example/override"
    _write_json(spec_path, spec)

    with pytest.raises(
        ProviderError,
        match=f"cannot override execution variable {name}",
    ) as exc_info:
        load_local_provider_spec(spec_path)

    assert "/example/override" not in str(exc_info.value)


def test_local_provider_accepts_exact_forced_isolation_environment(tmp_path):
    from sneeze.tpn.ocr_provider import load_local_provider_spec

    spec_path = _write_provider_spec(tmp_path)
    spec = json.loads(spec_path.read_text())
    spec["env"]["PYTHONNOUSERSITE"] = "1"
    _write_json(spec_path, spec)

    loaded = load_local_provider_spec(spec_path)

    assert loaded["env"]["PYTHONNOUSERSITE"] == "1"


def test_local_provider_rejects_changed_forced_isolation_environment(
    tmp_path,
):
    from sneeze.tpn.ocr_provider import (
        ProviderError,
        load_local_provider_spec,
    )

    spec_path = _write_provider_spec(tmp_path)
    spec = json.loads(spec_path.read_text())
    spec["env"]["PYTHONNOUSERSITE"] = "0"
    _write_json(spec_path, spec)

    with pytest.raises(
        ProviderError,
        match="local-only provider env requires PYTHONNOUSERSITE=1",
    ):
        load_local_provider_spec(spec_path)


def test_validate_provider_run_dir_requires_absolute_empty_path(tmp_path):
    from sneeze.tpn.ocr_provider import (
        ProviderError,
        validate_provider_run_dir,
    )

    created = validate_provider_run_dir(tmp_path / "new-run")
    assert created.is_dir()
    assert created.stat().st_mode & 0o777 == 0o700

    (created / "evidence.txt").write_text("occupied")
    with pytest.raises(ProviderError, match="empty"):
        validate_provider_run_dir(created)
    with pytest.raises(ProviderError, match="absolute"):
        validate_provider_run_dir("relative-run")


def test_validate_provider_run_dir_rejects_protected_corpus_before_write(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.ocr_provider import (
        ProviderError,
        validate_provider_run_dir,
    )

    protected = tmp_path / "canonical" / "papers"
    protected.mkdir(parents=True)
    config_home = tmp_path / "config"
    policy = config_home / "sneeze" / "tpn" / "ocr-protected-roots"
    policy.parent.mkdir(parents=True)
    policy.write_text(f"{protected}\n", encoding="utf-8")
    policy.chmod(0o600)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    run_dir = protected / "provider-evidence"

    with pytest.raises(ProviderError, match="protected corpora"):
        validate_provider_run_dir(run_dir)

    assert not run_dir.exists()


def test_validate_provider_run_dir_rejects_final_symlink(tmp_path):
    from sneeze.tpn.ocr_provider import (
        ProviderError,
        validate_provider_run_dir,
    )

    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "run-link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ProviderError, match="symlink"):
        validate_provider_run_dir(link)


def test_validate_provider_run_dir_rejects_unsafe_ancestry(tmp_path):
    from sneeze.tpn.ocr_provider import (
        ProviderError,
        validate_provider_run_dir,
    )

    shared_parent = tmp_path / "shared-parent"
    shared_parent.mkdir(mode=0o700)
    shared_parent.chmod(0o777)

    with pytest.raises(ProviderError, match="writable by other users"):
        validate_provider_run_dir(shared_parent / "run")


def test_provider_run_ancestor_rejects_arbitrary_owner(monkeypatch):
    import sneeze.tpn.ocr_provider as ocr_provider

    path_stat = os.stat_result((0o040700, 1, 1, 1, 12345, 12345, 0, 0, 0, 0))
    monkeypatch.setattr(Path, "lstat", lambda self: path_stat)
    monkeypatch.setattr(ocr_provider.os, "geteuid", lambda: 23456)

    with pytest.raises(
        ocr_provider.ProviderError,
        match="owned by root or the current user",
    ):
        ocr_provider._validate_provider_run_ancestor(
            Path("/synthetic-untrusted-owner")
        )


def test_provider_run_ancestor_allows_root_owned_sticky_directory(
    monkeypatch,
):
    import sneeze.tpn.ocr_provider as ocr_provider

    path_stat = os.stat_result((0o041777, 1, 1, 1, 0, 0, 0, 0, 0, 0))
    monkeypatch.setattr(Path, "lstat", lambda self: path_stat)
    monkeypatch.setattr(ocr_provider.os, "geteuid", lambda: 23456)

    ocr_provider._validate_provider_run_ancestor(Path("/synthetic-tmp"))


def test_validate_provider_run_dir_rejects_leaf_symlink_swap(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_provider as ocr_provider

    target = tmp_path / "swap-target"
    target.mkdir(mode=0o755)
    target.chmod(0o755)
    original_mkdir = os.mkdir
    swapped = False

    def replacing_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if dir_fd is None:
            result = original_mkdir(path, mode)
        else:
            result = original_mkdir(path, mode, dir_fd=dir_fd)
        if path == "run" and dir_fd is not None:
            os.rmdir(path, dir_fd=dir_fd)
            os.symlink(str(target), path, dir_fd=dir_fd)
            swapped = True
        return result

    monkeypatch.setattr(ocr_provider.os, "mkdir", replacing_mkdir)

    with pytest.raises(
        ocr_provider.ProviderError,
        match="could not create|symlink|safely open",
    ):
        ocr_provider.validate_provider_run_dir(tmp_path / "run")

    assert swapped is True
    assert target.stat().st_mode & 0o777 == 0o755


@pytest.mark.parametrize("mode", (0o755, 0o600))
def test_validate_provider_run_dir_rejects_non_private_or_unusable_mode(
    tmp_path,
    mode,
):
    from sneeze.tpn.ocr_provider import (
        ProviderError,
        validate_provider_run_dir,
    )

    run_dir = tmp_path / "existing-run"
    run_dir.mkdir(mode=0o700)
    run_dir.chmod(mode)

    with pytest.raises(ProviderError, match="private usable permissions"):
        validate_provider_run_dir(run_dir)


def test_validate_provider_run_dir_rejects_wrong_owner(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_provider as ocr_provider

    run_dir = tmp_path / "existing-run"
    run_dir.mkdir(mode=0o700)
    owner = run_dir.stat().st_uid
    monkeypatch.setattr(ocr_provider.os, "geteuid", lambda: owner + 1)

    with pytest.raises(ocr_provider.ProviderError, match="current user"):
        ocr_provider.validate_provider_run_dir(run_dir)


def test_validate_provider_run_dir_creation_ignores_ambient_umask(tmp_path):
    from sneeze.tpn.ocr_provider import validate_provider_run_dir

    previous_umask = os.umask(0o777)
    try:
        created = validate_provider_run_dir(tmp_path / "new-run")
    finally:
        os.umask(previous_umask)

    assert created.stat().st_mode & 0o777 == 0o700


def test_local_provider_runs_one_process_for_batch_and_normalizes_metadata(
    tmp_path,
):
    from sneeze.tpn.ocr_provider import run_local_provider_batch

    spec_path = _write_provider_spec(tmp_path)
    pages = [
        _write_page(tmp_path, "page-01"),
        _write_page(tmp_path, "page-02"),
    ]
    messages = []
    first = run_local_provider_batch(
        spec_path=spec_path,
        lane="best",
        run_dir=tmp_path / "run-1",
        pages=pages,
        out=messages.append,
    )

    assert first["ok"] is True
    assert first["process"]["returncode"] == 0
    assert len(first["results"]) == 2
    assert all(result["valid"] for result in first["results"])
    assert (tmp_path / "run-1" / "stdout.log").read_text().strip() == (
        "request_count=2"
    )
    assert messages == [
        "Running local OCR provider fake-local-provider "
        "lane=best for 2 page(s)."
    ]
    result = first["results"][0]
    metadata = result["normalized_metadata"]
    assert result["output_text"].startswith("# page-01")
    assert result["output_sha256"] == _sha256(result["output_text"].encode())
    assert metadata["backend"] == "local_provider"
    assert metadata["prompt_mode"] == "provider:best"
    assert metadata["generated_token_count"] == 8
    assert metadata["termination"] == {
        "eos": True,
        "length": False,
        "max_tokens": False,
    }
    assert metadata["model"]["revision"] == MODEL_REVISION
    assert (
        metadata["provider"]["provider_environment_sha256"]
        == (first["hashes"]["provider_environment_sha256"])
    )
    assert Path(first["paths"]["requests_jsonl"]).is_file()
    assert Path(first["paths"]["responses_jsonl"]).is_file()
    assert (tmp_path / "run-1" / "batch.json").is_file()
    assert Path(first["paths"]["provider_spec_source"]).read_bytes() == (
        spec_path.read_bytes()
    )
    expected_environment = {
        "FAKE_PROVIDER_MODE": "valid",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "NO_PROXY": "127.0.0.1,localhost",
        "PATH": "/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "no_proxy": "127.0.0.1,localhost",
    }
    expected_environment.update(
        {
            name: value
            for name, value in os.environ.items()
            if name in {"LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ"}
        }
    )
    assert first["provider_environment"] == expected_environment
    assert first["hashes"]["provider_environment_sha256"]

    second = run_local_provider_batch(
        spec_path=spec_path,
        lane="best",
        run_dir=tmp_path / "run-2",
        pages=pages,
    )
    assert [item["attempt_id"] for item in first["results"]] == [
        item["attempt_id"] for item in second["results"]
    ]


def test_provider_spec_hash_uses_prelaunch_bytes(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_provider as ocr_provider

    spec_path = _write_provider_spec(tmp_path)
    original_bytes = spec_path.read_bytes()
    original_hash = _sha256(original_bytes)
    original_runner = ocr_provider._run_provider_process

    def mutate_spec_then_run(**kwargs):
        spec_path.write_text('{"mutated": true}\n', encoding="utf-8")
        return original_runner(**kwargs)

    monkeypatch.setattr(
        ocr_provider,
        "_run_provider_process",
        mutate_spec_then_run,
    )
    batch = ocr_provider.run_local_provider_batch(
        spec_path=spec_path,
        lane="best",
        run_dir=tmp_path / "run",
        pages=[_write_page(tmp_path, "page-01")],
    )

    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert batch["ok"] is True
    assert run["provider_spec_source_sha256"] == original_hash
    assert batch["hashes"]["provider_spec_source_sha256"] == original_hash
    assert (tmp_path / "run" / "provider-spec.json").read_bytes() == (
        original_bytes
    )
    assert _sha256(spec_path.read_bytes()) != original_hash


@pytest.mark.parametrize(
    ("mode", "invalid_indexes", "protocol_fragment"),
    (
        ("duplicate", {0}, "duplicate response IDs"),
        ("missing", {1}, None),
        ("malformed", set(), "malformed response JSON"),
    ),
)
def test_response_mapping_handles_duplicate_missing_and_malformed_records(
    tmp_path,
    mode,
    invalid_indexes,
    protocol_fragment,
):
    from sneeze.tpn.ocr_provider import run_local_provider_batch

    spec_path = _write_provider_spec(tmp_path, mode=mode)
    pages = [
        _write_page(tmp_path, "page-01"),
        _write_page(tmp_path, "page-02"),
    ]
    batch = run_local_provider_batch(
        spec_path=spec_path,
        lane="best",
        run_dir=tmp_path / "run",
        pages=pages,
    )

    assert batch["ok"] is False
    assert {
        index
        for index, result in enumerate(batch["results"])
        if not result["valid"]
    } == invalid_indexes
    if protocol_fragment:
        assert any(
            protocol_fragment in error for error in batch["protocol_errors"]
        )


@pytest.mark.parametrize(
    ("mode", "metadata_gap"),
    (
        ("outside_artifact", "artifacts.0.relative_path"),
        ("bad_artifact_hash", "artifacts.0.sha256"),
        ("model_mismatch", "model.revision_mismatch"),
        ("null_backend", "runtime.attention_backend"),
        ("null_input_tokens", "generation.input_tokens"),
    ),
)
def test_response_provenance_and_artifact_validation(
    tmp_path,
    mode,
    metadata_gap,
):
    from sneeze.tpn.ocr_provider import run_local_provider_batch

    batch = run_local_provider_batch(
        spec_path=_write_provider_spec(tmp_path, mode=mode),
        lane="best",
        run_dir=tmp_path / "run",
        pages=[_write_page(tmp_path, "page-01")],
    )

    result = batch["results"][0]
    assert batch["ok"] is False
    assert result["valid"] is False
    assert metadata_gap in result["metadata_gaps"]
    assert result["normalized_metadata"]["status"] == "error"


def test_nonzero_provider_keeps_individually_valid_responses(tmp_path):
    from sneeze.tpn.ocr_provider import run_local_provider_batch

    batch = run_local_provider_batch(
        spec_path=_write_provider_spec(tmp_path, mode="nonzero"),
        lane="best",
        run_dir=tmp_path / "run",
        pages=[_write_page(tmp_path, "page-01")],
    )

    assert batch["ok"] is False
    assert batch["process"]["returncode"] == 3
    assert batch["results"][0]["valid"] is True
    assert "provider exited with status 3" in batch["protocol_errors"]


def test_timeout_terminates_the_provider_process_group(tmp_path):
    from sneeze.tpn.ocr_provider import run_local_provider_batch

    run_dir = tmp_path / "run"
    batch = run_local_provider_batch(
        spec_path=_write_provider_spec(tmp_path, mode="timeout"),
        lane="best",
        run_dir=run_dir,
        pages=[_write_page(tmp_path, "page-01")],
        timeout_seconds=1.0,
        terminate_grace_seconds=0.2,
    )

    process = batch["process"]
    assert batch["ok"] is False
    assert process["timed_out"] is True
    assert "SIGTERM" in process["termination_signals"]
    assert "SIGKILL" in process["termination_signals"]
    assert batch["results"][0]["valid"] is False
    marker = run_dir / "artifacts" / "grandchild-ticks.txt"
    assert marker.is_file()
    before = marker.stat().st_size
    time.sleep(0.15)
    assert marker.stat().st_size == before


def test_page_hash_prompt_and_settings_are_checked_before_process_start(
    tmp_path,
):
    from sneeze.tpn.ocr_provider import (
        ProviderError,
        run_local_provider_batch,
    )

    spec_path = _write_provider_spec(tmp_path)
    page = _write_page(tmp_path, "page-01")
    page["image_sha256"] = "0" * 64

    with pytest.raises(ProviderError, match="SHA-256 mismatch"):
        run_local_provider_batch(
            spec_path=spec_path,
            lane="best",
            run_dir=tmp_path / "run-hash",
            pages=[page],
        )
    assert not (tmp_path / "run-hash").exists()

    page = _write_page(tmp_path, "page-02")
    page["settings"] = {"temperature": 1.0}
    with pytest.raises(ProviderError, match="settings do not match"):
        run_local_provider_batch(
            spec_path=spec_path,
            lane="best",
            run_dir=tmp_path / "run-settings",
            pages=[page],
        )
    assert not (tmp_path / "run-settings").exists()
