import builtins
import json
import os
import subprocess
import sys
from io import StringIO

import pytest


def _assert_receipt_hash(receipt):
    from sneeze.tpn.ocr_backend_setup import sha256_json

    payload = dict(receipt)
    digest = payload.pop("receipt_sha256")
    assert digest == sha256_json(payload)


def _make_cache(tmp_path, backend):
    from sneeze.tpn.ocr_backend_setup import expected_snapshot_path

    cache = tmp_path / "hub"
    snapshot = expected_snapshot_path(backend, cache.resolve())
    snapshot.mkdir(parents=True)
    return cache, snapshot


def _mock_snapshot_integrity(monkeypatch, module, backend):
    monkeypatch.setattr(
        module,
        "_snapshot_integrity",
        lambda *_args, **_kwargs: {
            "files_sha256": backend.snapshot_identity_sha256,
            "weights_sha256": backend.weights_sha256,
            "error_type": None,
        },
    )


def test_registry_has_only_pinned_initial_backends():
    from sneeze.tpn.ocr_backends import (
        DEEPSEEK_MODEL_REVISION,
        MINERU_MODEL_REVISION,
        get_backend,
        list_backends,
    )

    backends = list_backends()
    assert [item.backend_id for item in backends] == [
        "deepseek-vllm",
        "mineru25-pro-2605",
    ]
    assert get_backend("deepseek").model_revision == DEEPSEEK_MODEL_REVISION
    mineru = get_backend("mineru25")
    assert mineru.model_revision == MINERU_MODEL_REVISION
    assert mineru.capability_status("markdown") == "qualified"
    assert mineru.qualified_adapter_sha256 == (
        "2ed21528fe34a01d5e897e1e50fe8162a71b14947dc1b6bf62f8120314db20d2"
    )
    assert mineru.snapshot_identity_sha256
    assert mineru.weights_sha256
    mineru_record = mineru.as_dict()
    assert mineru_record["environment"]["runtime_packages"]["torch"] == (
        "2.11.0+cu130"
    )
    assert mineru_record["provider"]["default_attention_backend"] == "sdpa"
    assert mineru_record["provider"]["settings"] == {
        "backend": "official-two-step-transformers",
        "batch_size": 1,
        "image_analysis": False,
    }
    assert get_backend("deepseek-vllm").capability_status("markdown") == (
        "guarded"
    )


def test_registry_rejects_unknown_backend():
    from sneeze.tpn.ocr_backends import BackendRegistryError, get_backend

    with pytest.raises(BackendRegistryError, match="unknown OCR backend"):
        get_backend("future-model")


def test_backend_cli_discovery_does_not_import_heavy_ml_packages(tmp_path):
    sentinel = tmp_path / "heavy-import.txt"
    code = f"""
import builtins
from pathlib import Path

blocked = {{"torch", "transformers", "vllm", "mineru_vl_utils"}}
original = builtins.__import__

def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in blocked:
        Path({str(sentinel)!r}).write_text(name, encoding='utf-8')
        raise RuntimeError('heavy import during CLI discovery: ' + name)
    return original(name, *args, **kwargs)

builtins.__import__ = guarded
from sneeze import cli

instance = cli.CLI(
    program_name='sne',
    module_names=['sneeze', 'sneeze.tpn'],
    introspect=True,
    auto_plugins=False,
)
expected = {{
    'ocr-backend-list',
    'ocr-backend-create-env',
    'ocr-backend-fetch-model',
    'ocr-backend-write-spec',
    'ocr-backend-doctor',
}}
assert expected <= set(instance._commands_by_name)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    assert completed.returncode == 0, completed.stderr
    assert not sentinel.exists()


def test_mineru_environment_plan_is_exactly_pinned():
    from sneeze.tpn.ocr_backend_setup import build_environment_commands
    from sneeze.tpn.ocr_backends import get_backend

    commands = build_environment_commands(
        get_backend("mineru25"),
        manager="mamba",
        env_name="mineru-env",
        python_version="3.12",
        torch_backend="cu130",
    )
    flattened = [part for command in commands for part in command]
    assert "torch==2.11.0+cu130" in flattened
    assert "torchvision==0.26.0+cu130" in flattened
    assert "transformers==4.57.6" in flattened
    assert "mineru-vl-utils==1.0.5" in flattened
    assert not any("sneeze-plugin-tpn" in part for part in flattened)


def test_mineru_environment_rejects_unqualified_torch_backend():
    from sneeze.tpn.ocr_backend_setup import (
        BackendSetupError,
        build_environment_commands,
    )
    from sneeze.tpn.ocr_backends import get_backend

    with pytest.raises(
        BackendSetupError,
        match="requires torch backend cu130",
    ):
        build_environment_commands(
            get_backend("mineru25"),
            manager="mamba",
            env_name="mineru-env",
            python_version="3.12",
            torch_backend="cu121",
        )


def test_deepseek_environment_plan_pins_qualified_runtime():
    from sneeze.tpn.ocr_backend_setup import build_environment_commands
    from sneeze.tpn.ocr_backends import get_backend

    commands = build_environment_commands(
        get_backend("deepseek"),
        manager="mamba",
        env_name="deepseek-env",
        python_version="3.12.13",
        torch_backend="cu130",
    )
    flattened = [part for command in commands for part in command]
    for requirement in (
        "python=3.12.13",
        "poppler=26.05.0",
        "torch==2.11.0",
        "torchvision==0.26.0",
        "torchaudio==2.11.0+cu130",
        "transformers==5.9.0",
        "vllm==0.21.1rc1.dev438+g4ff865c38",
    ):
        assert requirement in flattened

    with pytest.raises(
        ValueError,
        match="requires torch backend cu130",
    ):
        build_environment_commands(
            get_backend("deepseek"),
            manager="mamba",
            env_name="deepseek-env",
            python_version="3.12.13",
            torch_backend="cu121",
        )


def test_environment_plan_rejects_credential_bearing_package_url():
    from sneeze.tpn.ocr_backend_setup import build_environment_commands
    from sneeze.tpn.ocr_backends import get_backend

    with pytest.raises(ValueError, match="credential-like"):
        build_environment_commands(
            get_backend("deepseek"),
            manager="mamba",
            env_name="deepseek-env",
            python_version="3.12.13",
            torch_backend="cu130",
            package_spec=(
                "plugin @ https://example.invalid/"
                "wheel.whl?api_key=do-not-print-this"
            ),
        )


def test_environment_plan_receipt_is_content_hashed():
    from sneeze.tpn.ocr_backend_setup import (
        build_environment_commands,
        environment_plan_receipt,
    )
    from sneeze.tpn.ocr_backends import get_backend

    backend = get_backend("deepseek")
    commands = build_environment_commands(
        backend,
        manager="mamba",
        env_name="deepseek-env",
        python_version="3.12",
        torch_backend="cu130",
    )
    receipt = environment_plan_receipt(
        backend,
        manager="mamba",
        env_name="deepseek-env",
        python_version="3.12",
        torch_backend="cu130",
        commands=commands,
        dry_run=True,
    )
    assert receipt["status"] == "planned"
    assert receipt["model"]["revision"] == backend.model_revision
    assert receipt["workflow"] == {
        "standalone_adapter": False,
        "model_fetch_separate": True,
        "provider_spec_generation_separate": False,
    }
    _assert_receipt_hash(receipt)


def test_model_fetch_plan_uses_immutable_revision(tmp_path):
    from sneeze.tpn.ocr_backend_setup import (
        build_model_fetch_command,
        model_fetch_receipt,
    )
    from sneeze.tpn.ocr_backends import get_backend

    backend = get_backend("mineru25")
    cache = (tmp_path / "hub").resolve()
    command = build_model_fetch_command(
        backend,
        manager="mamba",
        env_name="mineru-env",
        cache_dir=cache,
    )
    assert backend.model_id in command
    assert backend.model_revision in command
    receipt = model_fetch_receipt(
        backend,
        cache_dir=cache,
        command=command,
        dry_run=True,
    )
    assert receipt["model"]["revision"] == backend.model_revision
    assert receipt["model"]["snapshot_present"] is False
    assert receipt["qualification_artifacts"] == {
        "snapshot_identity_sha256": backend.snapshot_identity_sha256,
        "weights_sha256": backend.weights_sha256,
    }
    _assert_receipt_hash(receipt)


def test_model_fetch_plan_rejects_relative_cache_path():
    from sneeze.tpn.ocr_backend_setup import (
        BackendSetupError,
        build_model_fetch_command,
    )
    from sneeze.tpn.ocr_backends import get_backend

    with pytest.raises(BackendSetupError, match="must be an absolute path"):
        build_model_fetch_command(
            get_backend("mineru25"),
            manager="mamba",
            env_name="mineru-env",
            cache_dir="relative/hub",
        )


def test_generated_mineru_spec_passes_production_loader(tmp_path):
    from sneeze.tpn.ocr_backend_setup import build_provider_spec
    from sneeze.tpn.ocr_backends import get_backend

    backend = get_backend("mineru25")
    cache, _ = _make_cache(tmp_path, backend)
    spec = build_provider_spec(
        backend,
        python_executable=sys.executable,
        cache_dir=cache.resolve(),
    )
    assert spec["egress"] == "none"
    assert spec["secret_env_names"] == []
    assert spec["model"] == {
        "requested": backend.model_id,
        "revision": backend.model_revision,
    }
    assert spec["lanes"]["best"]["settings"] == {
        "backend": "official-two-step-transformers",
        "batch_size": 1,
        "image_analysis": False,
    }
    assert spec["env"]["HF_HUB_OFFLINE"] == "1"
    assert spec["env"]["TRANSFORMERS_OFFLINE"] == "1"


def test_write_provider_spec_is_atomic_hashed_and_no_overwrite(tmp_path):
    from sneeze.tpn.ocr_backend_setup import (
        file_sha256,
        write_provider_spec,
    )
    from sneeze.tpn.ocr_backends import get_backend
    from sneeze.tpn.ocr_provider import load_local_provider_spec

    backend = get_backend("mineru25")
    cache, _ = _make_cache(tmp_path, backend)
    output = (tmp_path / "provider.json").resolve()
    receipt = write_provider_spec(
        backend,
        output_path=output,
        python_executable=sys.executable,
        cache_dir=cache.resolve(),
    )
    loaded = load_local_provider_spec(output)
    assert loaded["model"]["revision"] == backend.model_revision
    assert receipt["spec_sha256"] == file_sha256(output)
    _assert_receipt_hash(receipt)

    original = output.read_bytes()
    with pytest.raises(ValueError, match="refusing to replace"):
        write_provider_spec(
            backend,
            output_path=output,
            python_executable=sys.executable,
            cache_dir=cache.resolve(),
        )
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".provider.json.*.candidate"))


def test_in_process_backend_has_no_provider_spec(tmp_path):
    from sneeze.tpn.ocr_backend_setup import (
        BackendSetupError,
        build_provider_spec,
    )
    from sneeze.tpn.ocr_backends import get_backend

    with pytest.raises(BackendSetupError, match="runs in process"):
        build_provider_spec(
            get_backend("deepseek"),
            python_executable=sys.executable,
            cache_dir=tmp_path.resolve(),
        )


def test_doctor_passes_without_importing_runtime_and_scrubs_secrets(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn import ocr_backend_setup
    from sneeze.tpn.ocr_backends import get_backend

    backend = get_backend("mineru25")
    cache, _ = _make_cache(tmp_path, backend)
    spec_path = (tmp_path / "provider.json").resolve()
    ocr_backend_setup.write_provider_spec(
        backend,
        output_path=spec_path,
        python_executable=sys.executable,
        cache_dir=cache.resolve(),
    )
    captured = {"calls": []}
    monkeypatch.setenv("UNIT_TEST_SECRET_TOKEN", "must-not-appear")

    def fake_probe(argv, **kwargs):
        captured["calls"].append(argv)
        captured["env"] = kwargs["environment"]
        return {
            "returncode": 0,
            "stdout": json.dumps(dict(backend.runtime_packages)).encode(),
            "stderr": b"",
            "error_type": None,
            "stdout_truncated": False,
            "stderr_truncated": False,
        }

    monkeypatch.setattr(ocr_backend_setup, "_run_bounded_probe", fake_probe)
    _mock_snapshot_integrity(monkeypatch, ocr_backend_setup, backend)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "transformers", "vllm"}:
            raise AssertionError(f"doctor imported heavy runtime: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    receipt = ocr_backend_setup.doctor_provider_spec(spec_path)
    assert receipt["all_pass"] is True
    assert receipt["status"] == "pass"
    assert "UNIT_TEST_SECRET_TOKEN" not in captured["env"]
    rendered = json.dumps(receipt, sort_keys=True)
    assert "must-not-appear" not in rendered
    assert "UNIT_TEST_SECRET_TOKEN" not in rendered
    assert len(captured["calls"]) == 1
    assert receipt["model_probe"]["requested"] is False
    _assert_receipt_hash(receipt)


def test_doctor_reports_package_mismatch_without_raw_stderr(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn import ocr_backend_setup
    from sneeze.tpn.ocr_backends import get_backend

    backend = get_backend("mineru25")
    cache, _ = _make_cache(tmp_path, backend)
    spec_path = (tmp_path / "provider.json").resolve()
    ocr_backend_setup.write_provider_spec(
        backend,
        output_path=spec_path,
        python_executable=sys.executable,
        cache_dir=cache.resolve(),
    )

    def fake_probe(*_args, **_kwargs):
        return {
            "returncode": 9,
            "stdout": b"",
            "stderr": b"PRIVATE_VALUE_SHOULD_NOT_BE_RECORDED",
            "error_type": None,
            "stdout_truncated": False,
            "stderr_truncated": False,
        }

    monkeypatch.setattr(ocr_backend_setup, "_run_bounded_probe", fake_probe)
    _mock_snapshot_integrity(monkeypatch, ocr_backend_setup, backend)
    receipt = ocr_backend_setup.doctor_provider_spec(spec_path)
    assert receipt["all_pass"] is False
    assert receipt["status"] == "fail"
    assert "PRIVATE_VALUE_SHOULD_NOT_BE_RECORDED" not in json.dumps(receipt)
    _assert_receipt_hash(receipt)


def test_doctor_load_model_runs_exact_gpu_initialization_probe(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn import ocr_backend_setup
    from sneeze.tpn.ocr_backends import get_backend

    backend = get_backend("mineru25")
    cache, snapshot = _make_cache(tmp_path, backend)
    spec_path = (tmp_path / "provider.json").resolve()
    ocr_backend_setup.write_provider_spec(
        backend,
        output_path=spec_path,
        python_executable=sys.executable,
        cache_dir=cache.resolve(),
    )
    calls = []

    def fake_probe(argv, **_kwargs):
        calls.append(argv)
        if ocr_backend_setup._MODEL_PROBE_CODE in argv:
            result = {
                "ok": True,
                "model_class": "Qwen2VLForConditionalGeneration",
                "processor_class": "AutoProcessor",
                "client_class": "MinerUClient",
                "dtype": "bfloat16",
                "device": "cuda:0",
                "attention_backend": "sdpa",
                "backend": "transformers",
                "image_analysis": False,
                "batch_size": 1,
            }
        else:
            result = dict(backend.runtime_packages)
        return {
            "returncode": 0,
            "stdout": json.dumps(result).encode(),
            "stderr": b"",
            "error_type": None,
            "stdout_truncated": False,
            "stderr_truncated": False,
        }

    monkeypatch.setattr(ocr_backend_setup, "_run_bounded_probe", fake_probe)
    _mock_snapshot_integrity(monkeypatch, ocr_backend_setup, backend)
    receipt = ocr_backend_setup.doctor_provider_spec(
        spec_path,
        load_model=True,
    )
    assert receipt["all_pass"] is True
    assert receipt["model_probe"]["passed"] is True
    model_call = next(
        call for call in calls if ocr_backend_setup._MODEL_PROBE_CODE in call
    )
    assert str(snapshot) in model_call
    assert "cuda:0" in model_call
    assert "sdpa" in model_call
    _assert_receipt_hash(receipt)


def test_doctor_fails_when_qualified_adapter_hash_drifts(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn import ocr_backend_setup
    from sneeze.tpn.ocr_backends import get_backend

    backend = get_backend("mineru25")
    cache, _ = _make_cache(tmp_path, backend)
    spec_path = (tmp_path / "provider.json").resolve()
    ocr_backend_setup.write_provider_spec(
        backend,
        output_path=spec_path,
        python_executable=sys.executable,
        cache_dir=cache.resolve(),
    )
    real_hash = ocr_backend_setup.file_sha256

    def drifted_hash(path):
        if str(path).endswith("providers/mineru25.py"):
            return "0" * 64
        return real_hash(path)

    monkeypatch.setattr(ocr_backend_setup, "file_sha256", drifted_hash)
    monkeypatch.setattr(
        ocr_backend_setup,
        "_run_bounded_probe",
        lambda *_args, **_kwargs: {
            "returncode": 0,
            "stdout": json.dumps(dict(backend.runtime_packages)).encode(),
            "stderr": b"",
            "error_type": None,
            "stdout_truncated": False,
            "stderr_truncated": False,
        },
    )
    _mock_snapshot_integrity(monkeypatch, ocr_backend_setup, backend)
    receipt = ocr_backend_setup.doctor_provider_spec(spec_path)
    adapter_check = next(
        item for item in receipt["checks"] if item["name"] == "adapter_sha256"
    )
    assert adapter_check["passed"] is False
    assert receipt["all_pass"] is False


def test_backend_commands_emit_receipts_and_remain_visible(tmp_path):
    from sneeze import cli as sneeze_cli
    from sneeze.tpn.commands import (
        OcrBackendCreateEnv,
        OcrBackendFetchModel,
        OcrBackendList,
        OcrBackendWriteSpec,
    )

    out = StringIO()
    list_command = OcrBackendList(None, out, None)
    list_command.format = "json"
    list_command.run()
    registry_receipt = json.loads(out.getvalue())
    _assert_receipt_hash(registry_receipt)

    out = StringIO()
    env_command = OcrBackendCreateEnv(None, out, None)
    env_command.backend = "mineru25"
    env_command.manager = "mamba"
    env_command.dry_run = True
    env_command.run()
    assert '"kind": "ocr_backend_environment"' in out.getvalue()
    assert '"receipt_sha256"' in out.getvalue()

    cache = (tmp_path / "hub").resolve()
    out = StringIO()
    fetch_command = OcrBackendFetchModel(None, out, None)
    fetch_command.backend = "mineru25"
    fetch_command.cache_dir = str(cache)
    fetch_command.env_name = "mineru-env"
    fetch_command.manager = "mamba"
    fetch_command.dry_run = True
    fetch_command.run()
    assert '"kind": "ocr_backend_model"' in out.getvalue()

    cache.mkdir()
    output = (tmp_path / "provider.json").resolve()
    out = StringIO()
    spec_command = OcrBackendWriteSpec(None, out, None)
    spec_command.backend = "mineru25"
    spec_command.output = str(output)
    spec_command.python_executable = sys.executable
    spec_command.cache_dir = str(cache)
    spec_command.run()
    spec_receipt = json.loads(out.getvalue())
    assert spec_receipt["spec_sha256"]
    _assert_receipt_hash(spec_receipt)

    cli = sneeze_cli.CLI(
        program_name="sne",
        module_names=["sneeze", "sneeze.tpn"],
        introspect=True,
        auto_plugins=False,
    )
    for name in (
        "ocr-backend-list",
        "ocr-backend-create-env",
        "ocr-backend-fetch-model",
        "ocr-backend-write-spec",
        "ocr-backend-doctor",
    ):
        assert name in cli._commands_by_name
