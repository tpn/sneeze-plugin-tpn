from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest

GIB = 1024**3


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _runtime(backend_id):
    from sneeze.tpn.ocr_backends import (
        DEEPSEEK_IMPLEMENTATION_SHA256,
        MINERU_IMPLEMENTATION_SHA256,
    )

    if backend_id == "deepseek-vllm":
        return {
            "python": "3.12.13",
            "engine": "vllm",
            "engine_version": "0.21.1rc1.dev438+g4ff865c38",
            "torch": "2.11.0+cu130",
            "torch_cuda": "13.0",
            "packages": {
                "pillow": "12.2.0",
                "torch": "2.11.0+cu130",
                "transformers": "5.9.0",
                "vllm": "0.21.1rc1.dev438+g4ff865c38",
            },
            "attention_backend": "FLASH_ATTN",
            "dtype": "bfloat16",
            "implementation_sha256": DEEPSEEK_IMPLEMENTATION_SHA256,
            "installed_distributions_count": 4,
            "installed_distributions_sha256": _digest(
                "installed-distributions-deepseek"
            ),
        }
    return {
        "python": "3.12.13",
        "engine": "transformers",
        "engine_version": "4.57.6",
        "torch": "2.11.0+cu130",
        "torch_cuda": "13.0",
        "packages": {
            "torch": "2.11.0+cu130",
            "transformers": "4.57.6",
        },
        "attention_backend": "sdpa",
        "dtype": "bfloat16",
        "implementation_sha256": MINERU_IMPLEMENTATION_SHA256,
        "installed_distributions_count": 2,
        "installed_distributions_sha256": _digest(
            "installed-distributions-mineru"
        ),
    }


def _probe_runtime(backend_id):
    runtime = _runtime(backend_id)
    selected = runtime.pop("attention_backend")
    runtime["attention_backend_requested"] = (
        "auto" if backend_id == "deepseek-vllm" else selected
    )
    return runtime


def _profile(*, provisional=False):
    from sneeze.tpn.ocr_profiles import (
        get_pipeline_definition,
        qualification_policy_sha256,
        seal_execution_profile,
    )

    pipeline = get_pipeline_definition("local-mineru25")
    stages = {
        stage["stage_id"]: {
            "backend_id": stage["backend_id"],
            "capability": stage["capability"],
            "model": {
                **copy.deepcopy(stage["model"]),
                "snapshot_files_sha256": _digest(
                    f"snapshot-{stage['backend_id']}"
                ),
            },
            "batch_semantics": stage["batch_semantics"],
            "runtime_selector": _runtime(stage["backend_id"]),
        }
        for stage in pipeline["stages"]
    }
    budget = int(48 * GIB * 0.9)
    deepseek_settings = {
        "batch_size": 2,
        "max_num_seqs": 2,
        "max_num_batched_tokens": 16384,
        "engine_overhead_headroom_bytes": GIB,
        "gpu_memory_utilization": round((budget - GIB) / (48 * GIB), 8),
        "gpu_memory_utilization_denominator_bytes": 48 * GIB,
        "attention_backend": "auto",
        "mm_encoder_attn_backend": "auto",
        "enable_chunked_prefill": True,
        "max_tokens": 8192,
        "ngram_size": 10,
        "window_size": 50,
        "temperature": 0.0,
        "memory_tier": "max",
        "memory_budget_bytes": budget,
    }
    mineru_settings = {
        "backend": "official-two-step-transformers",
        "batch_size": 4,
        "image_analysis": False,
        "memory_tier": "max",
        "memory_budget_bytes": budget,
    }
    tiers = {}
    for name, fraction in (("25", 0.25), ("50", 0.5), ("max", 0.9)):
        supported = name == "max"
        tiers[name] = {
            "supported": supported,
            "reason": None if supported else "not qualified",
            "budget_fraction": fraction,
            "budget_bytes": (
                budget if supported else int(48 * GIB * fraction)
            ),
            "reserve_bytes": 2 * GIB,
            "stages": (
                {
                    "grounded_ocr": {
                        "settings": deepseek_settings,
                        "observed_peak_memory_bytes": 8 * GIB,
                        "confirmed_pages_per_second": 0.05,
                    },
                    "markdown": {
                        "settings": mineru_settings,
                        "observed_peak_memory_bytes": 10 * GIB,
                        "confirmed_pages_per_second": 0.04,
                    },
                }
                if supported
                else {}
            ),
        }
    evidence = []
    if not provisional:
        evidence = [
            {
                "suite_id": "hardware-smoke",
                "memory_tier": "max",
                "manifest_sha256": _digest("manifest"),
                "summary_sha256": _digest("summary"),
                "experiment_sha256": _digest("experiment"),
                "repeats": 5,
                "all_pass": True,
                "provenance_complete": True,
            }
        ]
    return seal_execution_profile(
        {
            "schema_version": 1,
            "kind": "ocr_execution_profile",
            "profile_id": "test-local-mineru25",
            "display_name": "Test local MinerU profile",
            "pipeline_id": "local-mineru25",
            "pipeline_definition_sha256": pipeline["definition_sha256"],
            "hardware_selector": {
                "schema_version": 1,
                "vendor": "nvidia",
                "product_name": "Test GPU",
                "pci_device_id": "0x1234",
                "compute_capability": [9, 0],
                "memory_model": "dedicated",
                "total_memory_bytes": 48 * GIB,
            },
            "stages": stages,
            "memory_tiers": tiers,
            "qualification": {
                "state": "provisional" if provisional else "qualified",
                "policy_sha256": qualification_policy_sha256(),
                "required_suites": ["hardware-smoke"],
                "evidence": evidence,
            },
        }
    )


def _observation(profile):
    return {
        "hardware": {"selector": copy.deepcopy(profile["hardware_selector"])},
        "stages": {
            stage_id: {
                "backend_id": stage["backend_id"],
                "capability": stage["capability"],
                "model": copy.deepcopy(stage["model"]),
                "runtime": copy.deepcopy(stage["runtime_selector"]),
            }
            for stage_id, stage in profile["stages"].items()
        },
    }


def _installation(tmp_path, backend_id):
    from sneeze.tpn.ocr_implementation import backend_implementation_record

    executable = tmp_path / f"{backend_id}-python"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)
    cache = tmp_path / f"{backend_id}-cache"
    cache.mkdir()
    snapshot = cache / "snapshot"
    snapshot.mkdir()
    adapter = tmp_path / "mineru25.py"
    adapter.write_text("# adapter\n", encoding="utf-8")
    if backend_id == "deepseek-vllm":
        model_id = "deepseek-ai/DeepSeek-OCR"
        model_revision = "9f30c71f441d010e5429c532364a86705536c53a"
    else:
        model_id = "opendatalab/MinerU2.5-Pro-2605-1.2B"
        model_revision = "bff20d4ae2bf202df9f45284b4d43681555a97ed"
    return {
        "backend_id": backend_id,
        "model": {"id": model_id, "revision": model_revision},
        "python_executable": {"path": str(executable)},
        "hf_cache": {"path": str(cache)},
        "snapshot": {
            "path": str(snapshot),
            "files_sha256": _digest(f"snapshot-{backend_id}"),
        },
        "implementation": backend_implementation_record(backend_id),
        "adapter": (
            {"path": str(adapter)}
            if backend_id == "mineru25-pro-2605"
            else None
        ),
        "cuda": {
            "visible_devices": "0",
            "device": "cuda:0",
            "attention_backend": (
                "sdpa" if backend_id == "mineru25-pro-2605" else "auto"
            ),
        },
    }


def test_runtime_probe_is_isolated_and_secret_scrubbed(tmp_path):
    from sneeze.tpn.ocr_execution import probe_backend_runtime

    installation = _installation(tmp_path, "deepseek-vllm")
    runtime = _probe_runtime("deepseek-vllm")
    captured = {}

    def runner(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps({"ok": True, "runtime": runtime}) + "\n",
            stderr="",
        )

    result = probe_backend_runtime(
        installation,
        runner=runner,
        environment={
            "PATH": "/usr/bin:/bin",
            "LD_LIBRARY_PATH": "/cuda/lib",
            "OPENAI_API_KEY": "do-not-copy",
        },
    )

    assert result == runtime
    assert captured["argv"][1:4] == ["-I", "-B", "-c"]
    assert "OPENAI_API_KEY" not in captured["env"]
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert captured["env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["argv"][6] == "auto"
    assert result["attention_backend_requested"] == "auto"


def test_runtime_probe_default_runner_streams_successfully():
    from sneeze.tpn.ocr_execution import _run_bounded_runtime_probe

    completed = _run_bounded_runtime_probe(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys; print('ready'); print('note', file=sys.stderr)",
        ],
        environment={"PATH": os.defpath},
        timeout_seconds=10,
        runner=subprocess.run,
    )

    assert completed.returncode == 0
    assert completed.stdout == "ready\n"
    assert completed.stderr == "note\n"


@pytest.mark.parametrize("stream_name", ("stdout", "stderr"))
def test_runtime_probe_stream_cap_stops_actual_producer(
    tmp_path,
    stream_name,
):
    from sneeze.tpn.ocr_execution import (
        OcrExecutionError,
        _run_bounded_runtime_probe,
    )

    completed_marker = tmp_path / "producer-completed"
    producer = "\n".join(
        (
            "import pathlib",
            "import sys",
            f"stream = sys.{stream_name}.buffer",
            "for _ in range(4096):",
            "    stream.write(b'x' * 1024)",
            "    stream.flush()",
            "pathlib.Path(sys.argv[1]).write_text('completed')",
        )
    )

    with pytest.raises(
        OcrExecutionError,
        match=rf"runtime probe {stream_name} exceeded 64 KiB",
    ):
        _run_bounded_runtime_probe(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                producer,
                str(completed_marker),
            ],
            environment={"PATH": os.defpath},
            timeout_seconds=10,
            runner=subprocess.run,
        )

    assert not completed_marker.exists()


def test_runtime_probe_timeout_reaps_actual_process(tmp_path):
    from sneeze.tpn.ocr_execution import (
        OcrExecutionError,
        _run_bounded_runtime_probe,
    )

    pid_path = tmp_path / "producer.pid"
    producer = "\n".join(
        (
            "import os",
            "import pathlib",
            "import sys",
            "import time",
            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))",
            "time.sleep(30)",
        )
    )

    with pytest.raises(OcrExecutionError, match="TimeoutExpired"):
        _run_bounded_runtime_probe(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                producer,
                str(pid_path),
            ],
            environment={"PATH": os.defpath},
            timeout_seconds=0.5,
            runner=subprocess.run,
        )

    producer_pid = int(pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(producer_pid, 0)


def test_preflight_auto_attention_matches_qualified_selected_backend():
    from sneeze.tpn.ocr_profiles import profile_match_report

    profile = _profile(provisional=True)
    observation = _observation(profile)
    observation["stages"]["grounded_ocr"]["runtime"] = _probe_runtime(
        "deepseek-vllm"
    )
    observation["stages"]["markdown"]["runtime"] = _probe_runtime(
        "mineru25-pro-2605"
    )

    report = profile_match_report(
        profile,
        observation,
        memory_tier="max",
        require_qualified=False,
    )

    assert report["matched"] is True
    assert report["reasons"] == []


def test_preflight_rejects_different_requested_attention_backend():
    from sneeze.tpn.ocr_profiles import profile_match_report

    profile = _profile(provisional=True)
    observation = _observation(profile)
    runtime = _probe_runtime("deepseek-vllm")
    runtime["attention_backend_requested"] = "TORCH_SDPA"
    observation["stages"]["grounded_ocr"]["runtime"] = runtime
    observation["stages"]["markdown"]["runtime"] = _probe_runtime(
        "mineru25-pro-2605"
    )

    report = profile_match_report(
        profile,
        observation,
        memory_tier="max",
        require_qualified=False,
    )

    assert report["matched"] is False
    assert report["reasons"] == [
        "stages.grounded_ocr.runtime.attention_backend_requested differs"
    ]


def test_execution_observation_binds_registered_model_snapshot(
    tmp_path,
):
    from sneeze.tpn.ocr_execution import (
        OcrExecutionError,
        build_execution_observation,
    )

    profile = _profile(provisional=True)
    installations = {
        backend_id: _installation(tmp_path, backend_id)
        for backend_id in ("deepseek-vllm", "mineru25-pro-2605")
    }

    def runner(argv, **_kwargs):
        backend_id = (
            "deepseek-vllm" if "vllm" in argv else "mineru25-pro-2605"
        )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(
                {"ok": True, "runtime": _probe_runtime(backend_id)}
            )
            + "\n",
            stderr="",
        )

    observation = build_execution_observation(
        pipeline_id="local-mineru25",
        hardware={"selector": profile["hardware_selector"]},
        installations=installations,
        runner=runner,
    )

    assert observation["stages"]["grounded_ocr"]["model"] == profile[
        "stages"
    ]["grounded_ocr"]["model"]
    assert observation["stages"]["markdown"]["model"] == profile[
        "stages"
    ]["markdown"]["model"]

    other_snapshot = _digest("different-registered-model-bytes")
    installations["deepseek-vllm"]["snapshot"][
        "files_sha256"
    ] = other_snapshot
    changed = build_execution_observation(
        pipeline_id="local-mineru25",
        hardware={"selector": profile["hardware_selector"]},
        installations=installations,
        runner=runner,
    )
    assert changed["stages"]["grounded_ocr"]["model"][
        "snapshot_files_sha256"
    ] == other_snapshot

    installations["deepseek-vllm"]["snapshot"]["files_sha256"] = "bad"
    with pytest.raises(OcrExecutionError, match="snapshot identity"):
        build_execution_observation(
            pipeline_id="local-mineru25",
            hardware={"selector": profile["hardware_selector"]},
            installations=installations,
            runner=runner,
        )


def test_in_process_deepseek_requires_registered_cuda_environment(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.ocr_execution import (
        OcrExecutionError,
        assert_in_process_deepseek_installation,
    )

    installation = _installation(tmp_path, "deepseek-vllm")
    monkeypatch.setattr(
        sys,
        "executable",
        installation["python_executable"]["path"],
    )
    expected = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
    }

    assert_in_process_deepseek_installation(installation, environ=expected)

    with pytest.raises(OcrExecutionError, match="CUDA_DEVICE_ORDER"):
        assert_in_process_deepseek_installation(
            installation,
            environ={"CUDA_VISIBLE_DEVICES": "0"},
        )
    with pytest.raises(OcrExecutionError, match="CUDA_VISIBLE_DEVICES"):
        assert_in_process_deepseek_installation(
            installation,
            environ={
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "1",
            },
        )


@pytest.mark.parametrize(
    "module_name",
    ["torch", "torch.cuda", "vllm", "vllm.engine.arg_utils"],
)
def test_in_process_deepseek_rejects_late_cuda_binding_check(
    tmp_path,
    monkeypatch,
    module_name,
):
    from sneeze.tpn.ocr_execution import (
        OcrExecutionError,
        assert_in_process_deepseek_installation,
    )

    installation = _installation(tmp_path, "deepseek-vllm")
    monkeypatch.setattr(
        sys,
        "executable",
        installation["python_executable"]["path"],
    )
    monkeypatch.setitem(sys.modules, module_name, object())

    with pytest.raises(
        OcrExecutionError,
        match="before importing CUDA runtime modules",
    ):
        assert_in_process_deepseek_installation(
            installation,
            environ={
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "0",
            },
        )


def test_structured_command_checks_cuda_binding_before_engine_start():
    import ast
    import inspect
    import textwrap

    from sneeze.tpn.commands import OcrPdfPages

    resolve_tree = ast.parse(
        textwrap.dedent(
            inspect.getsource(OcrPdfPages._resolve_structured_execution)
        )
    )
    run_tree = ast.parse(textwrap.dedent(inspect.getsource(OcrPdfPages.run)))

    def call_line(tree, name):
        return min(
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name)
                and node.func.id == name
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == name
            )
        )

    assert call_line(
        resolve_tree,
        "assert_in_process_deepseek_installation",
    ) < call_line(resolve_tree, "build_execution_observation")
    assert call_line(run_tree, "_resolve_structured_execution") < call_line(
        run_tree,
        "DeepSeekOcrOfflineProcessor",
    )
    assert call_line(run_tree, "_resolve_structured_execution") < call_line(
        run_tree,
        "_run_full_mode",
    )


def test_engine_metadata_records_startup_distribution_identity(monkeypatch):
    from sneeze.tpn import ocr_vllm

    identity = {
        "count": 2,
        "sha256": _digest("actual-in-process-distributions"),
        "distributions": ["one==1", "two==2"],
    }
    monkeypatch.setattr(
        ocr_vllm,
        "selected_attention_backend",
        lambda _llm: "FLASH_ATTN",
    )

    metadata = ocr_vllm.engine_metadata(
        object(),
        model="deepseek-ai/DeepSeek-OCR",
        model_revision="9f30c71f441d010e5429c532364a86705536c53a",
        requested_attention_backend="FLASH_ATTN",
        requested_mm_encoder_attn_backend=None,
        startup_installed_distributions={
            "count": identity["count"],
            "sha256": identity["sha256"],
        },
    )

    assert metadata["runtime"]["installed_distributions_count"] == 2
    assert metadata["runtime"]["installed_distributions_sha256"] == identity[
        "sha256"
    ]


def test_processor_captures_distribution_identity_before_engine(monkeypatch):
    from sneeze.tpn import ocr_vllm

    identity = {
        "count": 2,
        "sha256": _digest("processor-startup-distributions"),
        "distributions": ["one==1", "two==2"],
    }
    monkeypatch.setattr(
        ocr_vllm,
        "installed_distributions_identity",
        lambda: identity,
    )

    processor = ocr_vllm.DeepSeekOcrOfflineProcessor(
        model="deepseek-ai/DeepSeek-OCR",
        output_format="txt",
    )

    assert processor._startup_installed_distributions == {
        "count": identity["count"],
        "sha256": identity["sha256"],
    }


def test_stage_model_mismatches_rejects_provider_model_drift():
    from sneeze.tpn.ocr_execution import stage_model_mismatches

    profile = _profile(provisional=True)
    expected = copy.deepcopy(profile["stages"]["markdown"]["model"])
    loaded = {**expected, "resolved_revision": expected["revision"]}
    assert stage_model_mismatches(
        profile,
        stage_id="markdown",
        model=loaded,
    ) == []

    loaded["snapshot_files_sha256"] = _digest("different-loaded-model")
    loaded["resolved_revision"] = "0" * 40
    assert stage_model_mismatches(
        profile,
        stage_id="markdown",
        model=loaded,
    ) == [
        "stages.markdown.model.snapshot_files_sha256 differs",
        "stages.markdown.model.resolved_revision differs",
    ]


def test_deepseek_engine_identity_binds_actual_loaded_engine(tmp_path):
    from sneeze.tpn.ocr_execution import (
        deepseek_engine_identity_mismatches,
    )

    profile = _profile(provisional=True)
    installation = _installation(tmp_path, "deepseek-vllm")
    runtime = profile["stages"]["grounded_ocr"]["runtime_selector"]
    metadata = {
        "model": {
            "requested": installation["snapshot"]["path"],
            "revision": None,
            "resolved_path": installation["snapshot"]["path"],
            "snapshot": installation["model"]["revision"],
        },
        "runtime": {
            "python": runtime["python"],
            "torch": runtime["torch"],
            "torch_cuda": runtime["torch_cuda"],
            "vllm": runtime["engine_version"],
            "transformers": runtime["packages"]["transformers"],
            "pillow": runtime["packages"]["pillow"],
            "installed_distributions_count": runtime[
                "installed_distributions_count"
            ],
            "installed_distributions_sha256": runtime[
                "installed_distributions_sha256"
            ],
            "dtype": "torch.bfloat16",
            "attention_backend_selected": "FLASH_ATTN",
        },
    }

    assert deepseek_engine_identity_mismatches(
        profile,
        installation=installation,
        engine_metadata=metadata,
    ) == []

    wrong_backend = copy.deepcopy(metadata)
    wrong_backend["runtime"]["attention_backend_selected"] = "TORCH_SDPA"
    assert deepseek_engine_identity_mismatches(
        profile,
        installation=installation,
        engine_metadata=wrong_backend,
    ) == [
        "engine.runtime.attention_backend_selected differs from profile"
    ]

    wrong_snapshot = copy.deepcopy(metadata)
    wrong_snapshot["model"]["resolved_path"] = str(tmp_path / "other")
    wrong_snapshot["model"]["snapshot"] = "0" * 40
    assert deepseek_engine_identity_mismatches(
        profile,
        installation=installation,
        engine_metadata=wrong_snapshot,
    ) == [
        "engine.model.resolved_path differs from installation",
        "engine.model.snapshot differs from profile",
    ]

    wrong_runtime = copy.deepcopy(metadata)
    wrong_runtime["runtime"]["vllm"] = "0.0.0"
    assert deepseek_engine_identity_mismatches(
        profile,
        installation=installation,
        engine_metadata=wrong_runtime,
    ) == ["engine.runtime.vllm differs from profile"]

    wrong_distributions = copy.deepcopy(metadata)
    wrong_distributions["runtime"]["installed_distributions_sha256"] = (
        _digest("different-in-process-distributions")
    )
    assert deepseek_engine_identity_mismatches(
        profile,
        installation=installation,
        engine_metadata=wrong_distributions,
    ) == [
        "engine.runtime.installed_distributions_sha256 differs from profile"
    ]


def test_deepseek_engine_identity_checks_explicit_installation_backend(
    tmp_path,
):
    from sneeze.tpn.ocr_execution import (
        deepseek_engine_identity_mismatches,
    )

    profile = _profile(provisional=True)
    installation = _installation(tmp_path, "deepseek-vllm")
    installation["cuda"]["attention_backend"] = "FLASH_ATTN"
    runtime = profile["stages"]["grounded_ocr"]["runtime_selector"]
    metadata = {
        "model": {
            "requested": installation["snapshot"]["path"],
            "revision": None,
            "resolved_path": installation["snapshot"]["path"],
            "snapshot": installation["model"]["revision"],
        },
        "runtime": {
            "python": runtime["python"],
            "torch": runtime["torch"],
            "torch_cuda": runtime["torch_cuda"],
            "vllm": runtime["engine_version"],
            "transformers": runtime["packages"]["transformers"],
            "pillow": runtime["packages"]["pillow"],
            "installed_distributions_count": runtime[
                "installed_distributions_count"
            ],
            "installed_distributions_sha256": runtime[
                "installed_distributions_sha256"
            ],
            "dtype": runtime["dtype"],
            "attention_backend_selected": "TORCH_SDPA",
        },
    }

    reasons = deepseek_engine_identity_mismatches(
        profile,
        installation=installation,
        engine_metadata=metadata,
    )

    assert reasons == [
        "engine.runtime.attention_backend_selected differs from profile",
        (
            "engine.runtime.attention_backend_selected differs from "
            "installation"
        ),
    ]


def test_profile_resolution_supports_path_id_auto_and_provisional(tmp_path):
    from sneeze.tpn.ocr_execution import (
        OcrExecutionError,
        resolve_execution_profile,
    )
    from sneeze.tpn.ocr_profiles import write_execution_profile

    qualified = _profile()
    path = tmp_path / "qualified.json"
    write_execution_profile(qualified, path)
    observation = _observation(qualified)

    for selection in (str(path), qualified["profile_id"], "auto"):
        with pytest.raises(OcrExecutionError, match="not trusted"):
            resolve_execution_profile(
                selection,
                pipeline_id="local-mineru25",
                memory_tier="max",
                observation=observation,
                search_paths=[tmp_path],
            )
        with pytest.raises(OcrExecutionError, match="not trusted"):
            resolve_execution_profile(
                selection,
                pipeline_id="local-mineru25",
                memory_tier="max",
                observation=observation,
                search_paths=[tmp_path],
                allow_provisional=True,
            )

    provisional_dir = tmp_path / "provisional"
    provisional_dir.mkdir()
    provisional = _profile(provisional=True)
    provisional_path = provisional_dir / "profile.json"
    write_execution_profile(provisional, provisional_path)
    with pytest.raises(OcrExecutionError, match="provisional"):
        resolve_execution_profile(
            str(provisional_path),
            pipeline_id="local-mineru25",
            memory_tier="max",
            observation=_observation(provisional),
            search_paths=[provisional_dir],
        )
    assert resolve_execution_profile(
        str(provisional_path),
        pipeline_id="local-mineru25",
        memory_tier="max",
        observation=_observation(provisional),
        search_paths=[provisional_dir],
        allow_provisional=True,
    )["match"]["matched"]


def test_profile_provider_spec_is_bound_to_profile_and_installation(tmp_path):
    from sneeze.tpn.ocr_execution import build_profile_provider_spec

    profile = _profile(provisional=True)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    installation = _installation(tmp_path, "mineru25-pro-2605")

    spec = build_profile_provider_spec(
        profile=profile,
        profile_path=profile_path,
        memory_tier="max",
        installation=installation,
        allow_provisional=True,
    )

    assert spec["egress"] == "none"
    assert spec["secret_env_names"] == []
    assert (
        spec["lanes"]["best"]["settings"]
        == profile["memory_tiers"]["max"]["stages"]["markdown"]["settings"]
    )
    assert spec["argv"][-1] == "--allow-provisional-profile"
    assert profile["profile_sha256"] in spec["argv"]
    assert spec["env"]["HF_HUB_OFFLINE"] == "1"


@pytest.mark.parametrize(
    "timeout",
    (
        pytest.param(True, id="boolean"),
        pytest.param("30", id="string"),
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ),
)
def test_profile_provider_spec_rejects_invalid_timeout(tmp_path, timeout):
    from sneeze.tpn.ocr_execution import (
        OcrExecutionError,
        build_profile_provider_spec,
    )

    profile = _profile(provisional=True)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    installation = _installation(tmp_path, "mineru25-pro-2605")

    with pytest.raises(OcrExecutionError, match="finite positive"):
        build_profile_provider_spec(
            profile=profile,
            profile_path=profile_path,
            memory_tier="max",
            installation=installation,
            allow_provisional=True,
            timeout_seconds=timeout,
        )


def test_profile_provider_spec_preserves_positive_fractional_timeout(
    tmp_path,
):
    from sneeze.tpn.ocr_execution import build_profile_provider_spec

    profile = _profile(provisional=True)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    installation = _installation(tmp_path, "mineru25-pro-2605")

    spec = build_profile_provider_spec(
        profile=profile,
        profile_path=profile_path,
        memory_tier="max",
        installation=installation,
        allow_provisional=True,
        timeout_seconds=0.5,
    )

    assert spec["timeout_seconds"] == 0.5


def test_default_profile_search_prefers_bundled_then_machine_local(
    tmp_path,
):
    from sneeze.tpn.ocr_execution import (
        bundled_profile_directory,
        profile_search_paths,
    )

    paths = profile_search_paths(
        None,
    )
    assert paths[0] == bundled_profile_directory()
    assert paths[1].name == "ocr-profiles"

    explicit = (tmp_path / "profiles").resolve()
    assert profile_search_paths(str(explicit)) == (explicit,)


def test_explicit_profile_search_still_validates_bundled_trust(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn import ocr_execution

    def reject():
        raise ocr_execution.OcrExecutionError("synthetic trust mismatch")

    monkeypatch.setattr(
        ocr_execution,
        "validate_bundled_profile_registry",
        reject,
    )
    with pytest.raises(
        ocr_execution.OcrExecutionError,
        match="synthetic trust mismatch",
    ):
        ocr_execution.profile_search_paths(str(tmp_path.resolve()))


@pytest.mark.parametrize("selection", ["auto", "test-profile", "/x.json"])
def test_profile_resolution_cannot_bypass_bundled_trust(
    selection,
    monkeypatch,
):
    from sneeze.tpn import ocr_execution

    def reject():
        raise ocr_execution.OcrExecutionError("synthetic trust mismatch")

    monkeypatch.setattr(
        ocr_execution,
        "validate_bundled_profile_registry",
        reject,
    )
    with pytest.raises(
        ocr_execution.OcrExecutionError,
        match="synthetic trust mismatch",
    ):
        ocr_execution.resolve_execution_profile(
            selection,
            pipeline_id="local-mineru25",
            memory_tier="max",
            observation={},
            search_paths=[],
        )


def test_duplicate_profile_hash_cannot_be_accepted_as_trusted(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn import ocr_execution, ocr_profiles

    profile = _profile()
    payload = json.dumps(profile, indent=2, sort_keys=True)
    original = f'  "profile_sha256": "{profile["profile_sha256"]}",'
    duplicate = (
        f'  "profile_sha256": "{"0" * 64}",\n'
        f"{original}"
    )
    assert original in payload
    modified = payload.replace(original, duplicate, 1)
    assert modified.encode("utf-8") != payload.encode("utf-8")
    assert json.loads(modified) == profile
    path = tmp_path / "duplicate-trusted-hash.json"
    path.write_text(modified, encoding="utf-8")

    monkeypatch.setattr(
        ocr_execution,
        "validate_bundled_profile_registry",
        lambda: (),
    )
    monkeypatch.setattr(
        ocr_profiles,
        "TRUSTED_QUALIFIED_PROFILE_SHA256S",
        frozenset({profile["profile_sha256"]}),
    )
    with pytest.raises(
        ocr_execution.OcrExecutionError,
        match="duplicate JSON field in OCR profile: profile_sha256",
    ):
        ocr_execution.resolve_execution_profile(
            str(path),
            pipeline_id="local-mineru25",
            memory_tier="max",
            observation=_observation(profile),
            search_paths=[tmp_path],
        )


def test_bundled_profiles_and_reviewed_trust_registry_are_bijective():
    from sneeze.tpn.ocr_execution import validate_bundled_profile_registry
    from sneeze.tpn.ocr_profile_trust import (
        TRUSTED_QUALIFIED_PROFILE_SHA256S,
    )

    profiles = validate_bundled_profile_registry()
    assert isinstance(TRUSTED_QUALIFIED_PROFILE_SHA256S, frozenset)
    assert {profile["profile_sha256"] for profile in profiles} == set(
        TRUSTED_QUALIFIED_PROFILE_SHA256S
    )
    assert len({profile["profile_id"] for profile in profiles}) == len(
        profiles
    )


def test_bundled_profile_stages_match_backend_qualification_pins():
    from sneeze.tpn.ocr_backends import get_backend
    from sneeze.tpn.ocr_execution import validate_bundled_profile_registry

    for profile in validate_bundled_profile_registry():
        for stage in profile["stages"].values():
            backend = get_backend(stage["backend_id"])
            model = stage["model"]
            runtime = stage["runtime_selector"]
            packages = dict(backend.runtime_packages)

            assert model["requested"] == backend.model_id
            assert model["revision"] == backend.model_revision
            if backend.snapshot_identity_sha256 is not None:
                assert (
                    model["snapshot_files_sha256"]
                    == backend.snapshot_identity_sha256
                )

            assert (
                runtime["implementation_sha256"]
                == backend.qualified_implementation_sha256
            )
            assert runtime["python"] == backend.default_python
            assert runtime["engine"] == backend.runtime_engine
            assert (
                runtime["engine_version"]
                == packages[backend.runtime_engine_package]
            )
            assert runtime["torch"] == backend.qualified_torch_version
            assert runtime["torch_cuda"] == backend.qualified_torch_cuda
            assert runtime["dtype"] == backend.qualified_dtype
            assert runtime["packages"] == packages
