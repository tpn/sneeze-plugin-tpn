from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sneeze.tpn.providers import deepseek_vllm as adapter


def test_implementation_identity_matches_registry():
    from sneeze.tpn.ocr_implementation import (
        backend_implementation_record,
    )

    assert adapter.implementation_record() == backend_implementation_record(
        "deepseek-vllm"
    )


def test_isolated_adapter_binds_implementation_to_its_own_source_tree(
    tmp_path,
):
    source_tpn = Path(adapter.__file__).resolve().parents[1]
    copied_tpn = tmp_path / "copied" / "sneeze" / "tpn"
    shutil.copytree(
        source_tpn,
        copied_tpn,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    copied_component = copied_tpn / "ocr_vllm.py"
    copied_component.write_bytes(
        copied_component.read_bytes()
        + b"\n# Isolated source-binding regression marker.\n"
    )
    copied_adapter = copied_tpn / "providers" / "deepseek_vllm.py"
    probe = "\n".join(
        (
            "import json, pathlib, runpy, sys",
            "namespace = runpy.run_path(sys.argv[1], "
            "run_name='deepseek_source_binding_probe')",
            "record = namespace['implementation_record']()",
            "module = sys.modules['sneeze.tpn.ocr_vllm']",
            "print(json.dumps({'component': record['components']"
            "['ocr_vllm.py'], 'module': str(pathlib.Path("
            "module.__file__).resolve())}))",
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            probe,
            str(copied_adapter),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    copied_digest = hashlib.sha256(copied_component.read_bytes()).hexdigest()
    installed_digest = hashlib.sha256(
        (source_tpn / "ocr_vllm.py").read_bytes()
    ).hexdigest()
    assert copied_digest != installed_digest
    assert observed == {
        "component": copied_digest,
        "module": str(copied_component.resolve()),
    }


def test_adapter_source_binding_rejects_unexpected_layout(tmp_path):
    misplaced = tmp_path / "deepseek_vllm.py"
    misplaced.write_bytes(Path(adapter.__file__).read_bytes())

    with pytest.raises(
        adapter.DeepSeekBenchmarkAdapterError,
        match="unexpected package layout",
    ):
        adapter._source_root_for_adapter(misplaced)


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
    "x8AAwMCAO+Xk6cAAAAASUVORK5CYII="
)
REVISION = "9f30c71f441d010e5429c532364a86705536c53a"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_read_jsonl_rejects_duplicate_object_keys(tmp_path):
    requests_path = tmp_path / "duplicate-key.jsonl"
    requests_path.write_text(
        '{"attempt_id":"' + "a" * 64 + '","input":{"image":{},'
        '"image":{}}}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        adapter.DeepSeekBenchmarkAdapterError,
        match="duplicate JSON object key",
    ):
        adapter.read_jsonl(requests_path)


def _settings(**updates) -> dict:
    settings = {
        "attention_backend": "FLASH_ATTN",
        "batch_size": 2,
        "enable_chunked_prefill": False,
        "engine_overhead_headroom_bytes": 1,
        "gpu_memory_utilization": 0.499,
        "gpu_memory_utilization_denominator_bytes": 1000,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 2,
        "memory_budget_bytes": 500,
        "memory_tier": "50",
        "mm_encoder_attn_backend": None,
        "moe_backend": "triton",
        **adapter.FIXED_QUALITY_SETTINGS,
    }
    settings.update(updates)
    return settings


def test_installed_distribution_manifest_recomputes_identity():
    identity = adapter.installed_distributions_identity()
    assert identity["count"] == len(identity["distributions"])
    assert identity["sha256"] == adapter.sha256_json(
        identity["distributions"]
    )


def _request(image_path: Path, row: int, **updates) -> dict:
    instruction = adapter.FIXED_INSTRUCTION
    request = {
        "schema_version": 1,
        "attempt_id": hashlib.sha256(f"attempt-{row}".encode()).hexdigest(),
        "experiment_sha256": "e" * 64,
        "case_id": f"page-{row}",
        "lane": "grounded_ocr",
        "repeat": 1,
        "input": {
            "image": {
                "path": str(image_path.resolve()),
                "sha256": _sha256(PNG_BYTES),
                "dimensions": {"w": 1, "h": 1},
            },
            "raw_text": None,
        },
        "instruction": instruction,
        "instruction_sha256": hashlib.sha256(
            instruction.encode()
        ).hexdigest(),
        "settings": _settings(),
    }
    request.update(updates)
    return request


class FakeCuda:
    def __init__(self, *, integrated: int = 0) -> None:
        self.properties = SimpleNamespace(
            name="Synthetic GPU",
            major=8,
            minor=6,
            is_integrated=integrated,
            total_memory=1000,
        )
        self.reset_count = 0

    def is_available(self) -> bool:
        return True

    def current_device(self) -> int:
        return 0

    def set_device(self, device: int) -> None:
        assert device == 0

    def get_device_properties(self, _device: int):
        return self.properties

    def mem_get_info(self, _device: int) -> tuple[int, int]:
        return 600, 1000

    def memory_allocated(self, _device: int) -> int:
        return 101

    def memory_reserved(self, _device: int) -> int:
        return 202

    def max_memory_allocated(self, _device: int) -> int:
        return 303

    def max_memory_reserved(self, _device: int) -> int:
        return 404

    def reset_peak_memory_stats(self, _device: int) -> None:
        self.reset_count += 1


def _fake_torch(*, integrated: int = 0):
    return SimpleNamespace(
        __version__="2.test+cu130",
        cuda=FakeCuda(integrated=integrated),
        version=SimpleNamespace(cuda="13.0"),
    )


class FakeProcessor:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.prepare_count = 0
        self.shutdown_count = 0
        self.run_calls = []
        self.generation_metadata = {}
        self.__class__.instances.append(self)

    def prepare_engine(self):
        self.prepare_count += 1
        return object(), object()

    def run(self, tasks, *, llm, sampling_params) -> None:
        assert llm is not None
        assert sampling_params is not None
        self.run_calls.append(list(tasks))
        for task in tasks:
            Path(task.output_path).write_text(
                "Grounded page text\n", encoding="utf-8"
            )
            self.generation_metadata[task.output_path] = {
                "status": "completed",
                "prompt": "<image>\n<|grounding|>Free OCR.",
                "prompt_mode": "plain_ocr",
                "output_format": "txt",
                "runtime": {
                    "torch": "2.test+cu130",
                    "vllm": "0.test",
                    "transformers": "5.test",
                    "dtype": "torch.bfloat16",
                    "attention_backend_requested": "FLASH_ATTN",
                    "attention_backend_selected": "FLASH_ATTN",
                    "mm_encoder_attention_backend_requested": None,
                    "moe_backend_requested": "triton",
                    "moe_backend_configured": "triton",
                    "jit_environment": {
                        "MAX_JOBS": "1",
                        "FLASHINFER_NVCC_THREADS": "1",
                    },
                    "engine_init_kwargs": {
                        "moe_backend": "triton",
                    },
                },
                "sampling_params": {
                    "max_tokens": 8192,
                    "temperature": 0.0,
                },
                "logits_processor_extra_args": {
                    "ngram_size": 10,
                    "window_size": 50,
                },
                "prompt_token_count": 12,
                "generated_token_count": 34,
                "finish_reason": "stop",
                "stop_reason": None,
                "termination": {
                    "eos": True,
                    "length": False,
                    "max_tokens": False,
                },
                "repetition": {"tokens": {"token_count": 34}},
            }

    def shutdown_engine(self, _llm) -> None:
        self.shutdown_count += 1


@pytest.fixture(autouse=True)
def _reset_fake_processor():
    FakeProcessor.instances.clear()


def _snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / REVISION
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"synthetic weights")
    return snapshot


def test_runtime_record_requires_selected_attention_backend(monkeypatch):
    class FakeTorch:
        __version__ = "2.test+cu130"

        class version:
            cuda = "13.0"

    monkeypatch.setattr(adapter, "package_versions", lambda _names: {})
    with pytest.raises(
        adapter.DeepSeekBenchmarkAdapterError,
        match="selected attention backend",
    ):
        adapter._runtime_record(
            torch_module=FakeTorch,
            generation_metadata={
                "runtime": {
                    "dtype": "torch.bfloat16",
                    "attention_backend_requested": "FLASH_ATTN",
                    "attention_backend_selected": None,
                }
            },
            telemetry={},
            effective_settings={},
            startup_installed_distributions={
                "count": 1,
                "sha256": "a" * 64,
            },
        )


def test_success_response_preserves_explicit_non_eos_stop(
    tmp_path, monkeypatch
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    output = artifacts / "page.md"
    output.write_text("Stopped output\n", encoding="utf-8")
    monkeypatch.setattr(adapter, "_runtime_record", lambda **_kwargs: {})
    request = {
        "attempt_id": "a" * 64,
        "input": {
            "image": {
                "path": "/scratch/page.png",
                "sha256": "b" * 64,
                "dimensions": {"w": 100, "h": 100},
            }
        },
    }
    response = adapter._success_response(
        request=request,
        row_index=0,
        output_path=output,
        generation_metadata={
            "status": "completed",
            "prompt": "<image>\n<|grounding|>Free OCR.",
            "finish_reason": "stop",
            "stop_reason": "explicit_stop",
            "termination": {
                "eos": False,
                "length": False,
                "max_tokens": False,
            },
        },
        effective_settings={},
        snapshot_record={
            "path": "/scratch/snapshot",
            "commit": REVISION,
            "file_count": 1,
            "files_sha256": "c" * 64,
        },
        model_id=adapter.DEFAULT_OCR_MODEL,
        revision=REVISION,
        batch_wall_seconds=1.0,
        telemetry={"torch_module": object()},
        artifacts_dir=artifacts,
        startup_installed_distributions={
            "count": 1,
            "sha256": "d" * 64,
        },
    )

    assert response["generation"]["finish_reason"] == "stop"
    assert response["generation"]["stop_reason"] == "explicit_stop"
    assert response["generation"]["eos"] is False
    assert response["generation"]["max_tokens"] is False


def test_snapshot_identity_matches_installation_hash_for_hf_symlinks(
    tmp_path,
):
    from sneeze.tpn.ocr_installations import _snapshot_record

    cache = (tmp_path / "hub").resolve()
    model_root = cache / "models--deepseek-ai--DeepSeek-OCR"
    snapshot = model_root / "snapshots" / REVISION
    blobs = model_root / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    blob = blobs / "synthetic-blob"
    blob.write_bytes(b"resolved model bytes")
    (snapshot / "config.json").symlink_to(blob)
    (snapshot / "tokenizer.json").write_text("{}\n", encoding="utf-8")

    provider = adapter.snapshot_identity(snapshot, REVISION)
    installation = _snapshot_record(snapshot, cache)

    assert provider["files_sha256"] == installation["files_sha256"]
    assert provider["file_count"] == installation["file_count"]
    assert provider["size_bytes"] == installation["size_bytes"]
    assert set(provider["files"]["config.json"]) == {
        "sha256",
        "size_bytes",
    }


def _memory_snapshot(
    *,
    free_bytes: int,
    linux_available_bytes: int,
    cgroup_current_bytes: int,
    cgroup_peak_bytes: int,
    self_peak_rss_bytes: int,
    children_peak_rss_bytes: int,
    vm_hwm_bytes: int,
) -> dict:
    return {
        "cuda": {
            "memory_topology": "unified",
            "cuda_free_memory_bytes": free_bytes,
            "cuda_total_memory_bytes": 1000,
            "torch_allocator": {
                "peak_allocated_bytes": 200,
                "peak_reserved_bytes": 300,
            },
        },
        "linux": {
            "MemTotal": 2000,
            "MemAvailable": linux_available_bytes,
        },
        "cgroup_v2": {
            "available": True,
            "memory_max_bytes": 1000,
            "memory_max_is_unlimited": False,
            "memory_current_bytes": cgroup_current_bytes,
            "memory_peak_bytes": cgroup_peak_bytes,
        },
        "process": {
            "status_bytes": {"VmRSS": 50, "VmHWM": vm_hwm_bytes},
            "self_max_rss_bytes": self_peak_rss_bytes,
            "children_max_rss_bytes": children_peak_rss_bytes,
        },
    }


def test_provider_reuses_one_engine_and_records_complete_telemetry(
    tmp_path, monkeypatch
):
    from sneeze.tpn.ocr_benchmark import _provider_metadata_gaps

    image_path = tmp_path / "page.png"
    image_path.write_bytes(PNG_BYTES)
    requests = [_request(image_path, row) for row in range(3)]
    requests_path = tmp_path / "requests.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    _write_jsonl(requests_path, requests)
    fake_torch = _fake_torch()
    snapshot = _snapshot(tmp_path)
    requested_packages = []
    startup_distributions = {
        "count": 198,
        "sha256": hashlib.sha256(b"startup-distributions").hexdigest(),
        "distributions": [],
    }
    distribution_identity_calls = 0

    def package_versions(names):
        requested_packages.extend(names)
        return {}

    def installed_distributions_identity():
        nonlocal distribution_identity_calls
        distribution_identity_calls += 1
        if distribution_identity_calls == 1:
            return startup_distributions
        return {
            "count": 214,
            "sha256": hashlib.sha256(b"post-vllm-distributions").hexdigest(),
            "distributions": [],
        }

    monkeypatch.setattr(adapter, "package_versions", package_versions)
    monkeypatch.setattr(
        adapter,
        "installed_distributions_identity",
        installed_distributions_identity,
    )
    monkeypatch.setenv("MAX_JOBS", "1")
    monkeypatch.setenv("FLASHINFER_NVCC_THREADS", "1")

    adapter.run_provider(
        requests_path=requests_path,
        responses_path=responses_path,
        artifacts_dir=artifacts_dir,
        model_id=adapter.DEFAULT_OCR_MODEL,
        revision=REVISION,
        tuning=True,
        processor_class=FakeProcessor,
        torch_module=fake_torch,
        snapshot_resolver=lambda _model, _revision: snapshot,
    )

    assert len(FakeProcessor.instances) == 1
    assert distribution_identity_calls == 1
    assert set(requested_packages) >= {
        "huggingface-hub",
        "pdf2image",
        "pillow",
        "torch",
        "torchaudio",
        "torchvision",
        "tqdm",
        "transformers",
        "vllm",
    }
    processor = FakeProcessor.instances[0]
    assert processor.prepare_count == 1
    assert processor.shutdown_count == 1
    assert [len(tasks) for tasks in processor.run_calls] == [2, 1]
    assert fake_torch.cuda.reset_count == 2
    assert processor.kwargs["batch_size"] == 2
    assert processor.kwargs["max_num_seqs"] == 2
    assert processor.kwargs["max_num_batched_tokens"] == 8192
    assert processor.kwargs["gpu_memory_utilization"] == 0.499
    assert processor.kwargs["attention_backend"] == "FLASH_ATTN"
    assert processor.kwargs["mm_encoder_attn_backend"] is None
    assert processor.kwargs["enable_chunked_prefill"] is False
    assert processor.kwargs["moe_backend"] == "triton"
    assert processor.kwargs["max_tokens"] == 8192
    assert processor.kwargs["ngram_size"] == 10
    assert processor.kwargs["window_size"] == 50
    assert processor.kwargs["temperature"] == 0.0

    responses = _read_jsonl(responses_path)
    assert len(responses) == 3
    for request, response in zip(requests, responses, strict=True):
        assert response["attempt_id"] == request["attempt_id"]
        assert response["status"] == "completed"
        assert response["output"] == {
            "format": "markdown",
            "text": "Grounded page text\n",
        }
        assert response["effective_request"]["prompt"] == (
            "<image>\n<|grounding|>Free OCR."
        )
        assert response["model"]["revision"] == REVISION
        assert response["model"]["resolved_snapshot"] == str(snapshot)
        assert response["runtime"]["engine"] == "vllm"
        assert response["runtime"]["engine_version"] == "0.test"
        assert response["runtime"]["dtype"] == "bfloat16"
        assert response["runtime"]["installed_distributions"] == {
            "count": startup_distributions["count"],
            "sha256": startup_distributions["sha256"],
        }
        device = response["runtime"]["device"]
        assert device["memory_topology"] == "dedicated"
        assert device["cuda_free_memory_bytes"] == 600
        allocator = device["torch_allocator"]
        assert allocator["peak_allocated_bytes"] == 303
        assert allocator["peak_reserved_bytes"] == 404
        gpu_memory = response["runtime"]["gpu_memory"]
        assert gpu_memory["memory_kind"] == "dedicated_vram"
        assert gpu_memory["memory_budget_bytes"] == 500
        assert gpu_memory["torch_peak_reserved_bytes"] == 404
        assert gpu_memory["observed_cuda_used_delta_bytes"] == 0
        assert gpu_memory["process_peak_rss_bytes"] > 0
        assert (
            gpu_memory["parent_plus_children_peak_rss_bytes"]
            >= (gpu_memory["process_peak_rss_bytes"])
        )
        assert gpu_memory["observed_linux_mem_available_delta_bytes"] >= 0
        assert "cgroup_v2_memory" in gpu_memory
        memory = response["runtime"]["memory"]
        assert "before_engine" in memory
        assert "after_engine" in memory
        assert "batch_before" in memory
        assert "batch_after" in memory
        assert "after_shutdown" in memory
        assert memory["batch_after"]["process"]["self_max_rss_bytes"] > 0
        params = response["generation"]["params"]
        assert params["fixed_quality_settings"] == (
            adapter.FIXED_QUALITY_SETTINGS
        )
        assert params["execution_settings"]["memory_tier"] == "50"
        assert response["runtime"]["jit_environment"] == {
            "MAX_JOBS": "1",
            "FLASHINFER_NVCC_THREADS": "1",
        }
        assert params["execution_settings"]["memory_budget_fraction"] == 0.5
        assert (
            params["execution_settings"]["engine_overhead_headroom_bytes"]
            == 1
        )
        assert response["generation"]["input_tokens"] == 12
        assert response["generation"]["output_tokens"] == 34
        assert response["generation"]["eos"] is True
        assert response["generation"]["max_tokens"] is False
        assert response["generation"]["stop_reason"] == "eos"
        assert response["usage"]["cost_usd"] == 0.0
        assert len(response["artifacts"]) == 2
        for artifact in response["artifacts"]:
            path = artifacts_dir / artifact["relative_path"]
            assert path.is_file()
            assert adapter.sha256_path(path) == artifact["sha256"]
        native_path = (
            artifacts_dir / response["artifacts"][1]["relative_path"]
        )
        native = json.loads(native_path.read_text(encoding="utf-8"))
        assert "torch_module" not in native["memory_telemetry"]
        assert native["fixed_quality_settings"] == (
            adapter.FIXED_QUALITY_SETTINGS
        )
        assert (
            _provider_metadata_gaps(
                response,
                request,
                {
                    "model": {
                        "requested": adapter.DEFAULT_OCR_MODEL,
                        "revision": REVISION,
                    }
                },
                artifacts_dir,
            )
            == []
        )


def test_unified_memory_is_distinguished_from_dedicated_memory():
    telemetry = adapter.cuda_memory_telemetry(_fake_torch(integrated=1))

    assert telemetry["memory_topology"] == "unified"
    assert telemetry["is_integrated"] == 1
    assert telemetry["device_total_memory_bytes"] == 1000
    assert telemetry["cuda_total_memory_bytes"] == 1000


def test_cgroup_v2_memory_sampling_does_not_treat_max_as_capacity(tmp_path):
    membership = tmp_path / "self.cgroup"
    membership.write_text("0::/workload\n", encoding="utf-8")
    workload = tmp_path / "workload"
    workload.mkdir()
    (workload / "memory.max").write_text("max\n", encoding="utf-8")
    (workload / "memory.current").write_text("123\n", encoding="utf-8")
    (workload / "memory.peak").write_text("456\n", encoding="utf-8")

    telemetry = adapter.cgroup_v2_memory_telemetry(
        membership_path=membership,
        cgroup_root=tmp_path,
    )

    assert telemetry == {
        "available": True,
        "memory_max_bytes": None,
        "memory_max_is_unlimited": True,
        "memory_current_bytes": 123,
        "memory_peak_bytes": 456,
    }
    assert str(tmp_path) not in json.dumps(telemetry)


def test_unified_summary_records_linux_cgroup_and_aggregate_rss_deltas():
    baseline = _memory_snapshot(
        free_bytes=800,
        linux_available_bytes=1500,
        cgroup_current_bytes=100,
        cgroup_peak_bytes=400,
        self_peak_rss_bytes=100,
        children_peak_rss_bytes=200,
        vm_hwm_bytes=150,
    )
    after = _memory_snapshot(
        free_bytes=500,
        linux_available_bytes=900,
        cgroup_current_bytes=500,
        cgroup_peak_bytes=700,
        self_peak_rss_bytes=250,
        children_peak_rss_bytes=300,
        vm_hwm_bytes=275,
    )
    intermediate = _memory_snapshot(
        free_bytes=400,
        linux_available_bytes=700,
        cgroup_current_bytes=600,
        cgroup_peak_bytes=800,
        self_peak_rss_bytes=50,
        children_peak_rss_bytes=50,
        vm_hwm_bytes=50,
    )

    effective_settings = adapter.enforce_memory_budget(
        adapter.validate_tuning_settings(_settings()),
        1000,
    )
    summary = adapter._gpu_memory_summary(
        {
            "before_engine": baseline,
            "batch_after": after,
            "samples": [intermediate],
        },
        effective_settings,
    )

    assert summary["memory_kind"] == "unified_system"
    assert summary["linux_baseline_mem_available_bytes"] == 1500
    assert summary["linux_minimum_mem_available_bytes"] == 700
    assert summary["observed_linux_mem_available_delta_bytes"] == 800
    assert summary["process_peak_rss_bytes"] == 300
    assert summary["parent_peak_rss_bytes"] == 275
    assert summary["children_peak_rss_bytes"] == 300
    assert summary["parent_plus_children_peak_rss_bytes"] == 575
    assert (
        "not_simultaneous"
        in (summary["parent_plus_children_peak_rss_semantics"])
    )
    assert summary["cgroup_v2_memory"] == {
        "available": True,
        "capacity_bytes": 1000,
        "capacity_is_unlimited": False,
        "baseline_current_bytes": 100,
        "maximum_current_bytes": 600,
        "observed_current_delta_bytes": 500,
        "baseline_peak_bytes": 400,
        "maximum_peak_bytes": 800,
        "observed_peak_delta_bytes": 400,
        "baseline_headroom_bytes": 900,
        "minimum_headroom_bytes": 400,
        "observed_headroom_delta_bytes": 500,
        "memory_peak_semantics": (
            "cgroup_lifetime_high_water_mark_not_reset_by_adapter"
        ),
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"scheduler_delay_factor": 1}, "unsupported controls"),
        ({"max_tokens": 1}, "must remain fixed"),
        ({"batch_size": True}, "positive integer"),
        ({"batch_size": 3}, "cannot exceed max_num_seqs"),
        ({"gpu_memory_utilization": 0}, "greater than zero"),
        ({"memory_tier": "75"}, "must be 25, 50, or max"),
        ({"attention_backend": "arbitrary"}, "not allowlisted"),
        ({"moe_backend": "arbitrary"}, "moe_backend must be one of"),
        ({"moe_backend": "auto"}, "requires moe_backend=triton"),
        ({"enable_chunked_prefill": 1}, "must be boolean"),
    ],
)
def test_tuning_settings_fail_closed(updates, message):
    settings = _settings()
    settings.update(updates)

    with pytest.raises(adapter.DeepSeekBenchmarkAdapterError, match=message):
        adapter.validate_tuning_settings(settings)


def test_gpu_utilization_cannot_exceed_memory_budget():
    settings = adapter.validate_tuning_settings(
        _settings(gpu_memory_utilization=0.51)
    )

    with pytest.raises(
        adapter.DeepSeekBenchmarkAdapterError,
        match="exceeds its memory budget cap",
    ):
        adapter.enforce_memory_budget(settings, 1000)


def test_auto_backends_mean_no_explicit_vllm_backend():
    settings = adapter.validate_tuning_settings(
        _settings(
            attention_backend="auto",
            mm_encoder_attn_backend="auto",
        )
    )

    assert settings["attention_backend"] is None
    assert settings["mm_encoder_attn_backend"] is None


def test_device_selection_uses_cuda_visible_devices():
    assert adapter.validate_device("cuda:0") == 0
    with pytest.raises(
        adapter.DeepSeekBenchmarkAdapterError,
        match="CUDA_VISIBLE_DEVICES",
    ):
        adapter.validate_device("cuda:1")


def test_requests_are_grounded_image_only_and_share_one_configuration(
    tmp_path,
):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(PNG_BYTES)
    first = _request(image_path, 1)
    wrong_lane = _request(image_path, 2, lane="best")
    with pytest.raises(
        adapter.DeepSeekBenchmarkAdapterError,
        match="lane=grounded_ocr",
    ):
        adapter.validate_request(wrong_lane)

    with_raw = _request(image_path, 3)
    with_raw["input"]["raw_text"] = {"text": "not allowed"}
    with pytest.raises(
        adapter.DeepSeekBenchmarkAdapterError,
        match="cannot include raw text",
    ):
        adapter.validate_request(with_raw)

    wrong_prompt = _request(image_path, 4, instruction="Convert to markdown")
    with pytest.raises(
        adapter.DeepSeekBenchmarkAdapterError,
        match="exact Free OCR",
    ):
        adapter.validate_request(wrong_prompt)

    second = _request(image_path, 5)
    second["settings"] = _settings(batch_size=1)
    with pytest.raises(
        adapter.DeepSeekBenchmarkAdapterError,
        match="identical execution settings",
    ):
        adapter.validate_request_batch([first, second])

    duplicate = _request(image_path, 1)
    with pytest.raises(
        adapter.DeepSeekBenchmarkAdapterError,
        match="duplicate attempt IDs",
    ):
        adapter.validate_request_batch([first, duplicate])


def test_explicit_tuning_flag_is_required(tmp_path):
    with pytest.raises(SystemExit):
        adapter.parse_args(
            [
                "--requests-jsonl",
                str(tmp_path / "requests.jsonl"),
                "--responses-jsonl",
                str(tmp_path / "responses.jsonl"),
                "--artifacts-dir",
                str(tmp_path / "artifacts"),
            ]
        )

    with pytest.raises(
        adapter.DeepSeekBenchmarkAdapterError,
        match="requires explicit tuning mode",
    ):
        adapter.run_provider(
            requests_path=tmp_path / "missing.jsonl",
            responses_path=tmp_path / "responses.jsonl",
            artifacts_dir=tmp_path / "artifacts",
            model_id=adapter.DEFAULT_OCR_MODEL,
            revision=REVISION,
            tuning=False,
        )


def test_bad_image_hash_returns_artifact_without_starting_engine(tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(PNG_BYTES)
    request = _request(image_path, 1)
    request["input"]["image"]["sha256"] = "0" * 64
    requests_path = tmp_path / "requests.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    _write_jsonl(requests_path, [request])

    adapter.run_provider(
        requests_path=requests_path,
        responses_path=responses_path,
        artifacts_dir=artifacts_dir,
        model_id=adapter.DEFAULT_OCR_MODEL,
        revision=REVISION,
        tuning=True,
        processor_class=FakeProcessor,
        torch_module=_fake_torch(),
        snapshot_resolver=lambda _model, _revision: _snapshot(tmp_path),
    )

    assert not FakeProcessor.instances
    response = _read_jsonl(responses_path)[0]
    assert response["status"] == "error"
    assert response["error"]["phase"] == "input_validation"
    artifact = artifacts_dir / response["artifacts"][0]["relative_path"]
    assert artifact.is_file()
    assert adapter.sha256_path(artifact) == response["artifacts"][0]["sha256"]
