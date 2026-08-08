from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from sneeze.tpn.providers import mineru25


def test_standalone_implementation_identity_matches_registry():
    from sneeze.tpn.ocr_implementation import (
        backend_implementation_record,
    )

    assert mineru25.implementation_record() == backend_implementation_record(
        "mineru25-pro-2605"
    )


def test_standalone_implementation_attests_profile_validation_logic():
    assert {
        "ocr_backend_setup.py",
        "ocr_implementation.py",
        "ocr_profiles.py",
    } <= set(mineru25._IMPLEMENTATION_COMPONENT_PATHS)


def test_installed_distribution_manifest_recomputes_identity():
    identity = mineru25.installed_distributions_identity()
    assert identity["count"] == len(identity["distributions"])
    assert identity["sha256"] == mineru25.sha256_json(
        identity["distributions"]
    )


def test_startup_distribution_identity_discards_manifest_values(monkeypatch):
    monkeypatch.setattr(
        mineru25,
        "installed_distributions_identity",
        lambda: {
            "count": 64,
            "sha256": "a" * 64,
            "distributions": ["private-looking-name==1"],
        },
    )

    assert mineru25.startup_installed_distributions_identity() == {
        "count": 64,
        "sha256": "a" * 64,
    }


def _sealed_profile(tmp_path, *, state="provisional"):
    from sneeze.tpn.ocr_backends import (
        DEEPSEEK_IMPLEMENTATION_SHA256,
        MINERU_IMPLEMENTATION_SHA256,
    )
    from sneeze.tpn.ocr_profiles import (
        get_pipeline_definition,
        qualification_policy_sha256,
        seal_execution_profile,
    )

    pipeline = get_pipeline_definition("local-mineru25")
    runtime = {
        "python": "3.12.13",
        "engine": "transformers",
        "engine_version": "4.57.6",
        "torch": "2.11.0+cu130",
        "torch_cuda": "13.0",
        "packages": {"transformers": "4.57.6"},
        "attention_backend": "sdpa",
        "dtype": "bfloat16",
    }
    stages = {
        stage["stage_id"]: {
            "backend_id": stage["backend_id"],
            "capability": stage["capability"],
            "model": {
                **stage["model"],
                "snapshot_files_sha256": hashlib.sha256(
                    ("snapshot-" + stage["backend_id"]).encode("utf-8")
                ).hexdigest(),
            },
            "batch_semantics": stage["batch_semantics"],
            "runtime_selector": {
                **runtime,
                "implementation_sha256": (
                    DEEPSEEK_IMPLEMENTATION_SHA256
                    if stage["backend_id"] == "deepseek-vllm"
                    else MINERU_IMPLEMENTATION_SHA256
                ),
                "installed_distributions_count": 1,
                "installed_distributions_sha256": hashlib.sha256(
                    (
                        "installed-distributions-" + stage["backend_id"]
                    ).encode("utf-8")
                ).hexdigest(),
            },
        }
        for stage in pipeline["stages"]
    }
    total_memory = 48 * 1024**3
    budget = int(total_memory * 0.9)
    settings = {
        **mineru25.QUALIFIED_REQUEST_SETTINGS,
        "batch_size": 4,
        "memory_tier": "max",
        "memory_budget_bytes": budget,
    }
    grounded_settings = {
        "attention_backend": "FLASH_ATTN",
        "batch_size": 1,
        "enable_chunked_prefill": True,
        "engine_overhead_headroom_bytes": 1024**3,
        "gpu_memory_utilization": 0.8,
        "gpu_memory_utilization_denominator_bytes": total_memory,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 1,
        "max_tokens": 8192,
        "memory_budget_bytes": budget,
        "memory_tier": "max",
        "mm_encoder_attn_backend": "auto",
        "moe_backend": "triton",
        "ngram_size": 10,
        "temperature": 0.0,
        "window_size": 50,
    }
    tiers = {
        name: {
            "supported": name == "max",
            "reason": None if name == "max" else "not tested",
            "budget_fraction": {"25": 0.25, "50": 0.5, "max": 0.9}[name],
            "budget_bytes": {
                "25": 12 * 1024**3,
                "50": 24 * 1024**3,
                "max": budget,
            }[name],
            "reserve_bytes": 2 * 1024**3,
            "stages": (
                {
                    "grounded_ocr": {
                        "settings": grounded_settings,
                        "observed_peak_memory_bytes": 8 * 1024**3,
                        "confirmed_pages_per_second": 0.05,
                    },
                    "markdown": {
                        "settings": settings,
                        "observed_peak_memory_bytes": 10 * 1024**3,
                        "confirmed_pages_per_second": 0.04,
                    },
                }
                if name == "max"
                else {}
            ),
        }
        for name in ("25", "50", "max")
    }
    profile = seal_execution_profile(
        {
            "schema_version": 1,
            "kind": "ocr_execution_profile",
            "profile_id": "test-mineru-profile",
            "display_name": "Test MinerU profile",
            "pipeline_id": "local-mineru25",
            "pipeline_definition_sha256": pipeline["definition_sha256"],
            "hardware_selector": {
                "schema_version": 1,
                "vendor": "nvidia",
                "product_name": "Test GPU",
                "pci_device_id": None,
                "compute_capability": [9, 0],
                "memory_model": "dedicated",
                "total_memory_bytes": 48 * 1024**3,
            },
            "stages": stages,
            "memory_tiers": tiers,
            "qualification": {
                "state": state,
                "policy_sha256": qualification_policy_sha256(),
                "required_suites": ["hardware-smoke"],
                "evidence": [],
            },
        }
    )
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path, profile, settings


def _request(tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"png")
    return {
        "schema_version": 1,
        "attempt_id": "a" * 64,
        "lane": "best",
        "settings": dict(mineru25.QUALIFIED_REQUEST_SETTINGS),
        "input": {
            "image": {
                "path": str(image_path),
                "sha256": hashlib.sha256(b"png").hexdigest(),
                "dimensions": {"w": 1, "h": 1},
            }
        },
    }


def test_read_jsonl_rejects_empty_and_duplicate_object_keys(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL is empty"):
        mineru25.read_jsonl(str(empty))

    duplicate_key = tmp_path / "duplicate-key.jsonl"
    duplicate_key.write_text(
        '{"attempt_id":"' + "a" * 64 + '","input":{"image":{},'
        '"image":{}}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        mineru25.read_jsonl(str(duplicate_key))


def test_duplicate_attempt_ids_fail_before_artifact_or_model_work(
    tmp_path,
    monkeypatch,
):
    requests_path = tmp_path / "requests.jsonl"
    request = _request(tmp_path)
    requests_path.write_text(
        json.dumps(request) + "\n" + json.dumps(request) + "\n",
        encoding="utf-8",
    )
    responses_path = tmp_path / "output" / "responses.jsonl"
    artifacts_dir = tmp_path / "output" / "artifacts"
    monkeypatch.setattr(
        mineru25,
        "parse_args",
        lambda: SimpleNamespace(
            requests_jsonl=str(requests_path),
            responses_jsonl=str(responses_path),
            artifacts_dir=str(artifacts_dir),
        ),
    )
    monkeypatch.setattr(
        mineru25,
        "startup_installed_distributions_identity",
        lambda: pytest.fail("runtime inspection must not start"),
    )

    with pytest.raises(ValueError, match="duplicate attempt IDs"):
        mineru25.main()

    assert not responses_path.exists()
    assert not artifacts_dir.exists()


def test_validate_request_requires_pinned_native_lane(tmp_path):
    request = _request(tmp_path)
    mineru25.validate_request(request)

    request["lane"] = "parity"
    with pytest.raises(ValueError, match="only supports lane=best"):
        mineru25.validate_request(request)

    request["lane"] = "best"
    request["settings"]["backend"] = "direct-transformers"
    with pytest.raises(ValueError, match="exactly match"):
        mineru25.validate_request(request)

    request["settings"] = {
        **mineru25.QUALIFIED_REQUEST_SETTINGS,
        "batch_size": 8,
    }
    with pytest.raises(ValueError, match="exactly match"):
        mineru25.validate_request(request)

    request["settings"] = {
        **mineru25.QUALIFIED_REQUEST_SETTINGS,
        "unsupported_control": True,
    }
    with pytest.raises(ValueError, match="exactly match"):
        mineru25.validate_request(request)


def test_validate_request_accepts_bounded_explicit_tuning_settings(tmp_path):
    request = _request(tmp_path)
    request["settings"] = {
        **mineru25.QUALIFIED_REQUEST_SETTINGS,
        "batch_size": 8,
        "memory_tier": "50",
        "memory_budget_bytes": 24 * 1024**3,
    }

    mineru25.validate_request(request, tuning=True)

    request["settings"]["batch_size"] = 65
    with pytest.raises(ValueError, match="batch_size"):
        mineru25.validate_request(request, tuning=True)

    request["settings"]["batch_size"] = 1
    request["settings"]["unsupported_control"] = True
    with pytest.raises(ValueError, match="unsupported controls"):
        mineru25.validate_request(request, tuning=True)


def test_apply_tuning_memory_budget_uses_absolute_device_fraction():
    calls = []

    class FakeCuda:
        @staticmethod
        def get_device_properties(_device):
            return SimpleNamespace(total_memory=48 * 1024**3)

        @staticmethod
        def set_per_process_memory_fraction(fraction, device):
            calls.append((fraction, device))

    torch = SimpleNamespace(cuda=FakeCuda())
    settings = {
        "memory_tier": "25",
        "memory_budget_bytes": 12 * 1024**3,
    }

    result = mineru25.apply_tuning_memory_budget(
        torch,
        "cuda:0",
        settings,
    )

    assert calls == [(0.25, "cuda:0")]
    assert result == {
        "memory_tier": "25",
        "budget_bytes": 12 * 1024**3,
        "allocator_fraction": 0.25,
        "allocator_cap_applied": True,
    }


def test_runtime_metadata_exposes_profile_shaped_implementation_identity(
    tmp_path,
    monkeypatch,
):
    import sys

    from sneeze.tpn.ocr_execution import stage_runtime_mismatches

    _profile_path, profile, _settings = _sealed_profile(tmp_path)
    expected = profile["stages"]["markdown"]["runtime_selector"]

    properties = SimpleNamespace(
        total_memory=48 * 1024**3,
        name="Synthetic GPU",
        is_integrated=False,
    )
    fake_torch = SimpleNamespace(
        __version__=expected["torch"],
        version=SimpleNamespace(cuda=expected["torch_cuda"]),
        device=lambda value: value,
        cuda=SimpleNamespace(
            get_device_properties=lambda _device: properties,
            get_device_capability=lambda _device: (8, 6),
        ),
        are_deterministic_algorithms_enabled=lambda: True,
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(
                benchmark=False,
                deterministic=True,
                allow_tf32=False,
            ),
            cuda=SimpleNamespace(
                matmul=SimpleNamespace(allow_tf32=False),
            ),
        ),
    )
    fake_transformers = SimpleNamespace(
        __version__=expected["engine_version"]
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        mineru25.platform,
        "python_version",
        lambda: expected["python"],
    )
    monkeypatch.setattr(
        mineru25,
        "package_versions",
        lambda _names: dict(expected["packages"]),
    )
    model = SimpleNamespace(
        device="cuda:0",
        config=SimpleNamespace(_attn_implementation="sdpa"),
        dtype="torch.bfloat16",
    )

    runtime = mineru25.runtime_metadata(
        model,
        SimpleNamespace(),
        startup_installed_distributions={
            "count": expected["installed_distributions_count"],
            "sha256": expected["installed_distributions_sha256"],
        },
    )

    assert runtime["implementation_sha256"] == runtime["implementation"][
        "sha256"
    ]
    assert stage_runtime_mismatches(
        profile,
        stage_id="markdown",
        runtime=runtime,
    ) == []


def test_cgroup_v2_memory_sampling_does_not_treat_max_as_capacity(tmp_path):
    membership = tmp_path / "self.cgroup"
    membership.write_text("0::/workload\n", encoding="utf-8")
    workload = tmp_path / "workload"
    workload.mkdir()
    (workload / "memory.max").write_text("max\n", encoding="utf-8")
    (workload / "memory.current").write_text("123\n", encoding="utf-8")
    (workload / "memory.peak").write_text("456\n", encoding="utf-8")

    telemetry = mineru25._cgroup_v2_memory(
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


@pytest.mark.parametrize(
    ("integrated", "memory_kind", "capacity_source"),
    [
        (True, "unified_system", "linux_mem_available"),
        (False, "dedicated_vram", "cuda_device_total"),
    ],
)
def test_memory_tracker_records_linux_and_cgroup_deltas(
    monkeypatch,
    integrated,
    memory_kind,
    capacity_source,
):
    class FakeCuda:
        def __init__(self):
            self.free_values = iter((900, 600, 700))

        @staticmethod
        def get_device_properties(_device):
            return SimpleNamespace(
                total_memory=1000,
                is_integrated=integrated,
            )

        @staticmethod
        def reset_peak_memory_stats(_device):
            return None

        def mem_get_info(self, _device):
            return next(self.free_values), 1000

        @staticmethod
        def memory_allocated(_device):
            return 100

        @staticmethod
        def memory_reserved(_device):
            return 200

        @staticmethod
        def max_memory_allocated(_device):
            return 300

        @staticmethod
        def max_memory_reserved(_device):
            return 400

    linux_samples = iter(
        (
            {"total_bytes": 2000, "available_bytes": 1500},
            {"total_bytes": 2000, "available_bytes": 900},
            {"total_bytes": 2000, "available_bytes": 1100},
        )
    )
    cgroup_samples = iter(
        (
            {
                "available": True,
                "memory_max_bytes": 1000,
                "memory_max_is_unlimited": False,
                "memory_current_bytes": 100,
                "memory_peak_bytes": 400,
            },
            {
                "available": True,
                "memory_max_bytes": 1000,
                "memory_max_is_unlimited": False,
                "memory_current_bytes": 500,
                "memory_peak_bytes": 700,
            },
            {
                "available": True,
                "memory_max_bytes": 1000,
                "memory_max_is_unlimited": False,
                "memory_current_bytes": 400,
                "memory_peak_bytes": 700,
            },
        )
    )
    peak_rss_values = iter((100, 200, 150))
    monkeypatch.setattr(
        mineru25,
        "_linux_memory",
        lambda: next(linux_samples),
    )
    monkeypatch.setattr(
        mineru25,
        "_cgroup_v2_memory",
        lambda: next(cgroup_samples),
    )
    monkeypatch.setattr(
        mineru25,
        "_peak_process_rss_bytes",
        lambda: next(peak_rss_values),
    )
    torch = SimpleNamespace(cuda=FakeCuda())

    tracker = mineru25.CudaMemoryTracker(torch, "cuda:0")
    tracker.sample("after_model_load")
    telemetry = tracker.as_dict()

    assert telemetry["memory_kind"] == memory_kind
    assert telemetry["authoritative_capacity_source"] == capacity_source
    assert telemetry["baseline_cuda_free_bytes"] == 900
    assert telemetry["minimum_cuda_free_bytes"] == 600
    assert telemetry["observed_cuda_used_delta_bytes"] == 300
    assert telemetry["process_peak_rss_bytes"] == 200
    assert telemetry["linux_baseline_mem_available_bytes"] == 1500
    assert telemetry["linux_minimum_mem_available_bytes"] == 900
    assert telemetry["observed_linux_mem_available_delta_bytes"] == 600
    assert telemetry["cgroup_v2_memory"] == {
        "available": True,
        "capacity_bytes": 1000,
        "capacity_is_unlimited": False,
        "baseline_current_bytes": 100,
        "maximum_current_bytes": 500,
        "observed_current_delta_bytes": 400,
        "baseline_peak_bytes": 400,
        "maximum_peak_bytes": 700,
        "observed_peak_delta_bytes": 300,
        "baseline_headroom_bytes": 900,
        "minimum_headroom_bytes": 500,
        "observed_headroom_delta_bytes": 400,
        "memory_peak_semantics": (
            "cgroup_lifetime_high_water_mark_not_reset_by_adapter"
        ),
    }
    assert len(telemetry["samples"]) == 3
    assert all("cgroup_v2" in sample for sample in telemetry["samples"])


def test_profile_settings_require_explicit_provisional_approval(tmp_path):
    path, profile, expected = _sealed_profile(tmp_path)

    with pytest.raises(ValueError, match="provisional"):
        mineru25.load_profile_settings(
            str(path),
            expected_sha256=profile["profile_sha256"],
            memory_tier="max",
            allow_provisional=False,
            model_id=mineru25.DEFAULT_MODEL_ID,
            revision=mineru25.DEFAULT_REVISION,
        )

    settings, identity = mineru25.load_profile_settings(
        str(path),
        expected_sha256=profile["profile_sha256"],
        memory_tier="max",
        allow_provisional=True,
        model_id=mineru25.DEFAULT_MODEL_ID,
        revision=mineru25.DEFAULT_REVISION,
    )

    assert settings == expected
    assert identity == {
        "profile_id": "test-mineru-profile",
        "profile_sha256": profile["profile_sha256"],
        "qualification_state": "provisional",
        "memory_tier": "max",
    }


def test_profile_settings_reject_tampering_and_request_drift(tmp_path):
    path, profile, settings = _sealed_profile(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["memory_tiers"]["max"]["stages"]["markdown"]["settings"][
        "batch_size"
    ] = 8
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash"):
        mineru25.load_profile_settings(
            str(path),
            expected_sha256=profile["profile_sha256"],
            memory_tier="max",
            allow_provisional=True,
            model_id=mineru25.DEFAULT_MODEL_ID,
            revision=mineru25.DEFAULT_REVISION,
        )

    request = _request(tmp_path)
    request["settings"] = settings
    mineru25.validate_request(request, profile_settings=settings)
    request["settings"] = {**settings, "batch_size": 8}
    with pytest.raises(ValueError, match="sealed execution profile"):
        mineru25.validate_request(request, profile_settings=settings)


def test_generation_tracer_does_not_mark_eos_at_limit_as_max_tokens():
    class FakeSequence:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class FakeInputIds:
        shape = (1, 2)

    class FakeProcessor:
        def apply_chat_template(self, *args, **kwargs):
            return "rendered prompt"

        def decode(self, token_ids, **kwargs):
            return "decoded"

    class FakeModel:
        generation_config = SimpleNamespace(eos_token_id=9)

        def generate(self, *args, **kwargs):
            return [FakeSequence([4, 5, 9])]

    model = FakeModel()
    processor = FakeProcessor()
    tracer = mineru25.GenerationTracer(model, processor)
    tracer.install()

    processor.apply_chat_template([])
    model.generate(input_ids=FakeInputIds(), max_length=3)

    assert tracer.prompts == ["rendered prompt"]
    assert tracer.traces[0]["eos"] is True
    assert tracer.traces[0]["max_tokens"] is False
    assert tracer.traces[0]["finish_reason"] == "stop"
    assert tracer.traces[0]["stop_reason"] == "eos"


def test_generation_tracer_trims_padded_batch_rows_at_first_eos():
    class FakeSequence:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class FakeInputIds:
        shape = (2, 2)

    class FakeProcessor:
        def __init__(self):
            self.decoded_token_ids = []

        def apply_chat_template(self, *args, **kwargs):
            return "rendered prompt"

        def decode(self, token_ids, **kwargs):
            self.decoded_token_ids.append(token_ids)
            return ",".join(str(token_id) for token_id in token_ids)

    class FakeModel:
        generation_config = SimpleNamespace(eos_token_id=9)

        def generate(self, *args, **kwargs):
            return [
                FakeSequence([4, 5, 11, 9, 0, 0]),
                FakeSequence([6, 7, 21, 22, 23, 24]),
            ]

    model = FakeModel()
    processor = FakeProcessor()
    tracer = mineru25.GenerationTracer(model, processor)
    tracer.install()

    model.generate(input_ids=FakeInputIds(), max_new_tokens=4)

    assert processor.decoded_token_ids == [[11, 9], [21, 22, 23, 24]]
    assert tracer.traces[0]["token_ids"] == [11, 9]
    assert tracer.traces[0]["output_tokens"] == 2
    assert tracer.traces[0]["last_token_id"] == 9
    assert tracer.traces[0]["decoded_raw"] == "11,9"
    assert tracer.traces[0]["eos"] is True
    assert tracer.traces[0]["max_tokens"] is False
    assert tracer.traces[0]["finish_reason"] == "stop"
    assert tracer.traces[0]["stop_reason"] == "eos"
    assert tracer.traces[1]["token_ids"] == [21, 22, 23, 24]
    assert tracer.traces[1]["output_tokens"] == 4
    assert tracer.traces[1]["eos"] is False
    assert tracer.traces[1]["max_tokens"] is True
    assert tracer.traces[1]["finish_reason"] == "length"
    assert tracer.traces[1]["stop_reason"] == "max_tokens"


def test_snapshot_identity_hashes_critical_files(tmp_path):
    revision = "b" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()
    filenames = (
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "added_tokens.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "README.md",
    )
    for filename in filenames:
        (snapshot / filename).write_text(filename, encoding="utf-8")

    identity = mineru25.snapshot_identity(snapshot, revision)

    assert identity["commit"] == revision
    assert set(identity["files"]) == set(filenames)
    assert "added_tokens.json" in identity["files"]
    assert "README.md" in identity["files"]
    assert (
        identity["files"]["config.json"]["sha256"]
        == hashlib.sha256(b"config.json").hexdigest()
    )
    assert len(identity["files_sha256"]) == 64


def test_error_response_is_structured_and_preserves_trace(tmp_path):
    request = _request(tmp_path)
    tracer = SimpleNamespace(
        prompts=["prompt"],
        traces=[{"output_tokens": 1}],
    )
    error = RuntimeError("model initialization failed")

    response = mineru25.error_response(
        request,
        artifacts_dir=tmp_path,
        exc=error,
        phase="initialization",
        traceback_text="traceback",
        tracer=tracer,
    )

    assert response["status"] == "error"
    assert response["error"] == {
        "type": "RuntimeError",
        "message": "model initialization failed",
        "phase": "initialization",
    }
    artifact = tmp_path / response["artifacts"][0]["relative_path"]
    assert artifact.is_file()
    assert (
        mineru25.sha256_path(artifact) == response["artifacts"][0]["sha256"]
    )
