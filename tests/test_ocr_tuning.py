from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest


def _identity_digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _distribution_identity(backend_id: str) -> dict:
    distributions = [
        f"{backend_id}-runtime==1.0",
        "shared-runtime==2.0",
    ]
    digest = hashlib.sha256(
        json.dumps(
            distributions,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "count": len(distributions),
        "sha256": digest,
    }


def _reseal_receipt(value: dict) -> dict:
    from sneeze.tpn.ocr_backend_setup import make_receipt

    payload = copy.deepcopy(value)
    kind = payload.pop("kind")
    payload.pop("schema_version", None)
    payload.pop("receipt_sha256", None)
    return make_receipt(kind, payload)


def _hardware(total=48 * 1024**3, memory_model="dedicated"):
    return {
        "selector": {
            "schema_version": 1,
            "vendor": "nvidia",
            "product_name": "Synthetic GPU",
            "pci_device_id": "0x1234",
            "compute_capability": [8, 6],
            "memory_model": memory_model,
            "total_memory_bytes": total,
        }
    }


def _files(tmp_path):
    from sneeze.tpn.ocr_backends import get_backend

    manifest = tmp_path / "manifest.jsonl"
    manifest_records = [
        {
            "schema_version": 1,
            "case_id": f"case-{index}",
            "stem": f"page-{index}",
            "input": {
                "image_sha256": _identity_digest(f"image-{index}"),
                "raw_text_sha256": _identity_digest(f"raw-{index}"),
                "dimensions": {"w": 100, "h": 200},
            },
            "reference": {
                "markdown_sha256": _identity_digest(f"markdown-{index}"),
                "grounded_ocr_sha256": _identity_digest(
                    f"grounded-{index}"
                ),
            },
            "assertions": [
                {"type": "quality_status", "value": "pass"},
                {
                    "type": "metric_range",
                    "path": (
                        "quality.table_completion.raw_token_multiset_recall"
                    ),
                    "min": 0.5,
                    "lanes": ["best"],
                },
                {
                    "type": "metric_range",
                    "path": (
                        "quality.table_completion."
                        "grounded_ocr_token_multiset_recall"
                    ),
                    "min": 0.5,
                    "lanes": ["grounded_ocr"],
                },
            ],
        }
        for index in range(3)
    ]
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in manifest_records),
        encoding="utf-8",
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    installations = {
        backend: {
            "python_executable": sys.executable,
            "cache_dir": str(cache),
            "cuda_visible_devices": "0",
            "identity": {
                "record_sha256": _identity_digest(backend + "-record"),
                "python_executable_sha256": _identity_digest(
                    backend + "-python"
                ),
                "installed_distributions_count": _distribution_identity(
                    backend
                )["count"],
                "installed_distributions_sha256": _distribution_identity(
                    backend
                )["sha256"],
                "hf_cache_identity_sha256": _identity_digest(
                    backend + "-cache"
                ),
                "snapshot_files_sha256": _identity_digest(
                    backend + "-snapshot"
                ),
                "snapshot_weights_sha256": (
                    _identity_digest(backend + "-weights")
                    if backend.startswith("mineru")
                    else None
                ),
                "adapter_sha256": (
                    _identity_digest(backend + "-adapter")
                    if backend.startswith("mineru")
                    else None
                ),
                "implementation_sha256": get_backend(
                    backend
                ).qualified_implementation_sha256,
            },
        }
        for backend in ("deepseek-vllm", "mineru25-pro-2605")
    }
    return manifest, inputs, installations


def _candidate(
    *,
    candidate_id="a" * 64,
    backend_id="mineru25-pro-2605",
    capability="markdown",
    tier="25",
    batch_size=1,
    budget=12 * 1024**3,
):
    from sneeze.tpn.ocr_backends import get_backend

    settings = {
        "batch_size": batch_size,
        "memory_tier": tier,
        "memory_budget_bytes": budget,
    }
    if backend_id.startswith("mineru"):
        settings.update(
            {
                "backend": "official-two-step-transformers",
                "image_analysis": False,
            }
        )
    return {
        "candidate_id": candidate_id,
        "pipeline_id": "local-mineru25",
        "capability": capability,
        "backend_id": backend_id,
        "model_revision": "b" * 40,
        "hardware_selector": _hardware()["selector"],
        "installation_identity": {
            "snapshot_files_sha256": _identity_digest(
                backend_id + "-snapshot"
            ),
            "implementation_sha256": get_backend(
                backend_id
            ).qualified_implementation_sha256,
            "installed_distributions_count": _distribution_identity(
                backend_id
            )["count"],
            "installed_distributions_sha256": _distribution_identity(
                backend_id
            )["sha256"],
        },
        "tier": tier,
        "settings": settings,
        "batch_semantics": "layout_block_batch",
    }


def _benchmark_evidence(
    root: Path,
    *,
    all_pass=True,
    max_tokens=False,
    eos=None,
    wall_seconds=12.0,
    peak=4 * 1024**3,
    repeats=1,
    manifest_sha256="d" * 64,
    experiment_sha256="e" * 64,
    model_requested="opendatalab/MinerU2.5-Pro-2605-1.2B",
    model_revision="b" * 40,
    engine="transformers",
    engine_version="4.57.6",
    observed_attention_backend="sdpa",
    missing_runtime_package=None,
    settings=None,
    lane=None,
):
    from sneeze.tpn.ocr_backends import get_backend

    root.mkdir(parents=True)
    attempt_count = 3 * repeats
    summary = {
        "all_pass": all_pass,
        "provenance_complete": True,
        "result_count": attempt_count,
        "attempt_count": attempt_count,
        "passed_attempt_count": (
            attempt_count if all_pass else attempt_count - 1
        ),
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if engine == "vllm":
        packages = {
            "huggingface-hub": "1.17.0",
            "pdf2image": "1.17.0",
            "pillow": "12.2.0",
            "torch": "2.11.0+cu130",
            "torchaudio": "2.11.0+cu130",
            "torchvision": "0.26.0+cu130",
            "tqdm": "4.67.3",
            "transformers": "5.9.0",
            "vllm": engine_version,
            # Deliberate extras prove profile selectors use the canonical
            # backend package set rather than every installed distribution.
            "tokenizers": "0.23.0rc0",
            "nvidia-ml-py": "13.595.45",
        }
    else:
        packages = {
            "accelerate": "1.14.0",
            "huggingface-hub": "0.36.2",
            "mineru-vl-utils": "1.0.5",
            "pillow": "12.2.0",
            "torch": "2.11.0+cu130",
            "torchvision": "0.26.0+cu130",
            "transformers": engine_version,
        }
    if missing_runtime_package is not None:
        packages.pop(missing_runtime_package)
    backend_id = (
        "deepseek-vllm" if engine == "vllm" else "mineru25-pro-2605"
    )
    lane = lane or ("grounded_ocr" if engine == "vllm" else "best")
    backend = get_backend(backend_id)
    settings = dict(settings or _candidate()["settings"])
    if engine == "vllm":
        normalized = dict(settings)
        for key in ("attention_backend", "mm_encoder_attn_backend"):
            if normalized[key] == "auto":
                normalized[key] = None
        generation_params = {
            "execution_settings": normalized,
            "fixed_quality_settings": {
                key: normalized[key]
                for key in (
                    "max_tokens",
                    "ngram_size",
                    "temperature",
                    "window_size",
                )
            },
            "engine_init_kwargs": {
                key: normalized[key]
                for key in (
                    "attention_backend",
                    "enable_chunked_prefill",
                    "gpu_memory_utilization",
                    "max_num_batched_tokens",
                    "max_num_seqs",
                    "mm_encoder_attn_backend",
                )
            },
            "sampling_params": {
                key: normalized[key] for key in ("max_tokens", "temperature")
            },
            "logits_processor_extra_args": {
                key: normalized[key] for key in ("ngram_size", "window_size")
            },
        }
        memory_budget = None
    else:
        generation_params = {
            "request_settings": settings,
            "pipeline": {
                "backend": "transformers",
                "batch_size": settings["batch_size"],
                "image_analysis": settings["image_analysis"],
            },
        }
        memory_budget = {
            "memory_tier": settings["memory_tier"],
            "budget_bytes": settings["memory_budget_bytes"],
            "allocator_cap_applied": True,
        }
    rows = []
    for repeat in range(repeats):
        process = {
            "requests_jsonl_sha256": f"{repeat + 1:064x}",
            "wall_seconds": wall_seconds,
        }
        for index in range(3):
            rows.append(
                {
                    "case_id": f"case-{index}",
                    "lane": lane,
                    "repeat": repeat + 1,
                    "status": "completed",
                    "passed": all_pass or len(rows) < attempt_count - 1,
                    "metadata_complete": True,
                    "provenance": {
                        "provider": {
                            "model": {
                                "requested": model_requested,
                                "revision": model_revision,
                                "resolved_revision": model_revision,
                                "snapshot_files_sha256": _identity_digest(
                                    backend_id + "-snapshot"
                                ),
                            },
                            "runtime": {
                                "python": "3.12.13",
                                "engine": engine,
                                "engine_version": engine_version,
                                "packages": packages,
                                "attention_backend": (
                                    observed_attention_backend
                                ),
                                "torch_cuda": "13.0",
                                "dtype": "bfloat16",
                                "implementation": {
                                    "sha256": (
                                        backend.qualified_implementation_sha256
                                    )
                                },
                                "installed_distributions": (
                                    _distribution_identity(backend_id)
                                ),
                                "memory_budget": memory_budget,
                                "gpu_memory": {
                                    "memory_kind": "dedicated_vram",
                                    "torch_peak_reserved_bytes": peak,
                                    "observed_cuda_used_delta_bytes": (
                                        peak - 1
                                    ),
                                    "process_peak_rss_bytes": 99 * 1024**3,
                                },
                            },
                            "generation": {
                                "eos": (
                                    not max_tokens if eos is None else eos
                                ),
                                "max_tokens": max_tokens,
                                "params": generation_params,
                            },
                        },
                    },
                    "subprocess": process,
                }
            )
    with (root / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (root / "run.json").write_text(
        json.dumps(
            {
                "harness": {
                    "repeats": repeats,
                    "lanes": [lane],
                    "manifest_file_sha256": manifest_sha256,
                    "experiment_sha256": experiment_sha256,
                }
            }
        ),
        encoding="utf-8",
    )


def _write_evidence_for_runner(kwargs, *, all_pass=True):
    spec_path = Path(kwargs["provider_spec_path"])
    assert hashlib.sha256(spec_path.read_bytes()).hexdigest() == kwargs[
        "expected_provider_spec_file_sha256"
    ]
    spec = json.loads(spec_path.read_text())
    settings = next(iter(spec["lanes"].values()))["settings"]
    backend_id = spec["provider_id"].removesuffix("-profile-tuning")
    deepseek = backend_id == "deepseek-vllm"
    _benchmark_evidence(
        Path(kwargs["dest_dir"]),
        all_pass=all_pass,
        repeats=kwargs["repeats"],
        manifest_sha256=hashlib.sha256(
            Path(kwargs["manifest_path"]).read_bytes()
        ).hexdigest(),
        experiment_sha256=hashlib.sha256(
            json.dumps(
                {
                    "backend": backend_id,
                    "settings": settings,
                    "repeats": kwargs["repeats"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        model_requested=spec["model"]["requested"],
        model_revision=spec["model"]["revision"],
        engine="vllm" if deepseek else "transformers",
        engine_version=(
            "0.21.1rc1.dev438+g4ff865c38" if deepseek else "4.57.6"
        ),
        observed_attention_backend="FLASH_ATTN" if deepseek else "sdpa",
        settings=settings,
        lane=next(iter(spec["lanes"])),
    )


def _expected_evidence(*, repeats=1):
    return {
        "expected_case_ids": ["case-0", "case-1", "case-2"],
        "expected_repeats": repeats,
    }


def _manifest_contract(path: Path):
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "case_count": 3,
        "case_ids": ["case-0", "case-1", "case-2"],
    }


def test_tuning_plan_has_stage_specific_memory_candidates(tmp_path):
    from sneeze.tpn.ocr_tuning import (
        build_candidate_provider_spec,
        build_tuning_plan,
    )

    manifest, inputs, installations = _files(tmp_path)
    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("25", "max"),
        batch_sizes=(1, 4),
    )

    assert plan["kind"] == "ocr_profile_tuning_plan"
    assert len(plan["candidates"]) == 16
    assert plan["manifest"]["case_count"] == 3
    assert plan["policy"]["batch_bounds"] == {
        "manifest_case_count": 3,
        "page_batch_max_size": 3,
        "page_batch_sizes": [1, 3],
        "layout_block_batch_sizes": [1, 4],
    }
    assert plan["memory_budgets"]["25"]["budget_bytes"] == 12 * 1024**3
    assert plan["memory_budgets"]["max"]["budget_bytes"] == int(
        48 * 1024**3 * 0.9
    )
    deepseek = next(
        candidate
        for candidate in plan["candidates"]
        if candidate["backend_id"] == "deepseek-vllm"
        and candidate["tier"] == "25"
        and candidate["settings"]["batch_size"] == 1
    )
    assert deepseek["settings"]["gpu_memory_utilization"] == round(
        11 / 48, 8
    )
    assert deepseek["settings"]["engine_overhead_headroom_bytes"] == (
        1024**3
    )
    assert deepseek["settings"]["max_tokens"] == 8192
    assert {
        candidate["settings"]["batch_size"]
        for candidate in plan["candidates"]
        if candidate["backend_id"] == "deepseek-vllm"
    } == {1, 3}
    mineru = next(
        candidate
        for candidate in plan["candidates"]
        if candidate["backend_id"] == "mineru25-pro-2605"
    )
    assert mineru["batch_semantics"] == "layout_block_batch"
    assert {
        candidate["settings"]["batch_size"]
        for candidate in plan["candidates"]
        if candidate["backend_id"] == "mineru25-pro-2605"
    } == {1, 4}
    for backend_id in ("deepseek-vllm", "mineru25-pro-2605"):
        candidate = next(
            item
            for item in plan["candidates"]
            if item["backend_id"] == backend_id
        )
        spec = build_candidate_provider_spec(plan, candidate)
        assert spec["argv"][1:3] == ["-I", "-B"]


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
def test_tuning_plan_rejects_invalid_timeout(tmp_path, timeout):
    from sneeze.tpn.ocr_tuning import TuningError, build_tuning_plan

    manifest, inputs, installations = _files(tmp_path)

    with pytest.raises(TuningError, match="finite positive"):
        build_tuning_plan(
            pipeline_id="local-mineru25",
            hardware=_hardware(),
            installations=installations,
            manifest_path=manifest,
            input_root=inputs,
            memory_tiers=("25",),
            batch_sizes=(1,),
            timeout_seconds=timeout,
        )


def test_tuning_plan_preserves_positive_fractional_timeout(tmp_path):
    from sneeze.tpn.ocr_tuning import build_tuning_plan

    manifest, inputs, installations = _files(tmp_path)

    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("25",),
        batch_sizes=(1,),
        timeout_seconds=0.5,
    )

    assert plan["policy"]["timeout_seconds"] == 0.5


@pytest.mark.parametrize(
    ("field", "value"),
    (
        pytest.param("discovery_repeats", True, id="discovery-boolean"),
        pytest.param("discovery_repeats", 1.5, id="discovery-fraction"),
        pytest.param("confirmation_repeats", True, id="confirmation-boolean"),
        pytest.param("confirmation_repeats", 1.5, id="confirmation-fraction"),
    ),
)
def test_tuning_plan_rejects_noninteger_repeat_counts(tmp_path, field, value):
    from sneeze.tpn.ocr_tuning import TuningError, build_tuning_plan

    manifest, inputs, installations = _files(tmp_path)
    kwargs = {field: value}

    with pytest.raises(TuningError, match="repeat counts must be positive"):
        build_tuning_plan(
            pipeline_id="local-mineru25",
            hardware=_hardware(),
            installations=installations,
            manifest_path=manifest,
            input_root=inputs,
            memory_tiers=("25",),
            batch_sizes=(1,),
            **kwargs,
        )


def test_unified_max_budget_reserves_more_memory(tmp_path):
    from sneeze.tpn.ocr_tuning import build_tuning_plan

    manifest, inputs, installations = _files(tmp_path)
    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(
            total=128 * 1024**3,
            memory_model="unified",
        ),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("max",),
        batch_sizes=(1,),
    )

    assert plan["memory_budgets"]["max"]["budget_bytes"] == 96 * 1024**3


def test_tuning_api_rejects_withheld_deepseek_markdown_pipeline(tmp_path):
    from sneeze.tpn.ocr_tuning import TuningError, build_tuning_plan

    manifest, inputs, installations = _files(tmp_path)
    with pytest.raises(TuningError, match="unknown tuning pipeline"):
        build_tuning_plan(
            pipeline_id="deepseek-vllm-full",
            hardware=_hardware(),
            installations=installations,
            manifest_path=manifest,
            input_root=inputs,
        )


def test_tuning_plan_rejects_assertions_skipped_by_every_stage(tmp_path):
    from sneeze.tpn.ocr_tuning import TuningError, build_tuning_plan

    manifest, inputs, installations = _files(tmp_path)
    case = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    case["assertions"] = [
        {
            "type": "quality_status",
            "value": "pass",
            "lanes": ["stale-lane"],
        }
    ]
    manifest.write_text(
        json.dumps(case) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TuningError, match="not exercised by this pipeline"):
        build_tuning_plan(
            pipeline_id="local-mineru25",
            hardware=_hardware(),
            installations=installations,
            manifest_path=manifest,
            input_root=inputs,
        )


def test_tuning_plan_requires_completeness_gate_for_each_stage_lane(tmp_path):
    from sneeze.tpn.ocr_tuning import TuningError, build_tuning_plan

    manifest, inputs, installations = _files(tmp_path)
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["assertions"] = [
        assertion
        for assertion in records[0]["assertions"]
        if assertion.get("path")
        != (
            "quality.table_completion."
            "grounded_ocr_token_multiset_recall"
        )
    ]
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(TuningError, match="lane-scoped completeness gate"):
        build_tuning_plan(
            pipeline_id="local-mineru25",
            hardware=_hardware(),
            installations=installations,
            manifest_path=manifest,
            input_root=inputs,
        )


def test_candidate_evidence_deduplicates_process_and_memory_domains(tmp_path):
    from sneeze.tpn.ocr_tuning import evaluate_candidate_evidence

    benchmark = tmp_path / "benchmark"
    _benchmark_evidence(benchmark)
    result = evaluate_candidate_evidence(
        candidate=_candidate(),
        benchmark_dir=benchmark,
        **_expected_evidence(),
    )

    assert result["eligible"] is True
    assert result["process_count"] == 1
    assert result["cold_pages_per_second"] == 0.25
    assert result["peak_memory_bytes"] == 4 * 1024**3


def test_candidate_rejects_conflicting_duplicate_process_telemetry(tmp_path):
    from sneeze.tpn.ocr_tuning import evaluate_candidate_evidence

    benchmark = tmp_path / "benchmark"
    _benchmark_evidence(benchmark)
    results_path = benchmark / "results.jsonl"
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[1]["subprocess"]["wall_seconds"] = 99.0
    results_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = evaluate_candidate_evidence(
        candidate=_candidate(),
        benchmark_dir=benchmark,
        **_expected_evidence(),
    )

    assert result["eligible"] is False
    assert "inconsistent_process_telemetry" in result[
        "ineligible_reasons"
    ]


@pytest.mark.parametrize("peak", (0, True))
def test_candidate_rejects_nonpositive_or_boolean_memory_peak(tmp_path, peak):
    from sneeze.tpn.ocr_tuning import evaluate_candidate_evidence

    benchmark = tmp_path / "benchmark"
    _benchmark_evidence(benchmark, peak=peak)

    result = evaluate_candidate_evidence(
        candidate=_candidate(),
        benchmark_dir=benchmark,
        **_expected_evidence(),
    )

    assert result["eligible"] is False
    assert "missing_memory_telemetry" in result["ineligible_reasons"]


@pytest.mark.parametrize("wall_seconds", (0.0, True, float("inf")))
def test_candidate_rejects_invalid_throughput_telemetry(
    tmp_path,
    wall_seconds,
):
    from sneeze.tpn.ocr_tuning import evaluate_candidate_evidence

    benchmark = tmp_path / "benchmark"
    _benchmark_evidence(benchmark, wall_seconds=wall_seconds)

    result = evaluate_candidate_evidence(
        candidate=_candidate(),
        benchmark_dir=benchmark,
        **_expected_evidence(),
    )

    assert result["eligible"] is False
    assert "missing_throughput_telemetry" in result["ineligible_reasons"]


def test_candidate_requires_exact_case_repeat_lane_matrix(tmp_path):
    from sneeze.tpn.ocr_tuning import evaluate_candidate_evidence

    benchmark = tmp_path / "benchmark"
    _benchmark_evidence(benchmark, repeats=5)
    results_path = benchmark / "results.jsonl"
    first_row = results_path.read_text(encoding="utf-8").splitlines()[0]
    results_path.write_text(first_row + "\n", encoding="utf-8")

    result = evaluate_candidate_evidence(
        candidate=_candidate(),
        benchmark_dir=benchmark,
        **_expected_evidence(repeats=5),
    )

    assert result["eligible"] is False
    assert "result_matrix_count_mismatch" in result["ineligible_reasons"]
    assert "result_matrix_coverage_mismatch" in result["ineligible_reasons"]
    assert "benchmark_process_matrix_mismatch" in result[
        "ineligible_reasons"
    ]


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("all_pass", "quality_or_protocol_failure"),
        ("provenance_complete", "incomplete_provenance"),
    ],
)
def test_candidate_summary_boolean_gates_reject_integer_one(
    tmp_path,
    field,
    reason,
):
    from sneeze.tpn.ocr_tuning import evaluate_candidate_evidence

    benchmark = tmp_path / "benchmark"
    _benchmark_evidence(benchmark)
    summary_path = benchmark / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = evaluate_candidate_evidence(
        candidate=_candidate(),
        benchmark_dir=benchmark,
        **_expected_evidence(),
    )

    assert result["eligible"] is False
    assert reason in result["ineligible_reasons"]


def test_candidate_matrix_repeat_rejects_boolean_true(tmp_path):
    from sneeze.tpn.ocr_tuning import evaluate_candidate_evidence

    benchmark = tmp_path / "benchmark"
    _benchmark_evidence(benchmark)
    results_path = benchmark / "results.jsonl"
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["repeat"] = True
    results_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = evaluate_candidate_evidence(
        candidate=_candidate(),
        benchmark_dir=benchmark,
        **_expected_evidence(),
    )

    assert result["eligible"] is False
    assert "result_matrix_identity_invalid" in result[
        "ineligible_reasons"
    ]


def test_candidate_summary_count_rejects_boolean_true(tmp_path):
    from sneeze.tpn.ocr_tuning import evaluate_candidate_evidence

    benchmark = tmp_path / "benchmark"
    _benchmark_evidence(benchmark)
    summary_path = benchmark / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["attempt_count"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = evaluate_candidate_evidence(
        candidate=_candidate(),
        benchmark_dir=benchmark,
        **_expected_evidence(),
    )

    assert result["eligible"] is False
    assert "summary_count_mismatch" in result["ineligible_reasons"]


def test_candidate_results_reject_duplicate_json_fields(tmp_path):
    from sneeze.tpn.ocr_tuning import TuningError, evaluate_candidate_evidence

    benchmark = tmp_path / "benchmark"
    _benchmark_evidence(benchmark)
    results_path = benchmark / "results.jsonl"
    payload = results_path.read_text(encoding="utf-8").replace(
        '"case_id": "case-0"',
        '"case_id": "shadow", "case_id": "case-0"',
        1,
    )
    results_path.write_text(payload, encoding="utf-8")

    with pytest.raises(TuningError, match="duplicate JSON field"):
        evaluate_candidate_evidence(
            candidate=_candidate(),
            benchmark_dir=benchmark,
            **_expected_evidence(),
        )


def test_unified_candidate_uses_conservative_system_memory_peak(tmp_path):
    from sneeze.tpn.ocr_tuning import evaluate_candidate_evidence

    gib = 1024**3
    benchmark = tmp_path / "benchmark"
    _benchmark_evidence(benchmark, peak=2 * gib)
    rows = [
        json.loads(line)
        for line in (benchmark / "results.jsonl").read_text().splitlines()
    ]
    for row in rows:
        memory = row["provenance"]["provider"]["runtime"]["gpu_memory"]
        memory.update(
            {
                "memory_kind": "unified_system",
                "process_peak_rss_bytes": 5 * gib,
                "parent_plus_children_peak_rss_bytes": 6 * gib,
                "observed_linux_mem_available_delta_bytes": 7 * gib,
                "cgroup_v2_memory": {
                    "observed_current_delta_bytes": 8 * gib,
                    "observed_peak_delta_bytes": 9 * gib,
                    "observed_headroom_delta_bytes": 8 * gib,
                },
            }
        )
    (benchmark / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = evaluate_candidate_evidence(
        candidate=_candidate(budget=12 * gib),
        benchmark_dir=benchmark,
        **_expected_evidence(),
    )

    assert result["eligible"] is True
    assert result["peak_memory_bytes"] == 9 * gib
    assert result["peak_memory_sources"] == [
        "cgroup_v2.observed_peak_delta_bytes"
    ]


def test_unified_candidate_requires_aggregate_system_telemetry(tmp_path):
    from sneeze.tpn.ocr_tuning import evaluate_candidate_evidence

    benchmark = tmp_path / "benchmark"
    _benchmark_evidence(benchmark)
    rows = [
        json.loads(line)
        for line in (benchmark / "results.jsonl").read_text().splitlines()
    ]
    for row in rows:
        memory = row["provenance"]["provider"]["runtime"]["gpu_memory"]
        memory["memory_kind"] = "unified_system"
    (benchmark / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = evaluate_candidate_evidence(
        candidate=_candidate(),
        benchmark_dir=benchmark,
        **_expected_evidence(),
    )

    assert result["eligible"] is False
    assert (
        "missing_unified_system_memory_telemetry"
        in result["ineligible_reasons"]
    )


def test_candidate_quality_and_termination_are_hard_gates(tmp_path):
    from sneeze.tpn.ocr_tuning import evaluate_candidate_evidence

    benchmark = tmp_path / "benchmark"
    _benchmark_evidence(
        benchmark,
        all_pass=False,
        max_tokens=True,
        wall_seconds=1.0,
    )
    result = evaluate_candidate_evidence(
        candidate=_candidate(),
        benchmark_dir=benchmark,
        **_expected_evidence(),
    )

    assert result["eligible"] is False
    assert "quality_or_protocol_failure" in result["ineligible_reasons"]
    assert "max_tokens_termination" in result["ineligible_reasons"]
    assert "non_eos_termination" in result["ineligible_reasons"]


def test_candidate_rejects_non_eos_non_limit_termination(tmp_path):
    from sneeze.tpn.ocr_tuning import evaluate_candidate_evidence

    benchmark = tmp_path / "benchmark"
    _benchmark_evidence(benchmark, eos=False)
    result = evaluate_candidate_evidence(
        candidate=_candidate(),
        benchmark_dir=benchmark,
        **_expected_evidence(),
    )

    assert result["eligible"] is False
    assert result["ineligible_reasons"] == ["non_eos_termination"]


def test_candidate_rejects_provider_effective_setting_drift(tmp_path):
    from sneeze.tpn.ocr_tuning import evaluate_candidate_evidence

    candidate = _candidate(batch_size=2)
    benchmark = tmp_path / "benchmark"
    _benchmark_evidence(benchmark, settings=candidate["settings"])
    results_path = benchmark / "results.jsonl"
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["provenance"]["provider"]["generation"]["params"][
        "pipeline"
    ]["batch_size"] = 1
    results_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = evaluate_candidate_evidence(
        candidate=candidate,
        benchmark_dir=benchmark,
        **_expected_evidence(),
    )

    assert result["eligible"] is False
    assert (
        "effective_pipeline_settings_mismatch"
        in result["ineligible_reasons"]
    )


@pytest.mark.parametrize(
    ("section", "key", "value", "reason"),
    [
        (
            "execution_settings",
            "max_num_seqs",
            99,
            "effective_generation_settings_mismatch",
        ),
        (
            "engine_init_kwargs",
            "max_num_seqs",
            99,
            "effective_engine_settings_mismatch",
        ),
        (
            "sampling_params",
            "max_tokens",
            1,
            "effective_sampling_settings_mismatch",
        ),
        (
            "logits_processor_extra_args",
            "window_size",
            1,
            "effective_logits_settings_mismatch",
        ),
    ],
)
def test_deepseek_candidate_rejects_effective_setting_drift(
    tmp_path,
    section,
    key,
    value,
    reason,
):
    from sneeze.tpn.ocr_tuning import (
        build_tuning_plan,
        evaluate_candidate_evidence,
    )

    manifest, inputs, installations = _files(tmp_path)
    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("25",),
        batch_sizes=(1,),
        deepseek_engine_overhead_headroom_bytes=(1024**3,),
    )
    candidate = next(
        item
        for item in plan["candidates"]
        if item["backend_id"] == "deepseek-vllm"
    )
    benchmark = tmp_path / "deepseek-setting-drift"
    _benchmark_evidence(
        benchmark,
        engine="vllm",
        engine_version="0.21.1rc1.dev438+g4ff865c38",
        model_requested="deepseek-ai/DeepSeek-OCR",
        model_revision=candidate["model_revision"],
        observed_attention_backend="FLASH_ATTN",
        settings=candidate["settings"],
    )
    rows = [
        json.loads(line)
        for line in (benchmark / "results.jsonl").read_text().splitlines()
    ]
    rows[0]["provenance"]["provider"]["generation"]["params"][section][
        key
    ] = value
    (benchmark / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = evaluate_candidate_evidence(
        candidate=candidate,
        benchmark_dir=benchmark,
        **_expected_evidence(),
    )
    assert result["eligible"] is False
    assert reason in result["ineligible_reasons"]


def test_tuning_plan_rejects_live_unavailable_memory_tier(tmp_path):
    from sneeze.tpn.ocr_tuning import TuningError, build_tuning_plan

    manifest, inputs, installations = _files(tmp_path)
    hardware = _hardware()
    hardware["memory_budgets"] = {
        "tiers": [
            {
                "tier": "max",
                "target_fraction": 0.9,
                "reserve_bytes": 2 * 1024**3,
                "capacity_budget_bytes": 40 * 1024**3,
                "supported": True,
                "full_budget_available": False,
                "reason_codes": ["live_capacity_below_tier_budget"],
            }
        ]
    }

    with pytest.raises(TuningError, match="not available for tuning"):
        build_tuning_plan(
            pipeline_id="local-mineru25",
            hardware=hardware,
            installations=installations,
            manifest_path=manifest,
            input_root=inputs,
            memory_tiers=("max",),
            batch_sizes=(1,),
        )


def test_confirmation_requires_every_canonical_runtime_package(tmp_path):
    from sneeze.tpn.ocr_backends import get_backend
    from sneeze.tpn.ocr_tuning import (
        build_tuning_plan,
        evaluate_confirmation_evidence,
    )

    manifest, inputs, installations = _files(tmp_path)
    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("25",),
        batch_sizes=(1,),
        confirmation_repeats=2,
    )
    candidate = next(
        item
        for item in plan["candidates"]
        if item["backend_id"] == "deepseek-vllm"
    )
    backend = get_backend("deepseek-vllm")
    benchmark = tmp_path / "missing-package"
    _benchmark_evidence(
        benchmark,
        repeats=2,
        manifest_sha256=plan["manifest"]["sha256"],
        model_requested=backend.model_id,
        model_revision=candidate["model_revision"],
        engine="vllm",
        engine_version="0.21.1rc1.dev438+g4ff865c38",
        observed_attention_backend="FLASH_ATTN",
        missing_runtime_package="pdf2image",
        settings=candidate["settings"],
    )

    result = evaluate_confirmation_evidence(
        plan=plan,
        candidate=candidate,
        selection={
            "candidate_id": candidate["candidate_id"],
            "evidence_receipt_sha256": "a" * 64,
        },
        confirmation_id="b" * 64,
        benchmark_dir=benchmark,
    )

    assert result["eligible"] is False
    assert "missing_runtime_package:pdf2image" in result["ineligible_reasons"]
    assert result["runtime_selector"] is None


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (
            ("packages", "vllm"),
            "0.20.0",
            "runtime_package_version_mismatch:vllm",
        ),
        (("dtype",), "float16", "runtime_identity_mismatch:dtype"),
        (("python",), "3.12.12", "runtime_identity_mismatch:python"),
        (("torch_cuda",), "12.8", "runtime_identity_mismatch:torch_cuda"),
    ],
)
def test_confirmation_rejects_wrong_runtime_tuple(
    tmp_path,
    path,
    value,
    reason,
):
    from sneeze.tpn.ocr_backends import get_backend
    from sneeze.tpn.ocr_tuning import (
        build_tuning_plan,
        evaluate_confirmation_evidence,
    )

    manifest, inputs, installations = _files(tmp_path)
    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("25",),
        batch_sizes=(1,),
        confirmation_repeats=2,
        deepseek_engine_overhead_headroom_bytes=(1024**3,),
    )
    candidate = next(
        item
        for item in plan["candidates"]
        if item["backend_id"] == "deepseek-vllm"
    )
    backend = get_backend("deepseek-vllm")
    benchmark = tmp_path / "wrong-runtime"
    _benchmark_evidence(
        benchmark,
        repeats=2,
        manifest_sha256=plan["manifest"]["sha256"],
        model_requested=backend.model_id,
        model_revision=candidate["model_revision"],
        engine="vllm",
        engine_version="0.21.1rc1.dev438+g4ff865c38",
        observed_attention_backend="FLASH_ATTN",
        settings=candidate["settings"],
    )
    rows = [
        json.loads(line)
        for line in (benchmark / "results.jsonl").read_text().splitlines()
    ]
    for row in rows:
        current = row["provenance"]["provider"]["runtime"]
        for key in path[:-1]:
            current = current[key]
        current[path[-1]] = value
    (benchmark / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = evaluate_confirmation_evidence(
        plan=plan,
        candidate=candidate,
        selection={
            "candidate_id": candidate["candidate_id"],
            "evidence_receipt_sha256": "a" * 64,
        },
        confirmation_id="b" * 64,
        benchmark_dir=benchmark,
    )
    assert result["eligible"] is False
    assert reason in result["ineligible_reasons"]
    assert result["runtime_selector"] is None


def test_confirmation_rejects_configured_attention_backend_mismatch(
    tmp_path,
):
    from sneeze.tpn.ocr_backends import get_backend
    from sneeze.tpn.ocr_tuning import (
        build_tuning_plan,
        evaluate_confirmation_evidence,
    )

    manifest, inputs, installations = _files(tmp_path)
    installations["deepseek-vllm"]["attention_backend"] = "FLASH_ATTN"
    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("25",),
        batch_sizes=(1,),
        confirmation_repeats=2,
    )
    candidate = next(
        item
        for item in plan["candidates"]
        if item["backend_id"] == "deepseek-vllm"
    )
    backend = get_backend("deepseek-vllm")
    benchmark = tmp_path / "attention-mismatch"
    _benchmark_evidence(
        benchmark,
        repeats=2,
        manifest_sha256=plan["manifest"]["sha256"],
        model_requested=backend.model_id,
        model_revision=candidate["model_revision"],
        engine="vllm",
        engine_version="0.21.1rc1.dev438+g4ff865c38",
        observed_attention_backend="TORCH_SDPA",
        settings=candidate["settings"],
    )

    result = evaluate_confirmation_evidence(
        plan=plan,
        candidate=candidate,
        selection={
            "candidate_id": candidate["candidate_id"],
            "evidence_receipt_sha256": "a" * 64,
        },
        confirmation_id="b" * 64,
        benchmark_dir=benchmark,
    )

    assert result["eligible"] is False
    assert (
        "selected_attention_backend_mismatch"
        in (result["ineligible_reasons"])
    )
    assert result["runtime_selector"] is None


def test_confirmation_auto_seals_one_actual_attention_backend(tmp_path):
    from sneeze.tpn.ocr_backends import get_backend
    from sneeze.tpn.ocr_tuning import (
        build_tuning_plan,
        evaluate_confirmation_evidence,
    )

    manifest, inputs, installations = _files(tmp_path)
    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("25",),
        batch_sizes=(1,),
        confirmation_repeats=2,
    )
    candidate = next(
        item
        for item in plan["candidates"]
        if item["backend_id"] == "deepseek-vllm"
    )
    backend = get_backend("deepseek-vllm")
    benchmark = tmp_path / "auto-attention"
    _benchmark_evidence(
        benchmark,
        repeats=2,
        manifest_sha256=plan["manifest"]["sha256"],
        model_requested=backend.model_id,
        model_revision=candidate["model_revision"],
        engine="vllm",
        engine_version="0.21.1rc1.dev438+g4ff865c38",
        observed_attention_backend="FLASH_ATTN",
        settings=candidate["settings"],
    )

    result = evaluate_confirmation_evidence(
        plan=plan,
        candidate=candidate,
        selection={
            "candidate_id": candidate["candidate_id"],
            "evidence_receipt_sha256": "a" * 64,
        },
        confirmation_id="b" * 64,
        benchmark_dir=benchmark,
    )

    assert result["eligible"] is True
    assert result["runtime_selector"]["attention_backend"] == "FLASH_ATTN"


def test_confirmation_auto_rejects_selected_attention_backend_drift(
    tmp_path,
):
    from sneeze.tpn.ocr_backends import get_backend
    from sneeze.tpn.ocr_tuning import (
        build_tuning_plan,
        evaluate_confirmation_evidence,
    )

    manifest, inputs, installations = _files(tmp_path)
    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("25",),
        batch_sizes=(1,),
        confirmation_repeats=2,
    )
    candidate = next(
        item
        for item in plan["candidates"]
        if item["backend_id"] == "deepseek-vllm"
    )
    backend = get_backend("deepseek-vllm")
    benchmark = tmp_path / "attention-drift"
    _benchmark_evidence(
        benchmark,
        repeats=2,
        manifest_sha256=plan["manifest"]["sha256"],
        model_requested=backend.model_id,
        model_revision=candidate["model_revision"],
        engine="vllm",
        engine_version="0.21.1rc1.dev438+g4ff865c38",
        observed_attention_backend="FLASH_ATTN",
        settings=candidate["settings"],
    )
    results_path = benchmark / "results.jsonl"
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["provenance"]["provider"]["runtime"]["attention_backend"] = (
        "TORCH_SDPA"
    )
    results_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = evaluate_confirmation_evidence(
        plan=plan,
        candidate=candidate,
        selection={
            "candidate_id": candidate["candidate_id"],
            "evidence_receipt_sha256": "a" * 64,
        },
        confirmation_id="b" * 64,
        benchmark_dir=benchmark,
    )

    assert result["eligible"] is False
    assert (
        "selected_attention_backend_drift" in (result["ineligible_reasons"])
    )
    assert result["runtime_selector"] is None


def test_discovery_rejects_snapshot_identity_different_from_plan(tmp_path):
    from sneeze.tpn.ocr_backends import get_backend
    from sneeze.tpn.ocr_tuning import (
        build_tuning_plan,
        evaluate_candidate_evidence,
    )

    manifest, inputs, installations = _files(tmp_path)
    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("25",),
        batch_sizes=(1,),
    )
    candidate = next(
        item
        for item in plan["candidates"]
        if item["backend_id"] == "deepseek-vllm"
    )
    backend = get_backend("deepseek-vllm")
    benchmark = tmp_path / "snapshot-mismatch"
    _benchmark_evidence(
        benchmark,
        model_requested=backend.model_id,
        model_revision=candidate["model_revision"],
        engine="vllm",
        engine_version="0.21.1rc1.dev438+g4ff865c38",
        observed_attention_backend="FLASH_ATTN",
        settings=candidate["settings"],
    )
    results_path = benchmark / "results.jsonl"
    rows = [
        json.loads(line) for line in results_path.read_text().splitlines()
    ]
    for row in rows:
        row["provenance"]["provider"]["model"][
            "snapshot_files_sha256"
        ] = "a" * 64
    results_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = evaluate_candidate_evidence(
        candidate=candidate,
        benchmark_dir=benchmark,
        **_expected_evidence(),
    )

    assert result["eligible"] is False
    assert "model_snapshot_identity_mismatch" in result[
        "ineligible_reasons"
    ]


def test_confirmation_rejects_distribution_manifest_drift(tmp_path):
    from sneeze.tpn.ocr_backend_setup import sha256_json
    from sneeze.tpn.ocr_backends import get_backend
    from sneeze.tpn.ocr_tuning import (
        build_tuning_plan,
        evaluate_confirmation_evidence,
    )

    manifest, inputs, installations = _files(tmp_path)
    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("25",),
        batch_sizes=(1,),
        confirmation_repeats=2,
    )
    candidate = next(
        item
        for item in plan["candidates"]
        if item["backend_id"] == "deepseek-vllm"
    )
    backend = get_backend("deepseek-vllm")
    benchmark = tmp_path / "distribution-mismatch"
    _benchmark_evidence(
        benchmark,
        repeats=2,
        manifest_sha256=plan["manifest"]["sha256"],
        model_requested=backend.model_id,
        model_revision=candidate["model_revision"],
        engine="vllm",
        engine_version="0.21.1rc1.dev438+g4ff865c38",
        observed_attention_backend="FLASH_ATTN",
        settings=candidate["settings"],
    )
    results_path = benchmark / "results.jsonl"
    rows = [
        json.loads(line) for line in results_path.read_text().splitlines()
    ]
    changed = ["different-runtime==9.0", "shared-runtime==2.0"]
    changed.sort()
    for row in rows:
        identity = row["provenance"]["provider"]["runtime"][
            "installed_distributions"
        ]
        identity["count"] = len(changed)
        identity["sha256"] = sha256_json(changed)
    results_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = evaluate_confirmation_evidence(
        plan=plan,
        candidate=candidate,
        selection={
            "candidate_id": candidate["candidate_id"],
            "evidence_receipt_sha256": "a" * 64,
        },
        confirmation_id="b" * 64,
        benchmark_dir=benchmark,
    )

    assert result["eligible"] is False
    assert "installed_distributions_identity_mismatch" in result[
        "ineligible_reasons"
    ]


def test_summary_selects_slower_eligible_candidate_and_marks_missing_tiers():
    from sneeze.tpn.ocr_backend_setup import make_receipt
    from sneeze.tpn.ocr_tuning import summarize_tuning_results

    fast_bad = _candidate(candidate_id="a" * 64, batch_size=8)
    slow_good = _candidate(candidate_id="b" * 64, batch_size=1)
    plan = make_receipt(
        "ocr_profile_tuning_plan",
        {
            "pipeline_id": "local-mineru25",
            "policy": {
                "memory_tiers": ["25"],
                "suite_role": "hardware_smoke",
            },
            "candidates": [fast_bad, slow_good],
        },
    )
    summary = summarize_tuning_results(
        plan=plan,
        candidate_results=[
            {
                "candidate_id": fast_bad["candidate_id"],
                "eligible": False,
                "cold_pages_per_second": 100.0,
                "receipt_sha256": "d" * 64,
            },
            {
                "candidate_id": slow_good["candidate_id"],
                "eligible": True,
                "cold_pages_per_second": 1.0,
                "peak_memory_bytes": 1,
                "receipt_sha256": "e" * 64,
            },
        ],
    )

    markdown = next(
        item
        for item in summary["selections"]
        if item["capability"] == "markdown"
    )
    grounded = next(
        item
        for item in summary["selections"]
        if item["capability"] == "grounded_ocr"
    )
    assert markdown["candidate_id"] == slow_good["candidate_id"]
    assert grounded["supported"] is False
    assert summary["qualification_status"] == "provisional"


def test_confirmation_completion_requires_exact_selected_candidate_set():
    from sneeze.tpn.ocr_backend_setup import make_receipt, sha256_json
    from sneeze.tpn.ocr_tuning import (
        _confirmation_identity,
        summarize_onboarding_results,
    )

    expected_ids = ("a" * 64, "b" * 64)
    selection_rows = [
        {
            "capability": capability,
            "backend_id": backend_id,
            "tier": "25",
            "supported": True,
            "candidate_id": candidate_id,
            "settings": {"batch_size": 1},
            "evidence_receipt_sha256": _identity_digest(
                candidate_id + "-evidence"
            ),
        }
        for capability, backend_id, candidate_id in (
            ("grounded_ocr", "deepseek-vllm", expected_ids[0]),
            ("markdown", "mineru25-pro-2605", expected_ids[1]),
        )
    ]
    plan = make_receipt(
        "ocr_profile_tuning_plan",
        {
            "pipeline_id": "local-mineru25",
            "manifest": {"sha256": "d" * 64},
            "policy": {
                "memory_tiers": ["25"],
                "confirmation_repeats": 2,
                "suite_role": "hardware_smoke",
                "suite_id": "cuphoton-three-page-smoke",
            },
            "candidates": [
                {
                    key: selection[key]
                    for key in (
                        "candidate_id",
                        "capability",
                        "backend_id",
                        "tier",
                        "settings",
                    )
                }
                for selection in selection_rows
            ],
        },
    )
    discovery = make_receipt(
        "ocr_profile_tuning_summary",
        {
            "plan_receipt_sha256": plan["receipt_sha256"],
            "complete": True,
            "candidate_count": 2,
            "completed_candidate_count": 2,
            "selections": selection_rows,
        },
    )
    confirmation_id, _identity = _confirmation_identity(
        plan=plan,
        selection=selection_rows[0],
    )
    confirmations = [
        make_receipt(
            "ocr_profile_confirmation_result",
            {
                "confirmation_id": confirmation_id,
                "candidate_id": expected_ids[0],
                "capability": "grounded_ocr",
                "backend_id": "deepseek-vllm",
                "tier": "25",
                "repeats": 2,
                "manifest_sha256": "d" * 64,
                "settings_sha256": sha256_json({"batch_size": 1}),
                "eligible": True,
                "runtime_selector": {"engine": "vllm"},
            },
        ),
        make_receipt(
            "ocr_profile_confirmation_result",
            {
                "candidate_id": "d" * 64,
                "eligible": True,
                "runtime_selector": {"engine": "transformers"},
            },
        ),
    ]

    summary = summarize_onboarding_results(
        plan=plan,
        discovery=discovery,
        confirmation_results=confirmations,
    )

    assert summary["expected_confirmation_count"] == 2
    assert summary["completed_confirmation_count"] == 1
    assert summary["confirmation_complete"] is False
    assert summary["complete"] is False
    assert summary["profile_ready"] is False


def test_confirmation_receipts_cannot_replay_across_repeat_policies(tmp_path):
    from sneeze.tpn.ocr_tuning import (
        TuningError,
        build_tuning_plan,
        run_tuning_session,
        summarize_onboarding_results,
        summarize_tuning_results,
    )

    manifest, inputs, installations = _files(tmp_path)
    plan_arguments = {
        "pipeline_id": "local-mineru25",
        "hardware": _hardware(),
        "installations": installations,
        "manifest_path": manifest,
        "input_root": inputs,
        "memory_tiers": ("25",),
        "batch_sizes": (1,),
        "deepseek_engine_overhead_headroom_bytes": (1024**3,),
    }
    source_plan = build_tuning_plan(
        **plan_arguments,
        confirmation_repeats=2,
    )
    target_plan = build_tuning_plan(
        **plan_arguments,
        confirmation_repeats=5,
    )
    assert [item["candidate_id"] for item in source_plan["candidates"]] == [
        item["candidate_id"] for item in target_plan["candidates"]
    ]

    source_root = tmp_path / "source-session"
    source_summary = run_tuning_session(
        plan=source_plan,
        dest_dir=source_root,
        runner=lambda **kwargs: (
            _write_evidence_for_runner(kwargs) or {"all_pass": True}
        ),
    )
    candidate_results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(
            source_root.glob("candidates/*/attempt-*/candidate-result.json")
        )
    ]
    target_discovery = summarize_tuning_results(
        plan=target_plan,
        candidate_results=candidate_results,
    )

    with pytest.raises(
        TuningError,
        match="confirmation receipt does not match the current plan",
    ):
        summarize_onboarding_results(
            plan=target_plan,
            discovery=target_discovery,
            confirmation_results=source_summary["confirmation_results"],
        )


def test_tuning_session_resumes_completed_candidate_without_overwrite(
    tmp_path,
):
    from sneeze.tpn.ocr_backend_setup import make_receipt
    from sneeze.tpn.ocr_tuning import run_tuning_session

    manifest, inputs, installations = _files(tmp_path)
    candidate = _candidate()
    plan = make_receipt(
        "ocr_profile_tuning_plan",
        {
            "pipeline_id": "local-mineru25",
            "manifest": _manifest_contract(manifest),
            "input_root": str(inputs),
            "installations": installations,
            "policy": {
                "memory_tiers": ["25"],
                "discovery_repeats": 1,
                "timeout_seconds": 30,
                "suite_role": "hardware_smoke",
            },
            "candidates": [candidate],
        },
    )
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        _benchmark_evidence(Path(kwargs["dest_dir"]))
        return {"all_pass": True}

    destination = tmp_path / "session"
    first = run_tuning_session(
        plan=plan,
        dest_dir=destination,
        runner=runner,
    )
    second = run_tuning_session(
        plan=plan,
        dest_dir=destination,
        runner=runner,
    )

    assert len(calls) == 1
    assert first == second
    candidate_root = destination / "candidates" / candidate["candidate_id"]
    assert [path.name for path in candidate_root.iterdir()] == ["attempt-001"]


@pytest.mark.parametrize(
    ("artifact_name", "remove", "message"),
    (
        ("summary.json", False, "artifact was tampered"),
        ("results.jsonl", True, "artifact is missing"),
        ("run.json", False, "artifact was tampered"),
    ),
)
def test_resume_refuses_changed_discovery_artifacts(
    tmp_path,
    artifact_name,
    remove,
    message,
):
    from sneeze.tpn.ocr_backend_setup import make_receipt
    from sneeze.tpn.ocr_tuning import TuningError, run_tuning_session

    manifest, inputs, installations = _files(tmp_path)
    candidate = _candidate()
    plan = make_receipt(
        "ocr_profile_tuning_plan",
        {
            "pipeline_id": "local-mineru25",
            "manifest": _manifest_contract(manifest),
            "input_root": str(inputs),
            "installations": installations,
            "policy": {
                "memory_tiers": ["25"],
                "discovery_repeats": 1,
                "timeout_seconds": 30,
                "suite_role": "hardware_smoke",
            },
            "candidates": [candidate],
        },
    )
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        _benchmark_evidence(Path(kwargs["dest_dir"]))
        return {"all_pass": True}

    destination = tmp_path / "session"
    run_tuning_session(plan=plan, dest_dir=destination, runner=runner)
    attempt = (
        destination / "candidates" / candidate["candidate_id"] / "attempt-001"
    )
    artifact = attempt / "benchmark" / artifact_name
    if remove:
        artifact.unlink()
    else:
        artifact.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(TuningError, match=message):
        run_tuning_session(plan=plan, dest_dir=destination, runner=runner)
    assert len(calls) == 1


def test_onboarding_confirms_selected_candidates_and_exports_profile(
    tmp_path,
):
    from sneeze.tpn.ocr_profiles import load_execution_profile
    from sneeze.tpn.ocr_tuning import (
        build_tuning_plan,
        run_tuning_session,
        write_provisional_execution_profile,
    )

    manifest, inputs, installations = _files(tmp_path)
    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("25",),
        batch_sizes=(1,),
        discovery_repeats=1,
        confirmation_repeats=2,
    )
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        _write_evidence_for_runner(kwargs)
        return {"all_pass": True}

    destination = tmp_path / "session"
    first = run_tuning_session(
        plan=plan,
        dest_dir=destination,
        runner=runner,
    )
    call_count = len(calls)
    second = run_tuning_session(
        plan=plan,
        dest_dir=destination,
        runner=runner,
    )

    assert call_count == 6
    assert len(calls) == call_count
    assert first == second
    assert first["kind"] == "ocr_profile_onboarding_summary"
    assert first["qualification_status"] == "provisional"
    assert first["profile_ready"] is True
    assert first["completed_confirmation_count"] == 2
    assert all(
        result["repeats"] == 2 for result in first["confirmation_results"]
    )
    deepseek_confirmation = next(
        result
        for result in first["confirmation_results"]
        if result["backend_id"] == "deepseek-vllm"
    )
    assert (
        deepseek_confirmation["runtime_selector"]["attention_backend"]
        == "FLASH_ATTN"
    )
    assert deepseek_confirmation["observed_attention_backends"] == [
        "FLASH_ATTN"
    ]
    assert set(deepseek_confirmation["runtime_selector"]["packages"]) == {
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

    output = tmp_path / "profiles" / "synthetic-local-mineru25.json"
    profile = write_provisional_execution_profile(
        plan=plan,
        tuning_summary=first,
        profile_id="synthetic-local-mineru25",
        display_name="Synthetic local MinerU 2.5 profile",
        path=output,
    )
    reused = write_provisional_execution_profile(
        plan=plan,
        tuning_summary=first,
        profile_id="synthetic-local-mineru25",
        display_name="Synthetic local MinerU 2.5 profile",
        path=output,
    )

    assert profile == reused == load_execution_profile(output)
    assert profile["qualification"]["state"] == "provisional"
    assert profile["memory_tiers"]["25"]["supported"] is True
    assert profile["memory_tiers"]["50"] == {
        "supported": False,
        "reason": "tier_not_tuned",
        "budget_fraction": 0.5,
        "budget_bytes": 24 * 1024**3,
        "reserve_bytes": 2 * 1024**3,
        "stages": {},
    }
    assert profile["memory_tiers"]["max"]["supported"] is False
    assert (
        profile["stages"]["grounded_ocr"]["runtime_selector"]["dtype"]
        == "bfloat16"
    )
    assert {
        stage["model"]["snapshot_files_sha256"]
        for stage in profile["stages"].values()
    } == {
        _identity_digest("deepseek-vllm-snapshot"),
        _identity_digest("mineru25-pro-2605-snapshot"),
    }


def test_tuning_rejects_input_overlap_before_first_write(tmp_path):
    from sneeze.tpn.ocr_tuning import (
        TuningError,
        build_tuning_plan,
        run_tuning_session,
    )

    manifest, inputs, installations = _files(tmp_path)
    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("25",),
        batch_sizes=(1,),
    )
    destination = inputs / "forbidden-tuning-output"

    with pytest.raises(TuningError, match="separate from the input root"):
        run_tuning_session(
            plan=plan,
            dest_dir=destination,
            runner=lambda **_kwargs: {},
        )

    assert not destination.exists()


def test_tuning_rejects_symlink_destination_before_first_write(tmp_path):
    from sneeze.tpn.ocr_tuning import (
        TuningError,
        build_tuning_plan,
        run_tuning_session,
    )

    manifest, inputs, installations = _files(tmp_path)
    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("25",),
        batch_sizes=(1,),
    )
    target = tmp_path / "real-scratch"
    target.mkdir()
    destination = tmp_path / "linked-scratch"
    destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(TuningError, match="must not traverse symlinks"):
        run_tuning_session(
            plan=plan,
            dest_dir=destination,
            runner=lambda **_kwargs: {},
        )

    assert list(target.iterdir()) == []


def test_tuning_rejects_profile_output_in_input_before_first_write(tmp_path):
    from sneeze.tpn.ocr_tuning import (
        TuningError,
        build_tuning_plan,
        run_tuning_session,
    )

    manifest, inputs, installations = _files(tmp_path)
    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("25",),
        batch_sizes=(1,),
    )
    destination = tmp_path / "session"
    profile = inputs / "forbidden" / "profile.json"

    with pytest.raises(TuningError, match="separate from the input root"):
        run_tuning_session(
            plan=plan,
            dest_dir=destination,
            runner=lambda **_kwargs: {},
            profile_path=profile,
            profile_id="synthetic-local-mineru25",
            profile_display_name="Synthetic local MinerU profile",
        )

    assert not destination.exists()
    assert not profile.parent.exists()


def test_tuning_rejects_symlinked_profile_output_before_first_write(tmp_path):
    from sneeze.tpn.ocr_tuning import (
        TuningError,
        build_tuning_plan,
        run_tuning_session,
    )

    manifest, inputs, installations = _files(tmp_path)
    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("25",),
        batch_sizes=(1,),
    )
    target = tmp_path / "real-profile-dir"
    target.mkdir()
    linked = tmp_path / "linked-profile-dir"
    linked.symlink_to(target, target_is_directory=True)
    destination = tmp_path / "session"

    with pytest.raises(TuningError, match="must not traverse symlinks"):
        run_tuning_session(
            plan=plan,
            dest_dir=destination,
            runner=lambda **_kwargs: {},
            profile_path=linked / "profile.json",
            profile_id="synthetic-local-mineru25",
            profile_display_name="Synthetic local MinerU profile",
        )

    assert not destination.exists()
    assert list(target.iterdir()) == []


def test_confirmation_failure_marks_tier_unsupported(tmp_path):
    from sneeze.tpn.ocr_tuning import (
        TuningError,
        build_provisional_execution_profile,
        build_tuning_plan,
        run_tuning_session,
    )

    manifest, inputs, installations = _files(tmp_path)
    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("25", "50"),
        batch_sizes=(1,),
        confirmation_repeats=2,
    )

    def runner(**kwargs):
        spec = json.loads(Path(kwargs["provider_spec_path"]).read_text())
        backend_id = spec["provider_id"].removesuffix("-profile-tuning")
        settings = next(iter(spec["lanes"].values()))["settings"]
        confirmation = kwargs["repeats"] == 2
        failed_confirmation = (
            confirmation
            and backend_id.startswith("mineru")
            and settings["memory_tier"] == "25"
        )
        _write_evidence_for_runner(
            kwargs,
            all_pass=not failed_confirmation,
        )
        return {"all_pass": not failed_confirmation}

    summary = run_tuning_session(
        plan=plan,
        dest_dir=tmp_path / "session",
        runner=runner,
    )

    assert summary["complete"] is True
    assert summary["profile_ready"] is True
    assert summary["profile_blockers"] == []
    by_tier = {item["tier"]: item for item in summary["tiers"]}
    assert by_tier["25"]["supported"] is False
    assert by_tier["25"]["reason"] == "confirmation_failed"
    assert by_tier["50"]["supported"] is True

    profile = build_provisional_execution_profile(
        plan=plan,
        tuning_summary=summary,
        profile_id="synthetic-partial-tier-profile",
        display_name="Synthetic partial-tier profile",
    )
    assert profile["qualification"]["state"] == "provisional"
    assert profile["memory_tiers"]["25"]["supported"] is False
    assert profile["memory_tiers"]["25"]["reason"] == ("confirmation_failed")
    assert profile["memory_tiers"]["50"]["supported"] is True

    tampered_summary = copy.deepcopy(summary)
    tampered_summary["receipt_sha256"] = "0" * 64
    with pytest.raises(TuningError, match="content hash does not match"):
        build_provisional_execution_profile(
            plan=plan,
            tuning_summary=tampered_summary,
            profile_id="synthetic-tampered-summary-profile",
            display_name="Synthetic tampered summary profile",
        )

    drifted = copy.deepcopy(summary)
    markdown_confirmations = [
        item
        for item in drifted["confirmation_results"]
        if item["capability"] == "markdown"
    ]
    assert len(markdown_confirmations) == 2
    failed_confirmation = next(
        item for item in markdown_confirmations if item["tier"] == "25"
    )
    supported_confirmation = next(
        item for item in markdown_confirmations if item["tier"] == "50"
    )
    expected_snapshot = supported_confirmation["model_evidence"][
        "snapshot_files_sha256"
    ]
    previous_receipt = failed_confirmation["receipt_sha256"]
    failed_confirmation["model_evidence"]["snapshot_files_sha256"] = (
        "a" * 64
    )
    replacement = _reseal_receipt(failed_confirmation)
    drifted["confirmation_results"] = [
        replacement if item["receipt_sha256"] == previous_receipt else item
        for item in drifted["confirmation_results"]
    ]
    for tier in drifted["tiers"]:
        for stage in tier["stages"]:
            if stage.get("confirmation_receipt_sha256") == previous_receipt:
                stage["confirmation_receipt_sha256"] = replacement[
                    "receipt_sha256"
                ]
    drifted = _reseal_receipt(drifted)
    drifted_profile = build_provisional_execution_profile(
        plan=plan,
        tuning_summary=drifted,
        profile_id="synthetic-drifted-model-profile",
        display_name="Synthetic drifted model profile",
    )
    assert drifted_profile["stages"]["markdown"]["model"][
        "snapshot_files_sha256"
    ] == expected_snapshot


def test_resume_refuses_missing_confirmation_artifact(tmp_path):
    from sneeze.tpn.ocr_tuning import (
        TuningError,
        build_tuning_plan,
        run_tuning_session,
    )

    manifest, inputs, installations = _files(tmp_path)
    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("25",),
        batch_sizes=(1,),
        confirmation_repeats=2,
    )
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        _write_evidence_for_runner(kwargs)
        return {"all_pass": True}

    destination = tmp_path / "session"
    run_tuning_session(plan=plan, dest_dir=destination, runner=runner)
    confirmation_root = next((destination / "confirmations").iterdir())
    artifact = confirmation_root / "attempt-001" / "benchmark" / "run.json"
    artifact.unlink()

    with pytest.raises(TuningError, match="artifact is missing"):
        run_tuning_session(plan=plan, dest_dir=destination, runner=runner)
    assert len(calls) == 6


def test_resume_rejects_changed_installation_identity_at_same_paths(tmp_path):
    from sneeze.tpn.ocr_backend_setup import sha256_json
    from sneeze.tpn.ocr_tuning import (
        TuningError,
        build_tuning_plan,
        run_tuning_session,
    )

    manifest, inputs, installations = _files(tmp_path)
    arguments = {
        "pipeline_id": "local-mineru25",
        "hardware": _hardware(),
        "manifest_path": manifest,
        "input_root": inputs,
        "memory_tiers": ("25",),
        "batch_sizes": (1,),
        "deepseek_engine_overhead_headroom_bytes": (1024**3,),
    }
    first = build_tuning_plan(
        installations=installations,
        **arguments,
    )
    destination = tmp_path / "identity-bound-session"

    def runner(**kwargs):
        _write_evidence_for_runner(kwargs)
        return {"all_pass": True}

    run_tuning_session(plan=first, dest_dir=destination, runner=runner)
    changed = copy.deepcopy(installations)
    changed["deepseek-vllm"]["identity"]["record_sha256"] = "f" * 64
    second = build_tuning_plan(installations=changed, **arguments)
    assert first["receipt_sha256"] != second["receipt_sha256"]
    with pytest.raises(TuningError, match="belongs to another plan"):
        run_tuning_session(plan=second, dest_dir=destination, runner=runner)

    distribution_changed = copy.deepcopy(installations)
    identity = distribution_changed["deepseek-vllm"]["identity"]
    distributions = ["new-runtime==3.0", "shared-runtime==2.0"]
    identity["installed_distributions_count"] = len(distributions)
    identity["installed_distributions_sha256"] = sha256_json(distributions)
    third = build_tuning_plan(
        installations=distribution_changed,
        **arguments,
    )
    assert first["receipt_sha256"] != third["receipt_sha256"]
    assert {
        item["candidate_id"] for item in first["candidates"]
    } != {item["candidate_id"] for item in third["candidates"]}


def test_live_free_memory_changes_do_not_break_safe_resume_identity(tmp_path):
    from sneeze.tpn.ocr_tuning import build_tuning_plan

    manifest, inputs, installations = _files(tmp_path)

    def live_hardware(free_bytes):
        total = 48 * 1024**3
        return {
            **_hardware(),
            "memory_budgets": {
                "memory_evidence": {
                    "total_sources": {"torch_cuda": total},
                    "free_memory_bytes": free_bytes,
                },
                "tiers": [
                    {
                        "tier": "25",
                        "target_fraction": 0.25,
                        "reserve_bytes": 2 * 1024**3,
                        "total_memory_bytes": total,
                        "capacity_budget_bytes": 12 * 1024**3,
                        "minimum_required_bytes": 0,
                        "supported": True,
                        "live_free_memory_bytes": free_bytes,
                        "live_memory_after_reserve_bytes": (
                            free_bytes - 2 * 1024**3
                        ),
                        "minimum_fit_available": True,
                        "full_budget_available": True,
                        "reason_codes": ["supported_and_available"],
                    }
                ],
            },
        }

    arguments = {
        "pipeline_id": "local-mineru25",
        "installations": installations,
        "manifest_path": manifest,
        "input_root": inputs,
        "memory_tiers": ("25",),
        "batch_sizes": (1,),
        "deepseek_engine_overhead_headroom_bytes": (1024**3,),
    }
    first = build_tuning_plan(
        hardware=live_hardware(40 * 1024**3),
        **arguments,
    )
    second = build_tuning_plan(
        hardware=live_hardware(36 * 1024**3),
        **arguments,
    )
    assert first == second


def test_abandoned_confirmation_is_preserved_and_resumed(tmp_path):
    from sneeze.tpn.ocr_backend_setup import make_receipt
    from sneeze.tpn.ocr_tuning import run_tuning_session

    manifest, inputs, installations = _files(tmp_path)
    candidate = _candidate()
    plan = make_receipt(
        "ocr_profile_tuning_plan",
        {
            "pipeline_id": "local-mineru25",
            "hardware": {"selector": _hardware()["selector"]},
            "memory_budgets": {
                "25": {
                    "budget_fraction": 0.25,
                    "budget_bytes": 12 * 1024**3,
                    "reserve_bytes": 2 * 1024**3,
                }
            },
            "manifest": _manifest_contract(manifest),
            "input_root": str(inputs),
            "installations": installations,
            "policy": {
                "memory_tiers": ["25"],
                "discovery_repeats": 1,
                "confirmation_repeats": 2,
                "timeout_seconds": 30,
                "suite_role": "hardware_smoke",
                "suite_id": "cuphoton-three-page-smoke",
            },
            "candidates": [candidate],
        },
    )
    calls = []
    fail_confirmation = True

    def runner(**kwargs):
        nonlocal fail_confirmation
        calls.append(kwargs)
        if kwargs["repeats"] == 2 and fail_confirmation:
            fail_confirmation = False
            raise RuntimeError("synthetic confirmation interruption")
        _write_evidence_for_runner(kwargs)
        return {"all_pass": True}

    destination = tmp_path / "session"
    first = run_tuning_session(
        plan=plan,
        dest_dir=destination,
        runner=runner,
    )
    second = run_tuning_session(
        plan=plan,
        dest_dir=destination,
        runner=runner,
    )

    assert first["confirmation_complete"] is False
    assert second["confirmation_complete"] is True
    assert len(calls) == 3
    confirmation_root = next((destination / "confirmations").iterdir())
    attempts = sorted(confirmation_root.iterdir())
    assert [path.name for path in attempts] == [
        "attempt-001",
        "attempt-002",
    ]
    assert (attempts[0] / "abandoned.json").is_file()
    assert (attempts[1] / "confirmation-result.json").is_file()
    current = json.loads(
        (destination / "tuning-summary.json").read_text(encoding="utf-8")
    )
    assert current == second
    archived = {
        path.stem for path in (destination / "summaries").glob("*.json")
    }
    assert archived == {
        first["receipt_sha256"],
        second["receipt_sha256"],
    }


def _mock_tuning_command_runtime(monkeypatch, captured):
    from sneeze.tpn import ocr_hardware, ocr_tuning

    def probe(**kwargs):
        captured["probe"] = kwargs
        return {
            "selector": _hardware()["selector"],
            "selector_sha256": "a" * 64,
            "memory": {
                "total_sources": {"torch_cuda": 48 * 1024**3},
            },
        }

    def calculate(observation):
        captured["budget_observation"] = observation
        return {"tiers": []}

    def build(**kwargs):
        captured["build"] = kwargs
        return {"receipt_sha256": "b" * 64, "plan": True}

    def run(**kwargs):
        captured["run"] = kwargs
        return {"complete": True, "profile_ready": True}

    monkeypatch.setattr(ocr_hardware, "probe_nvidia_hardware", probe)
    monkeypatch.setattr(ocr_hardware, "calculate_memory_tiers", calculate)
    monkeypatch.setattr(ocr_tuning, "build_tuning_plan", build)
    monkeypatch.setattr(ocr_tuning, "run_tuning_session", run)


def _configured_tuning_command(tmp_path):
    from io import StringIO

    from sneeze.tpn.commands import OcrProfileTune

    manifest, inputs, _installations = _files(tmp_path)
    command = OcrProfileTune(None, StringIO(), None)
    command._manifest = str(manifest)
    command._input_root = str(inputs)
    command._dest_dir = str(tmp_path / "session")
    command.pipeline = "local-mineru25"
    return command


def _command_installation_record(
    backend_id,
    *,
    cache_dir,
    visible_devices="GPU-aabbccdd",
    attention_backend=None,
):
    mineru = backend_id.startswith("mineru")
    return {
        "record_sha256": _identity_digest(backend_id + "-record"),
        "python_executable": {
            "path": sys.executable,
            "sha256": _identity_digest(backend_id + "-python"),
            "installed_distributions_count": _distribution_identity(
                backend_id
            )["count"],
            "installed_distributions_sha256": _distribution_identity(
                backend_id
            )["sha256"],
        },
        "hf_cache": {
            "path": str(cache_dir),
            "identity_sha256": _identity_digest(backend_id + "-cache"),
        },
        "snapshot": {
            "files_sha256": _identity_digest(backend_id + "-snapshot"),
            "weights_sha256": (
                _identity_digest(backend_id + "-weights")
                if mineru
                else None
            ),
        },
        "adapter": (
            {"sha256": _identity_digest(backend_id + "-adapter")}
            if mineru
            else None
        ),
        "implementation": {
            "sha256": _identity_digest(backend_id + "-implementation")
        },
        "cuda": {
            "visible_devices": visible_devices,
            "device": "cuda:0",
            "attention_backend": attention_backend,
        },
    }


def _command_installation_identity(record):
    return {
        "record_sha256": record["record_sha256"],
        "python_executable_sha256": record["python_executable"]["sha256"],
        "installed_distributions_count": record["python_executable"][
            "installed_distributions_count"
        ],
        "installed_distributions_sha256": record["python_executable"][
            "installed_distributions_sha256"
        ],
        "hf_cache_identity_sha256": record["hf_cache"]["identity_sha256"],
        "snapshot_files_sha256": record["snapshot"]["files_sha256"],
        "snapshot_weights_sha256": record["snapshot"]["weights_sha256"],
        "adapter_sha256": (
            record["adapter"]["sha256"]
            if record["adapter"] is not None
            else None
        ),
        "implementation_sha256": record["implementation"]["sha256"],
    }


def test_tuning_command_consumes_nested_registry_and_exports_profile(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn import ocr_installations

    captured = {}
    _mock_tuning_command_runtime(monkeypatch, captured)
    deepseek_cache = tmp_path / "deepseek-cache"
    mineru_cache = tmp_path / "mineru-cache"
    deepseek_cache.mkdir()
    mineru_cache.mkdir()
    registry_path = (tmp_path / "installations.json").resolve()
    registry = {
        "installations": {
            "deepseek-vllm": _command_installation_record(
                "deepseek-vllm",
                cache_dir=deepseek_cache,
            ),
            "mineru25-pro-2605": _command_installation_record(
                "mineru25-pro-2605",
                cache_dir=mineru_cache,
                attention_backend="sdpa",
            ),
        }
    }

    def load(path, **kwargs):
        captured["registry"] = {"path": path, **kwargs}
        return registry

    monkeypatch.setattr(ocr_installations, "load_installation_registry", load)
    command = _configured_tuning_command(tmp_path)
    command._installation_registry = str(registry_path)
    command._profile_output = str(
        (tmp_path / "profiles" / "synthetic.json").resolve()
    )
    command._profile_id = "synthetic-local-mineru25"
    command._profile_display_name = "Synthetic local MinerU profile"

    command.run()

    assert captured["registry"] == {
        "path": str(registry_path),
        "validate_current": True,
        "missing_ok": False,
    }
    assert captured["probe"] == {
        "torch_python": sys.executable,
        "visible_devices": "GPU-aabbccdd",
        "cuda_device_order": "PCI_BUS_ID",
    }
    assert captured["build"]["pipeline_id"] == "local-mineru25"
    assert captured["build"]["installations"] == {
        "deepseek-vllm": {
            "python_executable": sys.executable,
            "cache_dir": str(deepseek_cache),
            "cuda_visible_devices": "GPU-aabbccdd",
            "device": "cuda:0",
            "attention_backend": "auto",
            "identity": _command_installation_identity(
                registry["installations"]["deepseek-vllm"]
            ),
        },
        "mineru25-pro-2605": {
            "python_executable": sys.executable,
            "cache_dir": str(mineru_cache),
            "cuda_visible_devices": "GPU-aabbccdd",
            "device": "cuda:0",
            "attention_backend": "sdpa",
            "identity": _command_installation_identity(
                registry["installations"]["mineru25-pro-2605"]
            ),
        },
    }
    assert captured["run"]["profile_path"] == command._profile_output
    assert captured["run"]["profile_id"] == command._profile_id
    assert captured["run"]["profile_display_name"] == (
        command._profile_display_name
    )


def test_tuning_command_keeps_complete_explicit_raw_fallback(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn import ocr_installations

    captured = {}
    _mock_tuning_command_runtime(monkeypatch, captured)

    def unexpected_registry(*_args, **_kwargs):
        raise AssertionError("raw fallback must not load the registry")

    monkeypatch.setattr(
        ocr_installations,
        "load_installation_registry",
        unexpected_registry,
    )
    deepseek_cache = tmp_path / "deepseek-cache"
    mineru_cache = tmp_path / "mineru-cache"
    deepseek_cache.mkdir()
    mineru_cache.mkdir()
    records = {
        "deepseek-vllm": _command_installation_record(
            "deepseek-vllm",
            cache_dir=deepseek_cache,
            visible_devices="2",
            attention_backend="FLASH_ATTN",
        ),
        "mineru25-pro-2605": _command_installation_record(
            "mineru25-pro-2605",
            cache_dir=mineru_cache,
            visible_devices="2",
            attention_backend="sdpa",
        ),
    }
    monkeypatch.setattr(
        ocr_installations,
        "build_backend_installation",
        lambda backend_id, **_kwargs: records[backend_id],
    )
    command = _configured_tuning_command(tmp_path)
    command._deepseek_python = sys.executable
    command._deepseek_cache_dir = str(deepseek_cache)
    command._mineru_python = sys.executable
    command._mineru_cache_dir = str(mineru_cache)
    command.cuda_visible_devices = "2"
    command.device = "cuda:0"
    command.deepseek_attention_backend = "FLASH_ATTN"

    command.run()

    installations = captured["build"]["installations"]
    assert installations["deepseek-vllm"]["attention_backend"] == (
        "FLASH_ATTN"
    )
    assert installations["deepseek-vllm"]["cuda_visible_devices"] == "2"
    assert installations["mineru25-pro-2605"]["cache_dir"] == str(
        mineru_cache
    )
    assert captured["run"]["profile_path"] is None


@pytest.mark.parametrize(
    ("attribute", "value", "option"),
    (
        ("cuda_visible_devices", "0", "--cuda-visible-devices"),
        ("device", "cuda:0", "--device"),
        (
            "deepseek_attention_backend",
            "auto",
            "--deepseek-attention-backend",
        ),
        (
            "mineru_attention_backend",
            "sdpa",
            "--mineru-attention-backend",
        ),
    ),
)
def test_registry_tuning_rejects_explicit_runtime_overrides(
    tmp_path,
    monkeypatch,
    attribute,
    value,
    option,
):
    from sneeze.command import CommandError

    from sneeze.tpn import ocr_installations

    def unexpected_registry(*_args, **_kwargs):
        raise AssertionError("override validation must precede registry IO")

    monkeypatch.setattr(
        ocr_installations,
        "load_installation_registry",
        unexpected_registry,
    )
    command = _configured_tuning_command(tmp_path)
    setattr(command, attribute, value)

    with pytest.raises(CommandError, match=option):
        command.run()


def test_raw_fallback_keeps_implicit_runtime_defaults(tmp_path, monkeypatch):
    from sneeze.tpn import ocr_installations

    captured = {}
    _mock_tuning_command_runtime(monkeypatch, captured)
    deepseek_cache = tmp_path / "deepseek-cache"
    mineru_cache = tmp_path / "mineru-cache"
    deepseek_cache.mkdir()
    mineru_cache.mkdir()
    calls = {}

    def build(backend_id, **kwargs):
        calls[backend_id] = kwargs
        return _command_installation_record(
            backend_id,
            cache_dir=kwargs["hf_cache"],
            visible_devices=kwargs["cuda_visible_devices"],
            attention_backend=kwargs["attention_backend"],
        )

    monkeypatch.setattr(
        ocr_installations,
        "build_backend_installation",
        build,
    )
    command = _configured_tuning_command(tmp_path)
    command._deepseek_python = sys.executable
    command._deepseek_cache_dir = str(deepseek_cache)
    command._mineru_python = sys.executable
    command._mineru_cache_dir = str(mineru_cache)

    command.run()

    assert calls["deepseek-vllm"] == {
        "python_executable": sys.executable,
        "hf_cache": str(deepseek_cache),
        "cuda_visible_devices": "0",
        "device": "cuda:0",
        "attention_backend": "auto",
    }
    assert calls["mineru25-pro-2605"] == {
        "python_executable": sys.executable,
        "hf_cache": str(mineru_cache),
        "cuda_visible_devices": "0",
        "device": "cuda:0",
        "attention_backend": "sdpa",
    }


@pytest.mark.parametrize(
    ("attributes", "message"),
    (
        (
            {"_profile_output": "/tmp/profile.json"},
            "must be supplied together",
        ),
        (
            {
                "_deepseek_python": sys.executable,
                "_deepseek_cache_dir": "/tmp",
            },
            "requires all Python/cache options",
        ),
    ),
)
def test_tuning_command_rejects_partial_input_groups_before_probe(
    tmp_path,
    monkeypatch,
    attributes,
    message,
):
    from sneeze.command import CommandError

    from sneeze.tpn import ocr_hardware

    def unexpected_probe(**_kwargs):
        raise AssertionError("validation must happen before GPU probing")

    monkeypatch.setattr(
        ocr_hardware,
        "probe_nvidia_hardware",
        unexpected_probe,
    )
    command = _configured_tuning_command(tmp_path)
    for name, value in attributes.items():
        setattr(command, name, value)

    with pytest.raises(CommandError, match=message):
        command.run()


def test_tuning_command_withholds_deepseek_markdown_pipeline(
    tmp_path,
    monkeypatch,
):
    from sneeze.command import CommandError

    from sneeze.tpn import ocr_installations

    def unexpected_registry(*_args, **_kwargs):
        raise AssertionError("withheld pipeline must fail before registry IO")

    monkeypatch.setattr(
        ocr_installations,
        "load_installation_registry",
        unexpected_registry,
    )
    command = _configured_tuning_command(tmp_path)
    command.pipeline = "deepseek-full"

    with pytest.raises(CommandError, match="true ungrounded Markdown lane"):
        command.run()


def test_tuning_command_new_paths_are_optional_long_options():
    from sneeze.tpn.commands import OcrProfileTune

    assert OcrProfileTune.DeepseekPythonArg._mandatory is False
    assert OcrProfileTune.MineruPythonArg._mandatory is False
    assert OcrProfileTune.InstallationRegistryArg._arg == (
        "--installation-registry"
    )
    assert OcrProfileTune.ProfileOutputArg._arg == "--profile-output"
    assert OcrProfileTune.ProfileIdArg._arg == "--profile-id"
    assert OcrProfileTune.ProfileDisplayNameArg._arg == (
        "--profile-display-name"
    )


def test_tuning_rejects_cross_user_writable_destination_before_runner(
    tmp_path,
):
    from sneeze.tpn.ocr_tuning import (
        TuningError,
        build_tuning_plan,
        run_tuning_session,
    )

    manifest, inputs, installations = _files(tmp_path)
    plan = build_tuning_plan(
        pipeline_id="local-mineru25",
        hardware=_hardware(),
        installations=installations,
        manifest_path=manifest,
        input_root=inputs,
        memory_tiers=("25",),
        batch_sizes=(1,),
        deepseek_engine_overhead_headroom_bytes=(1024**3,),
    )
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o775)
    called = False

    def runner(**_kwargs):
        nonlocal called
        called = True

    with pytest.raises(TuningError, match="writable by group"):
        run_tuning_session(
            plan=plan,
            dest_dir=shared / "session",
            runner=runner,
        )
    assert called is False
