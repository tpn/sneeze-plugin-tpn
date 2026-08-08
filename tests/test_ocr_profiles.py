import builtins
import copy
import hashlib
import importlib
import json
import os
import sys

import pytest

GIB = 1024**3
TOTAL_MEMORY = 48 * GIB


def _digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _runtime(
    engine,
    version,
    *,
    attention_backend,
    packages,
    implementation_sha256,
):
    return {
        "python": "3.12.13",
        "engine": engine,
        "engine_version": version,
        "torch": "2.11.0+cu130",
        "torch_cuda": "13.0",
        "packages": packages,
        "attention_backend": attention_backend,
        "dtype": "bfloat16",
        "implementation_sha256": implementation_sha256,
        "installed_distributions_count": len(packages),
        "installed_distributions_sha256": _digest(
            "installed-distributions-" + implementation_sha256
        ),
    }


def _stage_runtime(backend_id):
    from sneeze.tpn.ocr_backends import (
        DEEPSEEK_IMPLEMENTATION_SHA256,
        MINERU_IMPLEMENTATION_SHA256,
    )

    if backend_id == "deepseek-vllm":
        return _runtime(
            "vllm",
            "0.21.1rc1.dev438+g4ff865c38",
            attention_backend="FLASH_ATTN",
            packages={
                "transformers": "5.9.0",
                "vllm": "0.21.1rc1.dev438+g4ff865c38",
            },
            implementation_sha256=DEEPSEEK_IMPLEMENTATION_SHA256,
        )
    return _runtime(
        "transformers",
        "4.57.6",
        attention_backend="sdpa",
        packages={
            "mineru-vl-utils": "1.0.5",
            "transformers": "4.57.6",
        },
        implementation_sha256=MINERU_IMPLEMENTATION_SHA256,
    )


def _tier_stage(stage_id, budget, tier):
    if stage_id == "grounded_ocr":
        settings = {
            "attention_backend": "FLASH_ATTN",
            "batch_size": 1,
            "enable_chunked_prefill": True,
            "engine_overhead_headroom_bytes": GIB,
            "gpu_memory_utilization": round(
                (budget - GIB) / TOTAL_MEMORY, 8
            ),
            "gpu_memory_utilization_denominator_bytes": TOTAL_MEMORY,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 1,
            "max_tokens": 8192,
            "memory_budget_bytes": budget,
            "memory_tier": tier,
            "mm_encoder_attn_backend": "auto",
            "moe_backend": "triton",
            "ngram_size": 10,
            "temperature": 0.0,
            "window_size": 50,
        }
    else:
        settings = {
            "backend": "official-two-step-transformers",
            "batch_size": 4,
            "image_analysis": False,
            "memory_budget_bytes": budget,
            "memory_tier": tier,
        }
    return {
        "settings": settings,
        "observed_peak_memory_bytes": min(8 * GIB, budget),
        "confirmed_pages_per_second": 0.1,
    }


def _profile_payload(
    *,
    profile_id="rtx-pro-6000-local-mineru25",
    state="qualified",
    supported=("25", "50", "max"),
    include_evidence=True,
    memory_model="dedicated",
    evidence_repeats=5,
):
    from sneeze.tpn.ocr_profiles import (
        get_pipeline_definition,
        qualification_policy_sha256,
    )

    pipeline = get_pipeline_definition("local-mineru25")
    selector = {
        "schema_version": 1,
        "vendor": "nvidia",
        "product_name": "NVIDIA RTX PRO 6000 Blackwell",
        "pci_device_id": "0x2bb0",
        "compute_capability": [12, 0],
        "memory_model": memory_model,
        "total_memory_bytes": TOTAL_MEMORY,
    }
    stages = {}
    for stage in pipeline["stages"]:
        stages[stage["stage_id"]] = {
            "backend_id": stage["backend_id"],
            "capability": stage["capability"],
            "model": {
                **copy.deepcopy(stage["model"]),
                "snapshot_files_sha256": _digest(
                    f"snapshot-{stage['backend_id']}"
                ),
            },
            "batch_semantics": stage["batch_semantics"],
            "runtime_selector": _stage_runtime(stage["backend_id"]),
        }

    tiers = {}
    supported = set(supported)
    max_fraction = 0.9 if memory_model == "dedicated" else 0.75
    reserve = 2 * GIB if memory_model == "dedicated" else 16 * GIB
    for tier, fraction in (
        ("25", 0.25),
        ("50", 0.5),
        ("max", max_fraction),
    ):
        budget = min(
            int(TOTAL_MEMORY * fraction),
            TOTAL_MEMORY - reserve,
        )
        is_supported = tier in supported
        tiers[tier] = {
            "supported": is_supported,
            "reason": None if is_supported else "model_does_not_fit",
            "budget_fraction": fraction,
            "budget_bytes": budget,
            "reserve_bytes": reserve,
            "stages": (
                {
                    stage_id: _tier_stage(stage_id, budget, tier)
                    for stage_id in stages
                }
                if is_supported
                else {}
            ),
        }

    evidence = []
    if include_evidence:
        for tier in sorted(supported):
            evidence.append(
                {
                    "suite_id": "cuphoton-regressions",
                    "memory_tier": tier,
                    "manifest_sha256": _digest(f"manifest-{tier}"),
                    "summary_sha256": _digest(f"summary-{tier}"),
                    "experiment_sha256": _digest(f"experiment-{tier}"),
                    "repeats": evidence_repeats,
                    "all_pass": True,
                    "provenance_complete": True,
                }
            )

    return {
        "schema_version": 1,
        "kind": "ocr_execution_profile",
        "profile_id": profile_id,
        "display_name": "RTX PRO 6000 local MinerU profile",
        "pipeline_id": pipeline["pipeline_id"],
        "pipeline_definition_sha256": pipeline["definition_sha256"],
        "hardware_selector": selector,
        "stages": stages,
        "memory_tiers": tiers,
        "qualification": {
            "state": state,
            "policy_sha256": qualification_policy_sha256(),
            "required_suites": ["cuphoton-regressions"],
            "evidence": evidence,
        },
    }


def _profile(**kwargs):
    from sneeze.tpn.ocr_profiles import seal_execution_profile

    return seal_execution_profile(_profile_payload(**kwargs))


def _observation(profile):
    return {
        "hardware": {
            "selector": copy.deepcopy(profile["hardware_selector"]),
            "hostname": "qualification-host",
            "device": {
                "uuid": "GPU-AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE",
                "pci_bus_id": "00000000:01:00.0",
            },
        },
        "stages": {
            stage_id: {
                "backend_id": stage["backend_id"],
                "capability": stage["capability"],
                "model": copy.deepcopy(stage["model"]),
                "runtime": {
                    **copy.deepcopy(stage["runtime_selector"]),
                    "hostname": "qualification-host",
                    "driver_version": "590.00",
                },
            }
            for stage_id, stage in profile["stages"].items()
        },
    }


def test_module_import_is_stdlib_only(monkeypatch):
    blocked = {"torch", "transformers", "vllm", "pynvml"}
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in blocked:
            raise AssertionError(f"heavy import during discovery: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    sys.modules.pop("sneeze.tpn.ocr_profiles", None)
    module = importlib.import_module("sneeze.tpn.ocr_profiles")
    assert module.SCHEMA_VERSION == 1


def test_builtin_pipelines_pin_models_and_distinguish_batch_semantics():
    from sneeze.tpn.ocr_backends import get_backend
    from sneeze.tpn.ocr_profiles import (
        content_sha256,
        get_pipeline_definition,
        list_pipeline_definitions,
    )

    pipelines = list_pipeline_definitions()
    assert [item["pipeline_id"] for item in pipelines] == [
        "deepseek-vllm-full",
        "local-mineru25",
    ]
    for pipeline in pipelines:
        payload = dict(pipeline)
        digest = payload.pop("definition_sha256")
        assert digest == content_sha256(payload)

    legacy = get_pipeline_definition("legacy-deepseek-full")
    assert legacy["pipeline_id"] == "deepseek-vllm-full"
    assert [stage["batch_semantics"] for stage in legacy["stages"]] == [
        "page_batch",
        "page_batch",
    ]

    local = get_pipeline_definition("local-mineru25")
    assert local["execution_order"] == "sequential"
    assert [stage["batch_semantics"] for stage in local["stages"]] == [
        "page_batch",
        "layout_block_batch",
    ]
    expected = {
        "grounded_ocr": get_backend("deepseek-vllm"),
        "markdown": get_backend("mineru25-pro-2605"),
    }
    for stage in local["stages"]:
        backend = expected[stage["stage_id"]]
        assert stage["model"] == {
            "requested": backend.model_id,
            "revision": backend.model_revision,
        }


def test_profile_hash_is_canonical_and_detects_mutation():
    from sneeze.tpn.ocr_profiles import (
        OcrProfileError,
        content_sha256,
        profile_receipt_sha256,
        validate_execution_profile,
    )

    profile = _profile()
    payload = dict(profile)
    digest = payload.pop("profile_sha256")
    assert digest == content_sha256(payload)
    assert profile_receipt_sha256(profile) == digest

    profile["memory_tiers"]["max"]["stages"]["markdown"]["settings"][
        "batch_size"
    ] = 2
    with pytest.raises(
        OcrProfileError,
        match="does not match profile content",
    ):
        validate_execution_profile(profile)


def test_known_legacy_implementation_profile_without_moe_is_readable():
    from sneeze.tpn.ocr_profiles import seal_execution_profile

    legacy = _profile_payload()
    legacy["stages"]["grounded_ocr"]["runtime_selector"][
        "implementation_sha256"
    ] = (
        "1572663ce37a0735db51abb66454a0b1c5cbd8db7dc8bb96153698342f2e58be"
    )
    for tier in legacy["memory_tiers"].values():
        if tier["supported"]:
            tier["stages"]["grounded_ocr"]["settings"].pop(
                "moe_backend"
            )

    assert seal_execution_profile(legacy)["profile_sha256"]


def test_profile_requires_triton_moe_for_runtime_control_aware_source():
    from sneeze.tpn.ocr_profiles import (
        OcrProfileError,
        seal_execution_profile,
    )

    missing = _profile_payload()
    missing["stages"]["grounded_ocr"]["runtime_selector"][
        "implementation_sha256"
    ] = _digest("runtime-control-aware-deepseek")
    for tier in missing["memory_tiers"].values():
        if not tier["supported"]:
            continue
        settings = tier["stages"]["grounded_ocr"]["settings"]
        settings.pop("moe_backend")
    with pytest.raises(OcrProfileError, match="missing moe_backend"):
        seal_execution_profile(missing)

    invalid = _profile_payload()
    invalid["memory_tiers"]["max"]["stages"]["grounded_ocr"][
        "settings"
    ]["moe_backend"] = "auto"
    with pytest.raises(OcrProfileError, match="moe_backend must be triton"):
        seal_execution_profile(invalid)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda model: model.pop("snapshot_files_sha256"),
            "missing snapshot_files_sha256",
        ),
        (
            lambda model: model.__setitem__(
                "snapshot_files_sha256", "not-a-digest"
            ),
            "must be a lowercase SHA-256",
        ),
    ],
)
def test_profile_stage_requires_exact_model_snapshot_hash(mutation, message):
    from sneeze.tpn.ocr_profiles import (
        OcrProfileError,
        seal_execution_profile,
    )

    payload = _profile_payload()
    mutation(payload["stages"]["markdown"]["model"])
    with pytest.raises(OcrProfileError, match=message):
        seal_execution_profile(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["stages"]["markdown"][
                "runtime_selector"
            ].__setitem__("cache_dir", "/tmp/models"),
            "absolute host path",
        ),
        (
            lambda value: value["stages"]["markdown"][
                "runtime_selector"
            ].__setitem__("api_key", "synthetic-do-not-copy"),
            "credential-bearing field",
        ),
        (
            lambda value: value["stages"]["markdown"][
                "runtime_selector"
            ].__setitem__(
                "opaque_value", "hf_abcdefghijklmnopqrstuvwxyz012345"
            ),
            "credential-like data",
        ),
        (
            lambda value: value["stages"]["markdown"][
                "runtime_selector"
            ].__setitem__("build_flavor", "synthetic"),
            "runtime_selector fields are invalid: unexpected build_flavor",
        ),
        (
            lambda value: value.__setitem__("hostname", "build-host"),
            "fields are invalid",
        ),
        (
            lambda value: value["hardware_selector"].__setitem__(
                "uuid", "GPU-AAAA"
            ),
            "fields are invalid",
        ),
    ],
)
def test_portable_profile_rejects_paths_secrets_and_identity(
    mutation,
    message,
):
    from sneeze.tpn.ocr_profiles import (
        OcrProfileError,
        seal_execution_profile,
    )

    payload = _profile_payload()
    mutation(payload)
    with pytest.raises(OcrProfileError, match=message):
        seal_execution_profile(payload)


def test_memory_tiers_are_exact_and_may_be_unsupported():
    from sneeze.tpn.ocr_profiles import (
        OcrProfileError,
        seal_execution_profile,
    )

    profile = _profile(supported=("50", "max"))
    assert not profile["memory_tiers"]["25"]["supported"]
    assert profile["memory_tiers"]["25"]["stages"] == {}

    bad_budget = _profile_payload(supported=("50", "max"))
    bad_budget["memory_tiers"]["25"]["budget_bytes"] += 1
    with pytest.raises(OcrProfileError, match="must equal canonical budget"):
        seal_execution_profile(bad_budget)

    conservative = _profile_payload(supported=("50", "max"))
    conservative["memory_tiers"]["max"]["budget_bytes"] -= GIB
    conservative_budget = conservative["memory_tiers"]["max"][
        "budget_bytes"
    ]
    for stage in conservative["memory_tiers"]["max"]["stages"].values():
        stage["settings"]["memory_budget_bytes"] = conservative_budget
    conservative["memory_tiers"]["max"]["stages"]["grounded_ocr"][
        "settings"
    ]["gpu_memory_utilization"] = round(
        (conservative_budget - GIB) / TOTAL_MEMORY, 8
    )
    with pytest.raises(OcrProfileError, match="must equal canonical budget"):
        seal_execution_profile(conservative)

    wrong_reserve = _profile_payload(supported=("50", "max"))
    wrong_reserve["memory_tiers"]["50"]["reserve_bytes"] = 3 * GIB
    with pytest.raises(OcrProfileError, match="reserve_bytes must be"):
        seal_execution_profile(wrong_reserve)

    wrong_max_fraction = _profile_payload(supported=("50", "max"))
    wrong_max_fraction["memory_tiers"]["max"]["budget_fraction"] = 0.8
    with pytest.raises(OcrProfileError, match="budget_fraction must be 0.9"):
        seal_execution_profile(wrong_max_fraction)

    unified = _profile(memory_model="unified", supported=("50", "max"))
    assert unified["memory_tiers"]["max"]["budget_fraction"] == 0.75
    assert unified["memory_tiers"]["max"]["reserve_bytes"] == 16 * GIB
    assert unified["memory_tiers"]["max"]["budget_bytes"] == 32 * GIB

    wrong_unified_max = _profile_payload(
        memory_model="unified",
        supported=("50", "max"),
    )
    wrong_unified_max["memory_tiers"]["max"]["budget_fraction"] = 0.9
    with pytest.raises(OcrProfileError, match="budget_fraction must be 0.75"):
        seal_execution_profile(wrong_unified_max)

    bad_unsupported = _profile_payload(supported=("50", "max"))
    bad_unsupported["memory_tiers"]["25"]["stages"] = {
        "markdown": _tier_stage("markdown", TOTAL_MEMORY // 4, "25")
    }
    with pytest.raises(OcrProfileError, match="empty when unsupported"):
        seal_execution_profile(bad_unsupported)


def test_qualified_profiles_require_hashed_evidence_for_every_tier():
    from sneeze.tpn.ocr_profiles import (
        OcrProfileError,
        seal_execution_profile,
    )

    provisional = _profile(
        state="provisional",
        supported=("50", "max"),
        include_evidence=False,
    )
    assert provisional["qualification"]["state"] == "provisional"

    missing = _profile_payload(
        state="qualified",
        supported=("50", "max"),
        include_evidence=False,
    )
    with pytest.raises(OcrProfileError, match="lacks passing evidence"):
        seal_execution_profile(missing)

    qualified = _profile(state="qualified", supported=("50", "max"))
    evidence = qualified["qualification"]["evidence"]
    assert {(item["suite_id"], item["memory_tier"]) for item in evidence} == {
        ("cuphoton-regressions", "50"),
        ("cuphoton-regressions", "max"),
    }


def test_qualification_promotion_requires_five_repeats():
    from sneeze.tpn.ocr_profiles import (
        OcrProfileError,
        qualify_execution_profile,
    )

    provisional = _profile(
        state="provisional",
        supported=("50", "max"),
        evidence_repeats=5,
    )
    qualified = qualify_execution_profile(provisional)
    assert qualified["qualification"]["state"] == "qualified"
    assert qualified["profile_sha256"] != provisional["profile_sha256"]
    assert all(
        item["repeats"] == 5
        for item in qualified["qualification"]["evidence"]
    )

    insufficient = _profile(
        state="provisional",
        supported=("50", "max"),
        evidence_repeats=4,
    )
    with pytest.raises(OcrProfileError, match="repeats must be at least 5"):
        qualify_execution_profile(insufficient)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda obs: obs["hardware"]["selector"].__setitem__(
                "product_name", "NVIDIA RTX A6000"
            ),
            "hardware.product_name differs",
        ),
        (
            lambda obs: obs["hardware"]["selector"].__setitem__(
                "total_memory_bytes", 24 * GIB
            ),
            "hardware.total_memory_bytes differs",
        ),
        (
            lambda obs: obs["hardware"]["selector"].__setitem__(
                "compute_capability", [8, 6]
            ),
            "hardware.compute_capability differs",
        ),
        (
            lambda obs: obs["stages"]["markdown"]["runtime"].__setitem__(
                "engine_version", "4.58.0"
            ),
            "stages.markdown.runtime.engine_version differs",
        ),
        (
            lambda obs: obs["stages"]["markdown"]["model"].__setitem__(
                "revision", "f" * 40
            ),
            "stages.markdown.model differs",
        ),
        (
            lambda obs: obs["stages"]["markdown"]["model"].__setitem__(
                "snapshot_files_sha256", _digest("other-model-bytes")
            ),
            "stages.markdown.model differs",
        ),
    ],
)
def test_exact_matching_fails_on_class_memory_model_or_runtime(
    mutate,
    reason,
):
    from sneeze.tpn.ocr_profiles import profile_match_report

    profile = _profile()
    observation = _observation(profile)
    mutate(observation)
    report = profile_match_report(profile, observation)
    assert not report["matched"]
    assert reason in report["reasons"]


def test_exact_matching_ignores_host_uuid_and_observed_runtime_extras():
    from sneeze.tpn.ocr_profiles import (
        hardware_selector_from_observation,
        profile_matches,
    )

    profile = _profile()
    observation = _observation(profile)
    selector = hardware_selector_from_observation(observation["hardware"])
    assert selector == profile["hardware_selector"]
    assert "hostname" not in selector
    assert "uuid" not in selector
    assert profile_matches(
        profile,
        observation,
        trusted_qualified_profile_sha256s=(profile["profile_sha256"],),
    )

    raw_hardware_observation = {
        "hostname": "another-host",
        "device": {
            "uuid": "GPU-FFFFFFFF-BBBB-CCCC-DDDD-EEEEEEEEEEEE",
            "pci_bus_id": "00000000:02:00.0",
            "pci_device_id": "0x2BB0",
            "product_name": "NVIDIA RTX PRO 6000 Blackwell",
            "compute_capability": [12, 0],
        },
        "memory": {
            "model": "dedicated",
            "total_memory_bytes": TOTAL_MEMORY,
        },
    }
    assert (
        hardware_selector_from_observation(raw_hardware_observation)
        == profile["hardware_selector"]
    )


def test_selection_fails_closed_for_provisional_and_unsupported_tiers():
    from sneeze.tpn.ocr_profiles import (
        auto_select_execution_profile,
        profile_match_report,
        profile_matches,
    )

    profile = _profile(
        state="provisional",
        supported=("50", "max"),
        include_evidence=False,
    )
    observation = _observation(profile)
    assert not profile_matches(profile, observation)
    assert profile_matches(
        profile,
        observation,
        memory_tier="max",
        require_qualified=False,
    )
    selected = auto_select_execution_profile(
        [profile],
        observation,
        pipeline_id="mineru25-pipeline",
        require_qualified=False,
    )
    assert selected["profile_id"] == profile["profile_id"]
    report = profile_match_report(
        profile,
        observation,
        memory_tier="25",
        require_qualified=False,
    )
    assert report["reasons"] == ["memory tier 25 is unsupported"]


def test_trial_mode_never_trusts_a_self_declared_qualified_profile():
    from sneeze.tpn.ocr_profiles import profile_match_report, profile_matches

    qualified = _profile()
    observation = _observation(qualified)
    default_report = profile_match_report(qualified, observation)
    assert default_report["reasons"] == [
        "profile qualification is not trusted by this plugin build"
    ]
    assert not profile_matches(qualified, observation)
    report = profile_match_report(
        qualified,
        observation,
        require_qualified=False,
        trusted_qualified_profile_sha256s=(),
    )
    assert report["reasons"] == [
        "profile qualification is not trusted by this plugin build"
    ]
    assert not profile_matches(
        qualified,
        observation,
        require_qualified=False,
        trusted_qualified_profile_sha256s=(),
    )
    assert profile_matches(
        qualified,
        observation,
        require_qualified=False,
        trusted_qualified_profile_sha256s=(qualified["profile_sha256"],),
    )

    provisional = _profile(
        state="provisional",
        include_evidence=False,
    )
    assert profile_matches(
        provisional,
        _observation(provisional),
        require_qualified=False,
        trusted_qualified_profile_sha256s=(),
    )


def test_auto_select_reports_onboarding_and_ambiguity():
    from sneeze.tpn.ocr_profiles import (
        OcrProfileSelectionError,
        auto_select_execution_profile,
        seal_execution_profile,
    )

    profile = _profile()
    observation = _observation(profile)
    wrong_hardware = copy.deepcopy(observation)
    wrong_hardware["hardware"]["selector"]["product_name"] = (
        "NVIDIA RTX A6000"
    )
    with pytest.raises(OcrProfileSelectionError, match="tuning workflow"):
        auto_select_execution_profile(
            [profile],
            wrong_hardware,
            pipeline_id="local-mineru25",
            trusted_qualified_profile_sha256s=(
                profile["profile_sha256"],
            ),
        )

    duplicate_payload = dict(profile)
    duplicate_payload.pop("profile_sha256")
    duplicate_payload["profile_id"] = "rtx-pro-6000-local-mineru25-copy"
    duplicate = seal_execution_profile(duplicate_payload)
    with pytest.raises(OcrProfileSelectionError, match="multiple"):
        auto_select_execution_profile(
            [profile, duplicate],
            observation,
            pipeline_id="local-mineru25",
            trusted_qualified_profile_sha256s=(
                profile["profile_sha256"],
                duplicate["profile_sha256"],
            ),
        )


def test_profile_write_load_list_show_and_no_replace(tmp_path):
    from sneeze.tpn.ocr_profiles import (
        OcrProfileError,
        list_execution_profiles,
        load_execution_profile,
        show_execution_profile,
        write_execution_profile,
    )

    profile = _profile()
    path = tmp_path / "profile.json"
    written = write_execution_profile(profile, path)
    assert written == profile
    assert load_execution_profile(path) == profile
    assert list_execution_profiles(paths=[tmp_path]) == (profile,)
    assert (
        show_execution_profile(profile["profile_id"], paths=[tmp_path])
        == profile
    )

    original = path.read_bytes()
    with pytest.raises(OcrProfileError, match="refusing to replace"):
        write_execution_profile(profile, path)
    assert path.read_bytes() == original

    with pytest.raises(OcrProfileError, match="output must be absolute"):
        write_execution_profile(profile, "relative-profile.json")
    assert list_execution_profiles([profile, profile]) == (profile,)


def test_profile_file_rejects_duplicate_json_fields_at_any_depth(tmp_path):
    from sneeze.tpn.ocr_profiles import (
        OcrProfileError,
        load_execution_profile,
    )

    profile = _profile()
    payload = json.dumps(profile, indent=2, sort_keys=True)
    original = '    "state": "qualified"'
    duplicate = (
        '    "state": "provisional",\n'
        '    "state": "qualified"'
    )
    assert original in payload
    path = tmp_path / "duplicate-nested-field.json"
    path.write_text(payload.replace(original, duplicate, 1), encoding="utf-8")

    with pytest.raises(
        OcrProfileError,
        match="duplicate JSON field in OCR profile: state",
    ):
        load_execution_profile(path)


def test_profile_loader_rejects_unbounded_or_special_files(tmp_path):
    from sneeze.tpn.ocr_profiles import (
        OcrProfileError,
        list_execution_profiles,
        load_execution_profile,
    )

    profile = _profile()
    source = tmp_path / "profile.json"
    source.write_text(json.dumps(profile), encoding="utf-8")

    symlink = tmp_path / "profile-link.json"
    symlink.symlink_to(source)
    with pytest.raises(OcrProfileError, match="securely open|symlink"):
        load_execution_profile(symlink)

    profile_directory_link = tmp_path / "profile-directory-link"
    profile_directory_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(OcrProfileError, match="securely open|symlink"):
        list_execution_profiles(paths=[profile_directory_link])

    fifo = tmp_path / "profile-fifo.json"
    os.mkfifo(fifo)
    with pytest.raises(OcrProfileError, match="regular file"):
        load_execution_profile(fifo)

    oversized = tmp_path / "profile-oversized.json"
    oversized.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(OcrProfileError, match="exceeds 1048576 bytes"):
        load_execution_profile(oversized)

    crowded = tmp_path / "crowded"
    crowded.mkdir()
    for index in range(257):
        (crowded / f"entry-{index:03d}").touch()
    with pytest.raises(OcrProfileError, match="exceeds 256 entries"):
        list_execution_profiles(paths=[crowded])


def test_profile_search_deduplicates_identical_records_across_paths(tmp_path):
    from sneeze.tpn.ocr_profiles import (
        list_execution_profiles,
        show_execution_profile,
    )

    profile = _profile()
    bundled = tmp_path / "bundled"
    machine_local = tmp_path / "machine-local"
    bundled.mkdir()
    machine_local.mkdir()
    (bundled / "profile.json").write_text(
        json.dumps(profile, indent=2),
        encoding="utf-8",
    )
    (machine_local / "same-profile.json").write_text(
        json.dumps(profile, separators=(",", ":")),
        encoding="utf-8",
    )

    assert list_execution_profiles(paths=(bundled, machine_local)) == (
        profile,
    )
    assert show_execution_profile(
        profile["profile_id"],
        paths=(bundled, machine_local),
    ) == profile


def test_profile_search_rejects_same_id_with_conflicting_content(tmp_path):
    from sneeze.tpn.ocr_profiles import (
        OcrProfileError,
        list_execution_profiles,
        seal_execution_profile,
    )

    profile = _profile()
    conflicting_payload = copy.deepcopy(profile)
    conflicting_payload.pop("profile_sha256")
    conflicting_payload["display_name"] = "Conflicting profile"
    conflicting = seal_execution_profile(conflicting_payload)
    assert conflicting["profile_id"] == profile["profile_id"]
    assert conflicting["profile_sha256"] != profile["profile_sha256"]

    bundled = tmp_path / "bundled"
    machine_local = tmp_path / "machine-local"
    bundled.mkdir()
    machine_local.mkdir()
    (bundled / "profile.json").write_text(
        json.dumps(profile),
        encoding="utf-8",
    )
    (machine_local / "conflicting-profile.json").write_text(
        json.dumps(conflicting),
        encoding="utf-8",
    )

    with pytest.raises(
        OcrProfileError,
        match="duplicate OCR profile ID with conflicting content",
    ):
        list_execution_profiles(paths=(bundled, machine_local))


def test_profile_requires_exact_runtime_and_canonical_pipeline_id():
    from sneeze.tpn.ocr_profiles import (
        OcrProfileError,
        seal_execution_profile,
    )

    missing_runtime = _profile_payload()
    del missing_runtime["stages"]["markdown"]["runtime_selector"][
        "engine_version"
    ]
    with pytest.raises(OcrProfileError, match="missing exact runtime fields"):
        seal_execution_profile(missing_runtime)

    alias = _profile_payload()
    alias["pipeline_id"] = "mineru25-pipeline"
    with pytest.raises(OcrProfileError, match="canonical pipeline ID"):
        seal_execution_profile(alias)

    stale_pipeline = _profile_payload()
    stale_pipeline["pipeline_definition_sha256"] = "f" * 64
    with pytest.raises(OcrProfileError, match="built-in pipeline"):
        seal_execution_profile(stale_pipeline)
