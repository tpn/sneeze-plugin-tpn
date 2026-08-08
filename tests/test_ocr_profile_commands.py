from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
from io import StringIO
from pathlib import Path

import pytest

VALID_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
    "x8AAwMCAO+Xk6cAAAAASUVORK5CYII="
)
GIB = 1024**3


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _runtime(backend_id: str) -> dict:
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
                "torch": "2.11.0+cu130",
                "vllm": "0.21.1rc1.dev438+g4ff865c38",
            },
            "attention_backend": "FLASH_ATTN",
            "dtype": "bfloat16",
            "implementation_sha256": DEEPSEEK_IMPLEMENTATION_SHA256,
            "installed_distributions_count": 2,
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


def _qualified_profile() -> dict:
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
                    "snapshot-" + stage["backend_id"]
                ),
            },
            "batch_semantics": stage["batch_semantics"],
            "runtime_selector": _runtime(stage["backend_id"]),
        }
        for stage in pipeline["stages"]
    }
    budget = int(48 * GIB * 0.9)
    grounded_settings = {
        "attention_backend": "FLASH_ATTN",
        "batch_size": 2,
        "enable_chunked_prefill": True,
        "engine_overhead_headroom_bytes": GIB,
        "gpu_memory_utilization": 0.8,
        "gpu_memory_utilization_denominator_bytes": 48 * GIB,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 2,
        "max_tokens": 8192,
        "memory_budget_bytes": budget,
        "memory_tier": "max",
        "mm_encoder_attn_backend": "auto",
        "ngram_size": 10,
        "temperature": 0.0,
        "window_size": 50,
    }
    markdown_settings = {
        "backend": "official-two-step-transformers",
        "batch_size": 4,
        "image_analysis": False,
        "memory_budget_bytes": budget,
        "memory_tier": "max",
    }
    tiers = {}
    for tier, fraction in (("25", 0.25), ("50", 0.5), ("max", 0.9)):
        supported = tier == "max"
        tiers[tier] = {
            "supported": supported,
            "reason": None if supported else "not qualified on this tier",
            "budget_fraction": fraction,
            "budget_bytes": budget if supported else int(48 * GIB * fraction),
            "reserve_bytes": 2 * GIB,
            "stages": (
                {
                    "grounded_ocr": {
                        "settings": grounded_settings,
                        "observed_peak_memory_bytes": 8 * GIB,
                        "confirmed_pages_per_second": 0.05,
                    },
                    "markdown": {
                        "settings": markdown_settings,
                        "observed_peak_memory_bytes": 10 * GIB,
                        "confirmed_pages_per_second": 0.04,
                    },
                }
                if supported
                else {}
            ),
        }
    return seal_execution_profile(
        {
            "schema_version": 1,
            "kind": "ocr_execution_profile",
            "profile_id": "test-structured-local-mineru25",
            "display_name": "Structured command test profile",
            "pipeline_id": pipeline["pipeline_id"],
            "pipeline_definition_sha256": pipeline["definition_sha256"],
            "hardware_selector": {
                "schema_version": 1,
                "vendor": "nvidia",
                "product_name": "Synthetic GPU",
                "pci_device_id": "0x1234",
                "compute_capability": [9, 0],
                "memory_model": "dedicated",
                "total_memory_bytes": 48 * GIB,
            },
            "stages": stages,
            "memory_tiers": tiers,
            "qualification": {
                "state": "qualified",
                "policy_sha256": qualification_policy_sha256(),
                "required_suites": ["hardware-smoke"],
                "evidence": [
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
                ],
            },
        }
    )


def _mineru_installation(tmp_path) -> dict:
    from sneeze.tpn.ocr_implementation import backend_implementation_record

    executable = tmp_path / "mineru-python"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    adapter = tmp_path / "mineru25.py"
    adapter.write_text("# synthetic adapter\n", encoding="utf-8")
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    return {
        "backend_id": "mineru25-pro-2605",
        "python_executable": {"path": str(executable.resolve())},
        "hf_cache": {"path": str(cache.resolve())},
        "adapter": {"path": str(adapter.resolve())},
        "implementation": backend_implementation_record("mineru25-pro-2605"),
        "cuda": {
            "visible_devices": "GPU-SYNTHETIC",
            "device": "cuda:0",
            "attention_backend": "sdpa",
        },
    }


def test_structured_options_are_long_only_and_preserve_legacy_short_cli(
    tmp_path,
    monkeypatch,
):
    from sneeze import cli as sneeze_cli
    from sneeze.tpn.commands import OcrPdfPages

    captured = {}

    def fake_run(self):
        captured.update(
            pipeline=self.pipeline,
            execution_profile=self.execution_profile,
            memory_tier=self.memory_tier,
            explain_profile=self.explain_profile,
            prompt=self.prompt,
            prompt_mode=self.prompt_mode,
            offset=self._offset,
            scan_parallelism=self._scan_parallelism,
            attention_backend=self.attention_backend,
        )

    monkeypatch.setattr(OcrPdfPages, "run", fake_run)
    cli = sneeze_cli.CLI(
        program_name="sne",
        module_names=["sneeze", "sneeze.tpn"],
        introspect=True,
        auto_plugins=False,
    )
    commandline = cli._commands_by_name["ocr-pdf-pages"]
    for name in (
        "pipeline",
        "execution_profile",
        "memory_tier",
        "explain_profile",
    ):
        assert commandline.command._invariants[name]._opt_short is None

    commandline.run(
        [
            "--source-dir",
            str(tmp_path),
            "--pipeline",
            "local-mineru25",
            "--execution-profile",
            "auto",
            "--memory-tier",
            "50",
            "--explain-profile",
            "-p",
            "legacy prompt",
            "-r",
            "full",
            "-f",
            "2",
            "-a",
            "3",
            "-i",
            "FLASH_ATTN",
        ]
    )

    assert captured == {
        "pipeline": "local-mineru25",
        "execution_profile": "auto",
        "memory_tier": "50",
        "explain_profile": True,
        "prompt": "legacy prompt",
        "prompt_mode": "full",
        "offset": 2,
        "scan_parallelism": 3,
        "attention_backend": "FLASH_ATTN",
    }


def test_legacy_execution_bypasses_profile_infrastructure(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_hardware as ocr_hardware
    import sneeze.tpn.ocr_installations as ocr_installations
    import sneeze.tpn.ocr_vllm as ocr_vllm
    from sneeze.tpn.commands import OcrPdfPages

    source = tmp_path / "pages"
    source.mkdir()
    (source / "page-1.png").write_bytes(VALID_PNG_BYTES)
    captured = {}

    def unexpected(*_args, **_kwargs):
        raise AssertionError(
            "legacy OCR must not enter profile infrastructure"
        )

    class FakeProcessor:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def run(self, tasks):
            captured["tasks"] = list(tasks)

    monkeypatch.setattr(ocr_hardware, "probe_nvidia_hardware", unexpected)
    monkeypatch.setattr(
        ocr_installations,
        "load_installation_registry",
        unexpected,
    )
    monkeypatch.setattr(
        ocr_vllm,
        "DeepSeekOcrOfflineProcessor",
        FakeProcessor,
    )
    monkeypatch.setattr(
        OcrPdfPages,
        "_prepare_profile_markdown_provider",
        unexpected,
    )

    command = OcrPdfPages(None, StringIO(), None)
    command._source_dir = str(source)
    command.prompt_mode = "markdown"
    command.batch_size = 3
    command.attention_backend = "FLASH_ATTN"
    command.gpu_memory_utilization = "0.7"
    command.temperature = "0.2"
    command._scan_parallelism = 1
    command.run()

    assert captured["kwargs"]["batch_size"] == 3
    assert captured["kwargs"]["attention_backend"] == "FLASH_ATTN"
    assert captured["kwargs"]["gpu_memory_utilization"] == 0.7
    assert captured["kwargs"]["temperature"] == 0.2
    assert captured["kwargs"]["window_size"] == 128
    assert captured["tasks"][0].output_path == str(source / "page-1.md")


@pytest.mark.parametrize(
    ("name", "value", "option"),
    [
        ("model", "example/custom-model", "--model"),
        ("batch_size", 2, "--batch-size"),
        ("attention_backend", "FLASH_ATTN", "--attention-backend"),
        ("temperature", "0.1", "--temperature"),
        (
            "markdown_provider_spec",
            "/tmp/manual.json",
            "--markdown-provider-spec",
        ),
        ("keep_grounding", True, "--keep-grounding"),
    ],
)
def test_structured_execution_rejects_manual_model_runtime_and_provider_knobs(
    name,
    value,
    option,
):
    from sneeze.command import CommandError

    from sneeze.tpn.commands import OcrPdfPages

    command = OcrPdfPages(None, StringIO(), None)
    command.pipeline = "local-mineru25"
    setattr(command, name, value)

    with pytest.raises(CommandError, match=re.escape(option)):
        command._resolve_structured_execution(
            prompt_mode="markdown",
            output_format="md",
        )


@pytest.mark.parametrize(
    "name",
    [
        "VLLM_ATTENTION_BACKEND",
        "VLLM_DISABLE_FLASHINFER_PREFILL",
        "VLLM_BATCH_INVARIANT",
        "VLLM_FUTURE_UNMODELED_CONTROL",
    ],
)
def test_structured_preflight_rejects_ambient_vllm_controls_before_runtime(
    monkeypatch,
    name,
):
    from sneeze.command import CommandError

    from sneeze.tpn import ocr_installations
    from sneeze.tpn.commands import OcrPdfPages

    secret_value = "must-not-appear-in-errors"
    monkeypatch.setenv(name, secret_value)

    def unexpected(*_args, **_kwargs):
        raise AssertionError(
            "ambient controls must fail before registry load"
        )

    monkeypatch.setattr(
        ocr_installations,
        "load_installation_registry",
        unexpected,
    )
    command = OcrPdfPages(None, StringIO(), None)
    command.pipeline = "local-mineru25"

    with pytest.raises(CommandError) as raised:
        command._resolve_structured_execution(
            prompt_mode="markdown",
            output_format="md",
        )

    assert name in str(raised.value)
    assert secret_value not in str(raised.value)


def test_manual_mode_ignores_ambient_vllm_controls(monkeypatch):
    from sneeze.tpn.commands import OcrPdfPages

    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    command = OcrPdfPages(None, StringIO(), None)

    assert (
        command._resolve_structured_execution(
            prompt_mode="markdown",
            output_format="md",
        )
        is None
    )


@pytest.mark.parametrize(
    ("backend_id", "cuda", "message"),
    [
        (
            "deepseek-vllm",
            {"visible_devices": "0,1", "device": "cuda:0"},
            "must select exactly one GPU",
        ),
        (
            "mineru25-pro-2605",
            {"visible_devices": "0", "device": "cuda:1"},
            "must be cuda:0 for single-GPU OCR pipelines",
        ),
    ],
)
def test_structured_preflight_rejects_unsafe_stored_cuda_routing(
    monkeypatch,
    backend_id,
    cuda,
    message,
):
    from sneeze.command import CommandError

    from sneeze.tpn import ocr_execution, ocr_installations
    from sneeze.tpn.commands import OcrPdfPages

    installations = {
        "deepseek-vllm": {
            "cuda": {"visible_devices": "0", "device": "cuda:0"}
        },
        "mineru25-pro-2605": {
            "cuda": {"visible_devices": "0", "device": "cuda:0"}
        },
    }
    installations[backend_id]["cuda"] = cuda

    monkeypatch.setattr(
        ocr_installations,
        "load_installation_registry",
        lambda *_args, **_kwargs: {"installations": installations},
    )

    def unexpected(*_args, **_kwargs):
        raise AssertionError("unsafe CUDA routing must fail before runtime")

    monkeypatch.setattr(
        ocr_execution,
        "assert_in_process_deepseek_installation",
        unexpected,
    )
    command = OcrPdfPages(None, StringIO(), None)
    command.pipeline = "local-mineru25"

    with pytest.raises(CommandError, match=message):
        command._resolve_structured_execution(
            prompt_mode="markdown",
            output_format="md",
        )


def test_explain_stops_after_resolution_before_render_or_output_writes(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.commands as commands
    import sneeze.tpn.ocr_vllm as ocr_vllm
    from sneeze.tpn.commands import OcrPdfPages

    source = tmp_path / "pdfs"
    source.mkdir()
    destination = tmp_path / "must-not-exist"
    receipt = {
        "schema_version": 1,
        "kind": "ocr_structured_execution",
        "profile_id": "synthetic",
        "receipt_sha256": "a" * 64,
    }
    events = []

    def resolve(_self, *, prompt_mode, output_format):
        events.append(("resolve", prompt_mode, output_format))
        return {"receipt": receipt}

    def unexpected(*_args, **_kwargs):
        raise AssertionError(
            "explain must stop before rendering or inference"
        )

    monkeypatch.setattr(OcrPdfPages, "_resolve_structured_execution", resolve)
    monkeypatch.setattr(
        OcrPdfPages,
        "_prepare_profile_markdown_provider",
        unexpected,
    )
    monkeypatch.setattr(commands, "_run_pdf_to_pages", unexpected)
    monkeypatch.setattr(
        ocr_vllm,
        "DeepSeekOcrOfflineProcessor",
        unexpected,
    )

    out = StringIO()
    command = OcrPdfPages(None, out, None)
    command._source_dir = str(source)
    command.dest_dir = str(destination)
    command.pdf_to_pages = True
    command.explain_profile = True
    command.run()

    assert events == [("resolve", "markdown", "md")]
    assert json.loads(out.getvalue()) == receipt
    assert not destination.exists()


def test_structured_command_passes_receipt_to_full_collection(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_vllm as ocr_vllm
    from sneeze.tpn.commands import OcrPdfPages
    from sneeze.tpn.ocr_backend_setup import make_receipt

    source = tmp_path / "pages"
    source.mkdir()
    receipt = make_receipt(
        "ocr_structured_execution",
        {
            "profile_id": "synthetic",
            "profile_sha256": _digest("synthetic-profile"),
            "qualification_state": "qualified",
            "memory_tier": "max",
        },
    )
    structured = {"receipt": receipt}
    captured = {}

    monkeypatch.setattr(
        OcrPdfPages,
        "_resolve_structured_execution",
        lambda _self, **_kwargs: structured,
    )
    monkeypatch.setattr(
        OcrPdfPages,
        "_prepare_profile_markdown_provider",
        lambda _self, *_args, **_kwargs: {},
    )

    def collect(**kwargs):
        captured.update(kwargs)
        return [], 1

    monkeypatch.setattr(ocr_vllm, "collect_ocr_full_items", collect)

    command = OcrPdfPages(None, StringIO(), None)
    command._source_dir = str(source)
    command._scan_parallelism = 1
    command.run()

    assert captured["expected_execution_receipt"] is receipt


def test_structured_command_reports_unproven_reuse_as_command_error(
    tmp_path,
    monkeypatch,
):
    from sneeze.command import CommandError

    import sneeze.tpn.ocr_vllm as ocr_vllm
    from sneeze.tpn.commands import OcrPdfPages

    source = tmp_path / "pages"
    source.mkdir()
    structured = {"receipt": {"receipt_sha256": "b" * 64}}

    monkeypatch.setattr(
        OcrPdfPages,
        "_resolve_structured_execution",
        lambda _self, **_kwargs: structured,
    )
    monkeypatch.setattr(
        OcrPdfPages,
        "_prepare_profile_markdown_provider",
        lambda _self, *_args, **_kwargs: {},
    )

    def reject(**_kwargs):
        raise ocr_vllm.StructuredOutputReuseError(
            "Refusing to reuse structured OCR bundle page.json stage markdown"
        )

    monkeypatch.setattr(ocr_vllm, "collect_ocr_full_items", reject)

    command = OcrPdfPages(None, StringIO(), None)
    command._source_dir = str(source)
    command._scan_parallelism = 1

    with pytest.raises(CommandError, match="stage markdown"):
        command.run()


def test_structured_pdf_render_defaults_all_outputs_to_source(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.commands as commands
    import sneeze.tpn.ocr_vllm as ocr_vllm
    from sneeze.tpn.commands import OcrPdfPages

    source = tmp_path / "pdfs"
    source.mkdir()
    (source / "paper.pdf").write_bytes(b"synthetic PDF")
    captured = {}
    structured = {
        "receipt": {"receipt_sha256": "b" * 64},
        "tier": {
            "stages": {
                "grounded_ocr": {"settings": {}},
                "markdown": {"settings": {}},
            }
        },
        "installations": {
            "deepseek-vllm": {
                "hf_cache": {"path": str(tmp_path / "deepseek-cache")},
                "snapshot": {"path": str(tmp_path / "deepseek-snapshot")},
            }
        },
    }
    provider = {"provider_id": "synthetic-mineru"}

    def resolve(_self, **_kwargs):
        return structured

    def prepare_provider(_self, execution, *, source_dir, dest_dir):
        captured["provider_execution"] = execution
        captured["provider_source"] = source_dir
        captured["provider_dest"] = dest_dir
        return provider

    def render(**kwargs):
        captured["render"] = kwargs
        page_dir = source / "paper"
        page_dir.mkdir()
        image = page_dir / "page-01.png"
        image.write_bytes(VALID_PNG_BYTES)
        (page_dir / "page-01.txt").write_text(
            "extracted text",
            encoding="utf-8",
        )
        image_path = str(image)
        return commands.PdfPagesRunResult(
            image_paths=(image_path,),
            rendered_image_paths=frozenset({image_path}),
            rendered_text_image_paths=frozenset({image_path}),
        )

    class FakeProcessor:
        def __init__(self, **kwargs):
            captured["processor_kwargs"] = kwargs

    def run_full(_self, processor, items, **kwargs):
        captured["processor"] = processor
        captured["items"] = list(items)
        captured["full_kwargs"] = kwargs

    monkeypatch.setattr(OcrPdfPages, "_resolve_structured_execution", resolve)
    monkeypatch.setattr(
        OcrPdfPages,
        "_prepare_profile_markdown_provider",
        prepare_provider,
    )
    monkeypatch.setattr(OcrPdfPages, "_run_full_mode", run_full)
    monkeypatch.setattr(commands, "_run_pdf_to_pages", render)
    monkeypatch.setattr(
        ocr_vllm,
        "DeepSeekOcrOfflineProcessor",
        FakeProcessor,
    )

    command = OcrPdfPages(None, StringIO(), None)
    command._source_dir = str(source)
    command.pipeline = "local-mineru25"
    command.pdf_to_pages = True
    command._scan_parallelism = 1
    command.run()

    assert captured["render"]["source_dir"] == str(source)
    assert captured["render"]["dest_dir"] == str(source)
    assert captured["provider_execution"] is structured
    assert captured["provider_source"] == str(source)
    assert captured["provider_dest"] == str(source)
    assert captured["full_kwargs"] == {
        "markdown_provider": provider,
        "execution_plan": structured,
    }
    assert len(captured["items"]) == 1
    item = captured["items"][0]
    assert item.image_path == str(source / "paper/page-01.png")
    assert item.text_path == str(source / "paper/page-01.txt")
    assert item.ocr_path == str(source / "paper/page-01.ocr")
    assert item.markdown_path == str(source / "paper/page-01.md")
    assert item.json_path == str(source / "paper/page-01.json")


def test_selected_profile_drives_exact_deepseek_processor_arguments(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_vllm as ocr_vllm
    from sneeze.tpn.commands import OcrPdfPages
    from sneeze.tpn.ocr_implementation import backend_implementation_record

    source = tmp_path / "pages"
    destination = tmp_path / "ocr"
    source.mkdir()
    (source / "page-1.png").write_bytes(VALID_PNG_BYTES)
    (source / "page-1.txt").write_text("raw text", encoding="utf-8")
    settings = {
        "attention_backend": "FLASH_ATTN",
        "batch_size": 3,
        "enable_chunked_prefill": False,
        "engine_overhead_headroom_bytes": GIB,
        "gpu_memory_utilization": 0.4,
        "gpu_memory_utilization_denominator_bytes": 48 * GIB,
        "max_num_batched_tokens": 12288,
        "max_num_seqs": 3,
        "max_tokens": 8192,
        "memory_budget_bytes": 20 * GIB,
        "memory_tier": "50",
        "mm_encoder_attn_backend": "auto",
        "ngram_size": 10,
        "temperature": 0.0,
        "window_size": 50,
    }
    cache = tmp_path / "deepseek-cache"
    snapshot = (
        cache
        / "models--deepseek-ai--DeepSeek-OCR"
        / "snapshots"
        / ("9f30c71f441d010e5429c532364a86705536c53a")
    )
    snapshot.mkdir(parents=True)
    deepseek_installation = {
        "backend_id": "deepseek-vllm",
        "implementation": backend_implementation_record("deepseek-vllm"),
        "hf_cache": {"path": str(cache.resolve())},
        "snapshot": {
            "path": str(snapshot.resolve()),
            "files_sha256": "d" * 64,
        },
        "model": {
            "requested": "deepseek-ai/DeepSeek-OCR",
            "revision": "9f30c71f441d010e5429c532364a86705536c53a",
        },
    }
    structured = {
        "receipt": {"receipt_sha256": "b" * 64},
        "installations": {"deepseek-vllm": deepseek_installation},
        "tier": {
            "stages": {
                "grounded_ocr": {"settings": settings},
                "markdown": {"settings": {"batch_size": 4}},
            }
        },
    }
    provider = {"provider_id": "synthetic"}
    captured = {}

    class FakeProcessor:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    def resolve(_self, **_kwargs):
        return structured

    def prepare_provider(_self, execution, **_kwargs):
        assert execution is structured
        return provider

    def run_full(_self, processor, items, **kwargs):
        captured["processor"] = processor
        captured["items"] = list(items)
        captured["full_kwargs"] = kwargs
        captured["environment"] = {
            name: os.environ.get(name)
            for name in (
                "HF_HUB_CACHE",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_HUB_DISABLE_TELEMETRY",
            )
        }

    monkeypatch.setattr(OcrPdfPages, "_resolve_structured_execution", resolve)
    monkeypatch.setattr(
        OcrPdfPages,
        "_prepare_profile_markdown_provider",
        prepare_provider,
    )
    monkeypatch.setattr(OcrPdfPages, "_run_full_mode", run_full)
    monkeypatch.setattr(
        ocr_vllm,
        "DeepSeekOcrOfflineProcessor",
        FakeProcessor,
    )
    monkeypatch.setenv("HF_HUB_CACHE", "original-cache")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)

    command = OcrPdfPages(None, StringIO(), None)
    command._source_dir = str(source)
    command.dest_dir = str(destination)
    command._scan_parallelism = 1
    command.run()

    kwargs = captured["kwargs"]
    assert kwargs["model"] == str(snapshot.resolve())
    assert kwargs["model_revision"] is None
    assert kwargs["batch_size"] == 3
    assert kwargs["enable_chunked_prefill"] is False
    assert kwargs["max_num_batched_tokens"] == 12288
    assert kwargs["max_num_seqs"] == 3
    assert kwargs["attention_backend"] == "FLASH_ATTN"
    assert kwargs["mm_encoder_attn_backend"] is None
    assert kwargs["gpu_memory_utilization"] == 0.4
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 8192
    assert kwargs["ngram_size"] == 10
    assert kwargs["window_size"] == 50
    assert kwargs["output_format"] == "md"
    assert kwargs["prompt_mode"] == "plain_ocr"
    assert captured["full_kwargs"] == {
        "markdown_provider": provider,
        "execution_plan": structured,
    }
    assert len(captured["items"]) == 1
    assert captured["environment"] == {
        "HF_HUB_CACHE": str(cache.resolve()),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    assert os.environ["HF_HUB_CACHE"] == "original-cache"
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert "HF_HUB_DISABLE_TELEMETRY" not in os.environ


def test_full_mode_rejects_loaded_engine_drift_before_generation(tmp_path):
    from types import SimpleNamespace

    from sneeze.tpn.commands import OcrPdfPages
    from sneeze.tpn.ocr_implementation import backend_implementation_record

    profile = _qualified_profile()
    stage = profile["stages"]["grounded_ocr"]
    runtime = stage["runtime_selector"]
    snapshot = tmp_path / "deepseek-snapshot"
    snapshot.mkdir()
    installation = {
        "backend_id": "deepseek-vllm",
        "implementation": backend_implementation_record("deepseek-vllm"),
        "model": {
            "id": stage["model"]["requested"],
            "revision": stage["model"]["revision"],
        },
        "snapshot": {
            "path": str(snapshot),
            "files_sha256": stage["model"]["snapshot_files_sha256"],
        },
        "cuda": {"attention_backend": "FLASH_ATTN"},
    }
    engine = object()

    class FakeProcessor:
        generation_metadata = {}

        def __init__(self):
            self.run_calls = 0
            self.shutdown_calls = []
            self._engine_metadata = {
                "model": {
                    "requested": str(snapshot),
                    "revision": None,
                    "resolved_path": str(snapshot),
                    "snapshot": stage["model"]["revision"],
                },
                "runtime": {
                    "python": runtime["python"],
                    "torch": runtime["torch"],
                    "torch_cuda": runtime["torch_cuda"],
                    "vllm": runtime["engine_version"],
                    "dtype": runtime["dtype"],
                    "attention_backend_selected": "TORCH_SDPA",
                },
            }

        def init_progressbar(self, **_kwargs):
            return None

        def prepare_engine(self):
            return engine, object()

        def run(self, *_args, **_kwargs):
            self.run_calls += 1

        def close_progressbar(self):
            return None

        def shutdown_engine(self, value):
            self.shutdown_calls.append(value)

    processor = FakeProcessor()
    output = tmp_path / "page.ocr"
    item = SimpleNamespace(
        image_path=str(tmp_path / "page.png"),
        ocr_path=str(output),
        text_path=str(tmp_path / "page.txt"),
        markdown_path=str(tmp_path / "page.md"),
        need_ocr=True,
        need_markdown=False,
    )
    command = OcrPdfPages(None, StringIO(), None)

    with pytest.raises(RuntimeError, match="attention_backend_selected"):
        command._run_full_mode(
            processor,
            [item],
            execution_plan={
                "profile": profile,
                "installations": {"deepseek-vllm": installation},
            },
        )

    assert processor.run_calls == 0
    assert processor.shutdown_calls == [engine]
    assert not output.exists()


@pytest.mark.parametrize(
    ("termination", "expected_flag"),
    (
        (
            {"eos": False, "max_tokens": False},
            "grounded_ocr_non_eos_termination",
        ),
        (
            {"eos": True, "max_tokens": True},
            "grounded_ocr_max_tokens_termination",
        ),
    ),
)
def test_structured_full_mode_rejects_unqualified_grounded_termination(
    tmp_path,
    termination,
    expected_flag,
):
    from sneeze.tpn.commands import OcrPdfPages
    from sneeze.tpn.ocr_implementation import backend_implementation_record
    from sneeze.tpn.ocr_vllm import OcrFullItem

    profile = _qualified_profile()
    stage = profile["stages"]["grounded_ocr"]
    runtime = stage["runtime_selector"]
    snapshot = tmp_path / "deepseek-snapshot"
    snapshot.mkdir()
    installation = {
        "backend_id": "deepseek-vllm",
        "implementation": backend_implementation_record("deepseek-vllm"),
        "model": {
            "id": stage["model"]["requested"],
            "revision": stage["model"]["revision"],
        },
        "snapshot": {
            "path": str(snapshot),
            "files_sha256": stage["model"]["snapshot_files_sha256"],
        },
        "cuda": {"attention_backend": "FLASH_ATTN"},
    }
    engine = object()

    class FakeProcessor:
        def __init__(self):
            self.generation_metadata = {}
            self._engine_metadata = {
                "model": {
                    "requested": str(snapshot),
                    "revision": None,
                    "resolved_path": str(snapshot),
                    "snapshot": stage["model"]["revision"],
                },
                "runtime": {
                    "python": runtime["python"],
                    "vllm": runtime["engine_version"],
                    "torch": runtime["torch"],
                    "torch_cuda": runtime["torch_cuda"],
                    "installed_distributions_count": runtime[
                        "installed_distributions_count"
                    ],
                    "installed_distributions_sha256": runtime[
                        "installed_distributions_sha256"
                    ],
                    "dtype": runtime["dtype"],
                    "attention_backend_selected": runtime[
                        "attention_backend"
                    ],
                },
            }

        def init_progressbar(self, **_kwargs):
            return None

        def prepare_engine(self):
            return engine, object()

        def run(self, tasks, **_kwargs):
            for task in tasks:
                with open(task.output_path, "w", encoding="utf-8") as handle:
                    handle.write("complete grounded page prose\n")
                self.generation_metadata[task.output_path] = {
                    "status": "completed",
                    "termination": termination,
                }

        def update_progressbar(self, _amount):
            return None

        def close_progressbar(self):
            return None

        def shutdown_engine(self, _engine):
            return None

    image = tmp_path / "page.png"
    image.write_bytes(VALID_PNG_BYTES)
    raw = tmp_path / "page.txt"
    raw.write_text("complete grounded page prose\n", encoding="utf-8")
    output = tmp_path / "page.ocr"
    previous = b"previous validated grounded OCR\n"
    output.write_bytes(previous)
    previous_sha256 = hashlib.sha256(previous).hexdigest()
    item = OcrFullItem(
        image_path=str(image),
        base="paper/page",
        ocr_path=str(output),
        markdown_path=str(tmp_path / "page.md"),
        json_path=str(tmp_path / "page.json"),
        text_path=str(raw),
        need_ocr=True,
        need_markdown=False,
        image_dims={"w": 1, "h": 1},
        image_sha256=hashlib.sha256(VALID_PNG_BYTES).hexdigest(),
    )

    with pytest.raises(RuntimeError, match=expected_flag):
        OcrPdfPages(None, StringIO(), None)._run_full_mode(
            FakeProcessor(),
            [item],
            execution_plan={
                "profile": profile,
                "receipt": {"receipt_sha256": "b" * 64},
                "installations": {"deepseek-vllm": installation},
            },
        )

    assert output.read_bytes() == previous
    assert hashlib.sha256(output.read_bytes()).hexdigest() == previous_sha256
    payload = json.loads((tmp_path / "page.json").read_text())
    metadata = payload["generation_metadata"]["ocr"]
    assert expected_flag in metadata["validation"]["flags"]
    promotion = metadata["promotion"]
    assert promotion["promoted"] is False
    assert promotion["persisted_output_sha256_before"] == previous_sha256
    assert promotion["persisted_output_sha256"] == previous_sha256
    candidate = Path(promotion["candidate_path"])
    assert candidate.is_file()
    assert candidate.read_text() == "complete grounded page prose\n"


def test_structured_grounded_candidate_promotes_only_after_valid_eos(
    tmp_path,
):
    from types import SimpleNamespace

    from sneeze.tpn.commands import (
        _finalize_grounded_ocr_candidates,
        _stage_grounded_ocr_tasks,
    )
    from sneeze.tpn.ocr_vllm import OcrPageTask

    output = tmp_path / "page.ocr"
    output.write_text("previous grounded OCR\n", encoding="utf-8")
    tasks, candidates = _stage_grounded_ocr_tasks(
        [
            OcrPageTask(
                image_path=str(tmp_path / "page.png"),
                output_path=str(output),
                text_source_path=str(tmp_path / "page.txt"),
            )
        ]
    )
    candidate = Path(tasks[0].output_path)
    candidate.write_text("new validated grounded OCR\n", encoding="utf-8")
    processor = SimpleNamespace(
        generation_metadata={
            str(candidate): {
                "status": "completed",
                "termination": {"eos": True, "max_tokens": False},
            }
        }
    )

    _finalize_grounded_ocr_candidates(processor, candidates)

    assert output.read_text() == "new validated grounded OCR\n"
    assert not candidate.exists()
    metadata = processor.generation_metadata[str(output)]
    assert metadata["validation"] == {"status": "pass", "flags": []}
    assert metadata["promotion"]["promoted"] is True
    assert metadata["promotion"]["candidate_path"] is None
    assert (
        metadata["promotion"]["candidate_sha256"]
        == metadata["promotion"]["persisted_output_sha256"]
    )


def test_selected_profile_builds_exact_mineru_provider_arguments(
    tmp_path,
):
    from sneeze.tpn.commands import OcrPdfPages
    from sneeze.tpn.ocr_backends import get_backend
    from sneeze.tpn.ocr_profiles import load_execution_profile

    profile = _qualified_profile()
    installation = _mineru_installation(tmp_path)
    evidence = (tmp_path / "scratch" / "provider-evidence").resolve()
    command = OcrPdfPages(None, StringIO(), None)
    command.markdown_provider_run_dir = str(evidence)
    execution = {
        "profile": profile,
        "memory_tier": "max",
        "installations": {"mineru25-pro-2605": installation},
    }

    provider = command._prepare_profile_markdown_provider(
        execution,
        source_dir=str(tmp_path / "source"),
        dest_dir=str(tmp_path / "dest"),
    )

    backend = get_backend("mineru25-pro-2605")
    profile_path = provider["runtime_root"] + "/execution-profile.json"
    assert provider["lane"] == "best"
    assert provider["run_dir"] == str(evidence)
    assert provider["execution_profile"] is execution
    assert provider["spec"]["argv"] == [
        installation["python_executable"]["path"],
        "-I",
        "-B",
        installation["adapter"]["path"],
        "--model-id",
        backend.model_id,
        "--revision",
        backend.model_revision,
        "--device",
        "cuda:0",
        "--attention-backend",
        "sdpa",
        "--execution-profile",
        profile_path,
        "--profile-sha256",
        profile["profile_sha256"],
        "--memory-tier",
        "max",
    ]
    assert (
        provider["spec"]["lanes"]["best"]["settings"]
        == (profile["memory_tiers"]["max"]["stages"]["markdown"]["settings"])
    )
    assert provider["spec"]["env"] == {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "GPU-SYNTHETIC",
        "HF_HUB_CACHE": installation["hf_cache"]["path"],
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    assert provider["spec"]["secret_env_names"] == []
    assert load_execution_profile(profile_path) == profile


def test_structured_provider_timeout_scales_from_confirmed_throughput():
    from sneeze.tpn.commands import _structured_markdown_timeout_seconds

    profile = _qualified_profile()
    provider = {
        "spec": {"timeout_seconds": 3600},
        "execution_profile": {
            "profile": profile,
            "memory_tier": "max",
        },
    }
    page_count = 222
    throughput = profile["memory_tiers"]["max"]["stages"]["markdown"][
        "confirmed_pages_per_second"
    ]

    timeout = _structured_markdown_timeout_seconds(provider, page_count)

    assert timeout > 3600
    assert timeout >= 600 + 3 * page_count / throughput
    assert (
        _structured_markdown_timeout_seconds(
            {"spec": {"timeout_seconds": 30}},
            page_count,
        )
        is None
    )


def test_profile_provider_preflight_has_no_filesystem_side_effects(
    tmp_path,
):
    from sneeze.command import CommandError

    from sneeze.tpn.commands import OcrPdfPages

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    for path in (source, destination):
        path.mkdir()
    evidence = source / "must-not-be-created" / "evidence"
    before = sorted(
        path.relative_to(tmp_path) for path in tmp_path.rglob("*")
    )

    command = OcrPdfPages(None, StringIO(), None)
    command.markdown_provider_run_dir = str(evidence)
    with pytest.raises(CommandError, match="OCR source"):
        command._prepare_profile_markdown_provider(
            {},
            source_dir=str(source),
            dest_dir=str(destination),
        )

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before


@pytest.mark.parametrize(
    (
        "model_drift",
        "termination",
        "protocol_errors",
        "expected_flags",
    ),
    (
        (
            True,
            {"eos": True, "max_tokens": False},
            [],
            [
                "provider_model_mismatch: "
                "stages.markdown.model.snapshot_files_sha256 differs"
            ],
        ),
        (
            False,
            {"eos": False, "max_tokens": False},
            [],
            ["provider_non_eos_termination"],
        ),
        (
            False,
            {"eos": True, "max_tokens": True},
            [],
            ["provider_max_tokens_termination"],
        ),
        (
            False,
            {"eos": True, "max_tokens": False},
            ["synthetic nonzero provider exit"],
            [
                "provider_batch_protocol_error: "
                "synthetic nonzero provider exit"
            ],
        ),
    ),
)
def test_profile_provider_rejects_unqualified_completion(
    tmp_path,
    monkeypatch,
    model_drift,
    termination,
    protocol_errors,
    expected_flags,
):
    import sneeze.tpn.ocr_provider as ocr_provider
    from sneeze.tpn.commands import OcrPdfPages
    from sneeze.tpn.ocr_vllm import OcrFullItem

    profile = _qualified_profile()
    expected_model = copy.deepcopy(profile["stages"]["markdown"]["model"])
    expected_runtime = copy.deepcopy(
        profile["stages"]["markdown"]["runtime_selector"]
    )
    receipt = {
        "profile_id": profile["profile_id"],
        "profile_sha256": profile["profile_sha256"],
        "qualification_state": "qualified",
        "memory_tier": "max",
    }
    expected_runtime["execution_profile"] = receipt

    image = tmp_path / "page.png"
    image.write_bytes(VALID_PNG_BYTES)
    evidence_text = "real page prose\n"
    raw = tmp_path / "page.txt"
    raw.write_text(evidence_text, encoding="utf-8")
    grounded = tmp_path / "page.ocr"
    grounded.write_text(evidence_text, encoding="utf-8")
    markdown = tmp_path / "page.md"
    markdown.write_text("previous validated markdown\n", encoding="utf-8")
    item = OcrFullItem(
        image_path=str(image),
        base="paper/page",
        ocr_path=str(grounded),
        markdown_path=str(markdown),
        json_path=str(tmp_path / "page.json"),
        text_path=str(raw),
        need_markdown=True,
        image_dims={"w": 1, "h": 1},
        image_sha256=hashlib.sha256(VALID_PNG_BYTES).hexdigest(),
    )

    def fake_provider_batch(**kwargs):
        loaded_model = {
            **expected_model,
            "resolved_revision": expected_model["revision"],
            "snapshot_files_sha256": (
                _digest("different-loaded-model")
                if model_drift
                else expected_model["snapshot_files_sha256"]
            ),
        }
        return {
            "ok": True,
            "protocol_errors": protocol_errors,
            "results": [
                {
                    "page_id": kwargs["pages"][0]["page_id"],
                    "valid": True,
                    "errors": [],
                    "output_text": evidence_text,
                    "normalized_metadata": {
                        "status": "completed",
                        "runtime": expected_runtime,
                        "model": loaded_model,
                        "termination": termination,
                    },
                }
            ],
        }

    monkeypatch.setattr(
        ocr_provider,
        "run_local_provider_batch",
        fake_provider_batch,
    )

    class FakeProcessor:
        def __init__(self):
            self.generation_metadata = {}

        def update_progressbar(self, _amount):
            return None

    processor = FakeProcessor()
    provider = {
        "spec_path": str(tmp_path / "provider.json"),
        "spec": {
            "lanes": {"best": {"prompt": "native", "settings": {}}},
            "timeout_seconds": 3600,
        },
        "lane": "best",
        "run_dir": str(tmp_path / "evidence"),
        "execution_profile": {"profile": profile, "receipt": receipt},
    }
    failures = OcrPdfPages(None, StringIO(), None)._run_markdown_provider(
        processor,
        [item],
        provider,
    )

    expected_failures = [(str(markdown), expected_flags)]
    if protocol_errors:
        expected_failures.append((provider["run_dir"], expected_flags))
    assert failures == expected_failures
    assert markdown.read_text(encoding="utf-8") == (
        "previous validated markdown\n"
    )
    metadata = processor.generation_metadata[str(markdown)]
    assert metadata["validation"]["flags"] == expected_flags
    assert metadata["promotion"]["promoted"] is False
