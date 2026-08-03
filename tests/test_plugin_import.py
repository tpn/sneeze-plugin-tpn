import base64
import hashlib
import json
import os
import sys
from io import StringIO
from types import SimpleNamespace

import pytest

VALID_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+Xk6cAAAAASUVORK5CYII="
)


def test_plugin_package_imports():
    import importlib

    assert importlib.import_module("sneeze.tpn")


def test_pdf_page_tasks_split_existing_outputs(tmp_path, monkeypatch):
    from sneeze.tpn import pdf_pages

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    dest_dir = tmp_path / "pages"
    dest_dir.mkdir()
    pdf_path = source_dir / "alpha.pdf"
    pdf_path.write_text("", encoding="utf-8")

    tasks = pdf_pages.build_pdf_pages_tasks(
        str(source_dir),
        str(dest_dir),
        recursive=False,
        text_root=str(dest_dir),
    )
    assert len(tasks) == 1

    page_dir = dest_dir / "alpha"
    page_dir.mkdir()
    (page_dir / "page-01.png").write_bytes(VALID_PNG_BYTES)
    (page_dir / "page-01.txt").write_text("", encoding="utf-8")
    (page_dir / "page-02.png").write_bytes(VALID_PNG_BYTES)

    monkeypatch.setattr(pdf_pages, "get_pdf_page_count", lambda _path: 12)

    page_tasks = pdf_pages.build_pdf_page_tasks(
        tasks,
        extract_text=True,
        force=False,
    )
    by_page = {task.page_num: task for task in page_tasks}

    assert 1 not in by_page
    assert by_page[2].render_image is False
    assert by_page[2].render_text is True
    assert by_page[2].text_output_path == str(page_dir / "page-02.txt")
    assert by_page[3].page_stem == "page-03"


def test_extract_pdf_page_and_text_invokes_poppler(tmp_path, monkeypatch):
    from sneeze.tpn import pdf_pages

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("", encoding="utf-8")
    image_dir = tmp_path / "images"
    text_dir = tmp_path / "text"
    task = pdf_pages.PdfPageTask(
        source_path=str(pdf_path),
        dest_dir=str(image_dir),
        page_num=2,
        page_count=12,
        image_output_path=str(image_dir / "page-02.png"),
        render_image=True,
        render_text=True,
        text_output_path=str(text_dir / "page-02.txt"),
        dpi=300,
    )
    calls = []

    def fake_run(args, check, stdout, stderr, text):
        calls.append(args)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(pdf_pages.subprocess, "run", fake_run)

    assert pdf_pages.extract_pdf_page_and_text(task) is True
    assert calls == [
        [
            "pdftocairo",
            "-png",
            "-r",
            "300",
            "-singlefile",
            "-f",
            "2",
            "-l",
            "2",
            str(pdf_path),
            str(image_dir / "page-02"),
        ],
        [
            "pdftotext",
            "-f",
            "2",
            "-l",
            "2",
            str(pdf_path),
            str(text_dir / "page-02.txt"),
        ],
    ]


def test_pdf_to_pages_uses_public_cli_dest_dir(tmp_path, monkeypatch):
    from sneeze.tpn import pdf_pages
    from sneeze.tpn.commands import PdfToPages

    source_dir = tmp_path / "source"
    dest_dir = tmp_path / "pages"
    source_dir.mkdir()
    captured = {}

    def fake_build_pdf_pages_tasks(source, dest, **kwargs):
        captured["source"] = source
        captured["dest"] = dest
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(
        pdf_pages,
        "build_pdf_pages_tasks",
        fake_build_pdf_pages_tasks,
    )

    out = StringIO()
    command = PdfToPages(None, out, None)
    command._source_dir = str(source_dir)
    command.dest_dir = str(dest_dir)

    command.run()

    assert captured["source"] == str(source_dir)
    assert captured["dest"] == str(dest_dir)


def test_ocr_task_collection_skips_existing_outputs(tmp_path):
    from sneeze.tpn.ocr_vllm import collect_ocr_tasks

    source_dir = tmp_path / "pages"
    dest_dir = tmp_path / "ocr"
    source_dir.mkdir()
    dest_dir.mkdir()
    (source_dir / "page-1.png").write_bytes(VALID_PNG_BYTES)
    (source_dir / "page-2.png").write_bytes(VALID_PNG_BYTES)
    (dest_dir / "page-1.md").write_text("done", encoding="utf-8")

    tasks, pdf_count = collect_ocr_tasks(
        source_dir=str(source_dir),
        dest_dir=str(dest_dir),
        text_source_dir=None,
        output_ext="md",
        force=False,
        scan_parallelism=1,
    )

    assert pdf_count == 1
    assert len(tasks) == 1
    assert tasks[0].image_path == str(source_dir / "page-2.png")
    assert tasks[0].output_path == str(dest_dir / "page-2.md")


def test_ocr_full_collection_skips_only_complete_bundles(tmp_path):
    from sneeze.tpn.ocr_vllm import collect_ocr_full_items

    source_dir = tmp_path / "pages"
    dest_dir = tmp_path / "ocr"
    source_dir.mkdir()
    dest_dir.mkdir()
    (source_dir / "page-1.png").write_bytes(VALID_PNG_BYTES)
    (source_dir / "page-2.png").write_bytes(VALID_PNG_BYTES)
    for suffix in ("ocr", "md", "json"):
        (dest_dir / f"page-1.{suffix}").write_text(
            "done",
            encoding="utf-8",
        )
    (dest_dir / "page-2.ocr").write_text("ocr", encoding="utf-8")

    items, pdf_count = collect_ocr_full_items(
        source_dir=str(source_dir),
        dest_dir=str(dest_dir),
        text_source_dir=None,
        force=False,
        scan_parallelism=1,
    )

    assert pdf_count == 1
    assert len(items) == 1
    assert items[0].image_path == str(source_dir / "page-2.png")
    assert items[0].ocr_path == str(dest_dir / "page-2.ocr")
    assert items[0].markdown_path == str(dest_dir / "page-2.md")
    assert items[0].json_path == str(dest_dir / "page-2.json")
    assert items[0].need_ocr is False
    assert items[0].need_markdown is True
    assert items[0].need_json is True


def test_ocr_full_collection_rechecks_persisted_quality_failure(tmp_path):
    from sneeze.tpn.ocr_vllm import collect_ocr_full_items

    source_dir = tmp_path / "pages"
    dest_dir = tmp_path / "ocr"
    source_dir.mkdir()
    dest_dir.mkdir()
    (source_dir / "page-1.png").write_bytes(VALID_PNG_BYTES)
    (dest_dir / "page-1.ocr").write_text("ocr", encoding="utf-8")
    (dest_dir / "page-1.md").write_text("markdown", encoding="utf-8")
    (dest_dir / "page-1.json").write_text(
        json.dumps({"markdown_quality": {"status": "fail"}}),
        encoding="utf-8",
    )

    items, _pdf_count = collect_ocr_full_items(
        source_dir=str(source_dir),
        dest_dir=str(dest_dir),
        text_source_dir=None,
        force=False,
        scan_parallelism=1,
    )

    assert len(items) == 1
    assert items[0].need_ocr is False
    assert items[0].need_markdown is False
    assert items[0].need_json is True


def test_bundle_recheck_preserves_existing_generation_metadata(tmp_path):
    from sneeze.tpn.ocr_vllm import OcrFullItem, write_ocr_page_bundle

    image_path = tmp_path / "page.png"
    image_path.write_bytes(VALID_PNG_BYTES)
    ocr_path = tmp_path / "page.ocr"
    markdown_path = tmp_path / "page.md"
    json_path = tmp_path / "page.json"
    ocr_path.write_text("grounded evidence", encoding="utf-8")
    markdown_path.write_text("markdown", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "generation_metadata": {
                    "markdown": {"request_id": "preserved"}
                }
            }
        ),
        encoding="utf-8",
    )
    item = OcrFullItem(
        image_path=str(image_path),
        base="page",
        ocr_path=str(ocr_path),
        markdown_path=str(markdown_path),
        json_path=str(json_path),
        text_path=None,
        need_json=True,
    )

    payload = write_ocr_page_bundle(
        item,
        generation_metadata={"ocr": {"request_id": "new"}},
    )

    assert payload["generation_metadata"] == {
        "markdown": {"request_id": "preserved"},
        "ocr": {"request_id": "new"},
    }


def test_bundle_metadata_readers_ignore_non_object_json(tmp_path):
    from sneeze.tpn.ocr_vllm import (
        persisted_markdown_quality_status,
        read_existing_generation_metadata,
    )

    json_path = tmp_path / "page.json"
    json_path.write_text("[]", encoding="utf-8")

    assert persisted_markdown_quality_status(str(json_path)) is None
    assert read_existing_generation_metadata(str(json_path)) is None


def test_ocr_pdf_pages_uses_processor(tmp_path, monkeypatch):
    import sneeze.tpn.ocr_vllm as ocr_vllm
    from sneeze.tpn.commands import OcrPdfPages

    source_dir = tmp_path / "pages"
    dest_dir = tmp_path / "ocr"
    source_dir.mkdir()
    (source_dir / "page-1.png").write_bytes(VALID_PNG_BYTES)
    captured = {}

    class FakeProcessor:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def run(self, tasks):
            captured["tasks"] = list(tasks)

    monkeypatch.setattr(
        ocr_vllm,
        "DeepSeekOcrOfflineProcessor",
        FakeProcessor,
    )
    out = StringIO()
    command = OcrPdfPages(None, out, None)
    command._source_dir = str(source_dir)
    command.dest_dir = str(dest_dir)
    command.output_format = "md"
    command.prompt_mode = "markdown"
    command._scan_parallelism = 1
    command._max_tokens = 64
    command.temperature = "0.2"

    command.run()

    assert captured["kwargs"]["model_revision"] == (
        "9f30c71f441d010e5429c532364a86705536c53a"
    )
    assert captured["kwargs"]["max_tokens"] == 64
    assert captured["kwargs"]["temperature"] == 0.2
    assert captured["kwargs"]["window_size"] == 128
    assert len(captured["tasks"]) == 1
    assert captured["tasks"][0].output_path == str(dest_dir / "page-1.md")


def test_ocr_pdf_pages_full_mode_writes_bundle(tmp_path, monkeypatch):
    import sneeze.tpn.ocr_vllm as ocr_vllm
    from sneeze.tpn.commands import OcrPdfPages

    source_dir = tmp_path / "pages"
    dest_dir = tmp_path / "ocr"
    source_dir.mkdir()
    (source_dir / "page-1.png").write_bytes(VALID_PNG_BYTES)
    (source_dir / "page-1.txt").write_text("raw text", encoding="utf-8")
    captured = {"runs": []}

    class FakeProcessor:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.output_format = kwargs["output_format"]
            self.prompt_mode = kwargs["prompt_mode"]
            self.use_grounding = kwargs["use_grounding"]
            self.keep_grounding = kwargs["keep_grounding"]
            self.generation_metadata = {}

        def init_progressbar(self, total=None):
            captured["total"] = total

        def close_progressbar(self):
            captured["closed"] = True

        def update_progressbar(self, amount=1):
            captured["updates"] = captured.get("updates", 0) + amount

        def prepare_engine(self):
            captured["prepared"] = True
            return object(), SimpleNamespace(
                extra_args={"ngram_size": 10, "window_size": 50}
            )

        def shutdown_engine(self, llm):
            captured["shutdown"] = llm is not None

        def run(self, tasks, *, llm=None, sampling_params=None):
            tasks = list(tasks)
            captured["runs"].append(
                {
                    "prompt_mode": self.prompt_mode,
                    "use_grounding": self.use_grounding,
                    "keep_grounding": self.keep_grounding,
                    "outputs": [task.output_path for task in tasks],
                    "llm": llm is not None,
                    "sampling_params": sampling_params is not None,
                    "window_size": sampling_params.extra_args["window_size"],
                }
            )
            for task in tasks:
                with open(
                    task.output_path,
                    "w",
                    encoding="utf-8",
                ) as handle:
                    handle.write(f"{self.prompt_mode} output\n")
                self.generation_metadata[task.output_path] = {
                    "status": "completed"
                }
            self.update_progressbar(len(tasks))

    monkeypatch.setattr(
        ocr_vllm,
        "DeepSeekOcrOfflineProcessor",
        FakeProcessor,
    )
    out = StringIO()
    command = OcrPdfPages(None, out, None)
    command._source_dir = str(source_dir)
    command.dest_dir = str(dest_dir)
    command.output_format = "md"
    command.prompt_mode = "full"
    command._scan_parallelism = 1

    command.run()

    assert captured["prepared"] is True
    assert captured["total"] == 3
    assert captured["updates"] == 3
    assert [run["prompt_mode"] for run in captured["runs"]] == [
        "plain_ocr",
        "markdown",
    ]
    assert captured["runs"][0]["use_grounding"] is True
    assert captured["runs"][0]["keep_grounding"] is True
    assert captured["runs"][1]["use_grounding"] is False
    assert captured["runs"][1]["keep_grounding"] is False
    assert captured["runs"][0]["llm"] is True
    assert captured["runs"][1]["sampling_params"] is True
    assert captured["kwargs"]["window_size"] == 50
    assert [run["window_size"] for run in captured["runs"]] == [50, 128]
    payload = json.loads((dest_dir / "page-1.json").read_text())
    assert payload["ocr_text"] == "plain_ocr output\n"
    assert payload["markdown_text"] == "markdown output\n"
    assert payload["raw_text"] == "raw text"
    assert payload["image_dims"] == {"w": 1, "h": 1}


def test_full_mode_local_provider_releases_engine_before_markdown(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_provider as ocr_provider
    from sneeze.tpn.commands import OcrPdfPages
    from sneeze.tpn.ocr_vllm import OcrFullItem

    image_path = tmp_path / "page.png"
    image_path.write_bytes(VALID_PNG_BYTES)
    text_path = tmp_path / "page.txt"
    text_path.write_text("Table 8 rows Table 9 rows", encoding="utf-8")
    item = OcrFullItem(
        image_path=str(image_path),
        base="paper/page",
        ocr_path=str(tmp_path / "page.ocr"),
        markdown_path=str(tmp_path / "page.md"),
        json_path=str(tmp_path / "page.json"),
        text_path=str(text_path),
        need_ocr=True,
        need_markdown=True,
        need_json=True,
        image_dims={"w": 1, "h": 1},
        image_sha256=(
            "5c9b6e9f825b1ad005cbea8e70758bb8171d43bb66425b430254b9a0ef972d56"
        ),
    )
    events = []

    class FakeProcessor:
        def __init__(self):
            self.generation_metadata = {}

        def init_progressbar(self, total=None):
            events.append(("progress", total))

        def close_progressbar(self):
            events.append(("close", None))

        def update_progressbar(self, amount=1):
            events.append(("update", amount))

        def prepare_engine(self):
            events.append(("prepare", None))
            return object(), object()

        def shutdown_engine(self, llm):
            if llm is not None:
                events.append(("shutdown", None))

        def run(self, tasks, *, llm, sampling_params):
            tasks = list(tasks)
            events.append(("grounded", len(tasks)))
            assert all(task.output_path.endswith(".ocr") for task in tasks)
            for task in tasks:
                with open(task.output_path, "w", encoding="utf-8") as handle:
                    handle.write("Table 8 rows Table 9 rows")
                self.generation_metadata[task.output_path] = {
                    "status": "completed"
                }

    def fake_provider_batch(**kwargs):
        events.append(("provider", len(kwargs["pages"])))
        page = kwargs["pages"][0]
        return {
            "ok": True,
            "protocol_errors": [],
            "results": [
                {
                    "page_id": page["page_id"],
                    "valid": True,
                    "errors": [],
                    "output_text": "Table 8\nrows\n\nTable 9\nrows\n",
                    "normalized_metadata": {
                        "status": "completed",
                        "backend": "local_provider",
                        "termination": {
                            "eos": True,
                            "length": False,
                            "max_tokens": False,
                        },
                    },
                }
            ],
        }

    monkeypatch.setattr(
        ocr_provider,
        "run_local_provider_batch",
        fake_provider_batch,
    )
    provider = {
        "spec_path": str(tmp_path / "provider.json"),
        "spec": {
            "lanes": {
                "best": {
                    "prompt": "native two-stage",
                    "settings": {
                        "backend": "official-two-step-transformers"
                    },
                }
            }
        },
        "lane": "best",
        "run_dir": str(tmp_path / "evidence"),
    }
    command = OcrPdfPages(None, StringIO(), None)
    command._run_full_mode(
        FakeProcessor(),
        [item],
        markdown_provider=provider,
    )

    labels = [event[0] for event in events]
    assert labels.index("shutdown") < labels.index("provider")
    assert labels.count("grounded") == 1
    assert (tmp_path / "page.md").read_text() == (
        "Table 8\nrows\n\nTable 9\nrows\n"
    )
    payload = json.loads((tmp_path / "page.json").read_text())
    assert payload["generation_metadata"]["markdown"]["backend"] == (
        "local_provider"
    )
    promotion = payload["generation_metadata"]["markdown"]["promotion"]
    assert promotion["promoted"] is True
    assert promotion["candidate_sha256"] == promotion[
        "persisted_output_sha256"
    ]
    assert payload["markdown_quality"]["status"] == "pass"


def test_full_mode_local_provider_rejects_candidate_atomically(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_provider as ocr_provider
    from sneeze.tpn.commands import OcrPdfPages
    from sneeze.tpn.ocr_vllm import OcrFullItem

    image_path = tmp_path / "page.png"
    image_path.write_bytes(VALID_PNG_BYTES)
    ocr_path = tmp_path / "page.ocr"
    ocr_path.write_text("real page prose", encoding="utf-8")
    markdown_path = tmp_path / "page.md"
    markdown_path.write_text(
        "previous validated markdown\n",
        encoding="utf-8",
    )
    item = OcrFullItem(
        image_path=str(image_path),
        base="paper/page",
        ocr_path=str(ocr_path),
        markdown_path=str(markdown_path),
        json_path=str(tmp_path / "page.json"),
        text_path=None,
        need_ocr=False,
        need_markdown=True,
        need_json=True,
        image_dims={"w": 1, "h": 1},
        image_sha256=(
            "5c9b6e9f825b1ad005cbea8e70758bb8171d43bb66425b430254b9a0ef972d56"
        ),
    )

    class FakeProcessor:
        def __init__(self):
            self.generation_metadata = {}

        def init_progressbar(self, total=None):
            pass

        def close_progressbar(self):
            pass

        def update_progressbar(self, amount=1):
            pass

        def prepare_engine(self):
            raise AssertionError("provider-only Markdown must not start vLLM")

        def shutdown_engine(self, llm):
            pass

    def fake_provider_batch(**kwargs):
        page = kwargs["pages"][0]
        axis = " ".join(str(value) for value in range(20, 2226, 2))
        return {
            "ok": True,
            "protocol_errors": [],
            "results": [
                {
                    "page_id": page["page_id"],
                    "valid": True,
                    "errors": [],
                    "output_text": axis,
                    "normalized_metadata": {
                        "status": "completed",
                        "backend": "local_provider",
                        "termination": {
                            "eos": True,
                            "length": False,
                            "max_tokens": False,
                        },
                    },
                }
            ],
        }

    monkeypatch.setattr(
        ocr_provider,
        "run_local_provider_batch",
        fake_provider_batch,
    )
    provider = {
        "spec_path": str(tmp_path / "provider.json"),
        "spec": {
            "lanes": {
                "best": {
                    "prompt": "native two-stage",
                    "settings": {
                        "backend": "official-two-step-transformers"
                    },
                }
            }
        },
        "lane": "best",
        "run_dir": str(tmp_path / "evidence"),
    }
    command = OcrPdfPages(None, StringIO(), None)
    with pytest.raises(
        RuntimeError, match="runaway_monotonic_integer_axis"
    ) as exc_info:
        command._run_full_mode(
            FakeProcessor(),
            [item],
            markdown_provider=provider,
        )

    assert markdown_path.read_text() == "previous validated markdown\n"
    payload = json.loads((tmp_path / "page.json").read_text())
    markdown_metadata = payload["generation_metadata"]["markdown"]
    assert markdown_metadata["status"] == "error"
    assert markdown_metadata["validation"]["flags"] == [
        "runaway_monotonic_integer_axis"
    ]
    promotion = markdown_metadata["promotion"]
    assert promotion["promoted"] is False
    assert promotion["candidate_sha256"] != promotion[
        "persisted_output_sha256"
    ]
    assert promotion["persisted_output_sha256"] == hashlib.sha256(
        b"previous validated markdown\n"
    ).hexdigest()
    assert "1 page(s)" in str(exc_info.value)
    assert str(markdown_path) in str(exc_info.value)
    assert str(exc_info.value).count(str(markdown_path)) == 1


def test_markdown_provider_preparation_resolves_spec_and_protects_corpora(
    tmp_path,
    monkeypatch,
):
    from sneeze.command import CommandError

    from sneeze.tpn.commands import OcrPdfPages

    home = tmp_path / "home"
    home.mkdir()
    spec_path = home / "provider.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider_id": "fake-local-provider",
                "argv": [sys.executable],
                "egress": "none",
                "model": {
                    "requested": "example/model",
                    "revision": "a" * 40,
                },
                "lanes": {
                    "best": {
                        "prompt": "native pipeline",
                        "settings": {},
                    }
                },
                "secret_env_names": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    command = OcrPdfPages(None, StringIO(), None)
    command.markdown_provider_spec = "~/provider.json"
    command.markdown_provider_lane = "best"
    command.markdown_provider_run_dir = str(tmp_path / "evidence")

    provider = command._prepare_markdown_provider(
        prompt_mode="full",
        source_dir=str(tmp_path / "source"),
        dest_dir=str(tmp_path / "dest"),
    )
    assert provider["spec_path"] == str(spec_path.resolve())

    command.markdown_provider_run_dir = (
        "/home/trentn/src/cuphoton-papers/papers/"
        ".provider-evidence-must-not-be-created"
    )
    with pytest.raises(CommandError, match="protected canonical corpora"):
        command._prepare_markdown_provider(
            prompt_mode="full",
            source_dir=str(tmp_path / "source"),
            dest_dir=str(tmp_path / "dest"),
        )


def test_atomic_write_text_preserves_or_applies_normal_file_mode(tmp_path):
    from sneeze.tpn.commands import _atomic_write_text

    output_path = tmp_path / "page.md"
    previous_umask = os.umask(0o027)
    try:
        _atomic_write_text(str(output_path), "first\n")
    finally:
        os.umask(previous_umask)
    assert output_path.stat().st_mode & 0o777 == 0o640

    output_path.chmod(0o604)
    _atomic_write_text(str(output_path), "second\n")
    assert output_path.read_text(encoding="utf-8") == "second\n"
    assert output_path.stat().st_mode & 0o777 == 0o604


def test_ocr_pdf_pages_full_rejects_html_output(tmp_path):
    from sneeze.command import CommandError

    from sneeze.tpn.commands import OcrPdfPages

    source_dir = tmp_path / "pages"
    source_dir.mkdir()
    out = StringIO()
    command = OcrPdfPages(None, out, None)
    command._source_dir = str(source_dir)
    command.output_format = "html"
    command.prompt_mode = "full"

    with pytest.raises(CommandError, match="requires --output-format md"):
        command.run()


def test_bootstrap_commands_support_dry_run():
    from sneeze.tpn.commands import (
        DeepseekOcrCreateEnv,
        DeepseekOcrDownloadModel,
    )

    out = StringIO()
    command = DeepseekOcrCreateEnv(None, out, None)
    command.manager = "mamba"
    command.env_name = "ocr-env"
    command.python = "3.12"
    command.torch_backend = "cu130"
    command.dry_run = True
    command.run()
    text = out.getvalue()
    assert "mamba create -n ocr-env" in text
    assert "wheels.vllm.ai/nightly/cu130" in text

    out = StringIO()
    command = DeepseekOcrDownloadModel(None, out, None)
    command.manager = "mamba"
    command.env_name = "ocr-env"
    command.dry_run = True
    command.run()
    text = out.getvalue()
    assert "mamba run -n ocr-env python -c" in text
    assert "deepseek-ai/DeepSeek-OCR" in text


def test_command_docstrings_explain_workflow_prerequisites():
    from sneeze.tpn.commands import (
        DeepseekOcrCreateEnv,
        DeepseekOcrDownloadModel,
        OcrPdfPages,
        PdfToPages,
    )

    assert "pdftocairo" in PdfToPages.__doc__
    assert "pdftotext" in PdfToPages.__doc__
    assert "sne ocr-pdf-pages" in PdfToPages.__doc__
    assert "sne pdf-to-pages" in OcrPdfPages.__doc__
    assert "sne deepseek-ocr-create-env" in OcrPdfPages.__doc__
    assert "sne deepseek-ocr-download-model" in OcrPdfPages.__doc__
    assert "--prompt-mode full" in OcrPdfPages.__doc__
    assert "--output-format html" in OcrPdfPages.__doc__
    assert "sne ocr-pdf-pages" in DeepseekOcrCreateEnv.__doc__
    assert "sne ocr-pdf-pages" in DeepseekOcrDownloadModel.__doc__


def test_markdown_provider_options_belong_only_to_ocr_command():
    from sneeze.tpn.commands import OcrPdfPages, PdfToPages

    for argument in (
        "MarkdownProviderSpecArg",
        "MarkdownProviderLaneArg",
        "MarkdownProviderRunDirArg",
    ):
        assert hasattr(OcrPdfPages, argument)
        assert not hasattr(PdfToPages, argument)


def test_documented_commands_are_visible_to_sneeze_cli():
    from sneeze import cli as sneeze_cli

    cli = sneeze_cli.CLI(
        program_name="sneeze",
        module_names=["sneeze", "sneeze.tpn"],
        introspect=True,
        auto_plugins=False,
    )

    for command_name in (
        "pdf-to-pages",
        "pdfs-to-pages",
        "ocr-pdf-pages",
        "ocr-benchmark",
        "deepseek-ocr-create-env",
        "deepseek-ocr-download-model",
    ):
        assert command_name in cli._commands_by_name


def test_markdown_quality_detects_three_calibrated_failure_classes():
    from sneeze.tpn.ocr_vllm import analyze_markdown_quality

    axis = " ".join(str(value) for value in range(20, 2226, 2))
    quality = analyze_markdown_quality(
        axis,
        ocr_text="real page prose",
        raw_text="real page prose",
    )
    assert quality["flags"] == ["runaway_monotonic_integer_axis"]

    invented = " ".join(f"invented{index}" for index in range(50))
    quality = analyze_markdown_quality(
        "\n\n".join([invented] * 4),
        ocr_text="grounded evidence from the page",
        raw_text="grounded evidence from the page",
    )
    assert quality["flags"] == ["unsupported_long_span_repetition"]

    raw = "Table 8\n" + " ".join(f"row{index}" for index in range(80))
    raw += "\nTable 9\nmore table evidence"
    quality = analyze_markdown_quality(
        "| 1 | 2 |\n| 3 | 4 |",
        ocr_text=raw,
        raw_text=raw,
    )
    assert quality["flags"] == ["premature_table_truncation"]


def test_markdown_quality_accepts_supported_repetition():
    from sneeze.tpn.ocr_vllm import analyze_markdown_quality

    caption = " ".join(f"caption{index}" for index in range(50))
    quality = analyze_markdown_quality(
        "\n\n".join([caption] * 4),
        ocr_text=caption,
        raw_text=caption,
    )
    assert quality["status"] == "pass"
    assert quality["flags"] == []


def test_generation_metadata_keeps_vllm_completion_details(tmp_path):
    from sneeze.tpn.ocr_vllm import (
        DeepSeekOcrOfflineProcessor,
        OcrPageTask,
    )

    image_path = tmp_path / "page.png"
    image_path.write_bytes(VALID_PNG_BYTES)
    processor = DeepSeekOcrOfflineProcessor(
        model="deepseek-ai/DeepSeek-OCR",
        output_format="md",
        prompt_mode="markdown",
    )
    processor._engine_metadata = {
        "model": {"snapshot": "snapshot-sha"},
        "runtime": {"attention_backend_selected": "FLASH_ATTN"},
    }
    completion = SimpleNamespace(
        text="short output",
        token_ids=[10, 11, 12],
        finish_reason="length",
        stop_reason=None,
    )
    request_output = SimpleNamespace(
        request_id="request-1",
        prompt_token_ids=[1, 2],
    )
    sampling_params = SimpleNamespace(
        temperature=0.0,
        max_tokens=3,
        extra_args={"ngram_size": 10, "window_size": 50},
    )

    metadata = processor._generation_metadata(
        task=OcrPageTask(
            image_path=str(image_path),
            output_path=str(tmp_path / "page.md"),
        ),
        prompt="<image>\nConvert the document to markdown.",
        request_output=request_output,
        completion=completion,
        sampling_params=sampling_params,
    )

    assert metadata["image"]["dimensions"] == {"w": 1, "h": 1}
    assert metadata["prompt_mode"] == "markdown"
    assert metadata["generated_token_count"] == 3
    assert metadata["finish_reason"] == "length"
    assert metadata["status"] == "completed"
    assert metadata["termination"] == {
        "eos": False,
        "length": True,
        "max_tokens": True,
    }
    assert metadata["runtime"]["attention_backend_selected"] == "FLASH_ATTN"
    assert metadata["logits_processor_extra_args"] == {
        "ngram_size": 10,
        "window_size": 50,
    }

    completion.token_ids = [10, 11]
    metadata = processor._generation_metadata(
        task=OcrPageTask(
            image_path=str(image_path),
            output_path=str(tmp_path / "page.md"),
        ),
        prompt="<image>\nConvert the document to markdown.",
        request_output=request_output,
        completion=completion,
        sampling_params=sampling_params,
    )
    assert metadata["termination"] == {
        "eos": False,
        "length": True,
        "max_tokens": False,
    }


def test_full_mode_preserves_and_rejects_flagged_markdown(tmp_path):
    from sneeze.tpn.commands import OcrPdfPages
    from sneeze.tpn.ocr_vllm import OcrFullItem

    image_path = tmp_path / "page.png"
    image_path.write_bytes(VALID_PNG_BYTES)
    item = OcrFullItem(
        image_path=str(image_path),
        base="page",
        ocr_path=str(tmp_path / "page.ocr"),
        markdown_path=str(tmp_path / "page.md"),
        json_path=str(tmp_path / "page.json"),
        text_path=str(tmp_path / "page.txt"),
        need_ocr=True,
        need_markdown=True,
        need_json=True,
        image_dims={"w": 1, "h": 1},
    )
    (tmp_path / "page.txt").write_text("real prose", encoding="utf-8")

    class FakeProcessor:
        generation_metadata = {}

        def init_progressbar(self, total=None):
            pass

        def prepare_engine(self):
            return object(), SimpleNamespace(
                extra_args={"ngram_size": 10, "window_size": 50}
            )

        def run(self, tasks, *, llm, sampling_params):
            for task in tasks:
                if task.output_path.endswith(".ocr"):
                    text = "real prose"
                else:
                    text = " ".join(
                        str(value) for value in range(20, 2226, 2)
                    )
                with open(task.output_path, "w", encoding="utf-8") as handle:
                    handle.write(text)

        def update_progressbar(self, amount=1):
            pass

        def close_progressbar(self):
            pass

        def shutdown_engine(self, llm):
            pass

    command = OcrPdfPages(None, StringIO(), None)
    with pytest.raises(
        RuntimeError,
        match="page validation failed",
    ) as exc_info:
        command._run_full_mode(FakeProcessor(), [item])

    assert "grounded_ocr_generation_failed" in str(exc_info.value)
    assert "markdown_generation_failed" in str(exc_info.value)

    payload = json.loads((tmp_path / "page.json").read_text())
    assert payload["markdown_quality"]["status"] == "fail"
    assert payload["markdown_quality"]["flags"] == [
        "runaway_monotonic_integer_axis"
    ]
